"""
OCR + RAG Pipeline Module

This module handles:
- PDF to image conversion
- Tesseract OCR text extraction
- Text chunking with overlap
- Embedding generation using sentence-transformers
- FAISS vector indexing and retrieval
- Local LLM generation using TinyLlama
"""

import os
import tempfile
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import tiktoken

# PDF and OCR
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Embeddings and Vector Store
from sentence_transformers import SentenceTransformer
import faiss

# LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    page_num: int
    chunk_index: int

    def __str__(self):
        return f"[Page {self.page_num}, Chunk {self.chunk_index}]: {self.text[:100]}..."


class OCRProcessor:
    """Handles PDF to text conversion using Tesseract OCR."""

    def __init__(self, dpi: int = 300, lang: str = 'eng'):
        """
        Initialize OCR processor.

        Args:
            dpi: DPI for PDF to image conversion (higher = better quality, slower)
            lang: Tesseract language code
        """
        self.dpi = dpi
        self.lang = lang

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL Images."""
        images = convert_from_path(pdf_path, dpi=self.dpi)
        return images

    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from a single image using Tesseract."""
        text = pytesseract.image_to_string(image, lang=self.lang)
        return text.strip()

    def process_pdf(self, pdf_path: str, progress_callback=None) -> List[Tuple[int, str]]:
        """
        Process entire PDF and extract text from all pages.

        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            List of (page_number, extracted_text) tuples
        """
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        total_pages = len(images)

        results = []
        for i, image in enumerate(images):
            text = self.extract_text_from_image(image)
            results.append((i + 1, text))  # 1-indexed page numbers

            if progress_callback:
                progress_callback(i + 1, total_pages)

        return results


class TextChunker:
    """Handles text chunking with token-based sizing and overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize text chunker.

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Use cl100k_base encoding (same as GPT-4)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple implementation)."""
        # Simple sentence splitting - handles common cases
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, page_num: int) -> List[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            page_num: Page number for metadata

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        page_num=page_num,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0

                for word in words:
                    word_tokens = self._count_tokens(word + ' ')
                    if temp_tokens + word_tokens > self.chunk_size:
                        if temp_chunk:
                            chunk_text = ' '.join(temp_chunk)
                            chunks.append(Chunk(
                                text=chunk_text,
                                page_num=page_num,
                                chunk_index=chunk_index
                            ))
                            chunk_index += 1
                            # Keep overlap
                            overlap_words = temp_chunk[-10:]  # Approximate overlap
                            temp_chunk = overlap_words
                            temp_tokens = self._count_tokens(' '.join(temp_chunk))
                    temp_chunk.append(word)
                    temp_tokens += word_tokens

                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
                continue

            # Check if adding sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    page_num=page_num,
                    chunk_index=chunk_index
                ))
                chunk_index += 1

                # Start new chunk with overlap
                # Find sentences to keep for overlap
                overlap_text = ''
                overlap_sentences = []
                overlap_tokens = 0

                for sent in reversed(current_chunk):
                    sent_tokens = self._count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                page_num=page_num,
                chunk_index=chunk_index
            ))

        return chunks

    def chunk_pages(self, pages: List[Tuple[int, str]]) -> List[Chunk]:
        """
        Chunk text from multiple pages.

        Args:
            pages: List of (page_number, text) tuples

        Returns:
            List of all Chunk objects
        """
        all_chunks = []
        for page_num, text in pages:
            page_chunks = self.chunk_text(text, page_num)
            all_chunks.extend(page_chunks)
        return all_chunks


class EmbeddingModel:
    """Handles text embedding using sentence-transformers."""

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.model = None

    def load(self):
        """Load the embedding model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self

    def embed(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings
        """
        if self.model is None:
            self.load()

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed([query], show_progress=False)[0]


class VectorStore:
    """FAISS-based vector store for similarity search."""

    def __init__(self, dimension: int = 384):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self.index = None
        self.chunks: List[Chunk] = []

    def build_index(self, embeddings: np.ndarray, chunks: List[Chunk]):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: numpy array of embeddings
            chunks: List of corresponding Chunk objects
        """
        self.chunks = chunks

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create index using Inner Product (equivalent to cosine after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (Chunk, similarity_score) tuples
        """
        if self.index is None or len(self.chunks) == 0:
            return []

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distances[0][i])))

        return results


class LocalLLM:
    """Local LLM for generation using TinyLlama."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize local LLM.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None

    def load(self):
        """Load the LLM model."""
        if self.pipeline is None:
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model with appropriate settings
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=device if device == "cuda" else -1,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
        return self

    def generate(self, query: str, context_chunks: List[Tuple[Chunk, float]]) -> str:
        """
        Generate response using retrieved context.

        Args:
            query: User query
            context_chunks: List of (Chunk, score) tuples from retrieval

        Returns:
            Generated response text
        """
        if self.pipeline is None:
            self.load()

        # Build context from chunks
        context_parts = []
        for chunk, score in context_chunks:
            context_parts.append(f"[Page {chunk.page_num}]: {chunk.text}")

        context = "\n\n".join(context_parts)

        # Build prompt using TinyLlama chat format
        system_prompt = """You are a helpful assistant that answers questions based on the provided document context.
Only use information from the context to answer. If the answer is not in the context, say so.
Be concise and accurate in your responses."""

        user_prompt = f"""Context from the document:
{context}

Question: {query}

Please answer the question based only on the provided context."""

        # TinyLlama chat format
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"

        # Generate response
        outputs = self.pipeline(prompt, return_full_text=False)
        response = outputs[0]['generated_text'].strip()

        return response


class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.

        Args:
            chunk_size: Token size for chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Embedding model name
            llm_model: LLM model name
            top_k: Number of chunks to retrieve
        """
        self.ocr = OCRProcessor()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingModel(embedding_model)
        self.vector_store = VectorStore()
        self.llm = LocalLLM(llm_model)
        self.top_k = top_k

        self.is_indexed = False
        self.current_pdf_name = None

    def load_models(self, progress_callback=None):
        """Pre-load all models."""
        if progress_callback:
            progress_callback("Loading embedding model...")
        self.embedder.load()

        if progress_callback:
            progress_callback("Loading LLM model...")
        self.llm.load()

    def ingest_pdf(self, pdf_path: str, progress_callback=None) -> int:
        """
        Ingest a PDF file into the RAG system.

        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates

        Returns:
            Number of chunks created
        """
        self.current_pdf_name = os.path.basename(pdf_path)

        # Step 1: OCR
        if progress_callback:
            progress_callback("Extracting text from PDF pages...")

        def ocr_progress(current, total):
            if progress_callback:
                progress_callback(f"OCR processing page {current}/{total}...")

        pages = self.ocr.process_pdf(pdf_path, ocr_progress)

        # Step 2: Chunking
        if progress_callback:
            progress_callback("Chunking text...")

        chunks = self.chunker.chunk_pages(pages)

        if not chunks:
            raise ValueError("No text could be extracted from the PDF")

        # Step 3: Embedding
        if progress_callback:
            progress_callback(f"Embedding {len(chunks)} chunks...")

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(chunk_texts, show_progress=False)

        # Step 4: Index
        if progress_callback:
            progress_callback("Building vector index...")

        self.vector_store.build_index(embeddings, chunks)
        self.is_indexed = True

        if progress_callback:
            progress_callback(f"Indexing complete! {len(chunks)} chunks ready for querying.")

        return len(chunks)

    def query(self, question: str) -> Tuple[str, List[Tuple[Chunk, float]]]:
        """
        Query the RAG system.

        Args:
            question: User question

        Returns:
            Tuple of (generated_answer, retrieved_chunks)
        """
        if not self.is_indexed:
            raise ValueError("No document has been indexed yet. Please upload a PDF first.")

        # Retrieve relevant chunks
        query_embedding = self.embedder.embed_query(question)
        retrieved_chunks = self.vector_store.search(query_embedding, k=self.top_k)

        # Generate response
        answer = self.llm.generate(question, retrieved_chunks)

        return answer, retrieved_chunks

    def clear(self):
        """Clear the current index."""
        self.vector_store = VectorStore()
        self.is_indexed = False
        self.current_pdf_name = None


# Utility function for standalone testing
def main():
    """Test the pipeline with a sample PDF."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <pdf_path>")
        return

    pdf_path = sys.argv[1]

    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline()

    print(f"\nIngesting PDF: {pdf_path}")
    num_chunks = pipeline.ingest_pdf(pdf_path, lambda msg: print(f"  {msg}"))
    print(f"\nCreated {num_chunks} chunks")

    print("\nReady for questions! (type 'quit' to exit)")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'quit':
            break

        answer, chunks = pipeline.query(question)
        print(f"\nAnswer: {answer}")
        print("\nSource chunks:")
        for chunk, score in chunks:
            print(f"  - [Page {chunk.page_num}] (score: {score:.3f}): {chunk.text[:100]}...")


if __name__ == "__main__":
    main()
