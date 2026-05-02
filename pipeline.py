"""
OCR + RAG Pipeline Module (Enhanced)

This module handles:
- PDF to image conversion
- Tesseract OCR text extraction with confidence scoring
- Currency format preservation (USD, INR, EUR)
- Multi-column layout detection
- Text chunking with overlap
- Embedding generation using sentence-transformers
- FAISS vector indexing and retrieval
- Local LLM generation using TinyLlama
- Batch PDF processing support
"""

import os
import re
import tempfile
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
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
class OCRRegion:
    """Represents an OCR region with confidence information."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int] = None  # x, y, width, height

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 60.0


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    page_num: int
    chunk_index: int
    source_file: str = ""
    has_warning: bool = False
    warning_message: str = ""

    def __str__(self):
        warning = " ⚠️" if self.has_warning else ""
        return f"[Page {self.page_num}, Chunk {self.chunk_index}]{warning}: {self.text[:100]}..."


class CurrencyPreserver:
    """Handles preservation and normalization of currency formats."""

    # Currency patterns for USD, INR, EUR and common formats
    CURRENCY_PATTERNS = [
        # USD formats: $1,234.56 or $ 1,234.56 or USD 1,234.56
        (r'\$\s*[\d,]+\.?\d*', 'USD'),
        (r'USD\s*[\d,]+\.?\d*', 'USD'),
        (r'US\$\s*[\d,]+\.?\d*', 'USD'),
        # EUR formats: €1,234.56 or EUR 1,234.56
        (r'€\s*[\d,]+\.?\d*', 'EUR'),
        (r'EUR\s*[\d,]+\.?\d*', 'EUR'),
        # INR formats: ₹1,23,456.78 or Rs. 1,23,456.78 or INR 1,23,456.78
        (r'₹\s*[\d,]+\.?\d*', 'INR'),
        (r'Rs\.?\s*[\d,]+\.?\d*', 'INR'),
        (r'INR\s*[\d,]+\.?\d*', 'INR'),
        # Generic number formats that might be currency
        (r'[\d,]+\.\d{2}\b', 'NUMBER'),
    ]

    # Common OCR corruptions for currency symbols
    OCR_CORRECTIONS = {
        # USD
        r'\$\s+': '$',
        r'S\s*\$': '$',
        r'\$\s*,': '$',
        # EUR
        r'€\s+': '€',
        r'E\s*€': '€',
        # INR - Rupee symbol often gets corrupted
        r'[Rr][Ss]\.?\s*': 'Rs. ',
        r'INR\s+': 'INR ',
        # Number formatting
        r'(\d)\s+,\s*(\d)': r'\1,\2',  # Fix "1 , 234" -> "1,234"
        r'(\d)\s*,\s+(\d)': r'\1,\2',
        r'(\d)\s+\.\s*(\d)': r'\1.\2',  # Fix "1 . 23" -> "1.23"
    }

    @classmethod
    def fix_currency_formats(cls, text: str) -> str:
        """Fix common OCR corruptions in currency formats."""
        result = text

        # Apply OCR corrections
        for pattern, replacement in cls.OCR_CORRECTIONS.items():
            result = re.sub(pattern, replacement, result)

        # Normalize spacing around currency symbols
        result = re.sub(r'(\$|€|₹)\s+(\d)', r'\1\2', result)

        # Fix INR lakhs/crores format (Indian numbering: 1,23,456.78)
        # Preserve the Indian format if detected
        result = re.sub(r'(\d),(\d{2}),(\d{3})', r'\1,\2,\3', result)

        return result

    @classmethod
    def extract_currency_values(cls, text: str) -> List[Dict]:
        """Extract all currency values from text with their types."""
        currencies = []

        for pattern, currency_type in cls.CURRENCY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                currencies.append({
                    'value': match.group(),
                    'type': currency_type,
                    'position': match.span()
                })

        return currencies


class MultiColumnDetector:
    """Detects and handles multi-column layouts in documents."""

    @staticmethod
    def detect_columns(image: Image.Image) -> int:
        """
        Detect number of columns in an image using Tesseract's layout analysis.
        Returns estimated number of columns (1, 2, or 3).
        """
        try:
            # Get bounding box data from Tesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            if not data['left']:
                return 1

            # Get valid text blocks
            valid_indices = [i for i, conf in enumerate(data['conf'])
                           if int(conf) > 0 and data['text'][i].strip()]

            if len(valid_indices) < 10:
                return 1

            # Analyze x-positions of text blocks
            x_positions = [data['left'][i] for i in valid_indices]
            width = image.width

            # Check for distinct column regions
            left_count = sum(1 for x in x_positions if x < width * 0.35)
            middle_count = sum(1 for x in x_positions if width * 0.35 <= x < width * 0.65)
            right_count = sum(1 for x in x_positions if x >= width * 0.65)

            total = len(x_positions)

            # Determine column count based on distribution
            if left_count > total * 0.4 and right_count > total * 0.4:
                if middle_count > total * 0.15:
                    return 3
                return 2

            return 1

        except Exception:
            return 1

    @staticmethod
    def extract_columns(image: Image.Image, num_columns: int) -> List[str]:
        """
        Extract text from each column separately.
        """
        if num_columns == 1:
            return [pytesseract.image_to_string(image)]

        width = image.width
        height = image.height
        column_texts = []

        # Split image into columns
        column_width = width // num_columns

        for i in range(num_columns):
            left = i * column_width
            right = (i + 1) * column_width if i < num_columns - 1 else width

            # Crop column region
            column_img = image.crop((left, 0, right, height))

            # Extract text from column
            text = pytesseract.image_to_string(column_img)
            column_texts.append(text.strip())

        return column_texts


class OCRProcessor:
    """Handles PDF to text conversion using Tesseract OCR with enhanced features."""

    def __init__(self, dpi: int = 300, lang: str = 'eng'):
        """
        Initialize OCR processor.

        Args:
            dpi: DPI for PDF to image conversion (higher = better quality, slower)
            lang: Tesseract language code
        """
        self.dpi = dpi
        self.lang = lang
        self.currency_preserver = CurrencyPreserver()
        self.column_detector = MultiColumnDetector()

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL Images."""
        images = convert_from_path(pdf_path, dpi=self.dpi)
        return images

    def extract_text_with_confidence(self, image: Image.Image) -> Tuple[str, List[OCRRegion], float]:
        """
        Extract text from image with confidence scores.

        Returns:
            Tuple of (full_text, list_of_regions, average_confidence)
        """
        # Get detailed OCR data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=self.lang)

        regions = []
        confidences = []

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])

            if text and conf > 0:
                region = OCRRegion(
                    text=text,
                    confidence=conf,
                    bbox=(data['left'][i], data['top'][i],
                          data['width'][i], data['height'][i])
                )
                regions.append(region)
                confidences.append(conf)

        # Get full text
        full_text = pytesseract.image_to_string(image, lang=self.lang)

        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return full_text, regions, avg_confidence

    def extract_text_from_image(self, image: Image.Image) -> Tuple[str, float, List[str]]:
        """
        Extract text from a single image using Tesseract with enhancements.

        Returns:
            Tuple of (text, confidence, warnings)
        """
        warnings = []

        # Detect column layout
        num_columns = self.column_detector.detect_columns(image)

        if num_columns > 1:
            # Extract text from each column
            column_texts = self.column_detector.extract_columns(image, num_columns)
            text = "\n\n--- Column Break ---\n\n".join(column_texts)
        else:
            text, regions, avg_confidence = self.extract_text_with_confidence(image)

            # Check for low confidence regions
            low_conf_regions = [r for r in regions if r.is_low_confidence]
            if low_conf_regions:
                warnings.append(f"⚠️ {len(low_conf_regions)} regions with low OCR confidence detected")

        # Get overall confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Fix currency formats
        text = self.currency_preserver.fix_currency_formats(text)

        # Check for potential currency corruption
        if re.search(r'[S$]\s+\d|Rs\s+\d{1,2}\s+,', text):
            warnings.append("⚠️ Possible currency format corruption detected")

        return text.strip(), avg_confidence, warnings

    def process_pdf(self, pdf_path: str, progress_callback=None) -> List[Tuple[int, str, float, List[str]]]:
        """
        Process entire PDF and extract text from all pages.

        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            List of (page_number, extracted_text, confidence, warnings) tuples
        """
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        total_pages = len(images)

        results = []
        for i, image in enumerate(images):
            text, confidence, warnings = self.extract_text_from_image(image)
            results.append((i + 1, text, confidence, warnings))  # 1-indexed page numbers

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
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, page_num: int, source_file: str = "",
                   has_warning: bool = False, warning_message: str = "") -> List[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            page_num: Page number for metadata
            source_file: Source filename
            has_warning: Whether this text has OCR warnings
            warning_message: Warning message if any

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
                        chunk_index=chunk_index,
                        source_file=source_file,
                        has_warning=has_warning,
                        warning_message=warning_message
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
                                chunk_index=chunk_index,
                                source_file=source_file,
                                has_warning=has_warning,
                                warning_message=warning_message
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
                    chunk_index=chunk_index,
                    source_file=source_file,
                    has_warning=has_warning,
                    warning_message=warning_message
                ))
                chunk_index += 1

                # Start new chunk with overlap
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
                chunk_index=chunk_index,
                source_file=source_file,
                has_warning=has_warning,
                warning_message=warning_message
            ))

        return chunks

    def chunk_pages(self, pages: List[Tuple[int, str, float, List[str]]],
                    source_file: str = "") -> List[Chunk]:
        """
        Chunk text from multiple pages.

        Args:
            pages: List of (page_number, text, confidence, warnings) tuples
            source_file: Source filename

        Returns:
            List of all Chunk objects
        """
        all_chunks = []
        for page_num, text, confidence, warnings in pages:
            has_warning = len(warnings) > 0 or confidence < 70.0
            warning_message = "; ".join(warnings) if warnings else ""
            if confidence < 70.0:
                warning_message = f"Low OCR confidence: {confidence:.1f}%. " + warning_message

            page_chunks = self.chunk_text(
                text, page_num, source_file,
                has_warning, warning_message
            )
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

    def add_to_index(self, embeddings: np.ndarray, chunks: List[Chunk]):
        """
        Add more embeddings to existing index (for batch upload).

        Args:
            embeddings: numpy array of embeddings
            chunks: List of corresponding Chunk objects
        """
        if self.index is None:
            self.build_index(embeddings, chunks)
            return

        self.chunks.extend(chunks)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
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

    def get_stats(self) -> Dict:
        """Get statistics about the indexed documents."""
        if not self.chunks:
            return {"total_chunks": 0, "files": [], "pages": 0}

        files = list(set(c.source_file for c in self.chunks if c.source_file))
        pages = len(set((c.source_file, c.page_num) for c in self.chunks))
        warnings = sum(1 for c in self.chunks if c.has_warning)

        return {
            "total_chunks": len(self.chunks),
            "files": files,
            "total_files": len(files),
            "pages": pages,
            "chunks_with_warnings": warnings
        }


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
            source = f" ({chunk.source_file})" if chunk.source_file else ""
            warning = " [⚠️ Low confidence]" if chunk.has_warning else ""
            context_parts.append(f"[Page {chunk.page_num}{source}{warning}]: {chunk.text}")

        context = "\n\n".join(context_parts)

        # Build prompt using TinyLlama chat format
        system_prompt = """You are a helpful financial document assistant that answers questions based on the provided document context.
You are expert at reading invoices, balance sheets, and financial reports.
Only use information from the context to answer. If the answer is not in the context, say so.
Pay attention to currency formats (USD $, EUR €, INR ₹/Rs.) and preserve them in your answers.
Be concise and accurate in your responses."""

        user_prompt = f"""Context from the document(s):
{context}

Question: {query}

Please answer the question based only on the provided context. If you see any warnings about low confidence OCR, mention that the data might need verification."""

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
        self.indexed_files: List[str] = []

    def load_models(self, progress_callback=None):
        """Pre-load all models."""
        if progress_callback:
            progress_callback("Loading embedding model...")
        self.embedder.load()

        if progress_callback:
            progress_callback("Loading LLM model...")
        self.llm.load()

    def ingest_pdf(self, pdf_path: str, progress_callback=None, append: bool = False) -> Tuple[int, List[str]]:
        """
        Ingest a PDF file into the RAG system.

        Args:
            pdf_path: Path to PDF file
            progress_callback: Optional callback for progress updates
            append: If True, add to existing index instead of replacing

        Returns:
            Tuple of (number of chunks created, list of warnings)
        """
        filename = os.path.basename(pdf_path)
        all_warnings = []

        # Clear if not appending
        if not append:
            self.vector_store = VectorStore()
            self.indexed_files = []

        # Step 1: OCR
        if progress_callback:
            progress_callback("Extracting text from PDF pages...")

        def ocr_progress(current, total):
            if progress_callback:
                progress_callback(f"OCR processing page {current}/{total}...")

        pages = self.ocr.process_pdf(pdf_path, ocr_progress)

        # Collect warnings from pages
        for page_num, text, confidence, warnings in pages:
            for w in warnings:
                all_warnings.append(f"Page {page_num}: {w}")
            if confidence < 70.0:
                all_warnings.append(f"Page {page_num}: Low OCR confidence ({confidence:.1f}%)")

        # Step 2: Chunking
        if progress_callback:
            progress_callback("Chunking text...")

        chunks = self.chunker.chunk_pages(pages, source_file=filename)

        if not chunks and not append:
            raise ValueError("No text could be extracted from the PDF")

        if chunks:
            # Step 3: Embedding
            if progress_callback:
                progress_callback(f"Embedding {len(chunks)} chunks...")

            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.embed(chunk_texts, show_progress=False)

            # Step 4: Index
            if progress_callback:
                progress_callback("Building vector index...")

            self.vector_store.add_to_index(embeddings, chunks)
            self.indexed_files.append(filename)

        self.is_indexed = True

        if progress_callback:
            total_chunks = len(self.vector_store.chunks)
            progress_callback(f"Indexing complete! {total_chunks} total chunks ready for querying.")

        return len(chunks), all_warnings

    def ingest_multiple_pdfs(self, pdf_paths: List[str], progress_callback=None) -> Tuple[int, List[str]]:
        """
        Ingest multiple PDF files into a single index.

        Args:
            pdf_paths: List of paths to PDF files
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (total chunks created, list of all warnings)
        """
        total_chunks = 0
        all_warnings = []

        # Clear existing index
        self.vector_store = VectorStore()
        self.indexed_files = []

        for i, pdf_path in enumerate(pdf_paths):
            if progress_callback:
                progress_callback(f"Processing file {i+1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")

            chunks, warnings = self.ingest_pdf(pdf_path, progress_callback, append=True)
            total_chunks += chunks
            all_warnings.extend(warnings)

        return total_chunks, all_warnings

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

    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        stats = self.vector_store.get_stats()
        stats["indexed_files"] = self.indexed_files
        return stats

    def clear(self):
        """Clear the current index."""
        self.vector_store = VectorStore()
        self.is_indexed = False
        self.indexed_files = []


# Utility function for standalone testing
def main():
    """Test the pipeline with a sample PDF."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <pdf_path> [pdf_path2 ...]")
        return

    pdf_paths = sys.argv[1:]

    print("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline()

    if len(pdf_paths) == 1:
        print(f"\nIngesting PDF: {pdf_paths[0]}")
        num_chunks, warnings = rag_pipeline.ingest_pdf(pdf_paths[0], lambda msg: print(f"  {msg}"))
    else:
        print(f"\nIngesting {len(pdf_paths)} PDFs...")
        num_chunks, warnings = rag_pipeline.ingest_multiple_pdfs(pdf_paths, lambda msg: print(f"  {msg}"))

    print(f"\nCreated {num_chunks} chunks")
    if warnings:
        print(f"\nWarnings:")
        for w in warnings:
            print(f"  {w}")

    print("\nReady for questions! (type 'quit' to exit)")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'quit':
            break

        answer, chunks = rag_pipeline.query(question)
        print(f"\nAnswer: {answer}")
        print("\nSource chunks:")
        for chunk, score in chunks:
            warning = " ⚠️" if chunk.has_warning else ""
            print(f"  - [Page {chunk.page_num}]{warning} (score: {score:.3f}): {chunk.text[:100]}...")


if __name__ == "__main__":
    main()
