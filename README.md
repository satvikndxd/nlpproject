# PDF RAG Chat

A fully local, privacy-preserving RAG (Retrieval-Augmented Generation) pipeline that lets you chat with PDF documents using Tesseract OCR and local AI models.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Local Processing**: All data stays on your machine - no external API calls
- **OCR Support**: Extract text from scanned/image-based PDFs using Tesseract
- **Semantic Search**: Find relevant passages using sentence-transformers embeddings
- **Local LLM**: Generate answers using TinyLlama-1.1B-Chat
- **Web Interface**: User-friendly Gradio UI with PDF upload and chat
- **Source Attribution**: See which document chunks were used to generate answers

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  PDF Upload │────▶│  pdf2image   │────▶│  Tesseract    │
└─────────────┘     │  (Pages→Img) │     │  OCR (→Text)  │
                    └──────────────┘     └───────┬───────┘
                                                 │
                    ┌──────────────┐     ┌───────▼───────┐
                    │    FAISS     │◀────│   Chunking    │
                    │  (Vector DB) │     │  (500 tokens) │
                    └──────┬───────┘     └───────────────┘
                           │                     │
                    ┌──────▼───────┐     ┌───────▼───────┐
                    │  Top-K       │     │  Embeddings   │
                    │  Retrieval   │     │  (MiniLM-L6)  │
                    └──────┬───────┘     └───────────────┘
                           │
                    ┌──────▼───────┐     ┌───────────────┐
                    │  TinyLlama   │────▶│   Response    │
                    │  Generation  │     │   + Sources   │
                    └──────────────┘     └───────────────┘
```

## Prerequisites

### System Dependencies

**Tesseract OCR** must be installed on your system:

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng poppler-utils
```

#### macOS (Homebrew)
```bash
brew install tesseract poppler
```

#### Windows
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to PATH
3. Install poppler: https://github.com/osber/poppler-windows/releases

### Python Requirements

- Python 3.10 or higher
- CUDA-compatible GPU (optional, for faster inference)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pdf-rag-chat
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Tesseract installation**
   ```bash
   tesseract --version
   ```

## Usage

### Web UI (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

**How to use:**
1. Click "Drop your PDF here" or drag a PDF file
2. Wait for processing (OCR + indexing)
3. Type questions in the chat box
4. View answers and source chunks

### Command Line

For testing or scripting, use the pipeline directly:

```bash
python pipeline.py path/to/document.pdf
```

This starts an interactive session where you can type questions.

### Python API

```python
from pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    chunk_size=500,
    chunk_overlap=50,
    top_k=5
)

# Ingest a PDF
num_chunks = pipeline.ingest_pdf("document.pdf")
print(f"Created {num_chunks} chunks")

# Query
answer, sources = pipeline.query("What is the main topic?")
print(f"Answer: {answer}")

for chunk, score in sources:
    print(f"[Page {chunk.page_num}] Score: {score:.3f}")
```

## Configuration

Key parameters can be adjusted in the pipeline:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 500 | Maximum tokens per text chunk |
| `chunk_overlap` | 50 | Overlapping tokens between chunks |
| `top_k` | 5 | Number of chunks to retrieve |
| `dpi` | 300 | DPI for PDF to image conversion |

## Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Embeddings | `all-MiniLM-L6-v2` | ~80MB | Semantic text embeddings |
| LLM | `TinyLlama-1.1B-Chat` | ~2GB | Answer generation |
| OCR | Tesseract | System | Text extraction |

## Project Structure

```
pdf-rag-chat/
├── app.py              # Gradio web interface
├── pipeline.py         # Core RAG pipeline logic
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Troubleshooting

### "Tesseract not found" error
- Ensure Tesseract is installed and in your PATH
- On Linux: `which tesseract` should return a path
- Set path manually: `pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'`

### Out of memory errors
- Try a smaller chunk size (e.g., 300)
- Process smaller PDFs
- Use CPU instead of GPU if GPU memory is limited

### Slow processing
- First run downloads models (~2GB) - be patient
- Subsequent runs use cached models
- Consider using GPU for faster inference

### Poor OCR quality
- Increase DPI in `OCRProcessor(dpi=400)`
- Ensure PDF images are clear and well-lit
- Try different Tesseract language packs

## Performance Tips

1. **GPU Acceleration**: If you have an NVIDIA GPU with CUDA, the LLM will automatically use it
2. **Batch Processing**: For multiple PDFs, keep the pipeline instance alive to reuse loaded models
3. **Chunk Size**: Smaller chunks = more precise retrieval but more searches; larger chunks = better context but less precise

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [TinyLlama](https://github.com/jzhang38/TinyLlama)
- [Gradio](https://gradio.app/)
