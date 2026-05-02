"""
Gradio Web UI for PDF RAG Pipeline (Enhanced)

This module provides a user-friendly web interface for:
- Uploading single or multiple PDF documents (batch upload)
- Chatting with the document content
- Viewing source chunks with OCR confidence warnings
- Pre-built example queries for financial documents
"""

import gradio as gr
import tempfile
import os
from typing import List, Tuple, Optional
import time
import threading

from pipeline import RAGPipeline, Chunk

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None
models_loaded = False
models_loading = False

# Pre-built example queries for financial documents
EXAMPLE_QUERIES = [
    "What is the total revenue for Q3?",
    "List all outstanding invoices above $10,000",
    "What are the total liabilities?",
    "Show me the balance sheet summary",
    "What is the net profit margin?",
]


def preload_models():
    """Pre-load embedding model at startup (LLM loads lazily on first chat)."""
    global pipeline, models_loaded, models_loading

    if models_loading or models_loaded:
        return

    models_loading = True
    print("Pre-loading embedding model (LLM loads on first chat)...")

    pipeline = RAGPipeline(
        chunk_size=500,
        chunk_overlap=50,
        embedding_model='sentence-transformers/all-MiniLM-L6-v2',
        llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        top_k=5
    )
    # Only load embedding model at startup - LLM loads lazily
    pipeline.embedder.load()
    models_loaded = True
    models_loading = False
    print("✅ Embedding model ready! (LLM will load on first question)")


def initialize_pipeline():
    """Get the RAG pipeline (models should already be loaded)."""
    global pipeline, models_loaded

    # Wait for models if still loading
    while models_loading:
        time.sleep(0.5)

    # Fallback: load if not already loaded
    if not models_loaded:
        preload_models()

    return pipeline


def process_pdfs(files, progress=gr.Progress()) -> Tuple[str, str, str]:
    """
    Process uploaded PDF file(s) - supports batch upload.

    Args:
        files: Uploaded file(s) - can be single file or list
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, document_info, warnings)
    """
    global pipeline

    if files is None:
        return "❌ No file uploaded", "", ""

    # Handle both single file and multiple files
    if not isinstance(files, list):
        files = [files]

    if len(files) == 0:
        return "❌ No file uploaded", "", ""

    try:
        progress(0, desc="Initializing models...")

        # Initialize pipeline
        pipe = initialize_pipeline()

        # Get file paths
        pdf_paths = []
        for f in files:
            if hasattr(f, 'name'):
                pdf_paths.append(f.name)
            else:
                pdf_paths.append(f)

        filenames = [os.path.basename(p) for p in pdf_paths]

        # Clear previous index
        pipe.clear()

        all_warnings = []

        # Progress callback
        def progress_callback(msg):
            if "OCR processing page" in msg:
                try:
                    parts = msg.split()
                    page_info = parts[-1].replace("...", "")
                    current, total = map(int, page_info.split("/"))
                    progress(current / total * 0.6, desc=msg)
                except:
                    progress(0.3, desc=msg)
            elif "Processing file" in msg:
                try:
                    parts = msg.split()
                    file_info = parts[2].replace(":", "")
                    current, total = map(int, file_info.split("/"))
                    progress(current / total * 0.5, desc=msg)
                except:
                    progress(0.2, desc=msg)
            elif "Embedding" in msg:
                progress(0.7, desc=msg)
            elif "Building vector index" in msg:
                progress(0.9, desc=msg)
            elif "complete" in msg.lower():
                progress(1.0, desc=msg)
            else:
                progress(0.1, desc=msg)

        # Ingest PDFs
        if len(pdf_paths) == 1:
            num_chunks, warnings = pipe.ingest_pdf(pdf_paths[0], progress_callback)
        else:
            num_chunks, warnings = pipe.ingest_multiple_pdfs(pdf_paths, progress_callback)

        all_warnings.extend(warnings)

        # Get index stats
        stats = pipe.get_index_stats()

        # Build status message
        if len(filenames) == 1:
            status = f"✅ Successfully processed **{filenames[0]}**"
        else:
            status = f"✅ Successfully processed **{len(filenames)} files**"

        # Build info message
        info_parts = [
            f"📄 **Document Info:**",
            f"- Files: {', '.join(filenames)}",
            f"- Total chunks: {stats['total_chunks']}",
            f"- Total pages: {stats['pages']}",
        ]

        if stats.get('chunks_with_warnings', 0) > 0:
            info_parts.append(f"- ⚠️ Chunks with warnings: {stats['chunks_with_warnings']}")

        info_parts.append("- Ready for questions!")
        info = "\n".join(info_parts)

        # Build warnings message
        if all_warnings:
            warnings_text = "### ⚠️ OCR Warnings:\n\n"
            for w in all_warnings[:10]:  # Show max 10 warnings
                warnings_text += f"- {w}\n"
            if len(all_warnings) > 10:
                warnings_text += f"\n*...and {len(all_warnings) - 10} more warnings*"
        else:
            warnings_text = "✅ No OCR warnings detected"

        return status, info, warnings_text

    except Exception as e:
        error_msg = f"❌ Error processing PDF: {str(e)}"
        return error_msg, "", ""


def chat(message: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
    """
    Handle chat messages.

    Args:
        message: User message
        history: Chat history

    Returns:
        Tuple of (updated_history, source_chunks_display)
    """
    global pipeline

    if not message.strip():
        return history, ""

    if pipeline is None or not pipeline.is_indexed:
        history.append([message, "⚠️ Please upload a PDF document first before asking questions."])
        return history, ""

    try:
        # Query the pipeline
        answer, retrieved_chunks = pipeline.query(message)

        # Format source chunks for display
        sources = format_source_chunks(retrieved_chunks)

        # Add to history
        history.append([message, answer])

        return history, sources

    except Exception as e:
        error_response = f"❌ Error: {str(e)}"
        history.append([message, error_response])
        return history, ""


def use_example_query(example: str) -> str:
    """Set the example query in the input box."""
    return example


def format_source_chunks(chunks: List[Tuple[Chunk, float]]) -> str:
    """
    Format retrieved chunks for display with warning indicators.

    Args:
        chunks: List of (Chunk, score) tuples

    Returns:
        Formatted string for display
    """
    if not chunks:
        return "No source chunks retrieved."

    formatted = "### 📚 Source Chunks Used:\n\n"

    for i, (chunk, score) in enumerate(chunks, 1):
        # Add warning indicator if chunk has low confidence
        warning_badge = " ⚠️" if chunk.has_warning else ""
        source_info = f" ({chunk.source_file})" if chunk.source_file else ""

        formatted += f"**{i}. Page {chunk.page_num}{source_info}**{warning_badge} (relevance: {score:.3f})\n"

        # Show warning message if present
        if chunk.has_warning and chunk.warning_message:
            formatted += f"*⚠️ Warning: {chunk.warning_message}*\n"

        # Truncate long chunks for display
        text = chunk.text
        if len(text) > 300:
            text = text[:300] + "..."
        formatted += f"> {text}\n\n"
        formatted += "---\n\n"

    return formatted


def clear_chat():
    """Clear the chat history."""
    return [], ""


def clear_all():
    """Clear everything including uploaded documents."""
    global pipeline
    if pipeline:
        pipeline.clear()
    return [], "", "*No document loaded*", "", "✅ No OCR warnings detected"


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="PDF RAG Chat - Financial Document Assistant") as app:
        gr.Markdown("""
        # 📄 PDF RAG Chat - Financial Document Assistant

        **Upload invoices, balance sheets, or financial documents and chat with them using local AI.**

        ### ✨ Features:
        - 💰 **Currency Detection** - Preserves USD ($), EUR (€), INR (₹/Rs.) formats
        - 📊 **Multi-Column Support** - Handles two-column balance sheets and invoice layouts
        - ⚠️ **OCR Confidence** - Flags low-confidence regions that may need verification
        - 📁 **Batch Upload** - Process multiple invoices into a single searchable index
        - 🔒 **Fully Local** - No data leaves your machine

        ---
        """)

        with gr.Row():
            # Left column - Upload and Info
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload PDF(s)")

                pdf_upload = gr.File(
                    label="Drop your PDF(s) here or click to browse",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath"
                )

                process_btn = gr.Button("🚀 Process PDF(s)", variant="primary", size="lg")

                status_output = gr.Markdown(
                    value="*No document loaded*",
                    label="Status"
                )

                info_output = gr.Markdown(
                    value="",
                    label="Document Info"
                )

                warnings_output = gr.Markdown(
                    value="✅ No OCR warnings detected",
                    label="OCR Warnings"
                )

                clear_all_btn = gr.Button("🗑️ Clear All", variant="secondary")

            # Right column - Chat
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with Document(s)")

                # Example queries section
                gr.Markdown("**📝 Example queries** (click to use):")
                with gr.Row():
                    example_btns = []
                    for i, example in enumerate(EXAMPLE_QUERIES[:3]):
                        btn = gr.Button(example, size="sm", variant="secondary")
                        example_btns.append(btn)

                with gr.Row():
                    for i, example in enumerate(EXAMPLE_QUERIES[3:]):
                        btn = gr.Button(example, size="sm", variant="secondary")
                        example_btns.append(btn)

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=350,
                    show_label=False
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask a question about your documents...",
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                gr.Markdown("### 📚 Source Chunks")
                sources_output = gr.Markdown(
                    value="*Source chunks will appear here after you ask a question*",
                    label="Sources"
                )

        # Event handlers
        process_btn.click(
            fn=process_pdfs,
            inputs=[pdf_upload],
            outputs=[status_output, info_output, warnings_output],
            show_progress=True
        )

        # Process on file upload
        pdf_upload.change(
            fn=process_pdfs,
            inputs=[pdf_upload],
            outputs=[status_output, info_output, warnings_output],
            show_progress=True
        )

        # Chat handlers
        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, sources_output]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )

        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, sources_output]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )

        # Example query buttons
        for i, btn in enumerate(example_btns):
            query = EXAMPLE_QUERIES[i]
            btn.click(
                fn=lambda q=query: q,
                outputs=[msg_input]
            )

        # Clear handlers
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, sources_output]
        )

        clear_all_btn.click(
            fn=clear_all,
            outputs=[chatbot, sources_output, status_output, info_output, warnings_output]
        )

        gr.Markdown("""
        ---
        ### 💡 Tips for Best Results:
        - **Scanned documents**: Ensure good scan quality for better OCR accuracy
        - **Currency values**: The system detects and preserves USD, EUR, and INR formats
        - **Warnings**: Pay attention to ⚠️ warnings - those regions may need manual verification
        - **Batch upload**: Upload multiple related invoices to search across all of them

        *Built with Tesseract OCR, FAISS, sentence-transformers, and TinyLlama*
        """)

    return app


def main():
    """Launch the Gradio application."""
    print("=" * 60)
    print("📄 PDF RAG Chat - Financial Document Assistant")
    print("=" * 60)
    print()

    # Pre-load embedding model at startup (LLM loads lazily)
    print("Loading embedding model at startup...")
    preload_models()
    print()
    print("Starting Gradio server...")
    print()

    app = create_ui()

    # Launch with queue for better handling of long operations
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        pwa=False
    )


if __name__ == "__main__":
    main()
