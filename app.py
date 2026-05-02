"""
Gradio Web UI for PDF RAG Pipeline

This module provides a user-friendly web interface for:
- Uploading PDF documents
- Chatting with the document content
- Viewing source chunks used for answers
"""

import gradio as gr
import tempfile
import os
from typing import List, Tuple, Optional
import time

from pipeline import RAGPipeline, Chunk

# Global pipeline instance
pipeline: Optional[RAGPipeline] = None
models_loaded = False


def initialize_pipeline():
    """Initialize the RAG pipeline with models."""
    global pipeline, models_loaded

    if pipeline is None:
        pipeline = RAGPipeline(
            chunk_size=500,
            chunk_overlap=50,
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            top_k=5
        )

    if not models_loaded:
        pipeline.load_models()
        models_loaded = True

    return pipeline


def process_pdf(file, progress=gr.Progress()) -> Tuple[str, str]:
    """
    Process an uploaded PDF file.

    Args:
        file: Uploaded file object
        progress: Gradio progress tracker

    Returns:
        Tuple of (status_message, pdf_info)
    """
    global pipeline

    if file is None:
        return "❌ No file uploaded", ""

    try:
        progress(0, desc="Initializing models...")

        # Initialize pipeline
        pipe = initialize_pipeline()

        # Get file path
        if hasattr(file, 'name'):
            pdf_path = file.name
        else:
            pdf_path = file

        filename = os.path.basename(pdf_path)

        # Clear previous index
        pipe.clear()

        # Progress callback
        progress_messages = []

        def progress_callback(msg):
            progress_messages.append(msg)
            # Update progress bar based on message
            if "OCR processing page" in msg:
                try:
                    parts = msg.split()
                    page_info = parts[-1].replace("...", "")
                    current, total = map(int, page_info.split("/"))
                    progress(current / total * 0.6, desc=msg)  # OCR is 60% of work
                except:
                    progress(0.3, desc=msg)
            elif "Embedding" in msg:
                progress(0.7, desc=msg)
            elif "Building vector index" in msg:
                progress(0.9, desc=msg)
            elif "complete" in msg.lower():
                progress(1.0, desc=msg)
            else:
                progress(0.1, desc=msg)

        # Ingest PDF
        num_chunks = pipe.ingest_pdf(pdf_path, progress_callback)

        status = f"✅ Successfully processed **{filename}**"
        info = f"📄 **Document Info:**\n- Chunks created: {num_chunks}\n- Ready for questions!"

        return status, info

    except Exception as e:
        error_msg = f"❌ Error processing PDF: {str(e)}"
        return error_msg, ""


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


def format_source_chunks(chunks: List[Tuple[Chunk, float]]) -> str:
    """
    Format retrieved chunks for display.

    Args:
        chunks: List of (Chunk, score) tuples

    Returns:
        Formatted string for display
    """
    if not chunks:
        return "No source chunks retrieved."

    formatted = "### 📚 Source Chunks Used:\n\n"

    for i, (chunk, score) in enumerate(chunks, 1):
        formatted += f"**{i}. Page {chunk.page_num}** (relevance: {score:.3f})\n"
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


def create_ui():
    """Create the Gradio interface."""

    # Custom CSS for better styling
    css = """
    .container { max-width: 1200px; margin: auto; }
    .upload-section { border: 2px dashed #ccc; border-radius: 10px; padding: 20px; }
    .status-box { min-height: 100px; }
    .source-box { max-height: 400px; overflow-y: auto; }
    """

    with gr.Blocks() as app:
        gr.Markdown("""
        # 📄 PDF RAG Chat

        **Upload a PDF document and chat with its contents using local AI models.**

        This application uses:
        - 🔍 **Tesseract OCR** for text extraction from scanned PDFs
        - 🧠 **sentence-transformers** for semantic embeddings
        - 📊 **FAISS** for vector similarity search
        - 🤖 **TinyLlama** for answer generation

        *All processing is done locally - no data leaves your machine!*

        ---
        """)

        with gr.Row():
            # Left column - Upload and Info
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload PDF")

                pdf_upload = gr.File(
                    label="Drop your PDF here or click to browse",
                    file_types=[".pdf"],
                    type="filepath"
                )

                process_btn = gr.Button("🚀 Process PDF", variant="primary", size="lg")

                status_output = gr.Markdown(
                    value="*No document loaded*",
                    label="Status"
                )

                info_output = gr.Markdown(
                    value="",
                    label="Document Info"
                )

                gr.Markdown("""
                ---
                ### ℹ️ Tips
                - For best results, ensure your PDF has clear text
                - Processing may take a few minutes for large documents
                - The first query may be slow as models are loaded
                """)

            # Right column - Chat
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with Document")

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_label=False
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask a question about the document...",
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
            fn=process_pdf,
            inputs=[pdf_upload],
            outputs=[status_output, info_output],
            show_progress=True
        )

        # Also process when file is uploaded
        pdf_upload.change(
            fn=process_pdf,
            inputs=[pdf_upload],
            outputs=[status_output, info_output],
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

        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, sources_output]
        )

        gr.Markdown("""
        ---
        *Built with ❤️ using Gradio, Tesseract OCR, FAISS, and TinyLlama*
        """)

    return app


def main():
    """Launch the Gradio application."""
    print("=" * 60)
    print("📄 PDF RAG Chat Application")
    print("=" * 60)
    print()
    print("Starting Gradio server...")
    print()
    print("Note: Models will be loaded on first use.")
    print("The first PDF processing and query may take longer.")
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
