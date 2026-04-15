# ui.py

import os
from typing import List, Tuple

import gradio as gr

from tools.vector_search import build_vectorstore, get_existing_pdfs
from graph import build_graph
from memory import AgentState

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)

app = build_graph()


def save_and_index_files(files: List[str] | None, skip_existing: bool = True):
    """
    Save uploaded PDFs into data/ and rebuild the FAISS vectorstore.
    
    Returns tuple of (message, saved_count, skipped_count) for notifications.
    """
    if not files:
        return None, 0, 0

    existing_pdfs = get_existing_pdfs(DATA_DIR)
    saved_files: List[str] = []
    skipped_files: List[str] = []
    
    for f in files:
        src_path = str(f)
        if not os.path.isfile(src_path):
            continue

        orig_name = os.path.basename(src_path)
        if not orig_name.lower().endswith(".pdf"):
            continue

        dest_path = os.path.join(DATA_DIR, orig_name)
        
        if orig_name in existing_pdfs:
            skipped_files.append(orig_name)
            continue

        try:
            with open(src_path, "rb") as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
        except OSError:
            continue

        saved_files.append(orig_name)

    if not saved_files and skipped_files:
        return None, 0, len(skipped_files)

    if not saved_files:
        return None, 0, 0

    try:
        build_vectorstore(pdf_dir=DATA_DIR, save_path=VECTORSTORE_DIR, skip_existing=skip_existing)
    except Exception as e:
        return str(e), 0, 0

    return None, len(saved_files), len(skipped_files)


def handle_upload(files: List[str] | None, skip_existing: bool = True):
    """Handle file upload with notifications."""
    msg, saved, skipped = save_and_index_files(files, skip_existing)
    
    if msg:
        yield gr.Info, msg
        return
    
    if saved == 0 and skipped == 0:
        yield gr.Warning, "No valid PDF files found in upload."
        return
    
    if skipped > 0 and saved == 0:
        yield gr.Warning, f"Skipped {skipped} existing file(s). No new files uploaded."
        return
    
    if saved > 0 and skipped > 0:
        yield gr.Info, f"Uploaded {saved} new file(s), skipped {skipped} existing file(s)."
    elif saved > 0:
        yield gr.Info, f"Successfully uploaded and indexed {saved} file(s)."


def run_copilot(query: str) -> Tuple[str, str]:
    if not query.strip():
        return "Please enter a question.", ""

    state: AgentState = {
        "query": query,
        "sub_questions": [],
        "scratchpad": [],
        "draft_answer": "",
        "final_answer": "",
        "needs_revision": False,
        "iteration": 0,
    }

    final_state = app.invoke(state)

    answer = (
        final_state.get("final_answer")
        or final_state.get("draft_answer")
        or "No answer generated."
    )

    sub_qs = final_state.get("sub_questions", [])
    passages = final_state.get("scratchpad", [])

    trace_lines: List[str] = []

    if sub_qs:
        trace_lines.append("Sub-questions:")
        for i, sq in enumerate(sub_qs, 1):
            trace_lines.append(f"{i}. {sq}")
        trace_lines.append("")

    if passages:
        trace_lines.append("Retrieved passages (truncated):")
        for i, p in enumerate(passages[:5], 1):
            src = p.get("source", "unknown")
            page = p.get("page", "?")
            content = p.get("content", "")
            snippet = content[:300].replace("\n", " ")
            trace_lines.append(f"[{i}] {src} (p.{page}): {snippet}...")

    trace_text = "\n".join(trace_lines)
    return answer, trace_text


with gr.Blocks(title="Agentic Research Copilot") as demo:
    gr.Markdown("# Agentic Research Copilot\nUpload PDFs, then ask scholarly questions.")

    with gr.Tab("Upload PDFs"):
        file_uploader = gr.File(
            label="Upload one or more PDF papers",
            file_types=[".pdf"],
            file_count="multiple"
        )
        skip_existing_checkbox = gr.Checkbox(
            label="Skip existing files (only add new PDFs)",
            value=True,
        )
        upload_button = gr.Button("Upload and index")

        upload_button.click(
            fn=handle_upload,
            inputs=[file_uploader, skip_existing_checkbox],
        )

    with gr.Tab("Ask a Question"):
        query_box = gr.Textbox(
            label="Research question",
            placeholder="e.g., How do agentic AI systems improve reliability in RAG pipelines?",
            lines=3,
        )
        ask_button = gr.Button("Run copilot")
        answer_box = gr.Markdown(label="Answer")
        trace_box = gr.Textbox(
            label="Reasoning trace (planner, retrieval)",
            lines=15,
        )

        ask_button.click(
            fn=run_copilot,
            inputs=query_box,
            outputs=[answer_box, trace_box],
        )


if __name__ == "__main__":
    demo.launch()
