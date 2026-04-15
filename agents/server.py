# server.py

import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from tools.vector_search import build_vectorstore

app = FastAPI()

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)


def _get_safe_filename(file: UploadFile) -> str:
    """
    Ensure we have a non-empty filename string for the uploaded file.
    This narrows away Optional[str] so Pylance is happy.
    """
    filename: Optional[str] = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")
    return filename


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a single PDF, save it into data/, and rebuild the FAISS vectorstore.

    Returns a JSON message on success.
    """
    filename = _get_safe_filename(file)

    # Pylance-safe: filename is now a str, not Optional[str]
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    save_path = os.path.join(DATA_DIR, filename)

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Write the file to disk
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
    except OSError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {e}"
        ) from e

    # Rebuild vectorstore after upload
    # (You can later debounce this if uploads get frequent.)
    try:
        build_vectorstore(pdf_dir=DATA_DIR, save_path=VECTORSTORE_DIR)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to build vectorstore: {e}"
        ) from e

    return JSONResponse(
        {
            "message": "File uploaded and indexed successfully.",
            "filename": filename,
            "stored_path": save_path,
            "vectorstore_dir": VECTORSTORE_DIR,
        }
    )


@app.get("/health")
async def health():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}
