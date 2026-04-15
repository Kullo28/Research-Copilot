# tools/vector_search.py

import os
from typing import List, Set

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    Local HuggingFace embeddings (no external API).
    """
    model_name = "all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)


def get_existing_pdfs(pdf_dir: str = "data/") -> Set[str]:
    """Return set of PDF filenames already in the data directory."""
    os.makedirs(pdf_dir, exist_ok=True)
    return {f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")}


def build_vectorstore(pdf_dir: str = "data/", save_path: str = "vectorstore/", skip_existing: bool = True):
    """
    Build a FAISS vector store from PDFs in pdf_dir and save it to save_path.
    
    If skip_existing is True, only processes new PDFs not already in the vectorstore.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    existing_pdfs = get_existing_pdfs(pdf_dir)
    
    # Load existing vectorstore if it exists
    existing_db = None
    existing_files = set()
    if os.path.isdir(save_path) and os.listdir(save_path):
        try:
            existing_db = FAISS.load_local(
                save_path,
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            # Get existing file sources from vectorstore
            if existing_db:
                try:
                    docstore_dict = getattr(existing_db.docstore, '_dict', None)
                    if docstore_dict is None:
                        docstore_dict = getattr(existing_db.docstore, 'docs', {})
                    if docstore_dict:
                        existing_files = {
                            os.path.basename(doc.metadata.get("source", ""))
                            # type: ignore[union-attr]
                            for doc in docstore_dict.values()
                        }
                except Exception:
                    pass
        except Exception:
            pass
    
    # Determine which PDFs to process
    if skip_existing and existing_files:
        pdfs_to_process = existing_pdfs - existing_files
    else:
        pdfs_to_process = existing_pdfs
    
    if not pdfs_to_process:
        if existing_pdfs:
            return "All PDFs already indexed. No new files to process."
        return "No PDF documents found in data/."
    
    # Load new PDFs only
    docs = []
    for fname in pdfs_to_process:
        path = os.path.join(pdf_dir, fname)
        if os.path.isfile(path):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    
    if not docs:
        return "No new PDF documents to process."
    
    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )
    chunks = splitter.split_documents(docs)
    
    embeddings = get_embeddings()
    
    if existing_db is None:
        db = FAISS.from_documents(chunks, embeddings)
    else:
        db = existing_db
        db.add_documents(chunks)
    
    db.save_local(save_path)
    
    return f"Indexed {len(chunks)} new chunks. Total PDFs: {len(existing_pdfs)}."


def load_vectorstore(save_path: str = "vectorstore/"):
    """
    Load an existing FAISS vector store from disk.
    """
    if not os.path.isdir(save_path):
        raise FileNotFoundError(
            f"Vector store directory '{save_path}' not found. "
            f"Run build_vectorstore() first."
        )

    embeddings = get_embeddings()
    db = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return db


def search_corpus(query: str, k: int = 5, save_path: str = "vectorstore/"):
    """
    Perform a similarity search over the corpus for the given query.
    Returns a list of dicts: {content, source, page}.
    """
    db = load_vectorstore(save_path)
    results = db.similarity_search(query, k=k)

    out: List[dict] = []
    for r in results:
        out.append(
            {
                "content": r.page_content,
                "source": r.metadata.get("source", ""),
                "page": r.metadata.get("page", ""),
            }
        )
    return out
