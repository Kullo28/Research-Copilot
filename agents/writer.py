# agents/writer.py

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from memory import AgentState

load_dotenv()

BASE_URL = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LOCAL_OPENAI_API_KEY", "ollama")
MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3.1")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

WRITER_SYSTEM_PROMPT = """
You are a scholarly writing agent for an agentic RAG system.
You receive:
- A user query.
- A set of retrieved passages from academic PDFs, each with a numeric index and source info.

Your task:
- Write a concise, well-structured answer to the query (about 300–500 words).
- Use ONLY the information from the passages.
- Every factual statement must be supported by at least one passage.
- Use IEEE-style inline citations based on the passage indices, e.g., [1], [2], [1][3].

Do NOT invent references. If the passages do not support a claim, do not make it.
"""


def writer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Consumes state["query"] and state["scratchpad"] and returns {"draft_answer": ...}.
    """
    query = state["query"]
    passages: List[dict] = state.get("scratchpad", [])

    if not passages:
        return {"draft_answer": "I could not find any relevant passages in the corpus."}

    formatted_passages = []
    for i, p in enumerate(passages):
        idx = i + 1
        src = p.get("source", "unknown")
        page = p.get("page", "?")
        content = p.get("content", "")
        formatted_passages.append(
            f"[{idx}] (Source: {src}, p.{page})\n{content}"
        )

    passages_block = "\n\n".join(formatted_passages)

    user_prompt = f"""
User Query:
{query}

Retrieved Passages:
{passages_block}

Now write the answer:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    draft = (response.choices[0].message.content or "").strip()
    return {"draft_answer": draft}
