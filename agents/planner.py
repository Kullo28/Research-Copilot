# agents/planner.py

import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from memory import AgentState

load_dotenv()

BASE_URL = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LOCAL_OPENAI_API_KEY", "ollama")
MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3.1")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

PLANNER_SYSTEM_PROMPT = """
You are a research planning agent for an agentic RAG system.
Given a scholarly user question, decompose it into 3–5 focused
sub-questions that can be answered by retrieving passages from
a corpus of academic PDFs.

Return ONLY a JSON list of strings, with no extra text.
Example:
["What is retrieval-augmented generation (RAG)?",
 "How do multi-agent systems improve RAG reliability?",
 "What evaluation metrics are used for RAG?"]
"""


def planner_agent(state: AgentState) -> Dict[str, Any]:
    """
    Takes state["query"] and returns a partial update containing "sub_questions".
    """
    user_query = state["query"]

    prompt = (
        PLANNER_SYSTEM_PROMPT
        + "\n\nUser Query:\n"
        + user_query
        + "\n\nJSON list:"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content or ""

    # Strip possible code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[2]

    try:
        sub_questions: List[str] = json.loads(cleaned)
        if not isinstance(sub_questions, list):
            raise ValueError
    except Exception:
        sub_questions = [user_query]

    return {"sub_questions": sub_questions}
