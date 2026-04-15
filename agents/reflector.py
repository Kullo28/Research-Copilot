# agents/reflector.py

import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from memory import AgentState

load_dotenv()

BASE_URL = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LOCAL_OPENAI_API_KEY", "ollama")
MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "llama3.1")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

REFLECTOR_SYSTEM_PROMPT = """
You are a critical reflection agent for an agentic RAG system.
Your task is to evaluate whether the draft answer is supported by the retrieved passages.

For each factual claim in the draft answer, check if it can be directly supported 
by at least one passage in the scratchpad. Flag any claims that are not supported 
by the retrieved content.

Respond in JSON format with:
- "needs_revision": true if there are unsupported claims, false otherwise
- "unsupported_claims": list of specific unsupported claims (empty if none)
- "reflection": brief reasoning for your decision
"""


def reflector_agent(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates draft_answer against scratchpad to check for unsupported claims.
    Returns updated state with needs_revision flag and potentially final_answer.
    """
    query = state["query"]
    draft_answer = state.get("draft_answer", "")
    scratchpad = state.get("scratchpad", [])
    iteration = state.get("iteration", 0)
    max_iterations = 3

    if not draft_answer:
        return {
            "needs_revision": False,
            "final_answer": "No answer was generated.",
            "iteration": iteration + 1,
        }

    if not scratchpad:
        return {
            "needs_revision": False,
            "final_answer": draft_answer,
            "iteration": iteration + 1,
        }

    formatted_passages = []
    for i, p in enumerate(scratchpad):
        idx = i + 1
        src = p.get("source", "unknown")
        page = p.get("page", "?")
        content = p.get("content", "")
        formatted_passages.append(f"[{idx}] (Source: {src}, p.{page})\n{content}")

    passages_block = "\n\n".join(formatted_passages)

    user_prompt = f"""
User Query: {query}

Draft Answer:
{draft_answer}

Retrieved Passages:
{passages_block}

Evaluate whether each claim in the draft answer is supported by the passages.
Respond with JSON:
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content or ""

    import json

    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        
        result = json.loads(raw.strip())
        needs_revision = result.get("unsupported_claims", []) and result.get("needs_revision", False)
        reflection = result.get("reflection", "")
    except Exception:
        needs_revision = False
        reflection = "Failed to parse reflection response"

    if iteration >= max_iterations:
        return {
            "needs_revision": False,
            "final_answer": draft_answer,
            "iteration": iteration + 1,
            "reflection_notes": f"Max iterations reached. {reflection}",
        }

    if needs_revision:
        return {
            "needs_revision": True,
            "iteration": iteration + 1,
            "reflection_notes": reflection,
        }
    else:
        return {
            "needs_revision": False,
            "final_answer": draft_answer,
            "iteration": iteration + 1,
            "reflection_notes": reflection,
        }