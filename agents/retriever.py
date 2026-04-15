# agents/retriever.py

from typing import Any, Dict, List

from memory import AgentState
from tools.vector_search import search_corpus


def retriever_agent(state: AgentState) -> Dict[str, Any]:
    """
    For each sub-question, retrieves passages via search_corpus and appends them to scratchpad.
    """
    sub_questions: List[str] = state.get("sub_questions", [])
    existing = state.get("scratchpad", [])

    # To avoid duplicates
    seen_contents = {p.get("content", "") for p in existing}
    new_passages: List[dict] = []

    for sq in sub_questions:
        results = search_corpus(sq, k=3)
        for r in results:
            content = r.get("content", "")
            if content and content not in seen_contents:
                r["sub_question"] = sq
                new_passages.append(r)
                seen_contents.add(content)

    return {"scratchpad": new_passages}
