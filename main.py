# main.py

from typing import Any

from graph import build_graph


def run_query(query: str) -> Any:
    """
    Run a single query through the Planner -> Retriever -> Writer graph.
    Returns the final state dict.
    """
    app = build_graph()

    initial_state = {
        "query": query,
        "sub_questions": [],
        "scratchpad": [],
        "draft_answer": "",
        "final_answer": "",
        "needs_revision": False,
        "iteration": 0,
    }

    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    user_query = input("Enter your research question: ")
    state = run_query(user_query)

    print("\n=== Sub-questions ===")
    for sq in state.get("sub_questions", []):
        print(f"- {sq}")

    print("\n=== Retrieved Passages (first 3) ===")
    for i, p in enumerate(state.get("scratchpad", [])[:3], start=1):
        print(f"[{i}] {p.get('source', '')} p.{p.get('page', '?')}")
        print(p.get("content", "")[:300].replace("\n", " ") + "...")
        print()

    print("\n=== Answer ===\n")
    answer = state.get("final_answer") or state.get("draft_answer", "")
    print(answer)
