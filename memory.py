# memory.py

from typing import TypedDict, List, Annotated
import operator


class AgentState(TypedDict):
    """
    Shared state passed between agents in the LangGraph workflow.

    Fields:
    - query: original user question.
    - sub_questions: list of planner-generated sub-questions.
    - scratchpad: accumulated retrieved passages and intermediate notes.
    - draft_answer: latest answer produced by the writer.
    - final_answer: answer accepted after reflection (or same as draft).
    - needs_revision: flag set by reflector to trigger another loop.
    - iteration: how many reflection loops have run so far.
    """

    query: str
    sub_questions: List[str]
    scratchpad: Annotated[List[dict], operator.add]
    draft_answer: str
    final_answer: str
    needs_revision: bool
    iteration: int
