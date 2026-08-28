"""
Agent State Schemas for bib_rag agentic graph.

Defines TypedDict state definitions for the main graph and agent subgraph,
and the Pydantic QueryAnalysis model for structured query rewriting.
Adapted for the bib_rag (Eph/ephrin academic papers) domain.
"""

from typing import List, Annotated, Set
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
import operator


def accumulate_or_reset(
    existing: List[dict], new: List[dict]
) -> List[dict]:
    """Merge agent answers, resetting on marker."""
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new


def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b


class QueryAnalysis(BaseModel):
    """Structured output for query analysis and rewriting.

    Attributes:
        is_clear: Whether the user's question is clear and answerable.
        questions: List of rewritten, self-contained queries for retrieval.
        clarification_needed: Explanation if clarification is required.
    """

    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained queries for document retrieval."
    )
    clarification_needed: str = Field(
        description="Explanation of why the question needs clarification."
    )


class State(MessagesState):
    """State for the main (top-level) agent graph.

    Extends MessagesState with:
    - questionIsClear: whether the query is clear after rewriting
    - conversation_summary: 1-2 sentence summary of prior conversation
    - originalQuery: the user's raw original query
    - rewrittenQuestions: list of rewritten, self-contained queries
    - agent_answers: list of answer dicts from the agent subgraph
    """

    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []


class AgentState(MessagesState):
    """State for the individual agent subgraph.

    Extends MessagesState with:
    - question: the current query to answer
    - question_index: index of this question in the rewritten list
    - context_summary: compressed research context from prior iterations
    - retrieval_keys: set of already-retrieved parent IDs and search queries
    - final_answer: the extracted final answer
    - agent_answers: answers accumulated across subgraphs
    - tool_call_count: total tool calls (accumulated via operator.add)
    - iteration_count: iteration counter (accumulated via operator.add)
    """

    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    # Deterministic citation-guard note from collect_answer (empty when the
    # guard passed cleanly or is disabled via CITATION_GUARD=0).
    guard_note: str = ""
    # haiku.rag-style citation-policy redirect (P3b): when the deterministic
    # guard strips ALL Sources lines, collect_answer sets this instead of
    # shipping an uncitable answer; route_after_collect sends the agent back
    # to the orchestrator with the policy feedback. Capped by guard_retries.
    guard_redirect: bool = False
    guard_retries: int = 0
    agent_answers: List[dict] = []
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    # Tracks whether the orchestrator's mandatory first search has already been
    # performed. Critical: `compress_context` wipes `messages[1:]`, so we cannot
    # use "is messages empty?" as a proxy for "is this the first call" — that
    # branch would re-fire `force_search` and waste a retrieval on every resume.
    force_search_done: Annotated[bool, operator.or_] = False
