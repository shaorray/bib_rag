"""
Node functions for the bib_rag agentic graph.

Adapted from the reference implementation for the bib_rag domain:
- ChromaDB with bge-m3 embeddings instead of Qdrant
- OpenAI-compatible API via llama-server (Qwen3.6-35B) instead of Ollama
- Academic papers on Eph/ephrin instead of general PDFs
"""

from typing import Literal, Set
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command

from .agent_schemas import State, AgentState, QueryAnalysis
from .agent_prompts import (
    get_conversation_summary_prompt,
    get_rewrite_query_prompt,
    get_orchestrator_prompt,
    get_fallback_response_prompt,
    get_context_compression_prompt,
    get_aggregation_prompt,
)

# --- Configuration ---
MAX_ITERATIONS = 10
MAX_TOOL_CALLS = 8
BASE_TOKEN_THRESHOLD = 2000
TOKEN_GROWTH_FACTOR = 0.9


def estimate_context_tokens(messages) -> int:
    """Estimate the token count of a list of messages.

    Uses a simple character-based heuristic (approx. 4 chars per token)
    since we don't have tiktoken available.
    """
    total_chars = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if content:
            if isinstance(content, str):
                total_chars += len(content)
            else:
                total_chars += len(str(content))
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                total_chars += len(str(tc))
    # Rough heuristic: ~4 chars per token for mixed English/Chinese text
    return max(1, total_chars // 4)


def summarize_history(state: State, llm):
    """Compress the last 6 messages into a conversation summary.

    Args:
        state: Current State
        llm: LangChain chat model instance

    Returns:
        Dict with conversation_summary and agent_answers reset marker.
    """
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke(
        [
            SystemMessage(content=get_conversation_summary_prompt()),
            HumanMessage(content=conversation),
        ]
    )
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}


def rewrite_query(state: State, llm):
    """Rewrite the user query using conversation context, returning structured QueryAnalysis.

    Args:
        state: Current State
        llm: LangChain chat model instance

    Returns:
        Dict with questionIsClear, messages, originalQuery, rewrittenQuestions.
    """
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (
        f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else ""
    ) + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke(
        [
            SystemMessage(content=get_rewrite_query_prompt()),
            HumanMessage(content=context_section),
        ]
    )

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions,
        }

    clarification = (
        response.clarification_needed
        if response.clarification_needed and len(response.clarification_needed.strip()) > 10
        else "I need more information to understand your question."
    )
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}


def request_clarification(state: State):
    """Placeholder node — the graph interrupts before this so the user can respond.

    Returns empty dict to pass through.
    """
    return {}


# --- Agent Subgraph Nodes ---


def orchestrator(state: AgentState, llm_with_tools):
    """Main agent reasoning node.

    Calls the LLM with tools bound, injecting compressed context if available.
    On the first call, forces a search_child_chunks call.

    Args:
        state: Current AgentState
        llm_with_tools: LLM with tools bound via bind_tools()

    Returns:
        Dict with messages, tool_call_count, iteration_count, force_search_done.
    """
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())

    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary
        else []
    )

    # Branch on `force_search_done` (not on `messages` emptiness) because
    # `compress_context` removes messages[1:] and would otherwise trick us
    # into re-firing the mandatory first search on every post-compression
    # resume — wasting an LLM call and an embedding roundtrip.
    if not state.get("force_search_done", False):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(
            content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION."
        )
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {
            "messages": [human_msg, response],
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,
            "force_search_done": True,
        }

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {
        "messages": [response],
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "iteration_count": 1,
    }


def fallback_response(state: AgentState, llm):
    """Answer without further tool calls — called when iteration/tool limits are reached.

    Synthesizes the best possible answer from compressed context and retrieved data.

    Args:
        state: Current AgentState
        llm: LangChain chat model instance (no tools)

    Returns:
        Dict with messages containing the final answer AIMessage.
    """
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(
            f"## Compressed Research Context (from prior iterations)\n\n{context_summary}"
        )
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n"
            + "\n\n".join(
                f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1)
            )
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke(
        [
            SystemMessage(content=get_fallback_response_prompt()),
            HumanMessage(content=prompt_content),
        ]
    )
    return {"messages": [response]}


def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    """Check if context needs compression.

    Routes to compress_context if token threshold exceeded, otherwise back to orchestrator.

    Args:
        state: Current AgentState

    Returns:
        Command with goto and retrieval_keys update.
    """
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens(
        [HumanMessage(content=state.get("context_summary", ""))]
    )
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)


def compress_context(state: AgentState, llm):
    """Compress conversation context into a structured summary.

    Args:
        state: Current AgentState
        llm: LangChain chat model instance

    Returns:
        Dict with updated context_summary and removed old messages.
    """
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke(
        [
            SystemMessage(content=get_context_compression_prompt()),
            HumanMessage(content=conversation_text),
        ]
    )
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(
                f"- {p.replace('parent::', '')}" for p in parent_ids
            ) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}


def collect_answer(state: AgentState):
    """Extract the final answer from the last AI message.

    Args:
        state: Current AgentState

    Returns:
        Dict with final_answer and agent_answers.
    """
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [
            {"index": state["question_index"], "question": state["question"], "answer": answer}
        ],
    }


def aggregate_answers(state: State, llm):
    """Merge answers from multiple agent subgraphs into a single response.

    Args:
        state: Current State (main graph)
        llm: LangChain chat model instance

    Returns:
        Dict with messages containing the aggregated AIMessage.
    """
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += f"\nAnswer {i}:\n{ans['answer']}\n"

    user_message = HumanMessage(
        content=f"Original user question: {state['originalQuery']}\nRetrieved answers:{formatted_answers}"
    )
    synthesis_response = llm.invoke(
        [SystemMessage(content=get_aggregation_prompt()), user_message]
    )
    return {"messages": [AIMessage(content=synthesis_response.content)]}
