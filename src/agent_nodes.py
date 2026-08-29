"""
Node functions for the bib_rag agentic graph.

Adapted from the reference implementation for the bib_rag domain:
- ChromaDB with bge-m3 embeddings instead of Qdrant
- OpenAI-compatible API via llama-server (Qwen3.6-35B) instead of a small local model
- Academic papers on Eph/ephrin instead of general PDFs
"""

from typing import Literal, Set
import os
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
# Resolve iteration/tool-call limits from env, defaulting by model speed.
# A slow local model (e.g. Qwen3.8-27B at ~33 t/s) must do FEWER serial LLM
# calls per query or it times out; a fast cloud model (glm-5.2) can afford more.
# Explicit env overrides always win:
#   AGENT_MAX_ITERATIONS, AGENT_MAX_TOOL_CALLS
def _resolve_limits():
    # Each env override is honored independently; the unset one falls back to
    # a per-model-speed default.
    def _int_env(name):
        try:
            v = int(os.environ.get(name, "").strip() or 0)
            return v if v > 0 else 0
        except ValueError:
            return 0

    iters, tools = _int_env("AGENT_MAX_ITERATIONS"), _int_env("AGENT_MAX_TOOL_CALLS")
    model = os.environ.get("LLM_MODEL", "glm-5.2:cloud")
    llm_url = os.environ.get("LLM_URL", "")
    # A local GGUF file (e.g. Qwen3.8-27B, ~33 t/s decode) is a slow dense
    # model → conservative limits. Detect it by the .gguf suffix OR the local
    # llama-server port (:5015); cloud models are served through the
    # OpenAI-compatible gateway (localhost:11434) and are fast.
    is_local = model.endswith(".gguf") or ":5015" in llm_url
    if is_local:
        return iters or 3, tools or 4   # slow local model: conservative, avoid timeout
    return iters or 10, tools or 8      # fast cloud model: generous


MAX_ITERATIONS, MAX_TOOL_CALLS = _resolve_limits()


def is_local_llm() -> bool:
    """True when LLM_URL/LLM_MODEL points at a local llama-server slot
    (GGUF path or :5015) — shared with agent_tools budget defaults."""
    model = os.environ.get("LLM_MODEL", "")
    llm_url = os.environ.get("LLM_URL", "")
    return model.endswith(".gguf") or ":5015" in llm_url


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

    # Explicit method="function_calling": the OpenAI-compat gateway endpoint does
    # not honor the default structured-output method (json_mode / response_format
    # is ignored by cloud models like glm-5.2), which caused pydantic
    # "Invalid JSON" errors. function_calling routes through tool_calls, which
    # glm-5.2 / deepseek-v4-flash support correctly.
    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(
        QueryAnalysis, method="function_calling"
    )
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


def _cap_parallel_tool_calls(response):
    """Round-level tool budget for local slots: a 4096-token llama-server slot
    cannot hold several parallel tool RESULTS on top of prompts + history
    (measured: Qwen emits ~2 chars/token, so 3×3000-char tool outputs alone ≈
    4500 tokens → server-side 400). Keep only the first tool call per round;
    parallelism is a cloud-model luxury. Mutates + returns the response."""
    tcs = getattr(response, "tool_calls", None) or []
    if is_local_llm() and len(tcs) > 1:
        # Dropped calls simply never execute; the model re-requests them next
        # round. No note injection — the AIMessage keeps its original content.
        response.tool_calls = tcs[:1]
    return response


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
        response = _cap_parallel_tool_calls(
            llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search]))
        return {
            "messages": [human_msg, response],
            "tool_call_count": len(response.tool_calls or []),
            "iteration_count": 1,
            "force_search_done": True,
        }

    response = _cap_parallel_tool_calls(
        llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"]))
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

    # Evidence gate (ragent mechanism, zero LLM): audit whether the session's
    # retrievals actually produced evidence, and force the fallback answer to
    # say so when they didn't. Env kill-switch: EVIDENCE_GATE=0.
    gate_note = ""
    coverage = {}
    if os.environ.get("EVIDENCE_GATE", "1") != "0":
        try:
            from .evidence_gate import evidence_coverage, gap_instruction, coverage_block
            coverage = evidence_coverage(state["messages"], state.get("retrieval_keys", set()))
            gate_note = gap_instruction(coverage)
        except Exception:
            gate_note = ""  # gate must never break answering

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
        f"INSTRUCTION:\n"
        f"Provide the best possible answer using only the data above."
        + (f"\n\n{gate_note}" if gate_note else "")
    )
    response = llm.invoke(
        [
            SystemMessage(content=get_fallback_response_prompt()),
            HumanMessage(content=prompt_content),
        ]
    )
    answer = response.content
    if coverage and os.environ.get("EVIDENCE_GATE", "1") != "0":
        try:
            cb = coverage_block(coverage)
            if cb:
                answer = answer.rstrip() + "\n\n" + cb
        except Exception:
            pass
    return {"messages": [AIMessage(content=answer)]}


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
                        raw = [raw]
                    # Real parent ids look like '<source>#<section>#<hash>'.
                    # Models occasionally pass author surnames or titles here
                    # (tool then returns NO_PARENT_DOCUMENT); ledgering those
                    # pollutes the whitelist AND makes the harvest gate below
                    # believe real keys already exist, so ToolMessage
                    # harvesting is skipped and genuine ids never land.
                    new_ids.update(f"parent::{r}" for r in raw
                                   if isinstance(r, str) and "#" in r)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    # Parent-ID harvesting from ToolMessages (same source the citation guard
    # uses): search results list the parent_ids whose excerpts were shown,
    # so a session that answers from search excerpts alone would otherwise
    # carry ONLY search:: keys in the evidence ledger — and then
    # evaluate.citation_faithfulness (which sees only the keys, not the tool
    # messages) would score every Sources line as unresolvable (whitelist
    # rate 0.0 despite fully grounded retrieval). Mirrors
    # citation_guard.parent_ids_from_tool_messages.
    if not any(k.startswith("parent::") for k in new_ids):
        try:
            from .citation_guard import parent_ids_from_tool_messages
            harvested = parent_ids_from_tool_messages(messages)
            new_ids.update(f"parent::{p}" for p in harvested)
        except Exception:
            pass  # ledger hygiene must never break the routing loop

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    # Short loops don't benefit from mid-loop compression: compressing costs
    # one extra slow LLM call (~30-40s on a local 27B) and the lossy summary
    # actually *hurts* answer grounding. The tool outputs are already capped
    # (agent_tools budgets), so the final orchestrator can prefill the raw
    # evidence directly. Only compress when the loop is long enough to
    # re-prefill the accumulated context many times.
    if MAX_ITERATIONS <= 4:
        return Command(update={"retrieval_keys": updated_ids}, goto="orchestrator")

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

    Runs the deterministic citation guard (zero LLM) before returning:
    the Sources section is whitelisted against retrieval_keys (paper-qa
    mechanism) and lexically spot-checked against the cited parents
    (LumiCite/citelocal-agent cheap tier). Env kill-switch:
    CITATION_GUARD=0 disables the guard entirely.

    Args:
        state: Current AgentState

    Returns:
        Dict with final_answer and agent_answers.
    """
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    guard_note = ""
    guard_redirect = False
    if is_valid and os.environ.get("CITATION_GUARD", "1") != "0":
        try:
            from .citation_guard import enforce_citation_guard
            answer, report = enforce_citation_guard(
                answer, state.get("retrieval_keys", set()),
                tool_messages=state.get("messages") or [])
            if report.get("dropped"):
                guard_note = f"[citation_guard] removed {report['dropped']} unverifiable source line(s)"
            # haiku.rag-style citation policy: an answer whose Sources were
            # ALL hallucinated (every line dropped) must not ship — set the
            # redirect flag so the graph can send the agent back to the
            # orchestrator with feedback instead of finalizing this answer.
            if report.get("dropped") and not report.get("kept"):
                guard_redirect = True
        except Exception as e:  # guard must never break answering
            guard_note = f"[citation_guard] skipped ({type(e).__name__})"
        # Answer-side hygiene (paper-qa / CogDoc mechanism): strip inline
        # (Author, year) parentheticals that match no retrieved parent and
        # flag malformed citation tokens. Zero LLM; separate kill-switch.
        try:
            from .citation_guard import (
                enforce_answer_side_hygiene, parent_ids_from_keys,
                parent_ids_from_tool_messages)
            known = parent_ids_from_keys(state.get("retrieval_keys", set()))
            known |= parent_ids_from_tool_messages(state.get("messages") or [])
            answer, side_report = enforce_answer_side_hygiene(answer, known)
            n_stripped = len(side_report.get("stripped_inline", []))
            n_malformed = len(side_report.get("malformed_tokens", []))
            if n_stripped or n_malformed:
                parts = []
                if n_stripped:
                    parts.append(f"{n_stripped} inline citation(s)")
                if n_malformed:
                    parts.append(f"{n_malformed} malformed token(s)")
                note = f"[citation_guard] answer-side: removed " + " + flagged ".join(parts)
                guard_note = f"{guard_note}; {note}" if guard_note else note
        except Exception:
            pass  # answer-side hygiene is best-effort; never break answering

    # Citation-policy redirect (haiku.rag mechanism): an answer with zero
    # surviving Sources goes back to the orchestrator with feedback instead of
    # shipping. Capped — after GUARD_REDIRECT_MAX retries the answer ships
    # with its guard_note so the loop can never wedge the graph.
    max_redirects = int(os.environ.get("GUARD_REDIRECT_MAX", "1"))
    if guard_redirect:
        note = (guard_note + "; " if guard_note else "") \
            + "[citation_guard] redirected: all Sources lines unverifiable"
        if state.get("guard_retries", 0) < max_redirects:
            feedback = (
                "Your previous answer failed the citation policy: every entry in "
                "its Sources section was unverifiable against retrieved evidence "
                "and was removed. Answer ONLY from the retrieved evidence, and "
                "cite only papers that actually appear in your retrieval results. "
                "If the evidence does not support an answer, say so explicitly "
                "and do not invent sources."
            )
            return {
                "guard_redirect": True,
                "guard_retries": state.get("guard_retries", 0) + 1,
                "final_answer": "",
                "guard_note": note,
                "messages": ([
                    # Drop the failed answer so the retry does not see it as
                    # the latest AI turn; feedback rides as a user turn.
                    RemoveMessage(id=last_message.id),
                ] if getattr(last_message, "id", None) else []) + [
                    HumanMessage(content=feedback),
                ],
            }
        # budget exhausted → ship the (source-less) answer, transparently noted
        return {
            "final_answer": answer,
            "guard_note": note,
            "agent_answers": [
                {
                    "index": state["question_index"],
                    "question": state["question"],
                    "answer": answer,
                    "retrieval_keys": sorted(state.get("retrieval_keys", set()) or set()),
                }
            ],
        }

    # Citation-policy redirect note: if a previous round already redirected,
    # keep that fact in the final note (guard_note is a plain channel — a
    # fresh '' here would erase the redirect audit trail).
    prior_note = state.get("guard_note", "") or ""
    if guard_note and prior_note and prior_note not in guard_note:
        guard_note = f"{prior_note}; {guard_note}"
    elif not guard_note and prior_note:
        guard_note = prior_note

    return {
        "final_answer": answer,
        "guard_note": guard_note,
        # LangGraph channels persist values unless overwritten — an earlier
        # redirect on this subgraph run must be cleared here or
        # route_after_collect would loop back to the orchestrator forever.
        "guard_redirect": False,
        "agent_answers": [
            {
                "index": state["question_index"],
                "question": state["question"],
                "answer": answer,
                # Evidence-ledger passthrough for evaluate.citation_faithfulness:
                # Send()-spawned subgraph AgentState fields are not visible in
                # the main-graph checkpoint, so the whitelist keys ride along
                # here (LangGraph: only State-level channels persist).
                "retrieval_keys": sorted(state.get("retrieval_keys", set()) or set()),
            }
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

    # A single agent answer is already a complete, sourced response — an
    # extra aggregation round-trip would only re-generate it (~25-30s on a
    # slow local model). Return it as-is.
    if len(state["agent_answers"]) == 1:
        return {"messages": [AIMessage(content=state["agent_answers"][0]["answer"])]}

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
