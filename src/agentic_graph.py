"""
Main LangGraph workflow for bib_rag agentic RAG.

Builds a two-level graph architecture:
1. Main graph: summarize_history → rewrite_query → [clarification?] → agent subgraphs → aggregate_answers
2. Agent subgraph: orchestrator → tools → compress_context? → orchestrator (loop) → collect_answer

Uses ChromaDB (bge-m3 embeddings) and OpenAI-compatible API via llama-server (Qwen3.6-35B).

Usage:
    from src.agentic_graph import create_agent_graph
    from src.agent_tools import ToolFactory

    llm = ChatOpenAI(base_url="http://localhost:5015/v1", model="qwen3.6-35b")
    factory = ToolFactory()
    tools = factory.create_tools()  # list of LangChain tools
    graph = create_agent_graph(llm, tools)

    result = graph.invoke({"messages": [("user", "What is the role of Eph receptors?")]})
    print(result["messages"][-1].content)
"""

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from functools import partial

from .agent_schemas import State, AgentState
from .agent_nodes import (
    summarize_history,
    rewrite_query,
    request_clarification,
    orchestrator,
    fallback_response,
    should_compress_context,
    compress_context,
    collect_answer,
    aggregate_answers,
)
from .agent_edges import route_after_rewrite, route_after_orchestrator_call, route_after_collect


def create_agent_graph(llm, tools_list):
    """Create and compile the agentic RAG graph.

    Args:
        llm: LangChain chat model (OpenAI-compatible, e.g. via ChatOpenAI with custom base_url)
        tools_list: List of LangChain tool objects (created via ToolFactory.create_tools())

    Returns:
        Compiled LangGraph graph with InMemorySaver checkpointer.
        Interrupts before request_clarification to allow user feedback.
    """
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    checkpointer = InMemorySaver()

    # --- Agent subgraph ---
    print("Building agent subgraph...")
    agent_builder = StateGraph(AgentState)
    agent_builder.add_node(
        "orchestrator",
        partial(orchestrator, llm_with_tools=llm_with_tools),
    )
    agent_builder.add_node("tools", tool_node)
    agent_builder.add_node(
        "compress_context",
        partial(compress_context, llm=llm),
    )
    agent_builder.add_node(
        "fallback_response",
        partial(fallback_response, llm=llm),
    )
    agent_builder.add_node(should_compress_context)
    agent_builder.add_node(collect_answer)

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {
            "tools": "tools",
            "fallback_response": "fallback_response",
            "collect_answer": "collect_answer",
        },
    )
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_conditional_edges(
        "collect_answer",
        route_after_collect,
        {
            "orchestrator": "orchestrator",
            "__end__": END,
        },
    )

    agent_subgraph = agent_builder.compile()

    # --- Main graph ---
    print("Building main graph...")
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "summarize_history",
        partial(summarize_history, llm=llm),
    )
    graph_builder.add_node(
        "rewrite_query",
        partial(rewrite_query, llm=llm),
    )
    graph_builder.add_node(request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node(
        "aggregate_answers",
        partial(aggregate_answers, llm=llm),
    )

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
    )
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"],
    )

    print("✓ Agent graph compiled successfully.")
    return agent_graph
