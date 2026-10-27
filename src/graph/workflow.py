"""LangGraph workflow assembly and compilation for multi-agent debate."""
from typing import Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.judge import JudgeAgent
from src.agents.opponent import OpponentAgent
from src.agents.proponent import ProponentAgent
from src.graph.edges import should_continue_debate
from src.graph.nodes import DebateNodes, initialize_node
from src.graph.state import DebateState


def build_debate_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    proponent_agent: Optional[ProponentAgent] = None,
    opponent_agent: Optional[OpponentAgent] = None,
    judge_agent: Optional[JudgeAgent] = None,
):
    """
    Constructs and compiles the cyclic multi-agent debate StateGraph.

    Workflow:
        START -> initialize -> proponent -> opponent -> (conditional check)
                                  ^            |
                                  |--(continue)-
                                  |
                                  +--(conclude)-> judge -> END

    Args:
        checkpointer: Optional checkpoint saver (defaults to in-memory MemorySaver).
        proponent_agent: Optional pre-configured ProponentAgent instance.
        opponent_agent: Optional pre-configured OpponentAgent instance.
        judge_agent: Optional pre-configured JudgeAgent instance.

    Returns:
        Compiled LangGraph Pregel application.
    """
    nodes = DebateNodes(
        proponent_agent=proponent_agent,
        opponent_agent=opponent_agent,
        judge_agent=judge_agent,
    )

    builder = StateGraph(DebateState)

    # Register Nodes
    builder.add_node("initialize", initialize_node)
    builder.add_node("proponent", nodes.proponent_node)
    builder.add_node("opponent", nodes.opponent_node)
    builder.add_node("judge", nodes.judge_node)

    # Standard Edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "proponent")
    builder.add_edge("proponent", "opponent")

    # Conditional turn routing
    builder.add_conditional_edges(
        "opponent",
        should_continue_debate,
        {
            "continue": "proponent",
            "conclude": "judge",
        },
    )

    # Conclusion Edge
    builder.add_edge("judge", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)

