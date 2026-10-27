"""LangGraph state schema for the debate workflow."""
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DebateState(TypedDict):
    """
    Central state passed across all LangGraph nodes in the debate workflow.
    """

    # Debate configuration
    debate_id: str
    claim: str
    rules: str
    max_rounds: int

    # Dynamic workflow tracking
    current_round: int
    current_speaker: Literal["proponent", "opponent", "judge"]

    # History of turns
    turns: List[Dict[str, Any]]

    # LangChain message trace with append reducer
    messages: Annotated[List[BaseMessage], add_messages]

    # Cumulative extracted arguments
    proponent_arguments: List[str]
    opponent_arguments: List[str]

    # Debate status and resolution
    is_completed: bool
    verdict: Optional[Dict[str, Any]]


def initialize_state(
    claim: str,
    max_rounds: int = 3,
    rules: Optional[str] = None,
    debate_id: Optional[str] = None,
) -> DebateState:
    """
    Initializes a clean, standardized DebateState at the start of a debate conversation.

    Args:
        claim: The central claim or motion to be debated.
        max_rounds: Maximum debate rounds before proceeding to adjudication (default: 3).
        rules: Debate conduct guidelines and rules.
        debate_id: Optional unique identifier. If not provided, a UUID will be generated.

    Returns:
        A pristine DebateState ready for LangGraph invocation.
    """
    default_rules = (
        "Maintain civil, logic-driven debate. Refute assertions with evidence. "
        "Avoid logical fallacies."
    )

    return {
        "debate_id": debate_id or f"deb-{uuid.uuid4().hex[:10]}",
        "claim": claim.strip(),
        "rules": rules.strip() if rules else default_rules,
        "max_rounds": max(1, max_rounds),
        "current_round": 1,
        "current_speaker": "proponent",
        "turns": [],
        "messages": [],
        "proponent_arguments": [],
        "opponent_arguments": [],
        "is_completed": False,
        "verdict": None,
    }



