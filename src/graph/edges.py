"""Conditional routing edges for the debate graph."""
import logging
from typing import Literal
from src.graph.state import DebateState

logger = logging.getLogger(__name__)


def should_continue_debate(state: DebateState) -> Literal["continue", "conclude"]:
    """
    Determines whether to continue to the next debate round or conclude with judgment.

    Args:
        state: Current DebateState.

    Returns:
        "continue" if current_round <= max_rounds and debate is active.
        "conclude" if max_rounds reached or debate marked completed.
    """
    if state.get("is_completed", False):
        logger.info("Debate %s already marked complete. Routing to conclude.", state.get("debate_id"))
        return "conclude"

    current_round = state.get("current_round", 1)
    max_rounds = state.get("max_rounds", 3)

    if current_round > max_rounds:
        logger.info(
            "Debate %s reached maximum rounds (%d > %d). Routing to judge.",
            state.get("debate_id"),
            current_round - 1,
            max_rounds,
        )
        return "conclude"

    logger.info(
        "Debate %s proceeding to round %d of %d.",
        state.get("debate_id"),
        current_round,
        max_rounds,
    )
    return "continue"

