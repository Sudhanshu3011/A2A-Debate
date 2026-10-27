"""Unit tests for conditional edges in the debate graph."""
import pytest
from src.graph.edges import should_continue_debate
from src.graph.state import initialize_state


def test_should_continue_debate_returns_continue():
    """Verify routing continues when current round is within max rounds."""
    state = initialize_state(claim="Test claim", max_rounds=3)
    state["current_round"] = 2
    assert should_continue_debate(state) == "continue"


def test_should_continue_debate_returns_conclude_when_rounds_exceeded():
    """Verify routing concludes when current round exceeds max rounds."""
    state = initialize_state(claim="Test claim", max_rounds=2)
    state["current_round"] = 3
    assert should_continue_debate(state) == "conclude"


def test_should_continue_debate_returns_conclude_when_completed():
    """Verify routing concludes when debate is already flagged as completed."""
    state = initialize_state(claim="Test claim", max_rounds=3)
    state["is_completed"] = True
    assert should_continue_debate(state) == "conclude"

