"""Unit tests for DebateState and schema validations."""
import pytest
from src.schemas.debate import ArgumentTurn, DebateRequest, DebateResponse
from src.schemas.verdict import DebateVerdict
from src.graph.state import DebateState


def test_argument_turn_schema_valid():
    """Verify ArgumentTurn model instantiates and validates correctly."""
    turn = ArgumentTurn(
        speaker="proponent",
        round_number=1,
        content="Nuclear energy provides base-load zero-emission power.",
        key_points=["Low carbon footprint", "Reliable base-load"],
        rebuttals=[],
    )
    assert turn.speaker == "proponent"
    assert turn.round_number == 1
    assert len(turn.key_points) == 2
    assert turn.timestamp is not None


def test_debate_verdict_schema_bounds():
    """Verify DebateVerdict validates score constraints."""
    verdict = DebateVerdict(
        winner="proponent",
        confidence_score=0.85,
        proponent_score=8.5,
        opponent_score=7.0,
        key_strengths_proponent=["Solid empirical evidence on capacity factors"],
        key_strengths_opponent=["Accurate points regarding high initial capital outlay"],
        deciding_factors=["Superior rebuttal of waste management solutions"],
        synthesis_summary="The proponent effectively addressed the economic challenges.",
    )
    assert verdict.winner == "proponent"
    assert 0.0 <= verdict.confidence_score <= 1.0
    assert 0.0 <= verdict.proponent_score <= 10.0


def test_debate_state_dict_structure():
    """Verify DebateState dictionary structure matches expected contract."""
    state: DebateState = {
        "debate_id": "deb-123",
        "claim": "Remote work increases overall productivity",
        "rules": "Standard civil debate",
        "max_rounds": 2,
        "current_round": 1,
        "current_speaker": "proponent",
        "turns": [],
        "messages": [],
        "proponent_arguments": [],
        "opponent_arguments": [],
        "is_completed": False,
        "verdict": None,
    }
    assert state["debate_id"] == "deb-123"
    assert state["current_round"] == 1
    assert state["is_completed"] is False


def test_initialize_state_default_values():
    """Verify initialize_state produces clean state with defaults."""
    from src.graph.state import initialize_state

    state = initialize_state(claim="Universal Basic Income reduces poverty")

    assert state["claim"] == "Universal Basic Income reduces poverty"
    assert state["max_rounds"] == 3
    assert state["current_round"] == 1
    assert state["current_speaker"] == "proponent"
    assert state["turns"] == []
    assert state["messages"] == []
    assert state["proponent_arguments"] == []
    assert state["opponent_arguments"] == []
    assert state["is_completed"] is False
    assert state["verdict"] is None
    assert state["debate_id"].startswith("deb-")
    assert "Maintain civil, logic-driven debate" in state["rules"]

    # Verify custom parameters
    state_custom = initialize_state(claim="Test claim", max_rounds=5, debate_id="custom-id-999")
    assert state_custom["debate_id"] == "custom-id-999"
    assert state_custom["max_rounds"] == 5
    assert state_custom["claim"] == "Test claim"



