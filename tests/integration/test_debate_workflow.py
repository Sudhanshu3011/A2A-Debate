"""Integration tests for the LangGraph multi-agent debate workflow."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.proponent import ProponentAgent
from src.agents.opponent import OpponentAgent
from src.agents.judge import JudgeAgent
from src.agents.provider import LLMProvider
from src.graph.state import initialize_state
from src.graph.workflow import build_debate_graph
from src.schemas.debate import ArgumentTurn
from src.schemas.verdict import DebateVerdict


@pytest.fixture
def mock_debate_agents():
    """Builds mocked agents for fast, deterministic end-to-end graph execution."""
    # Mock Proponent
    proponent_mock = MagicMock(spec=ProponentAgent)
    async def mock_proponent_speech(round_number, opponent_previous_turn=None, prior_proponent_points=None):
        return ArgumentTurn(
            speaker="proponent",
            round_number=round_number,
            content=f"Proponent speech for round {round_number}.",
            key_points=[f"Pro point {round_number}A", f"Pro point {round_number}B"],
            rebuttals=[f"Rebuttal {round_number}"] if round_number > 1 else [],
        )
    proponent_mock.generate_speech = AsyncMock(side_effect=mock_proponent_speech)

    # Mock Opponent
    opponent_mock = MagicMock(spec=OpponentAgent)
    async def mock_opponent_speech(round_number, proponent_current_turn, prior_opponent_points=None):
        return ArgumentTurn(
            speaker="opponent",
            round_number=round_number,
            content=f"Opponent speech for round {round_number}.",
            key_points=[f"Opp point {round_number}A"],
            rebuttals=[f"Counter-rebuttal {round_number}"],
        )
    opponent_mock.generate_speech = AsyncMock(side_effect=mock_opponent_speech)

    # Mock Judge
    judge_mock = MagicMock(spec=JudgeAgent)
    async def mock_evaluate(turns):
        return DebateVerdict(
            winner="proponent",
            confidence_score=0.9,
            proponent_score=8.5,
            opponent_score=7.2,
            key_strengths_proponent=["Strong opening arguments"],
            key_strengths_opponent=["Valid critique"],
            deciding_factors=["Better evidence"],
            synthesis_summary="The proponent presented a more coherent empirical defense.",
        )
    judge_mock.evaluate_debate = AsyncMock(side_effect=mock_evaluate)

    return proponent_mock, opponent_mock, judge_mock


@pytest.mark.asyncio
async def test_single_round_debate_execution(mock_debate_agents):
    """Test full execution of a 1-round debate graph."""
    pro_agent, opp_agent, judge_agent = mock_debate_agents

    graph = build_debate_graph(
        proponent_agent=pro_agent,
        opponent_agent=opp_agent,
        judge_agent=judge_agent,
    )

    initial_state = initialize_state(
        claim="Space exploration should be publicly funded",
        max_rounds=1,
        debate_id="test-single-round-101",
    )

    config = {"configurable": {"thread_id": "thread-1"}}
    final_state = await graph.ainvoke(initial_state, config=config)

    assert final_state["is_completed"] is True
    assert final_state["debate_id"] == "test-single-round-101"
    assert len(final_state["turns"]) == 2  # 1 Proponent, 1 Opponent

    # Verify turn sequence
    turns = final_state["turns"]
    assert turns[0]["speaker"] == "proponent"
    assert turns[0]["round_number"] == 1
    assert turns[1]["speaker"] == "opponent"
    assert turns[1]["round_number"] == 1

    # Verify verdict synthesis
    assert final_state["verdict"] is not None
    assert final_state["verdict"]["winner"] == "proponent"
    assert final_state["verdict"]["confidence_score"] == 0.9

    # Verify mock call counts
    assert pro_agent.generate_speech.await_count == 1
    assert opp_agent.generate_speech.await_count == 1
    assert judge_agent.evaluate_debate.await_count == 1


@pytest.mark.asyncio
async def test_multi_round_debate_execution(mock_debate_agents):
    """Test full execution of a 2-round debate graph verifying cyclical iteration."""
    pro_agent, opp_agent, judge_agent = mock_debate_agents

    graph = build_debate_graph(
        proponent_agent=pro_agent,
        opponent_agent=opp_agent,
        judge_agent=judge_agent,
    )

    initial_state = initialize_state(
        claim="Remote work is superior to in-office work",
        max_rounds=2,
        debate_id="test-multi-round-202",
    )

    config = {"configurable": {"thread_id": "thread-2"}}
    final_state = await graph.ainvoke(initial_state, config=config)

    assert final_state["is_completed"] is True
    # 2 rounds * 2 speeches per round = 4 total speeches
    assert len(final_state["turns"]) == 4

    turns = final_state["turns"]
    assert [t["speaker"] for t in turns] == ["proponent", "opponent", "proponent", "opponent"]
    assert [t["round_number"] for t in turns] == [1, 1, 2, 2]

    # Verify arguments accumulated
    assert len(final_state["proponent_arguments"]) == 4  # 2 points in R1 + 2 in R2
    assert len(final_state["opponent_arguments"]) == 2   # 1 point in R1 + 1 in R2

    assert pro_agent.generate_speech.await_count == 2
    assert opp_agent.generate_speech.await_count == 2
    assert judge_agent.evaluate_debate.await_count == 1

