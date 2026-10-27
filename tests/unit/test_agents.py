"""Unit tests for ProponentAgent, OpponentAgent, and JudgeAgent."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.proponent import ProponentAgent
from src.agents.opponent import OpponentAgent
from src.agents.judge import JudgeAgent
from src.agents.provider import LLMProvider
from src.schemas.debate import ArgumentTurn
from src.schemas.verdict import DebateVerdict


@pytest.mark.asyncio
async def test_proponent_agent_opening_speech():
    """Verify ProponentAgent produces round 1 opening speech."""
    mock_turn = ArgumentTurn(
        speaker="proponent",
        round_number=1,
        content="AI accelerates medical discovery and code development.",
        key_points=["Health innovation", "Productivity gains"],
    )
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_turn)
    mock_client = MagicMock()
    mock_client.with_structured_output = MagicMock(return_value=mock_runnable)

    agent = ProponentAgent(
        claim="AI offers net positive benefits to society.",
        provider=LLMProvider.OPENROUTER,
        client=mock_client,
    )

    turn = await agent.generate_speech(round_number=1)
    assert turn.speaker == "proponent"
    assert turn.round_number == 1
    assert "AI accelerates" in turn.content
    assert len(turn.key_points) == 2


@pytest.mark.asyncio
async def test_opponent_agent_rebuttal_speech():
    """Verify OpponentAgent rebuts proponent's argument in round 1."""
    proponent_speech = ArgumentTurn(
        speaker="proponent",
        round_number=1,
        content="AI accelerates medical discovery and code development.",
        key_points=["Health innovation", "Productivity gains"],
    )

    mock_opp_turn = ArgumentTurn(
        speaker="opponent",
        round_number=1,
        content="AI poses significant alignment risks, job displacement, and data privacy issues.",
        key_points=["Job loss", "Privacy violations"],
        rebuttals=["Productivity gains are offset by structural unemployment."],
    )
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_opp_turn)
    mock_client = MagicMock()
    mock_client.with_structured_output = MagicMock(return_value=mock_runnable)

    agent = OpponentAgent(
        claim="AI offers net positive benefits to society.",
        provider=LLMProvider.GROQ,
        client=mock_client,
    )

    turn = await agent.generate_speech(
        round_number=1,
        proponent_current_turn=proponent_speech,
    )
    assert turn.speaker == "opponent"
    assert turn.round_number == 1
    assert "alignment risks" in turn.content
    assert len(turn.rebuttals) == 1


@pytest.mark.asyncio
async def test_judge_agent_verdict_evaluation():
    """Verify JudgeAgent correctly aggregates turns and outputs DebateVerdict."""
    turns = [
        ArgumentTurn(
            speaker="proponent",
            round_number=1,
            content="AI helps cure diseases.",
            key_points=["Medical advancements"],
        ),
        ArgumentTurn(
            speaker="opponent",
            round_number=1,
            content="AI poses existential risk.",
            key_points=["Existential safety"],
        ),
    ]

    mock_verdict = DebateVerdict(
        winner="proponent",
        confidence_score=0.8,
        proponent_score=8.0,
        opponent_score=6.5,
        key_strengths_proponent=["Strong empirical examples"],
        key_strengths_opponent=["Valid long-term safety concerns"],
        deciding_factors=["Proponent focused on immediate verifiable benefits"],
        synthesis_summary="Both sides debated well, but proponent was more pragmatic.",
    )

    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=mock_verdict)
    mock_client = MagicMock()
    mock_client.with_structured_output = MagicMock(return_value=mock_runnable)

    judge = JudgeAgent(
        claim="AI offers net positive benefits to society.",
        provider=LLMProvider.OPENROUTER,
        client=mock_client,
    )

    verdict = await judge.evaluate_debate(turns=turns)
    assert verdict.winner == "proponent"
    assert verdict.confidence_score == 0.8
    assert verdict.proponent_score == 8.0
    assert verdict.opponent_score == 6.5


def test_agents_resolve_models_from_env_variables(monkeypatch):
    """Verify that Proponent, Opponent, and Judge agents read model names from environment variables."""
    from src.core.config import get_settings

    monkeypatch.setenv("PROPONENT_MODEL", "meta-llama/llama-3.3-70b-instruct")
    monkeypatch.setenv("OPPONENT_MODEL", "mixtral-8x7b-32768")
    monkeypatch.setenv("JUDGE_MODEL", "openai/gpt-4o")
    get_settings.cache_clear()

    proponent = ProponentAgent(claim="Test claim", client=MagicMock())
    opponent = OpponentAgent(claim="Test claim", client=MagicMock())
    judge = JudgeAgent(claim="Test claim", client=MagicMock())

    assert proponent.model_name == "meta-llama/llama-3.3-70b-instruct"
    assert opponent.model_name == "mixtral-8x7b-32768"
    assert judge.model_name == "openai/gpt-4o"


