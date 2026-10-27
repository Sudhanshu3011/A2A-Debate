"""Tests for concurrency, rate limiting failover, and structured output recovery."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.proponent import ProponentAgent
from src.agents.provider import LLMProvider
from src.schemas.debate import ArgumentTurn, DebateRequest
from src.schemas.verdict import DebateVerdict
from src.services.debate_service import DebateService


class MockTargetSchema(BaseModel):
    content: str
    key_points: list[str] = []


@pytest.mark.asyncio
async def test_structured_output_recovery_from_markdown():
    """Verify that malformed/markdown structured output is recovered using regex extraction."""
    mock_client = MagicMock()
    # Simulate structured parser throwing OutputParserException
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(side_effect=OutputParserException("Failed to parse JSON"))
    mock_client.with_structured_output = MagicMock(return_value=mock_runnable)

    # Raw response containing markdown json
    raw_markdown = 'Here is the response:\n```json\n{"content": "Extracted from markdown successfully", "key_points": ["Point A"]}\n```'
    mock_client.ainvoke = AsyncMock(return_value=AIMessage(content=raw_markdown))

    agent = ProponentAgent(
        claim="Defend claim",
        client=mock_client,
        enable_failover=False,
    )

    result = await agent.call_llm(
        messages=[HumanMessage(content="Hello")],
        structured_schema=MockTargetSchema,
    )

    assert result.content == "Extracted from markdown successfully"
    assert result.key_points == ["Point A"]


@pytest.mark.asyncio
async def test_structured_output_fallback_construction():
    """Verify that unparseable raw text gracefully falls back to structured object construction."""
    mock_client = MagicMock()
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(side_effect=OutputParserException("Invalid JSON"))
    mock_client.with_structured_output = MagicMock(return_value=mock_runnable)

    # Completely non-JSON unstructured plain text
    raw_plain_text = "Just a raw speech without any JSON formatting."
    mock_client.ainvoke = AsyncMock(return_value=AIMessage(content=raw_plain_text))

    agent = ProponentAgent(
        claim="Defend claim",
        client=mock_client,
        enable_failover=False,
    )

    result: ArgumentTurn = await agent.call_llm(
        messages=[HumanMessage(content="Hello")],
        structured_schema=ArgumentTurn,
    )

    assert result.content == raw_plain_text
    assert result.speaker == "proponent"
    assert result.round_number == 1


@pytest.mark.asyncio
async def test_rate_limit_failover(monkeypatch):
    """Verify that hitting a 429 rate limit triggers failover to the secondary provider."""
    # Mock primary client with 429 Rate Limit error
    primary_client = MagicMock()
    primary_client.ainvoke = AsyncMock(side_effect=RuntimeError("HTTP 429: Too Many Requests - TPM limit reached"))

    # Mock failover client with successful response
    failover_client = MagicMock()
    failover_client.ainvoke = AsyncMock(return_value=AIMessage(content="Failover provider speech success."))

    agent = ProponentAgent(
        claim="Defend claim",
        client=primary_client,
        enable_failover=True,
    )

    # Mock failover client initialization
    agent._init_failover_client = MagicMock(return_value=failover_client)

    result = await agent.call_llm(messages=[HumanMessage(content="Hello")])
    assert result.content == "Failover provider speech success."


@pytest.mark.asyncio
async def test_concurrent_debate_runs():
    """Verify multiple concurrent debates run in parallel with isolated states."""
    service = DebateService()

    # Create mock agents
    async def create_mock_agents():
        pro = MagicMock()
        pro.generate_speech = AsyncMock(
            return_value=ArgumentTurn(speaker="proponent", round_number=1, content="Pro speech")
        )
        opp = MagicMock()
        opp.generate_speech = AsyncMock(
            return_value=ArgumentTurn(speaker="opponent", round_number=1, content="Opp speech")
        )
        judge = MagicMock()
        judge.evaluate_debate = AsyncMock(
            return_value=DebateVerdict(
                winner="draw",
                confidence_score=0.7,
                proponent_score=7.0,
                opponent_score=7.0,
                synthesis_summary="Evenly matched.",
            )
        )
        return pro, opp, judge

    agents1 = await create_mock_agents()
    agents2 = await create_mock_agents()
    agents3 = await create_mock_agents()

    req1 = DebateRequest(claim="Claim A: Solar is best", max_rounds=1)
    req2 = DebateRequest(claim="Claim B: Wind is best", max_rounds=1)
    req3 = DebateRequest(claim="Claim C: Hydro is best", max_rounds=1)

    # Run 3 debates concurrently
    res1, res2, res3 = await asyncio.gather(
        service.run_debate(req1, proponent_agent=agents1[0], opponent_agent=agents1[1], judge_agent=agents1[2]),
        service.run_debate(req2, proponent_agent=agents2[0], opponent_agent=agents2[1], judge_agent=agents2[2]),
        service.run_debate(req3, proponent_agent=agents3[0], opponent_agent=agents3[1], judge_agent=agents3[2]),
    )

    # Verify each debate maintains state isolation
    assert res1.claim == "Claim A: Solar is best"
    assert res2.claim == "Claim B: Wind is best"
    assert res3.claim == "Claim C: Hydro is best"
    assert res1.debate_id != res2.debate_id != res3.debate_id
    assert res1.is_completed and res2.is_completed and res3.is_completed

