"""Unit tests for individual agent LLM provider connections and call_llm methods."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.proponent import ProponentAgent
from src.agents.opponent import OpponentAgent
from src.agents.judge import JudgeAgent
from src.agents.provider import LLMProvider
from src.core.exceptions import ProviderAuthenticationError, ProviderCallError
from src.core.config import get_settings


class SampleArgument(BaseModel):
    statement: str
    strength: float


@pytest.mark.asyncio
async def test_proponent_agent_init_openrouter():
    """Verify OpenRouter client is properly initialized with reasoning disabled."""
    from langchain_openrouter import ChatOpenRouter

    agent = ProponentAgent(
        claim="Nuclear energy is clean.",
        model_name="anthropic/claude-3.5-sonnet",
    )
    assert agent.name == "proponent"
    assert agent.provider == LLMProvider.OPENROUTER
    assert agent.model_name == "anthropic/claude-3.5-sonnet"
    assert agent.client is not None
    assert isinstance(agent.client, ChatOpenRouter)
    assert agent.client.reasoning == {"effort": "none"}


@pytest.mark.asyncio
async def test_opponent_agent_init_groq():
    """Verify Groq client is properly initialized for OpponentAgent."""
    agent = OpponentAgent(
        claim="Nuclear energy is clean.",
        model_name="llama-3.3-70b-versatile",
    )
    assert agent.name == "opponent"
    assert agent.provider == LLMProvider.GROQ
    assert agent.model_name == "llama-3.3-70b-versatile"
    assert agent.client is not None


def test_missing_groq_api_key_raises_error(monkeypatch):
    """Test that missing GROQ_API_KEY raises ProviderAuthenticationError for Opponent."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        OpponentAgent(claim="Nuclear energy is safe.")
    assert "GROQ_API_KEY is required" in str(exc_info.value)


def test_missing_openrouter_api_key_raises_error(monkeypatch):
    """Test that missing OPENROUTER_API_KEY raises ProviderAuthenticationError for Proponent."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        ProponentAgent(claim="Nuclear energy is safe.")
    assert "OPENROUTER_API_KEY is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_call_llm_plain_text():
    """Test standard call_llm invocation returning an AIMessage on ProponentAgent."""
    mock_client = MagicMock()
    mock_response = AIMessage(content="Nuclear energy is reliable and clean.")
    mock_client.ainvoke = AsyncMock(return_value=mock_response)

    agent = ProponentAgent(
        claim="Nuclear energy is clean.",
        client=mock_client,
        enable_failover=False,
    )

    messages = [HumanMessage(content="Is nuclear energy viable?")]
    result = await agent.call_llm(messages)

    assert result == mock_response
    assert result.content == "Nuclear energy is reliable and clean."
    mock_client.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_llm_structured_output():
    """Test call_llm using structured output schema on OpponentAgent."""
    expected_output = SampleArgument(statement="Capital expenditure is too high.", strength=8.5)
    mock_runnable = MagicMock()
    mock_runnable.ainvoke = AsyncMock(return_value=expected_output)

    mock_client = MagicMock()
    mock_client.with_structured_output = MagicMock(return_value=mock_runnable)

    agent = OpponentAgent(
        claim="Nuclear energy is clean.",
        client=mock_client,
    )

    messages = [HumanMessage(content="Challenge the cost of nuclear energy.")]
    result = await agent.call_llm(messages, structured_schema=SampleArgument)

    assert result.statement == "Capital expenditure is too high."
    assert result.strength == 8.5
    mock_client.with_structured_output.assert_called_once_with(SampleArgument)
    mock_runnable.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_llm_failure_raises_provider_call_error():
    """Test that runtime exceptions during LLM call trigger retries and raise ProviderCallError."""
    mock_client = MagicMock()
    mock_client.ainvoke = AsyncMock(side_effect=RuntimeError("Connection timeout"))

    agent = ProponentAgent(
        claim="Nuclear energy is clean.",
        client=mock_client,
        enable_failover=False,
    )

    messages = [HumanMessage(content="Hello")]
    with pytest.raises(ProviderCallError) as exc_info:
        await agent.call_llm(messages)

    assert "Connection timeout" in str(exc_info.value)
    assert exc_info.value.details["model"] == agent.model_name
    assert mock_client.ainvoke.await_count >= 1


def test_format_exception_message_connect_timeout():
    """Verify that empty-message exceptions like httpx.ConnectTimeout produce clear error messages."""
    import httpx
    from src.core.exceptions import format_exception_message

    timeout_exc = httpx.ConnectTimeout("")
    msg = format_exception_message(timeout_exc)
    assert "ConnectTimeout" in msg
    assert "Network connection timed out" in msg

    runtime_exc = RuntimeError("Service unavailable")
    msg2 = format_exception_message(runtime_exc)
    assert msg2 == "RuntimeError: Service unavailable"


