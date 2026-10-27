"""Shared pytest fixtures."""
import os
import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.messages import AIMessage
from src.core.config import get_settings


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Sets dummy environment variables for tests."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-openrouter-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def mock_chat_model():
    """Returns a mock ChatModel simulating LangChain chat models."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=AIMessage(content="Debate statement mock response"))
    mock.with_structured_output = MagicMock()
    return mock

