"""Unit tests for configuration and custom exceptions."""
import pytest
from src.core.config import Settings, get_settings
from src.core.exceptions import DebateAppException, ProviderCallError


def test_settings_defaults():
    """Verify default configurations."""
    assert Settings.model_fields["DEFAULT_MAX_ROUNDS"].default == 3
    assert Settings.model_fields["REQUEST_TIMEOUT_SECONDS"].default == 30.0
    assert bool(Settings.model_fields["PROPONENT_MODEL"].default) is True
    assert bool(Settings.model_fields["OPPONENT_MODEL"].default) is True
    assert bool(Settings.model_fields["JUDGE_MODEL"].default) is True


def test_settings_environment_override(monkeypatch):
    """Verify settings pick up environment overrides."""
    monkeypatch.setenv("DEFAULT_MAX_ROUNDS", "5")
    monkeypatch.setenv("PROPONENT_MODEL", "meta-llama/llama-3.3-70b-instruct")
    monkeypatch.setenv("OPPONENT_MODEL", "mixtral-8x7b-32768")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.DEFAULT_MAX_ROUNDS == 5
    assert settings.PROPONENT_MODEL == "meta-llama/llama-3.3-70b-instruct"
    assert settings.OPPONENT_MODEL == "mixtral-8x7b-32768"


def test_custom_exception_details():
    """Verify custom exception details dict propagation."""
    exc = ProviderCallError("Call timed out", details={"provider": "groq", "status": 504})
    assert isinstance(exc, DebateAppException)
    assert exc.message == "Call timed out"
    assert exc.details["provider"] == "groq"
    assert exc.details["status"] == 504

