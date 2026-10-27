"""Custom application exceptions for the debate system."""
from typing import Optional, Dict, Any


def format_exception_message(exc: Exception) -> str:
    """
    Extracts a non-empty, informative description from an exception,
    handling cases like httpx.ConnectTimeout where str(exc) is empty.
    """
    exc_type = type(exc).__name__
    exc_str = str(exc).strip()
    if not exc_str:
        if "Timeout" in exc_type:
            return f"{exc_type}: Network connection timed out or host unreachable"
        return f"{exc_type}: No error message provided by runtime"
    return f"{exc_type}: {exc_str}"


class DebateAppException(Exception):
    """Base exception for all debate application errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProviderAuthenticationError(DebateAppException):
    """Raised when API credentials for an LLM provider are missing or invalid."""
    pass


class ProviderCallError(DebateAppException):
    """Raised when an LLM provider invocation fails after retries."""
    pass


class InvalidStateError(DebateAppException):
    """Raised when graph execution encounters invalid or corrupted state."""
    pass
