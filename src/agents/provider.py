"""LLM provider enumeration."""
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM service providers."""
    GROQ = "groq"
    OPENROUTER = "openrouter"

