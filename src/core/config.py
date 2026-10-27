"""Application configuration using Pydantic Settings."""
import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Preload .env into os.environ for LangChain & LangSmith background tracers
load_dotenv()


class Settings(BaseSettings):
    """Central settings definition populated from environment variables or .env."""

    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # LangSmith Tracing & Observability
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "debate-a2a"

    # Groq Provider Settings
    GROQ_API_KEY: Optional[str] = None

    # OpenRouter Provider Settings
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_SITE_URL: str = "https://antigravity.internal"
    OPENROUTER_APP_NAME: str = "MultiAgent-Debate"

    # Agent Specific Model Configurations (from env)
    PROPONENT_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    OPPONENT_MODEL: str = "openai/gpt-oss-120b"
    JUDGE_MODEL: str = "nvidia/nemotron-3.5-super-vl:free"

    # Runtime Execution Defaults
    DEFAULT_MAX_ROUNDS: int = 3
    REQUEST_TIMEOUT_SECONDS: float = 30.0
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 4096

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of Settings and configures LangSmith."""
    settings = Settings()

    # Synchronize LangSmith / LangChain tracing variables into os.environ
    if settings.LANGSMITH_TRACING or os.getenv("LANGCHAIN_TRACING_V2") == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        if settings.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        if settings.LANGSMITH_PROJECT:
            os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
            os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
            # Remove conflicting agent address keys if set
            os.environ.pop("LANGSMITH_AGENT_ID", None)
            os.environ.pop("LANGSMITH_AGENT_ENVIRONMENT", None)
        if settings.LANGSMITH_ENDPOINT:
            os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
            os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

    return settings


