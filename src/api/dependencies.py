"""FastAPI dependency injection providers."""
from functools import lru_cache
from src.services.debate_service import DebateService


@lru_cache()
def get_debate_service() -> DebateService:
    """Provides a singleton instance of DebateService."""
    return DebateService()

