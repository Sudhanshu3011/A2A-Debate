"""Pydantic schemas for debate request, response, and turns."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from src.schemas.verdict import DebateVerdict


class ArgumentTurn(BaseModel):
    """Structured representation of a single speech in the debate."""

    speaker: Literal["proponent", "opponent", "judge"] = Field(
        description="Active speaker persona."
    )
    round_number: int = Field(ge=1, description="Debate round index.")
    content: str = Field(description="Full text speech delivered in this turn.")
    key_points: List[str] = Field(
        default_factory=list,
        description="Core assertions and positive arguments made.",
    )
    rebuttals: List[str] = Field(
        default_factory=list,
        description="Direct counterarguments attacking the opponent's previous points.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of the turn.",
    )

    @field_validator("key_points", "rebuttals", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            lines = [line.strip("- *").strip() for line in v.split("\n") if line.strip("- *").strip()]
            return lines if lines else [v]
        elif isinstance(v, list):
            return [str(item) for item in v]
        return []


class DebateRequest(BaseModel):
    """Input payload to initiate a multi-round debate."""

    claim: str = Field(
        ...,
        min_length=5,
        description="The central proposition or claim to be debated.",
        examples=["Artificial general intelligence will be achieved by 2030."],
    )
    rules: Optional[str] = Field(
        default="Maintain civil, logic-driven debate. Refute assertions with evidence. Avoid logical fallacies.",
        description="Ground rules and debate conduct guidelines.",
    )
    max_rounds: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum debate cycles before moving to judgment.",
    )


class DebateResponse(BaseModel):
    """Output payload returning the completed debate results and verdict."""

    debate_id: str
    claim: str
    total_rounds: int
    turns: List[ArgumentTurn]
    verdict: Optional[DebateVerdict] = None
    is_completed: bool = True

