"""Schemas package exports."""
from src.schemas.debate import ArgumentTurn, DebateRequest, DebateResponse
from src.schemas.verdict import DebateVerdict

__all__ = [
    "ArgumentTurn",
    "DebateRequest",
    "DebateResponse",
    "DebateVerdict",
]

