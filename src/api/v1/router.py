"""API v1 router registry."""
from fastapi import APIRouter
from src.api.v1.endpoints.debate import router as debate_router

api_v1_router = APIRouter()
api_v1_router.include_router(debate_router)

