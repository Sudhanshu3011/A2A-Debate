"""Debate execution and retrieval API endpoints."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_debate_service
from src.core.exceptions import DebateAppException, ProviderCallError, format_exception_message
from src.schemas.debate import DebateRequest, DebateResponse
from src.services.debate_service import DebateService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debate", tags=["Debate"])


@router.post(
    "/run",
    response_model=DebateResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute full debate",
    description="Initiates a multi-round debate between Proponent and Opponent agents, concluding with an objective Judge verdict.",
)
async def run_debate(
    request: DebateRequest,
    service: DebateService = Depends(get_debate_service),
) -> DebateResponse:
    """Executes a full non-streaming multi-round debate."""
    try:
        response = await service.run_debate(request)
        return response
    except DebateAppException as exc:
        logger.error(
            "Debate execution failed [%s]: %s | Details: %s",
            exc.__class__.__name__,
            exc.message,
            exc.details,
        )
        status_code = (
            status.HTTP_502_BAD_GATEWAY
            if isinstance(exc, ProviderCallError)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        ) from exc
    except Exception as exc:
        err_msg = format_exception_message(exc)
        logger.error("Unexpected error during debate execution: %s", err_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debate execution encountered an unexpected error: {err_msg}",
        ) from exc


@router.get(
    "/{debate_id}",
    response_model=DebateResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve debate by ID",
    description="Fetches completed debate transcript and verdict from cache or storage.",
)
async def get_debate(
    debate_id: str,
    service: DebateService = Depends(get_debate_service),
) -> DebateResponse:
    """Retrieves an existing debate result."""
    debate = await service.get_debate(debate_id)
    if not debate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Debate with ID '{debate_id}' not found.",
        )
    return debate

