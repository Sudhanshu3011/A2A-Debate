"""FastAPI application initialization and middleware configuration."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1.router import api_v1_router
from src.core.config import get_settings
from src.core.exceptions import DebateAppException, ProviderCallError
from src.core.logging import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown routines."""
    settings = get_settings()
    logger.info("Starting Multi-Agent Debate Engine in %s mode...", settings.ENVIRONMENT)
    yield
    logger.info("Shutting down Multi-Agent Debate Engine...")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title="Multi-Agent Debate Service",
        description="Dialectical Multi-Agent Debate Architecture with LangGraph, LangChain, Groq, and OpenRouter.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom Exception Handler
    @app.exception_handler(DebateAppException)
    async def debate_exception_handler(request: Request, exc: DebateAppException):
        logger.error(
            "Debate domain exception [%s]: %s | Details: %s",
            exc.__class__.__name__,
            exc.message,
            exc.details,
        )
        status_code = (
            status.HTTP_502_BAD_GATEWAY
            if isinstance(exc, ProviderCallError)
            else status.HTTP_400_BAD_REQUEST
        )
        return JSONResponse(
            status_code=status_code,
            content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
        )

    # Health Check
    @app.get("/health", tags=["Health"], summary="Service health probe")
    async def health_check():
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "version": "0.1.0",
        }

    # Mount API v1 Routes
    app.include_router(api_v1_router, prefix="/api/v1")

    return app


app = create_app()

