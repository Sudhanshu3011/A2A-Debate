"""Integration tests for FastAPI endpoints."""
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock

from src.main import app
from src.api.dependencies import get_debate_service
from src.schemas.debate import ArgumentTurn, DebateResponse
from src.schemas.verdict import DebateVerdict
from src.services.debate_service import DebateService
from src.core.exceptions import ProviderCallError


@pytest.fixture
def mock_service():
    """Returns a mock DebateService with deterministic responses."""
    service = MagicMock(spec=DebateService)

    sample_verdict = DebateVerdict(
        winner="proponent",
        confidence_score=0.92,
        proponent_score=8.8,
        opponent_score=7.4,
        key_strengths_proponent=["Strong empirical analysis"],
        key_strengths_opponent=["Good skepticism on timelines"],
        deciding_factors=["Comprehensive technological viability demonstration"],
        synthesis_summary="The proponent effectively substantiated the claim.",
    )

    sample_turns = [
        ArgumentTurn(
            speaker="proponent",
            round_number=1,
            content="Renewable energy is now cheaper than fossil fuels.",
            key_points=["LCOE metrics", "Grid storage breakthroughs"],
        ),
        ArgumentTurn(
            speaker="opponent",
            round_number=1,
            content="Intermittency and rare mineral supply chain bottlenecks persist.",
            key_points=["Intermittency", "Mineral dependency"],
        ),
    ]

    sample_response = DebateResponse(
        debate_id="test-deb-555",
        claim="Renewable energy can fully replace fossil fuels by 2040.",
        total_rounds=1,
        turns=sample_turns,
        verdict=sample_verdict,
        is_completed=True,
    )

    service.run_debate = AsyncMock(return_value=sample_response)
    service.get_debate = AsyncMock(
        side_effect=lambda deb_id: sample_response if deb_id == "test-deb-555" else None
    )
    return service


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Verify GET /health returns 200 with healthy status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.asyncio
async def test_run_debate_endpoint_success(mock_service):
    """Verify POST /api/v1/debate/run executes and returns DebateResponse."""
    app.dependency_overrides[get_debate_service] = lambda: mock_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "claim": "Renewable energy can fully replace fossil fuels by 2040.",
            "max_rounds": 1,
        }
        response = await client.post("/api/v1/debate/run", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["debate_id"] == "test-deb-555"
        assert len(data["turns"]) == 2
        assert data["verdict"]["winner"] == "proponent"
        assert data["verdict"]["confidence_score"] == 0.92
        assert data["is_completed"] is True

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_run_debate_provider_call_error(mock_service):
    """Verify POST /api/v1/debate/run returns 502 with structured error details when provider fails."""
    mock_service.run_debate = AsyncMock(
        side_effect=ProviderCallError(
            "Proponent LLM call failed with openrouter: ConnectTimeout",
            details={"provider": "openrouter", "model": "test-model", "error": "ConnectTimeout"},
        )
    )
    app.dependency_overrides[get_debate_service] = lambda: mock_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"claim": "Renewable energy can fully replace fossil fuels by 2040.", "max_rounds": 1}
        response = await client.post("/api/v1/debate/run", json=payload)
        assert response.status_code == 502
        data = response.json()
        assert data["detail"]["error"] == "ProviderCallError"
        assert "ConnectTimeout" in data["detail"]["message"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_run_debate_validation_error():
    """Verify POST /api/v1/debate/run validates claim minimum length."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Invalid claim (less than 5 chars)
        payload = {"claim": "No", "max_rounds": 1}
        response = await client.post("/api/v1/debate/run", json=payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_debate_by_id_found(mock_service):
    """Verify GET /api/v1/debate/{id} returns debate details when found."""
    app.dependency_overrides[get_debate_service] = lambda: mock_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/debate/test-deb-555")
        assert response.status_code == 200
        data = response.json()
        assert data["debate_id"] == "test-deb-555"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_debate_by_id_not_found(mock_service):
    """Verify GET /api/v1/debate/{id} returns 404 when ID does not exist."""
    app.dependency_overrides[get_debate_service] = lambda: mock_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/debate/non-existent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    app.dependency_overrides.clear()
