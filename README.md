# Multi-Agent Debate Engine

An adversarial multi-agent debate architecture powered by **LangGraph**, **LangChain**, and **FastAPI**, featuring a **Proponent Agent (Thesis)**, an **Opponent Agent (Antithesis)**, and an impartial **Judge Node (Synthesis)**.

Supports **Groq** (low-latency inference) and **OpenRouter** (frontier and open-weights models).

---

## Getting Started

### 1. Environment Setup
Copy the environment template and configure your API keys:
```bash
cp .env.example .env
```
Provide your `GROQ_API_KEY` and/or `OPENROUTER_API_KEY` in `.env`.

### 2. Run the Service

#### Option A: Run with Docker Compose (Recommended)
```bash
docker compose up -d --build
```

#### Option B: Run with Python Locally
```bash
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation will be available at:
`http://localhost:8000/docs`

---

## API Endpoints

### 1. Health Probe
```http
GET /health
```

### 2. Execute Debate
```http
POST /api/v1/debate/run
Content-Type: application/json

{
  "claim": "Artificial general intelligence will be achieved by 2030.",
  "max_rounds": 2,
  "proponent_provider": "openrouter",
  "opponent_provider": "groq"
}
```

### 3. Retrieve Debate
```http
GET /api/v1/debate/{debate_id}
```

---

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

