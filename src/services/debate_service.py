"""Debate service handling graph execution and persistence."""
import logging
from typing import Dict, Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from src.agents.judge import JudgeAgent
from src.agents.opponent import OpponentAgent
from src.agents.proponent import ProponentAgent
from src.graph.state import initialize_state
from src.graph.workflow import build_debate_graph
from src.schemas.debate import ArgumentTurn, DebateRequest, DebateResponse
from src.schemas.verdict import DebateVerdict

logger = logging.getLogger(__name__)


class DebateService:
    """
    Orchestration service coordinating agent initialization,
    graph execution (non-streaming), and state retrieval.
    """

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        self.checkpointer = checkpointer or MemorySaver()
        self._debates_cache: Dict[str, DebateResponse] = {}

    async def run_debate(
        self,
        request: DebateRequest,
        proponent_agent: Optional[ProponentAgent] = None,
        opponent_agent: Optional[OpponentAgent] = None,
        judge_agent: Optional[JudgeAgent] = None,
    ) -> DebateResponse:
        """
        Executes a complete multi-round debate workflow and returns the response.
        """
        initial_state = initialize_state(
            claim=request.claim,
            max_rounds=request.max_rounds,
            rules=request.rules,
        )
        debate_id = initial_state["debate_id"]

        logger.info(
            "DebateService starting debate '%s' with claim: '%s'",
            debate_id,
            request.claim,
        )

        # Initialize agents using environment-configured models
        pro_agent = proponent_agent or ProponentAgent(
            claim=request.claim,
            rules=request.rules or "",
        )
        opp_agent = opponent_agent or OpponentAgent(
            claim=request.claim,
            rules=request.rules or "",
        )
        j_agent = judge_agent or JudgeAgent(
            claim=request.claim,
        )

        graph = build_debate_graph(
            checkpointer=self.checkpointer,
            proponent_agent=pro_agent,
            opponent_agent=opp_agent,
            judge_agent=j_agent,
        )

        config = {"configurable": {"thread_id": debate_id}}
        final_state = await graph.ainvoke(initial_state, config=config)

        # Parse turns
        turns = [ArgumentTurn(**t) for t in final_state.get("turns", [])]
        raw_verdict = final_state.get("verdict")
        verdict = DebateVerdict(**raw_verdict) if raw_verdict else None

        response = DebateResponse(
            debate_id=debate_id,
            claim=request.claim,
            total_rounds=request.max_rounds,
            turns=turns,
            verdict=verdict,
            is_completed=final_state.get("is_completed", True),
        )

        self._debates_cache[debate_id] = response
        logger.info("Debate '%s' successfully completed.", debate_id)
        return response

    async def get_debate(self, debate_id: str) -> Optional[DebateResponse]:
        """
        Retrieves a cached debate response by debate_id.
        """
        return self._debates_cache.get(debate_id)

