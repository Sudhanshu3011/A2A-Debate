"""Graph node execution handlers for the debate workflow."""
import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from src.agents.proponent import ProponentAgent
from src.agents.opponent import OpponentAgent
from src.agents.judge import JudgeAgent
from src.graph.state import DebateState
from src.schemas.debate import ArgumentTurn
from src.schemas.verdict import DebateVerdict

logger = logging.getLogger(__name__)


def initialize_node(state: DebateState) -> Dict[str, Any]:
    """
    Validates and prepares state for debate execution.
    """
    logger.info(
        "Initializing debate '%s' on claim: '%s' (max_rounds=%d)",
        state.get("debate_id"),
        state.get("claim"),
        state.get("max_rounds", 3),
    )
    return {
        "current_round": 1,
        "current_speaker": "proponent",
        "is_completed": False,
    }


class DebateNodes:
    """
    Encapsulates node execution methods with optional agent dependency injection.
    """

    def __init__(
        self,
        proponent_agent: Optional[ProponentAgent] = None,
        opponent_agent: Optional[OpponentAgent] = None,
        judge_agent: Optional[JudgeAgent] = None,
    ):
        self.proponent_agent = proponent_agent
        self.opponent_agent = opponent_agent
        self.judge_agent = judge_agent

    def _get_proponent(self, state: DebateState, config: Optional[RunnableConfig] = None) -> ProponentAgent:
        if self.proponent_agent:
            return self.proponent_agent
        configurable = (config or {}).get("configurable", {})
        if "proponent_agent" in configurable:
            return configurable["proponent_agent"]
        return ProponentAgent(claim=state["claim"], rules=state.get("rules", ""))

    def _get_opponent(self, state: DebateState, config: Optional[RunnableConfig] = None) -> OpponentAgent:
        if self.opponent_agent:
            return self.opponent_agent
        configurable = (config or {}).get("configurable", {})
        if "opponent_agent" in configurable:
            return configurable["opponent_agent"]
        return OpponentAgent(claim=state["claim"], rules=state.get("rules", ""))

    def _get_judge(self, state: DebateState, config: Optional[RunnableConfig] = None) -> JudgeAgent:
        if self.judge_agent:
            return self.judge_agent
        configurable = (config or {}).get("configurable", {})
        if "judge_agent" in configurable:
            return configurable["judge_agent"]
        return JudgeAgent(claim=state["claim"])

    async def proponent_node(
        self,
        state: DebateState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """
        Executes the proponent's speech turn.
        """
        current_round = state.get("current_round", 1)
        logger.info(
            "Debate '%s': Proponent delivering speech for Round %d",
            state.get("debate_id"),
            current_round,
        )

        agent = self._get_proponent(state, config)

        # Extract latest opponent turn if round > 1
        opponent_previous_turn: Optional[ArgumentTurn] = None
        turns = state.get("turns", [])
        for past_turn in reversed(turns):
            if past_turn.get("speaker") == "opponent":
                opponent_previous_turn = ArgumentTurn(**past_turn)
                break

        turn: ArgumentTurn = await agent.generate_speech(
            round_number=current_round,
            opponent_previous_turn=opponent_previous_turn,
            prior_proponent_points=state.get("proponent_arguments", []),
        )

        new_turn_dict = turn.model_dump()
        return {
            "turns": state.get("turns", []) + [new_turn_dict],
            "proponent_arguments": state.get("proponent_arguments", []) + turn.key_points,
            "messages": [AIMessage(content=turn.content, name="proponent")],
            "current_speaker": "opponent",
        }

    async def opponent_node(
        self,
        state: DebateState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """
        Executes the opponent's speech turn and increments round counter.
        """
        current_round = state.get("current_round", 1)
        logger.info(
            "Debate '%s': Opponent delivering rebuttal for Round %d",
            state.get("debate_id"),
            current_round,
        )

        agent = self._get_opponent(state, config)

        # Extract latest proponent turn
        proponent_current_turn: Optional[ArgumentTurn] = None
        turns = state.get("turns", [])
        for past_turn in reversed(turns):
            if past_turn.get("speaker") == "proponent":
                proponent_current_turn = ArgumentTurn(**past_turn)
                break

        if proponent_current_turn is None:
            raise ValueError("Opponent node invoked without a preceding Proponent speech.")

        turn: ArgumentTurn = await agent.generate_speech(
            round_number=current_round,
            proponent_current_turn=proponent_current_turn,
            prior_opponent_points=state.get("opponent_arguments", []),
        )

        new_turn_dict = turn.model_dump()
        return {
            "turns": state.get("turns", []) + [new_turn_dict],
            "opponent_arguments": state.get("opponent_arguments", []) + turn.key_points,
            "messages": [AIMessage(content=turn.content, name="opponent")],
            "current_round": current_round + 1,
            "current_speaker": "proponent",
        }

    async def judge_node(
        self,
        state: DebateState,
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, Any]:
        """
        Executes the judge's evaluation across all debate turns and synthesizes verdict.
        """
        logger.info(
            "Debate '%s': Judge evaluating complete debate (%d turns)",
            state.get("debate_id"),
            len(state.get("turns", [])),
        )

        agent = self._get_judge(state, config)
        turns = [ArgumentTurn(**t) for t in state.get("turns", [])]

        verdict: DebateVerdict = await agent.evaluate_debate(turns=turns)

        return {
            "verdict": verdict.model_dump(),
            "is_completed": True,
            "current_speaker": "judge",
            "messages": [AIMessage(content=verdict.synthesis_summary, name="judge")],
        }

