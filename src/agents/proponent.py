"""Proponent agent defending the debate claim."""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq

from src.agents.prompt_loader import render_prompt
from src.agents.provider import LLMProvider
from src.core.config import get_settings
from src.core.exceptions import (
    ProviderAuthenticationError,
    ProviderCallError,
    format_exception_message,
)
from src.schemas.debate import ArgumentTurn

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ProponentAgent:
    """Agent advocating and defending the motion/claim."""

    def __init__(
        self,
        claim: str,
        rules: str = "Maintain civil, logic-driven debate. Refute assertions with evidence.",
        provider: Optional[LLMProvider] = LLMProvider.OPENROUTER,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
        client: Optional[BaseChatModel] = None,
        enable_failover: bool = True,
    ):
        settings = get_settings()
        self.name = "proponent"
        self.role = "Proponent"
        self.claim = claim
        self.rules = rules
        self.provider = provider or LLMProvider.OPENROUTER
        self.model_name = model_name or settings.PROPONENT_MODEL
        self.temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else settings.DEFAULT_MAX_TOKENS
        self.request_timeout = request_timeout if request_timeout is not None else settings.REQUEST_TIMEOUT_SECONDS
        self.enable_failover = enable_failover

        self.system_prompt = render_prompt(
            "proponent.jinja2",
            {"claim": claim, "rules": rules},
        )

        self.client: BaseChatModel = client if client is not None else self._init_client(self.model_name)

    def _init_client(self, model_name: str) -> BaseChatModel:
        """Initializes direct ChatOpenRouter client with reasoning disabled."""
        settings = get_settings()
        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            raise ProviderAuthenticationError(
                "OPENROUTER_API_KEY is required to initialize the Proponent agent client. Please configure it in .env or environment."
            )

        logger.info("Initializing ChatOpenRouter client for proponent with model '%s'", model_name)
        openrouter_timeout_ms = int(self.request_timeout * 1000) if self.request_timeout else None
        return ChatOpenRouter(
            openrouter_api_key=api_key,
            model_name=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=openrouter_timeout_ms,
            reasoning={"effort": "none"},
        )

    def _init_failover_client(self) -> BaseChatModel:
        """Initializes fallback ChatGroq client in case of OpenRouter outage."""
        settings = get_settings()
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ProviderAuthenticationError("GROQ_API_KEY is required for failover client.")

        return ChatGroq(
            api_key=api_key,
            model_name=settings.OPPONENT_MODEL,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
            max_retries=2,
        )

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Checks if exception indicates rate limiting."""
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in ["429", "rate limit", "rate_limit", "quota exceeded", "tpm", "rpm", "resource_exhausted"]
        )

    def _attempt_manual_json_extraction(self, text: str, schema: Type[T]) -> Optional[T]:
        """Attempts to extract json from raw output if structured parsing failed."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return schema.model_validate(data)
            except Exception:
                pass

        match_braces = re.search(r"(\{.*\})", text, re.DOTALL)
        if match_braces:
            try:
                data = json.loads(match_braces.group(1))
                return schema.model_validate(data)
            except Exception:
                pass

        return None

    def _create_fallback_structured_object(self, text: str, schema: Type[T]) -> T:
        """Creates fallback structured object to keep debate flow alive."""
        logger.warning("Constructing fallback structured object for schema '%s'", schema.__name__)
        schema_fields = getattr(schema, "model_fields", {})
        fallback_kwargs: Dict[str, Any] = {}
        if "content" in schema_fields:
            fallback_kwargs["content"] = text
        if "speaker" in schema_fields:
            fallback_kwargs["speaker"] = "proponent"
        if "round_number" in schema_fields:
            fallback_kwargs["round_number"] = 1

        try:
            return schema.model_validate(fallback_kwargs)
        except Exception as exc:
            raise ProviderCallError(
                f"Failed to generate or recover structured response for {schema.__name__}: {exc}",
                details={"raw_text": text},
            ) from exc

    async def _execute_with_client(
        self,
        client: BaseChatModel,
        full_messages: List[BaseMessage],
        structured_schema: Optional[Type[T]] = None,
    ) -> Any:
        if structured_schema is not None:
            try:
                runnable = client.with_structured_output(structured_schema)
                response = await runnable.ainvoke(full_messages)
                if response is not None:
                    return response
            except (OutputParserException, ValidationError, json.JSONDecodeError) as parse_err:
                logger.warning(
                    "Structured parsing failed for proponent (%s). Attempting raw recovery...",
                    str(parse_err),
                )
                raw_msg = await client.ainvoke(full_messages)
                raw_text = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)
                recovered = self._attempt_manual_json_extraction(raw_text, structured_schema)
                if recovered is not None:
                    return recovered
                return self._create_fallback_structured_object(raw_text, structured_schema)

        return await client.ainvoke(full_messages)

    @retry(
        # stop=stop_after_attempt(3),
        stop=stop_after_attempt(1),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=0.5),
        retry=retry_if_exception_type(ProviderCallError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def call_llm(
        self,
        messages: List[BaseMessage],
        structured_schema: Optional[Type[T]] = None,
    ) -> Any:
        """Executes LLM call with retry and error recovery."""
        full_messages = [SystemMessage(content=self.system_prompt)] + list(messages)
        try:
            return await self._execute_with_client(self.client, full_messages, structured_schema)
        except ProviderAuthenticationError:
            raise
        except Exception as exc:
            err_msg = format_exception_message(exc)
            if self.enable_failover and (self._is_rate_limit_error(exc) or isinstance(exc, Exception)):
                logger.warning(
                    "Primary provider '%s' failed for proponent (%s). Triggering failover...",
                    self.provider,
                    err_msg,
                )
                try:
                    failover_client = self._init_failover_client()
                    return await self._execute_with_client(failover_client, full_messages, structured_schema)
                except Exception as failover_err:
                    failover_msg = format_exception_message(failover_err)
                    logger.error("Failover also failed for proponent: %s", failover_msg)

            logger.error("LLM call failed for proponent: %s", err_msg)
            raise ProviderCallError(
                f"Proponent LLM call failed with {self.provider}: {err_msg}",
                details={
                    "agent": "proponent",
                    "provider": str(self.provider),
                    "model": self.model_name,
                    "error_type": type(exc).__name__,
                    "error": err_msg,
                },
            ) from exc

    async def generate_speech(
        self,
        round_number: int,
        opponent_previous_turn: Optional[ArgumentTurn] = None,
        prior_proponent_points: Optional[List[str]] = None,
    ) -> ArgumentTurn:
        """Generates a structured speech supporting the claim, refuting previous opposition."""
        if round_number == 1 or opponent_previous_turn is None:
            user_content = (
                f"Motion: '{self.claim}'\n\n"
                f"This is Round 1. Deliver your opening constructive arguments supporting the motion."
            )
        else:
            prior_summary = ""
            if prior_proponent_points:
                prior_summary = "Your prior key points were:\n" + "\n".join(
                    f"- {pt}" for pt in prior_proponent_points
                ) + "\n\n"

            opp_speech = opponent_previous_turn.content
            opp_points = "\n".join(f"- {pt}" for pt in opponent_previous_turn.key_points)

            user_content = (
                f"Motion: '{self.claim}'\n"
                f"Current Round: {round_number}\n\n"
                f"{prior_summary}"
                f"The Opponent's previous speech:\n"
                f"\"\"\"\n{opp_speech}\n\"\"\"\n"
                f"Opponent's Key Points:\n{opp_points}\n\n"
                f"Please directly rebut their arguments and provide further evidence supporting the motion."
            )

        messages = [HumanMessage(content=user_content)]
        turn: ArgumentTurn = await self.call_llm(
            messages=messages,
            structured_schema=ArgumentTurn,
        )
        turn.speaker = "proponent"
        turn.round_number = round_number
        return turn
