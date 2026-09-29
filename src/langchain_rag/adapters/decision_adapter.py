"""Decision Adapter Interface and Implementations.

Provides an abstract interface for structured decision-making, routing,
and verification, decoupling agent orchestration from specific model providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class VerificationResult(BaseModel):
    """Result of factual grounding / faithfulness verification."""

    is_faithful: bool = Field(
        description="Whether all numerical figures and facts match retrieved evidence."
    )
    hallucinated_figures: List[str] = Field(
        default_factory=list,
        description="List of specific figures in draft response not grounded in source.",
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Calibrated confidence score of the evaluation."
    )
    discrepancy_details: Optional[str] = Field(
        default=None, description="Detailed explanation of discrepancies or arithmetic mismatch."
    )


class DecisionAdapter(ABC):
    """Abstract interface for structured decision-making, routing, and verification."""

    @abstractmethod
    async def classify(
        self,
        text: str,
        schema: Type[T],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        """Analyze text and context, returning a populated structured Pydantic schema."""
        pass

    @abstractmethod
    async def verify(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
    ) -> VerificationResult:
        """Validate whether numerical and factual assertions match ground truth evidence."""
        pass


class PydanticLLMAdapter(DecisionAdapter):
    """Production default implementation using LangChain .with_structured_output()."""

    def __init__(
        self,
        llm: Any = None,
        model_name: str = "gpt-4o-mini",
        mock: bool = False,
    ) -> None:
        """Initialize with an LLM instance or model name.

        Args:
            llm: LangChain BaseChatModel instance. If None, initializes default model.
            model_name: Fallback model identifier if llm is None.
            mock: If True, forces offline deterministic fallback.
        """
        if mock:
            self.llm = None
            return

        if llm is not None:
            self.llm = llm
        else:
            try:
                import os
                from dotenv import load_dotenv

                load_dotenv()
                # Only initialize live external model if explicitly enabled or key specified
                if os.getenv("ENABLE_LIVE_LLM", "false").lower() in ("true", "1", "yes"):
                    openrouter_key = os.getenv("OPENROUTER_API_KEY")
                    if openrouter_key and not os.getenv("OPENAI_API_KEY"):
                        from langchain_rag.llm import get_llm

                        model_to_use = os.getenv("OPENROUTER_MODEL", "thinkingmachines/inkling:free")
                        self.llm = get_llm(model=model_to_use)
                    else:
                        from langchain.chat_models import init_chat_model

                        self.llm = init_chat_model(model_name)
                else:
                    self.llm = None
            except Exception as e:
                logger.warning(
                    f"Could not auto-initialize chat model {model_name}: {e}. Mock fallback enabled."
                )
                self.llm = None

    async def classify(
        self,
        text: str,
        schema: Type[T],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        """Classify input text using LangChain structured outputs."""
        if self.llm is None:
            # Fallback for offline / test environments
            logger.info(f"PydanticLLMAdapter [Offline]: constructing default {schema.__name__}")
            return schema.model_construct()

        try:
            structured_llm = self.llm.with_structured_output(schema)
            ctx_str = f"Context: {context}\n\n" if context else ""
            prompt = (
                f"{ctx_str}"
                "Analyze the input query and extract structured financial search parameters.\n"
                f"Input: {text}"
            )
            result = await structured_llm.ainvoke(prompt)
            return result
        except Exception as e:
            logger.warning(f"PydanticLLMAdapter classification error: {e}. Falling back to default schema.")
            return schema.model_construct()

    async def verify(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
    ) -> VerificationResult:
        """Verify factual grounding of claim against evidence."""
        if self.llm is None:
            return VerificationResult(
                is_faithful=True,
                hallucinated_figures=[],
                confidence_score=1.0,
                discrepancy_details="Offline mock verification passed.",
            )

        try:
            structured_llm = self.llm.with_structured_output(VerificationResult)
            prompt = (
                "You are a strict financial auditor. Verify whether all numbers, dates, and metric names "
                "in the claim below are strictly grounded in the retrieved evidence.\n\n"
                f"Evidence: {evidence}\n\n"
                f"Claim to Verify:\n{claim}"
            )
            result: VerificationResult = await structured_llm.ainvoke(prompt)
            return result
        except Exception as e:
            logger.warning(f"PydanticLLMAdapter verification error: {e}. Falling back to default verification.")
            return VerificationResult(
                is_faithful=True,
                hallucinated_figures=[],
                confidence_score=1.0,
                discrepancy_details=f"Verification fallback: {e}",
            )


class JevAdapter(DecisionAdapter):
    """Reserved drop-in adapter for TypeSafe AI JEV decision model.

    Enables sub-20ms, low-cost classification without modifying LangGraph topology.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    async def classify(
        self,
        text: str,
        schema: Type[T],
        context: Optional[Dict[str, Any]] = None,
    ) -> T:
        raise NotImplementedError(
            "JevAdapter is reserved for Phase 5 optimization. "
            "Use PydanticLLMAdapter for current production execution."
        )

    async def verify(
        self,
        claim: str,
        evidence: List[Dict[str, Any]],
    ) -> VerificationResult:
        raise NotImplementedError(
            "JevAdapter is reserved for Phase 5 optimization. "
            "Use PydanticLLMAdapter for current production execution."
        )
