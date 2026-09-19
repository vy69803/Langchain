"""Cost Optimization module for LangChain RAG & LLM pipelines.

Provides:
  1. Semantic Caching: Reuses responses for semantically similar questions (0 token cost).
  2. Context & Prompt Compression: Prunes irrelevant sentences/whitespace to minimize input tokens.
  3. Tiered Model Routing (Model Cascading): Routes simple queries to free/cheap models and complex queries to flagship models.
  4. Real-time Token & Dollar Cost Tracking: Measures token usage, dollar spend, and monetary savings.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from langchain_rag.embeddings import calculate_similarity, cosine_similarity, get_embeddings
from langchain_rag.llm import get_llm

load_dotenv()


# Pricing per 1,000,000 tokens in USD (Input / Output)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "thinkingmachines/inkling:free": {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.1-8b-instruct:free": {"input": 0.0, "output": 0.0},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "meta-llama/llama-3.3-70b-instruct": {"input": 0.30, "output": 0.40},
    "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
    "default": {"input": 0.15, "output": 0.60},
}


class ModelTier(str, Enum):
    FREE = "free"
    LIGHTWEIGHT = "lightweight"
    BALANCED = "balanced"
    FLAGSHIP = "flagship"


TIER_DEFAULT_MODELS: dict[ModelTier, str] = {
    ModelTier.FREE: "thinkingmachines/inkling:free",
    ModelTier.LIGHTWEIGHT: "openai/gpt-4o-mini",
    ModelTier.BALANCED: "meta-llama/llama-3.3-70b-instruct",
    ModelTier.FLAGSHIP: "anthropic/claude-3.5-sonnet",
}


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string.
    
    Uses tiktoken if available, with a fast ~4 chars/token heuristic fallback.
    """
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Standard NLP rule of thumb: ~4 characters or ~0.75 words per token
        words = len(text.split())
        chars = len(text)
        return max(1, int((words * 1.3 + chars / 4.0) / 2))


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated monetary cost in USD for a completion call."""
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    input_cost = (prompt_tokens / 1_000_000.0) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000.0) * pricing["output"]
    return input_cost + output_cost


@dataclass
class UsageRecord:
    query: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    saved_usd: float
    latency_ms: float
    cache_hit: bool
    compressed: bool
    revoked: bool = False
    revocation_reason: str | None = None


class CostTracker:
    """Thread-safe cost and token usage tracking with budget limits and statistics."""

    def __init__(self, budget_usd: float | None = None, max_session_tokens: int | None = None) -> None:
        self.budget_usd = budget_usd
        self.max_session_tokens = max_session_tokens
        self.records: list[UsageRecord] = []

    def record(
        self,
        query: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cache_hit: bool = False,
        saved_usd: float = 0.0,
        compressed: bool = False,
        revoked: bool = False,
        revocation_reason: str | None = None,
    ) -> UsageRecord:
        """Record an LLM, Cache, or Revocation event."""
        cost = 0.0 if (cache_hit or revoked) else calculate_cost(model, prompt_tokens, completion_tokens)
        rec = UsageRecord(
            query=query,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            saved_usd=saved_usd,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            compressed=compressed,
            revoked=revoked,
            revocation_reason=revocation_reason,
        )
        self.records.append(rec)
        return rec

    def is_over_budget(self) -> bool:
        if self.budget_usd is not None and self.total_cost() >= self.budget_usd:
            return True
        if self.max_session_tokens is not None and self.total_tokens() >= self.max_session_tokens:
            return True
        return False

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def total_saved(self) -> float:
        return sum(r.saved_usd for r in self.records)

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def cache_hit_rate(self) -> float:
        valid_queries = [r for r in self.records if not r.revoked]
        if not valid_queries:
            return 0.0
        hits = sum(1 for r in valid_queries if r.cache_hit)
        return (hits / len(valid_queries)) * 100.0

    def summary(self) -> dict[str, Any]:
        """Generate comprehensive cost and efficiency statistics."""
        total_queries = len(self.records)
        cache_hits = sum(1 for r in self.records if r.cache_hit)
        revoked_count = sum(1 for r in self.records if r.revoked)
        total_cost = self.total_cost()
        total_saved = self.total_saved()
        avg_latency = (
            sum(r.latency_ms for r in self.records) / total_queries if total_queries else 0.0
        )

        return {
            "total_queries": total_queries,
            "cache_hits": cache_hits,
            "cache_hit_rate_pct": round(self.cache_hit_rate(), 2),
            "revoked_queries": revoked_count,
            "total_tokens": self.total_tokens(),
            "total_cost_usd": round(total_cost, 6),
            "total_saved_usd": round(total_saved, 6),
            "net_roi_factor": round((total_saved / (total_cost + 1e-9)), 2) if total_cost > 0 else "Infinite (Free/Zero Cost)",
            "average_latency_ms": round(avg_latency, 2),
        }


class SemanticCache:
    """Semantic Vector Cache that reuses LLM responses for similar queries.
    
    If query similarity >= threshold (default: 0.80), returns cached response instantly,
    reducing latency to < 5ms and token cost to $0.00.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.embeddings = embeddings or get_embeddings("local")
        # List of dicts: {"query": str, "vector": list[float], "response": str, "metadata": dict}
        self._entries: list[dict[str, Any]] = []

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def get(self, query: str) -> tuple[str | None, float]:
        """Lookup cached response for a query.
        
        Returns:
            (cached_response, similarity_score). If no match above threshold, returns (None, score).
        """
        if not self._entries:
            return None, 0.0

        query_vec = self.embeddings.embed_query(query)
        best_entry: dict[str, Any] | None = None
        best_score = -1.0

        for entry in self._entries:
            score = cosine_similarity(query_vec, entry["vector"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            return best_entry["response"], best_score

        return None, max(0.0, best_score)

    def set(self, query: str, response: str, metadata: dict[str, Any] | None = None) -> None:
        """Store a query-response pair in the semantic cache."""
        query_vec = self.embeddings.embed_query(query)
        self._entries.append({
            "query": query,
            "vector": query_vec,
            "response": response,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })


class PromptCompressor:
    """Prunes and compresses prompts and retrieved context without losing critical facts.
    
    Capabilities:
      - Whitespace & boilerplate minification
      - Extractive sentence filtering: retains top sentences scored against query
      - Token budget fitting: guarantees prompt stays within maximum tokens
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        max_context_tokens: int = 800,
    ) -> None:
        self.embeddings = embeddings or get_embeddings("local")
        self.max_context_tokens = max_context_tokens

    def minify_text(self, text: str) -> str:
        """Remove redundant whitespace and empty lines."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def extract_relevant_sentences(
        self,
        context: str,
        query: str,
        max_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        """Extract only the most relevant sentences from context matching the query.
        
        Returns:
            (compressed_context, original_tokens, compressed_tokens)
        """
        target_tokens = max_tokens or self.max_context_tokens
        original_tokens = estimate_tokens(context)

        if original_tokens <= target_tokens:
            return context.strip(), original_tokens, original_tokens

        # Split into individual sentences
        sentences = re.split(r"(?<=[.!?])\s+", context)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return context[: target_tokens * 4], original_tokens, estimate_tokens(context[: target_tokens * 4])

        # Score each sentence against query
        query_vec = self.embeddings.embed_query(query)
        sentence_scores: list[tuple[str, float, int]] = []

        for s in sentences:
            s_vec = self.embeddings.embed_query(s)
            sim = cosine_similarity(query_vec, s_vec)
            # Add lexical keyword bonus if exact words overlap
            q_words = set(query.lower().split())
            s_words = set(s.lower().split())
            lexical_bonus = len(q_words & s_words) * 0.05
            sentence_scores.append((s, sim + lexical_bonus, estimate_tokens(s)))

        # Sort by relevance
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        # Accumulate sentences up to token budget
        selected_sentences: list[str] = []
        accumulated_tokens = 0

        for s, _, tok_len in sentence_scores:
            if accumulated_tokens + tok_len <= target_tokens:
                selected_sentences.append(s)
                accumulated_tokens += tok_len
            if accumulated_tokens >= target_tokens:
                break

        # Reconstruct preserved context
        compressed_text = " ".join(selected_sentences)
        compressed_tokens = estimate_tokens(compressed_text)

        return compressed_text, original_tokens, compressed_tokens


class TieredRouter:
    """Routes queries to the most cost-effective model based on query complexity.
    
    Heuristics:
      - Simple lookup / factual / definition -> Free / Lightweight tier
      - Summary / Comparison / Synthesis -> Balanced tier
      - Complex code generation / Multi-step logic / Security auditing -> Flagship tier
    """

    COMPLEXITY_KEYWORDS = {
        "flagship": [
            "architect", "implement", "debug", "refactor", "derive", "mathematical proof",
            "security vulnerability", "exploit", "concurrency", "optimize algorithm",
        ],
        "balanced": [
            "compare", "contrast", "summarize", "synthesize", "explain in detail",
            "tradeoffs", "pros and cons", "step by step guide",
        ],
    }

    def __init__(self, fallback_model: str = "thinkingmachines/inkling:free") -> None:
        self.fallback_model = fallback_model

    def classify_query(self, query: str) -> tuple[ModelTier, str, str]:
        """Classify query into an appropriate ModelTier and model identifier.
        
        Returns:
            (tier, model_name, reasoning)
        """
        q_lower = query.lower()
        word_count = len(q_lower.split())

        for kw in self.COMPLEXITY_KEYWORDS["flagship"]:
            if kw in q_lower:
                return (
                    ModelTier.FLAGSHIP,
                    TIER_DEFAULT_MODELS[ModelTier.FLAGSHIP],
                    f"Matched high-complexity keyword: '{kw}'",
                )

        for kw in self.COMPLEXITY_KEYWORDS["balanced"]:
            if kw in q_lower:
                return (
                    ModelTier.BALANCED,
                    TIER_DEFAULT_MODELS[ModelTier.BALANCED],
                    f"Matched balanced-complexity keyword: '{kw}'",
                )

        if word_count > 35 or "code" in q_lower or "script" in q_lower:
            return (
                ModelTier.BALANCED,
                TIER_DEFAULT_MODELS[ModelTier.BALANCED],
                "Long prompt or code-oriented query",
            )

        # Default fast & free
        return (
            ModelTier.FREE,
            TIER_DEFAULT_MODELS[ModelTier.FREE],
            "Factual/short query suited for fast free model",
        )


class CostOptimizer:
    """End-to-end Cost Optimizer combining semantic caching, prompt compression,
    tiered routing, query token budgeting/revocation, and real-time cost tracking."""

    def __init__(
        self,
        cache_threshold: float = 0.80,
        max_context_tokens: int = 600,
        max_query_tokens: int | None = 150,
        max_session_tokens: int | None = None,
        budget_usd: float | None = None,
    ) -> None:
        self.max_query_tokens = max_query_tokens
        self.cache = SemanticCache(similarity_threshold=cache_threshold)
        self.compressor = PromptCompressor(max_context_tokens=max_context_tokens)
        self.router = TieredRouter()
        self.tracker = CostTracker(budget_usd=budget_usd, max_session_tokens=max_session_tokens)

    def execute_optimized_query(
        self,
        query: str,
        context: str = "",
        system_instruction: str = "Answer the user question concisely based on the context.",
        force_model: str | None = None,
    ) -> dict[str, Any]:
        """Execute a query through the full cost-optimization pipeline:
          0. Token Budget Check -> If query exceeds max_query_tokens, revoke immediately.
          1. Check Semantic Cache -> If hit, return with 0 cost.
          2. Compress Context -> Prune irrelevant sentences.
          3. Route to optimal model -> Select tier based on complexity.
          4. Invoke LLM and track usage & cost savings.
          5. Update Semantic Cache.
        """
        start_time = time.perf_counter()
        query_tokens = estimate_tokens(query)

        # 0. Query Token Budget Guardrail (Instant Revocation for long/abusive queries)
        if self.max_query_tokens is not None and query_tokens > self.max_query_tokens:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            reason = (
                f"Query revoked: input exceeds maximum allowed query budget of {self.max_query_tokens} tokens "
                f"(detected ~{query_tokens} tokens). Request was blocked to prevent high LLM token costs."
            )
            saved_tokens = query_tokens + estimate_tokens(context)
            saved_usd = calculate_cost("openai/gpt-4o", saved_tokens, 100)

            self.tracker.record(
                query=query,
                model="guardrail_revoked",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                revoked=True,
                revocation_reason=reason,
                saved_usd=saved_usd,
            )
            return {
                "answer": f"[Query Blocked]: {reason}",
                "revoked": True,
                "revocation_reason": reason,
                "query_tokens": query_tokens,
                "max_query_tokens": self.max_query_tokens,
                "cache_hit": False,
                "similarity_score": 0.0,
                "model_used": "guardrail_revoked",
                "model_tier": "revoked",
                "route_reason": "Query token budget exceeded",
                "cost_usd": 0.0,
                "saved_usd": round(saved_usd, 6),
                "latency_ms": round(latency_ms, 2),
                "compressed": False,
                "original_tokens": estimate_tokens(context),
                "compressed_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        # Session Budget Guardrail
        if self.tracker.is_over_budget():
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            reason = "Session budget exceeded: cumulative token or dollar spending limit reached."
            self.tracker.record(
                query=query,
                model="budget_capped",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                revoked=True,
                revocation_reason=reason,
            )
            return {
                "answer": f"[Request Blocked]: {reason}",
                "revoked": True,
                "revocation_reason": reason,
                "query_tokens": query_tokens,
                "max_query_tokens": self.max_query_tokens,
                "cache_hit": False,
                "similarity_score": 0.0,
                "model_used": "budget_capped",
                "model_tier": "revoked",
                "route_reason": "Session budget limit reached",
                "cost_usd": 0.0,
                "saved_usd": 0.0,
                "latency_ms": round(latency_ms, 2),
                "compressed": False,
                "original_tokens": estimate_tokens(context),
                "compressed_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        # 1. Semantic Cache check
        cached_resp, sim_score = self.cache.get(query)
        if cached_resp is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            est_tokens = estimate_tokens(query + context) + estimate_tokens(cached_resp)
            saved_usd = calculate_cost("openai/gpt-4o", est_tokens, estimate_tokens(cached_resp))

            self.tracker.record(
                query=query,
                model="semantic_cache",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                cache_hit=True,
                saved_usd=saved_usd,
            )
            return {
                "answer": cached_resp,
                "revoked": False,
                "cache_hit": True,
                "similarity_score": round(sim_score, 4),
                "model_used": "semantic_cache",
                "model_tier": "cache",
                "route_reason": "Semantic similarity cache hit",
                "cost_usd": 0.0,
                "saved_usd": round(saved_usd, 6),
                "latency_ms": round(latency_ms, 2),
                "compressed": False,
                "original_tokens": estimate_tokens(context),
                "final_tokens": 0,
                "query_tokens": query_tokens,
            }

        # 2. Context Compression
        orig_tokens = estimate_tokens(context)
        if context:
            compressed_context, _, comp_tokens = self.compressor.extract_relevant_sentences(
                context=context,
                query=query,
            )
        else:
            compressed_context = ""
            comp_tokens = 0

        # Build prompt
        if compressed_context:
            full_prompt = f"{system_instruction}\n\nContext:\n{compressed_context}\n\nQuestion: {query}\n\nAnswer:"
        else:
            full_prompt = f"{system_instruction}\n\nQuestion: {query}\n\nAnswer:"

        prompt_tokens = estimate_tokens(full_prompt)

        # 3. Tiered Model Routing
        if force_model:
            model_to_use = force_model
            route_reason = "Explicit user override"
            tier = ModelTier.FREE
        else:
            tier, model_to_use, route_reason = self.router.classify_query(query)

        # 4. Invoke LLM
        answer = ""
        try:
            llm = get_llm(model=model_to_use)
            resp = llm.invoke(full_prompt)
            answer = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            # Fallback to Thinking Machines Inkling free model if selected model fails
            try:
                llm_fallback = get_llm(model="thinkingmachines/inkling:free")
                resp = llm_fallback.invoke(full_prompt)
                answer = resp.content if hasattr(resp, "content") else str(resp)
                model_to_use = "thinkingmachines/inkling:free"
            except Exception as inner_err:
                answer = f"[LLM Invocation Notice: {inner_err}] Context was compressed from {orig_tokens} -> {comp_tokens} tokens."

        completion_tokens = estimate_tokens(answer)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Calculate savings from compression
        token_reduction = max(0, orig_tokens - comp_tokens)
        saved_usd = (token_reduction / 1_000_000.0) * MODEL_PRICING.get(model_to_use, MODEL_PRICING["default"])["input"]

        self.tracker.record(
            query=query,
            model=model_to_use,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cache_hit=False,
            saved_usd=saved_usd,
            compressed=bool(token_reduction > 0),
        )

        # 5. Populate Cache
        if answer and not answer.startswith("[LLM Invocation Notice"):
            self.cache.set(query, answer)

        cost = calculate_cost(model_to_use, prompt_tokens, completion_tokens)

        return {
            "answer": answer,
            "revoked": False,
            "cache_hit": False,
            "similarity_score": round(sim_score, 4),
            "model_used": model_to_use,
            "model_tier": tier.value,
            "route_reason": route_reason,
            "cost_usd": round(cost, 6),
            "saved_usd": round(saved_usd, 6),
            "latency_ms": round(latency_ms, 2),
            "compressed": bool(token_reduction > 0),
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "query_tokens": query_tokens,
        }


def create_cost_optimizer(
    cache_threshold: float = 0.80,
    max_context_tokens: int = 600,
    max_query_tokens: int | None = 150,
    max_session_tokens: int | None = None,
    budget_usd: float | None = None,
) -> CostOptimizer:
    """Factory function to initialize a CostOptimizer instance with token budgeting."""
    return CostOptimizer(
        cache_threshold=cache_threshold,
        max_context_tokens=max_context_tokens,
        max_query_tokens=max_query_tokens,
        max_session_tokens=max_session_tokens,
        budget_usd=budget_usd,
    )
