"""Production Agent and LangGraph Workflow Engine.

Orchestrates enterprise RAG, dynamic query routing, multi-tiered semantic caching,
guardrails/security validation, citation synthesis, and observability metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    TypedDict,
    Union,
)

from fastapi import HTTPException

# LangChain & LangGraph dependencies
try:
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
except ImportError:
    StateGraph = None
    END = "__end__"

    def add_messages(left: list, right: list) -> list:
        return list(left) + list(right)

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from production_api.cache import (
    CacheKeyBuilder,
    default_cache,
    semantic_cache,
    tiered_cache,
)
from production_api.config import Settings, get_settings
from production_api.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MessageRole,
    SourceDocument,
    TokenUsage,
)
from production_api.monitoring import get_logger, metrics_collector
from production_api.security import (
    ThreatLevel,
    UnifiedSecurityGuard,
    detect_prompt_injection,
    scan_code_injection,
    scan_secrets,
)

logger = get_logger("production_api.agent")


# ============================================================================
# 1. Agent State Definition
# ============================================================================

class AgentState(TypedDict, total=False):
    """Execution state schema for the Production Agent and LangGraph workflow."""

    # Conversation history & current turn
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    session_id: Optional[str]
    user_id: Optional[str]
    tenant_id: str

    # Retrieval & context
    context: str
    documents: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]

    # Routing & control flow
    route: str  # "cache_hit" | "rag" | "direct_llm" | "clarify" | "blocked"
    confidence_score: float
    is_cached: bool

    # Security & Guardrails
    security_status: str  # "safe" | "suspicious" | "threat_detected" | "blocked"
    security_metadata: Dict[str, Any]
    security_notes: List[str]

    # Reasoning loop control
    iteration_count: int
    max_iterations: int

    # Final outputs & telemetry
    final_response: str
    token_usage: Dict[str, int]
    latency_ms: float
    metadata: Dict[str, Any]
    error: Optional[str]


def create_initial_state(
    query: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tenant_id: str = "default",
    history: Optional[Sequence[Union[ChatMessage, BaseMessage, dict]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_iterations: int = 5,
) -> AgentState:
    """Helper to initialize an AgentState dictionary with standardized defaults."""
    messages: List[BaseMessage] = []

    if history:
        for item in history:
            if isinstance(item, BaseMessage):
                messages.append(item)
            elif isinstance(item, ChatMessage):
                if item.role == MessageRole.USER:
                    messages.append(HumanMessage(content=item.content))
                elif item.role == MessageRole.ASSISTANT:
                    messages.append(AIMessage(content=item.content))
                elif item.role == MessageRole.SYSTEM:
                    messages.append(SystemMessage(content=item.content))
                else:
                    messages.append(HumanMessage(content=item.content))
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))

    # Append current turn
    messages.append(HumanMessage(content=query))

    return {
        "messages": messages,
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "context": "",
        "documents": [],
        "citations": [],
        "route": "init",
        "confidence_score": 1.0,
        "is_cached": False,
        "security_status": "safe",
        "security_metadata": {},
        "security_notes": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "final_response": "",
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "latency_ms": 0.0,
        "metadata": metadata or {},
        "error": None,
    }


# ============================================================================
# 2. Production Agent Implementation
# ============================================================================

class ProductionAgent:
    """Production-grade Agent orchestrator integrating LangGraph, RAG, Security, and Caching."""

    def __init__(
        self,
        rag_pipeline: Optional[Any] = None,
        security_guard: Optional[UnifiedSecurityGuard] = None,
        cache_instance: Optional[Any] = None,
        semantic_cache_instance: Optional[Any] = None,
        llm: Optional[Any] = None,
        settings: Optional[Settings] = None,
        max_iterations: int = 5,
    ) -> None:
        """Initialize ProductionAgent with required pipeline components."""
        self.settings = settings or get_settings()
        self.max_iterations = max_iterations
        self._llm = llm

        # Security Guardrails
        self.security_guard = security_guard or UnifiedSecurityGuard()

        # Multi-tiered and Semantic Cache
        self.cache = cache_instance or tiered_cache
        self.semantic_cache = semantic_cache_instance or semantic_cache

        # RAG Knowledge Pipeline (lazy loaded if not provided)
        self._rag_pipeline = rag_pipeline

        # Build and compile LangGraph workflow
        self.graph = self._build_graph()

    @property
    def rag_pipeline(self) -> Any:
        """Lazily initialize RAGPipeline if not injected."""
        if self._rag_pipeline is None:
            try:
                from langchain_rag.rag_pipeline import create_rag_pipeline
                self._rag_pipeline = create_rag_pipeline(collection_name="production_kb")
            except Exception as e:
                logger.warning(f"RAGPipeline initialization deferred: {e}")
                self._rag_pipeline = None
        return self._rag_pipeline

    @property
    def llm(self) -> Any:
        """Lazily initialize LLM instance if not injected."""
        if self._llm is None:
            try:
                from langchain_rag.llm import get_llm
                self._llm = get_llm(model=self.settings.primary_model)
            except Exception as e:
                logger.warning(f"LLM initialization deferred or running in offline mode: {e}")
                self._llm = None
        return self._llm

    @property
    def model_name(self) -> str:
        """Return the active model identifier."""
        if self._llm is not None:
            return getattr(self._llm, "model_name", getattr(self._llm, "model", self.settings.primary_model))
        return self.settings.primary_model

    # ------------------------------------------------------------------------
    # LangGraph State Graph Builder
    # ------------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Construct the compiled LangGraph StateGraph workflow."""
        if StateGraph is None:
            return None

        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("guardrail_node", self.guardrail_node)
        workflow.add_node("cache_lookup_node", self.cache_lookup_node)
        workflow.add_node("router_node", self.router_node)
        workflow.add_node("retrieve_node", self.retrieve_node)
        workflow.add_node("generate_node", self.generate_node)
        workflow.add_node("cache_store_node", self.cache_store_node)
        workflow.add_node("output_guard_node", self.output_guard_node)

        # Set Entry Point
        workflow.set_entry_point("guardrail_node")

        # Conditional Edges from Guardrails
        workflow.add_conditional_edges(
            "guardrail_node",
            self._decide_after_guardrail,
            {
                "cache_lookup": "cache_lookup_node",
                "blocked": "output_guard_node",
            },
        )

        # Conditional Edges from Cache Lookup
        workflow.add_conditional_edges(
            "cache_lookup_node",
            self._decide_after_cache,
            {
                "hit": "output_guard_node",
                "miss": "router_node",
            },
        )

        # Conditional Edges from Router
        workflow.add_conditional_edges(
            "router_node",
            self._decide_route,
            {
                "rag": "retrieve_node",
                "direct_llm": "generate_node",
                "clarify": "output_guard_node",
            },
        )

        # Standard Transitions
        # Flow: retrieve → generate → output validation → cache store → END
        workflow.add_edge("retrieve_node", "generate_node")
        workflow.add_edge("generate_node", "output_guard_node")
        workflow.add_edge("output_guard_node", "cache_store_node")
        workflow.add_edge("cache_store_node", END)

        return workflow.compile()

    # ------------------------------------------------------------------------
    # Graph Routing Logic
    # ------------------------------------------------------------------------

    @staticmethod
    def _decide_after_guardrail(state: AgentState) -> str:
        """Route to cache lookup or directly block request."""
        if state.get("security_status") == "blocked":
            return "blocked"
        return "cache_lookup"

    @staticmethod
    def _decide_after_cache(state: AgentState) -> str:
        """Route to output if cached, otherwise proceed to router."""
        if state.get("is_cached", False):
            return "hit"
        return "miss"

    @staticmethod
    def _decide_route(state: AgentState) -> str:
        """Determine downstream reasoning node based on intent classification."""
        return state.get("route", "rag")

    # ------------------------------------------------------------------------
    # Graph Workflow Nodes
    # ------------------------------------------------------------------------

    def guardrail_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Inbound Security Guardrail validation."""
        query = state.get("query", "")
        start_t = time.perf_counter()
        security_notes: List[str] = list(state.get("security_notes") or [])

        try:
            # 1. Code exploit scan
            is_exploit, exploits = self.security_guard.code_guard.scan(query)
            if is_exploit:
                self.security_guard.audit_logger.record_event(
                    SecurityEventType.CODE_INJECTION,
                    ThreatLevel.MALICIOUS,
                    f"Blocked code exploit: {', '.join(exploits)}",
                )
                note = f"Blocked code exploit: {exploits}"
                return {
                    "security_status": "blocked",
                    "security_metadata": {
                        "exploits": exploits,
                        "security_notes": [note],
                    },
                    "security_notes": [note],
                    "final_response": "I cannot fulfill this request as it contains prohibited code patterns.",
                }

            # 2. Check prompt injection scan
            inj_result = detect_prompt_injection(query)
            if not inj_result.is_safe:
                logger.warning(f"Inbound prompt injection detected: {inj_result.reason}")
                note = f"Prompt injection blocked: {inj_result.reason}"
                return {
                    "security_status": "blocked",
                    "security_metadata": {
                        "threat_level": str(inj_result.threat_level),
                        "reason": inj_result.reason,
                        "flagged_patterns": inj_result.flagged_patterns,
                        "security_notes": [note],
                    },
                    "security_notes": [note],
                    "final_response": "I cannot fulfill this request as it violates security and safety policies.",
                }
            elif inj_result.flagged_patterns:
                security_notes.append(f"Prompt injection patterns flagged: {inj_result.flagged_patterns}")

            sanitized_query = inj_result.sanitized_text or query

            # 3. Secret scan
            has_secrets, secrets, sanitized_query = self.security_guard.secret_scanner.scan(sanitized_query)
            if has_secrets:
                self.security_guard.audit_logger.record_event(
                    SecurityEventType.SECRET_DETECTED,
                    ThreatLevel.SUSPICIOUS,
                    f"Redacted secrets in user input: {', '.join(secrets)}",
                )
                security_notes.append(f"Input secrets redacted: {secrets}")

            # 4. PII scan & masking
            pii_result = self.security_guard.pii_detector.scan(sanitized_query)
            if pii_result.contains_pii:
                sanitized_query = pii_result.masked_text
                masked_entities = sorted(list(pii_result.entity_counts.keys()))
                security_notes.append(f"Input PII masked: {masked_entities}")

            inbound_latency = round((time.perf_counter() - start_t) * 1000, 2)
            sec_metadata = {
                "inbound_latency_ms": inbound_latency,
                "security_notes": security_notes,
            }

            return {
                "query": sanitized_query,
                "security_status": "safe",
                "security_metadata": sec_metadata,
                "security_notes": security_notes,
            }

        except Exception as e:
            logger.error(f"Error in guardrail_node: {e}", exc_info=True)
            return {
                "security_status": "safe",
                "security_metadata": {"guardrail_error": str(e), "security_notes": security_notes},
                "security_notes": security_notes,
            }

    def cache_lookup_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Multi-Tier Exact & Semantic Cache Lookup."""
        query = state.get("query", "")
        tenant_id = state.get("tenant_id", "default")
        use_cache = state.get("metadata", {}).get("use_cache", True)

        if not use_cache:
            return {"is_cached": False}

        # 1. Exact composite key cache lookup
        cache_key = CacheKeyBuilder.build_key(
            prompt=query,
            model=self.model_name,
            tenant_id=tenant_id,
        )

        try:
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                logger.info(f"L1/L2 Cache hit for query: {query[:40]}")
                metrics_collector.record_cache(hit=True)
                response_text = cached_val if isinstance(cached_val, str) else cached_val.get("response", str(cached_val))
                sources = cached_val.get("sources", []) if isinstance(cached_val, dict) else []

                return {
                    "is_cached": True,
                    "final_response": response_text,
                    "documents": sources,
                    "route": "cache_hit",
                }

            # 2. Semantic vector cache lookup
            if self.semantic_cache:
                sem_hit = self.semantic_cache.lookup(query)
                if sem_hit is not None:
                    response_text, sim_score = sem_hit
                    logger.info(f"Semantic Cache hit (score={sim_score}) for query: {query[:40]}")
                    metrics_collector.record_cache(hit=True)
                    return {
                        "is_cached": True,
                        "final_response": response_text,
                        "confidence_score": sim_score,
                        "route": "semantic_cache_hit",
                    }

        except Exception as e:
            logger.warning(f"Cache lookup failed, proceeding to generation: {e}")

        metrics_collector.record_cache(hit=False)
        return {"is_cached": False}

    def router_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Dynamic Complexity & Intent Classification."""
        query = state.get("query", "").strip().lower()

        # Simple heuristics for routing; can be extended with an LLM classifier
        conversational_greetings = {"hello", "hi", "hey", "how are you", "who are you", "good morning", "good evening"}
        
        if query in conversational_greetings:
            return {
                "route": "direct_llm",
                "final_response": "Hello! I am your Enterprise AI Assistant. How can I assist you with your documentation, knowledge base, or system queries today?",
            }

        if len(query) < 3:
            return {
                "route": "clarify",
                "final_response": "Could you please provide more detail or elaborate on your question?",
            }

        # Default to RAG knowledge retrieval
        return {"route": "rag"}

    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 4: RAG Knowledge Retrieval & Context Assembly."""
        query = state.get("query", "")
        top_k = state.get("metadata", {}).get("top_k", 4)
        filter_dict = state.get("metadata", {}).get("filter")

        retrieved_docs: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        formatted_context = ""

        try:
            if self.rag_pipeline is not None:
                retrieved = self.rag_pipeline.retrieve(query=query, k=top_k, where=filter_dict)
                formatted_context = self.rag_pipeline.format_context(retrieved)

                for idx, r in enumerate(retrieved, start=1):
                    meta = r.get("metadata", {}) or {}
                    source_name = meta.get("source") or meta.get("filename") or f"doc_{idx}"
                    doc_item = {
                        "id": r.get("id", f"doc_{idx}"),
                        "content": r.get("text", ""),
                        "source": source_name,
                        "score": r.get("distance", 0.0),
                        "metadata": meta,
                    }
                    retrieved_docs.append(doc_item)
                    citations.append({
                        "citation": f"[{idx}]",
                        "source": source_name,
                        "metadata": meta,
                        "content_preview": " ".join(r.get("text", "").split())[:120] + "...",
                    })
            else:
                formatted_context = "No vector store connected."

        except Exception as e:
            logger.error(f"Error during document retrieval: {e}", exc_info=True)
            formatted_context = f"[Retrieval Error: {e}]"

        return {
            "context": formatted_context,
            "documents": retrieved_docs,
            "citations": citations,
        }

    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 5: Grounded LLM Response Generation & Synthesis."""
        # Check if router already set a response
        if state.get("final_response"):
            return {}

        query = state.get("query", "")
        context = state.get("context", "")
        route = state.get("route", "rag")

        prompt_text = (
            f"Context Information:\n{context}\n\n"
            f"User Question: {query}\n\n"
            f"Provide a comprehensive, accurate answer. When citing facts from the context, "
            f"cite the relevant source numbers using brackets (e.g. [1], [2])."
        ) if route == "rag" and context else query

        response_content = ""
        token_usage = {"prompt_tokens": len(prompt_text.split()), "completion_tokens": 0, "total_tokens": 0}

        try:
            llm_instance = self.llm
            if llm_instance is not None:
                start_llm = time.perf_counter()
                llm_response = llm_instance.invoke(prompt_text)
                llm_latency_sec = time.perf_counter() - start_llm

                if hasattr(llm_response, "content"):
                    response_content = str(llm_response.content)
                else:
                    response_content = str(llm_response)

                token_usage["completion_tokens"] = len(response_content.split())
                token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                metrics_collector.record_llm_call(
                    model=self.model_name,
                    prompt_tokens=token_usage["prompt_tokens"],
                    completion_tokens=token_usage["completion_tokens"],
                    duration_seconds=llm_latency_sec,
                )
            else:
                # Offline / RAG retrieval fallback
                doc_count = len(state.get("documents", []))
                if doc_count > 0:
                    response_content = (
                        f"Retrieved {doc_count} relevant knowledge document(s) for query '{query}':\n\n"
                        f"{context}"
                    )
                else:
                    response_content = f"Processed query: {query}"

        except Exception as e:
            logger.error(f"Error invoking LLM: {e}", exc_info=True)
            metrics_collector.record_error("llm_error")
            response_content = f"[Generation Notice: {e}]. Retrieved relevant context:\n\n{context}"

        return {
            "final_response": response_content,
            "token_usage": token_usage,
        }

    def cache_store_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 6: Save response to exact & semantic cache."""
        query = state.get("query", "")
        response_text = state.get("final_response", "")
        tenant_id = state.get("tenant_id", "default")
        is_cached = state.get("is_cached", False)
        security_status = state.get("security_status", "safe")

        if is_cached or security_status != "safe" or not response_text:
            return {}

        try:
            cache_key = CacheKeyBuilder.build_key(
                prompt=query,
                model=self.model_name,
                tenant_id=tenant_id,
            )
            cache_payload = {
                "response": response_text,
                "sources": state.get("documents", []),
            }
            self.cache.set(cache_key, cache_payload, ttl=self.settings.cache_ttl_seconds)

            if self.semantic_cache:
                self.semantic_cache.store(prompt=query, response=response_text)

        except Exception as e:
            logger.warning(f"Failed to store result in cache: {e}")

        return {}

    def output_guard_node(self, state: AgentState) -> Dict[str, Any]:
        """Node 7: Outbound Security & Leak Prevention."""
        response_text = state.get("final_response", "")
        if not response_text:
            return {}

        security_notes = list(state.get("security_notes") or [])
        try:
            cleaned, warnings = self.security_guard.process_outbound(response_text)
            if warnings:
                logger.warning(f"Outbound guardrail warning: {warnings}")
                security_notes.extend(warnings)
            
            sec_meta = dict(state.get("security_metadata", {}))
            sec_meta["security_notes"] = security_notes
            return {
                "final_response": cleaned,
                "security_notes": security_notes,
                "security_metadata": sec_meta,
            }
        except Exception as e:
            logger.error(f"Error in output_guard_node: {e}", exc_info=True)

        return {}

    # ------------------------------------------------------------------------
    # Public Execution APIs
    # ------------------------------------------------------------------------

    async def arun(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: str = "default",
        history: Optional[Sequence[Union[ChatMessage, BaseMessage, dict]]] = None,
        use_cache: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the production agent asynchronously on a user query."""
        start_time = time.perf_counter()
        meta = dict(metadata or {})
        meta["use_cache"] = use_cache

        initial_state = create_initial_state(
            query=query,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            history=history,
            metadata=meta,
            max_iterations=self.max_iterations,
        )

        try:
            if self.graph is not None:
                # Execute compiled LangGraph StateGraph
                final_state = await self.graph.ainvoke(initial_state)
            else:
                # Direct fallback pipeline if LangGraph is uncompiled
                # Flow: security → cache lookup → agent → output validation → cache store
                s = dict(initial_state)
                s.update(self.guardrail_node(s))
                if s.get("security_status") != "blocked":
                    s.update(self.cache_lookup_node(s))
                    if not s.get("is_cached"):
                        s.update(self.router_node(s))
                        if s.get("route") == "rag":
                            s.update(self.retrieve_node(s))
                        s.update(self.generate_node(s))
                        s.update(self.output_guard_node(s))
                        s.update(self.cache_store_node(s))
                    else:
                        s.update(self.output_guard_node(s))
                final_state = s

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            final_state["latency_ms"] = latency_ms

            security_notes = final_state.get("security_notes") or final_state.get("security_metadata", {}).get("security_notes", [])
            now_iso = datetime.now(timezone.utc).isoformat()

            return {
                "answer": final_state.get("final_response", ""),
                "sources": final_state.get("documents", []),
                "citations": final_state.get("citations", []),
                "is_cached": final_state.get("is_cached", False),
                "security_status": final_state.get("security_status", "safe"),
                "security_notes": security_notes,
                "model_used": self.model_name,
                "timestamp": now_iso,
                "token_usage": final_state.get("token_usage", {}),
                "latency_ms": latency_ms,
                "processing_time_ms": latency_ms,
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Unhandled exception in ProductionAgent.arun: {e}", exc_info=True)
            metrics_collector.record_error("agent_execution_error")
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            now_iso = datetime.now(timezone.utc).isoformat()
            return {
                "answer": f"An error occurred while processing your request: {str(e)}",
                "sources": [],
                "citations": [],
                "is_cached": False,
                "security_status": "error",
                "security_notes": [f"Execution error: {str(e)}"],
                "model_used": self.model_name,
                "timestamp": now_iso,
                "error": str(e),
                "latency_ms": latency_ms,
                "processing_time_ms": latency_ms,
                "session_id": session_id,
            }

    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for agent execution."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(self.arun(query, **kwargs)))
                    return future.result()
            else:
                return loop.run_until_complete(self.arun(query, **kwargs))
        except RuntimeError:
            return asyncio.run(self.arun(query, **kwargs))

    async def process_request(self, request: ChatRequest) -> ChatResponse:
        """Process incoming FastAPI ChatRequest model into a validated ChatResponse."""
        result = await self.arun(
            query=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
            history=request.history,
            use_cache=request.use_cache,
            metadata=request.metadata,
        )

        source_docs: List[SourceDocument] = []
        for doc in result.get("sources", []):
            if isinstance(doc, SourceDocument):
                source_docs.append(doc)
            elif isinstance(doc, dict):
                source_docs.append(
                    SourceDocument(
                        document_id=doc.get("id"),
                        title=doc.get("source"),
                        content=doc.get("content", doc.get("text", "")),
                        score=doc.get("score"),
                        metadata=doc.get("metadata", {}),
                    )
                )

        usage = TokenUsage(**result.get("token_usage", {})) if result.get("token_usage") else None
        security_notes = result.get("security_notes", [])
        now_ts = result.get("timestamp") or datetime.now(timezone.utc).isoformat()
        latency = result.get("latency_ms")

        return ChatResponse(
            response=result.get("answer", ""),
            session_id=result.get("session_id"),
            cached=result.get("is_cached", False),
            model_used=result.get("model_used", self.model_name),
            processing_time_ms=latency,
            latency_ms=latency,
            security_notes=security_notes,
            timestamp=now_ts,
            sources=source_docs,
            usage=usage,
            metadata={
                "security_status": result.get("security_status"),
                "security_notes": security_notes,
                "model_used": result.get("model_used", self.model_name),
                "processing_time_ms": latency,
                "timestamp": now_ts,
                "citations": result.get("citations", []),
            },
        )

    async def astream(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated response tokens asynchronously."""
        result = await self.arun(
            query=query,
            session_id=session_id,
            user_id=user_id,
            use_cache=use_cache,
            **kwargs,
        )
        answer = result.get("answer", "")
        # Stream response in natural word chunks
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.01)


# ============================================================================
# 3. Compatibility Aliases & Default Singleton
# ============================================================================

class AgentService(ProductionAgent):
    """Backward-compatible wrapper for legacy AgentService consumers."""
    pass


# Global singleton agent instances
agent_service = AgentService()
production_agent = agent_service

__all__ = [
    "AgentState",
    "ProductionAgent",
    "AgentService",
    "create_initial_state",
    "production_agent",
    "agent_service",
]
