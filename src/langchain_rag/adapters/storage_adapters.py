"""Storage Adapter Interfaces and Implementations.

Defines unified adapter interfaces for vector databases (ChromaDB, Qdrant)
and graph databases (Neo4j), ensuring incremental migration and testability.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseVectorStoreAdapter(ABC):
    """Abstract interface for vector retrieval stores."""

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search vector database for top-k semantically relevant chunks."""
        pass

    @abstractmethod
    def upsert_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add or update text chunks with associated metadata."""
        pass


class ChromaVectorAdapter(BaseVectorStoreAdapter):
    """Vector store adapter wrapping ChromaDB."""

    def __init__(self, collection_name: str = "financebench", persist_directory: Optional[str] = None) -> None:
        from langchain_rag.vector_stores import VectorStore

        self.store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        where = None
        if filter_metadata:
            if len(filter_metadata) == 1 or "$and" in filter_metadata or "$or" in filter_metadata:
                where = filter_metadata
            else:
                where = {"$and": [{k_field: v} for k_field, v in filter_metadata.items()]}
        results = self.store.query(query_text=query, n_results=k, where=where)
        formatted: List[Dict[str, Any]] = []
        for r in results:
            dist = r.get("distance")
            score = 1.0 - dist if dist is not None else 1.0
            formatted.append(
                {
                    "id": r.get("id"),
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata") or {},
                    "score": score,
                }
            )
        return formatted

    def upsert_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        return self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


class SupabaseVectorAdapter(BaseVectorStoreAdapter):
    """Production vector store adapter for Supabase pgvector using halfvec(384) & IVFFlat."""

    def __init__(
        self,
        db_url: Optional[str] = None,
        table_name: str = "financebench_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        ivfflat_probes: int = 10,
    ) -> None:
        import os
        from dotenv import load_dotenv

        load_dotenv()
        self.db_url = db_url or os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("SUPABASE_DB_URL or DATABASE_URL must be provided.")
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.ivfflat_probes = ivfflat_probes
        self._embedder = None

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from langchain_rag.embeddings import LocalEmbeddings

            self._embedder = LocalEmbeddings(model_name=self.embedding_model)
        return self._embedder

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search Supabase pgvector table using halfvec cosine similarity and indexed filters."""
        import psycopg2

        # 1. Embed query
        query_vec = self.embedder.embed_query(query)
        vec_str = f"[{','.join(f'{x:.5f}' for x in query_vec)}]"

        # 2. Build parameterized WHERE clause targeting B-tree columns
        where_clauses: List[str] = []
        filter_params: List[Any] = []

        if filter_metadata:
            for key, val in filter_metadata.items():
                if key == "company":
                    where_clauses.append("company = %s")
                    filter_params.append(str(val))
                elif key in ("filing_year", "year", "doc_period"):
                    where_clauses.append("filing_year = %s")
                    filter_params.append(int(val))
                elif key == "doc_type":
                    where_clauses.append("doc_type = %s")
                    filter_params.append(str(val))
                else:
                    where_clauses.append("metadata->>%s = %s")
                    filter_params.extend([str(key), str(val)])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query_sql = f"""
            SELECT 
                id, 
                content, 
                company, 
                filing_year, 
                doc_type, 
                metadata, 
                1 - (embedding <=> %s::halfvec) AS score
            FROM {self.table_name}
            {where_sql}
            ORDER BY embedding <=> %s::halfvec ASC
            LIMIT %s;
        """

        conn = psycopg2.connect(self.db_url)
        try:
            with conn.cursor() as cur:
                # Set probe count for accuracy vs. speed balance
                cur.execute(f"SET ivfflat.probes = {self.ivfflat_probes};")

                params = [vec_str] + filter_params + [vec_str, k]
                cur.execute(query_sql, params)
                rows = cur.fetchall()

                results = []
                for r in rows:
                    meta = r[5] or {}
                    if r[2]:
                        meta["company"] = r[2]
                    if r[3]:
                        meta["filing_year"] = r[3]
                    if r[4]:
                        meta["doc_type"] = r[4]

                    results.append({
                        "id": r[0],
                        "text": r[1],
                        "metadata": meta,
                        "score": float(r[6]),
                    })
                return results
        finally:
            conn.close()

    def upsert_chunks(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert or update chunks directly into Supabase with halfvec embeddings."""
        import json
        import uuid
        import psycopg2
        from psycopg2.extras import execute_values

        if not texts:
            return []

        doc_ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self.embedder.embed_documents(texts)

        def clean_s(val):
            return str(val).replace("\x00", "") if val is not None else None

        def clean_d(d):
            if isinstance(d, str):
                return d.replace("\x00", "")
            elif isinstance(d, dict):
                return {clean_s(k): clean_d(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_d(x) for x in d]
            return d

        records = []
        for doc_id, text, meta, emb in zip(doc_ids, texts, metadatas, embeddings):
            meta_dict = clean_d(meta or {})
            company = clean_s(meta_dict.get("company"))
            raw_period = meta_dict.get("doc_period") or meta_dict.get("year") or meta_dict.get("filing_year")
            filing_year = int(raw_period) if raw_period and str(raw_period).isdigit() else None
            doc_type = clean_s(meta_dict.get("doc_type"))
            emb_str = f"[{','.join(f'{x:.5f}' for x in emb)}]"
            records.append((clean_s(doc_id), clean_s(text) or "", company, filing_year, doc_type, json.dumps(meta_dict), emb_str))

        insert_sql = f"""
            INSERT INTO {self.table_name} (id, content, company, filing_year, doc_type, metadata, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE 
            SET content = EXCLUDED.content,
                company = EXCLUDED.company,
                filing_year = EXCLUDED.filing_year,
                doc_type = EXCLUDED.doc_type,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding;
        """

        conn = psycopg2.connect(self.db_url)
        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    insert_sql,
                    records,
                    template="(%s, %s, %s, %s, %s, %s::jsonb, %s::halfvec)",
                )
            conn.commit()
            return doc_ids
        finally:
            conn.close()


class BaseGraphStoreAdapter(ABC):
    """Abstract interface for knowledge graph stores."""

    @abstractmethod
    def execute_cypher(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        read_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the knowledge graph."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify graph database connectivity."""
        pass


class Neo4jGraphAdapter(BaseGraphStoreAdapter):
    """Neo4j Knowledge Graph adapter using official neo4j bolt driver."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        import os

        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
        password = password or os.getenv("NEO4J_PASSWORD", "financial_graph_rag_password")
        database = database or os.getenv("NEO4J_DATABASE")
        timeout = float(
            os.getenv(
                "NEO4J_TIMEOUT",
                "20.0" if ("databases.neo4j.io" in uri or "neo4j+s" in uri or "+s" in uri) else "2.0",
            )
        )

        try:
            import neo4j

            self._driver = neo4j.GraphDatabase.driver(
                uri,
                auth=(user, password),
                connection_timeout=timeout,
                max_connection_lifetime=300.0,
            )
            self._database = database
            logger.info(f"Initialized Neo4j driver at {uri}")
        except Exception as e:
            logger.warning(f"Neo4j driver initialization failed (offline mode): {e}")
            self._driver = None
            self._database = database

        self._available: Optional[bool] = None

    def close(self) -> None:
        """Close driver connection."""
        if self._driver:
            self._driver.close()

    def _get_session(self):
        """Helper to create session with or without explicit database name."""
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def health_check(self) -> bool:
        """Verify Neo4j connectivity."""
        if not self._driver or self._available is False:
            return False
        try:
            with self._get_session() as session:
                result = session.run("RETURN 1 AS ping")
                record = result.single()
                is_ok = record is not None and record["ping"] == 1
                self._available = is_ok
                return is_ok
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")
            self._available = False
            return False

    def execute_cypher(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        read_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Safely execute a Cypher query with read-only validation."""
        if read_only:
            query_upper = query.strip().upper()
            forbidden = ["DELETE", "DETACH", "DROP", "CREATE", "SET", "REMOVE", "MERGE"]
            for keyword in forbidden:
                # Basic token boundary check
                if f" {keyword} " in f" {query_upper} ":
                    raise PermissionError(f"Operation '{keyword}' forbidden in read-only Cypher query.")

        if not self._driver or self._available is False:
            logger.warning("Neo4j driver offline. Returning empty query result.")
            return []

        params = params or {}
        try:
            with self._get_session() as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Cypher query '{query}': {e}")
            raise
