-- ============================================================================
-- Supabase Schema: Production Optimized Vector Store (pgvector 0.8+)
-- Features:
--   1. halfvec(384) for 50% vector memory & storage reduction
--   2. Dedicated B-Tree indexed columns for high-speed company/year filtering
--   3. TOAST extended compression for document text chunks
--   4. IVFFlat cosine similarity index (~18 MB vs. 220 MB for HNSW)
-- ============================================================================

-- Step 1: Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create the document table with halfvec(384)
CREATE TABLE IF NOT EXISTS financebench_docs (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    company VARCHAR(64),
    filing_year SMALLINT,
    doc_type VARCHAR(16),
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding halfvec(384),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Step 3: Enable TOAST compression for the content column
ALTER TABLE financebench_docs ALTER COLUMN content SET STORAGE EXTENDED;

-- Step 4: Lightweight B-Tree indexes for fast exact metadata filtering (~3 MB each)
CREATE INDEX IF NOT EXISTS idx_financebench_company ON financebench_docs (company);
CREATE INDEX IF NOT EXISTS idx_financebench_year ON financebench_docs (filing_year);
CREATE INDEX IF NOT EXISTS idx_financebench_doc_type ON financebench_docs (doc_type);

-- Step 5: IVFFlat Vector Index
-- IMPORTANT: Run this index creation AFTER inserting the 134,663 chunks!
-- Building IVFFlat on populated data trains optimal Voronoi centroid clusters.
-- lists = sqrt(134663) ≈ 366 (360 is optimal for batch clustering)
--
-- CREATE INDEX IF NOT EXISTS idx_financebench_ivfflat 
-- ON financebench_docs USING ivfflat (embedding halfvec_cosine_ops)
-- WITH (lists = 360);

-- Step 6: Diagnostic query to check disk usage and verify < 250 MB
-- SELECT 
--     pg_size_pretty(pg_relation_size('financebench_docs')) AS table_size,
--     pg_size_pretty(pg_total_relation_size('financebench_docs')) AS total_size_with_indexes;
