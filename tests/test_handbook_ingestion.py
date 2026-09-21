"""Unit tests for GitLab Handbook Parser and Ingestion Pipeline."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from langchain_rag.handbook_parser import (
    HandbookParser,
    clean_hugo_shortcodes,
    extract_department_and_breadcrumbs,
    parse_frontmatter,
    resolve_handbook_url,
)
from langchain_rag.handbook_pipeline import (
    HandbookIngestionPipeline,
    sanitize_metadata_for_chroma,
)
from langchain_rag.vector_stores import VectorStore


class TestHandbookParser(unittest.TestCase):
    def setUp(self):
        self.parser = HandbookParser()

    def test_parse_frontmatter(self):
        content = """---
title: "GitLab Values"
description: "Core values of GitLab"
tags:
  - values
  - culture
---

# Values
GitLab's values are CREDIT.
"""
        meta, body = parse_frontmatter(content)
        self.assertEqual(meta.get("title"), "GitLab Values")
        self.assertEqual(meta.get("description"), "Core values of GitLab")
        self.assertIn("values", meta.get("tags", []))
        self.assertIn("# Values", body)
        self.assertNotIn("---", body)

    def test_clean_hugo_shortcodes(self):
        raw = """
{{% alert title="Important Notice" color="warning" %}}
This is a critical alert.
{{% /alert %}}

{{< panel header="**Security Guidelines**" >}}
Do not share secrets.
{{< /panel >}}

{{% note %}}
Remember to submit MRs promptly.
{{% /note %}}

{{< youtube "dQw4w9WgXcQ" >}}

{{% alert %}}
Generic alert without title.
{{% /alert %}}
"""
        cleaned = clean_hugo_shortcodes(raw)
        self.assertIn("> **[Important Notice]** This is a critical alert.", cleaned)
        self.assertIn("> **[Security Guidelines]** Do not share secrets.", cleaned)
        self.assertIn("> **Note**: Remember to submit MRs promptly.", cleaned)
        self.assertIn("[Video Resource: https://youtube.com/watch?v=dQw4w9WgXcQ]", cleaned)
        self.assertIn("> **[Alert]** Generic alert without title.", cleaned)
        self.assertNotIn("{{%", cleaned)
        self.assertNotIn("{{<", cleaned)

    def test_resolve_handbook_url(self):
        url1 = resolve_handbook_url("handbook/content/handbook/_index.md")
        self.assertEqual(url1, "https://handbook.gitlab.com/handbook/")

        url2 = resolve_handbook_url("handbook/content/handbook/values/index.md")
        self.assertEqual(url2, "https://handbook.gitlab.com/handbook/values/")

        url3 = resolve_handbook_url("handbook/content/handbook/engineering/architecture.md")
        self.assertEqual(url3, "https://handbook.gitlab.com/handbook/engineering/architecture/")

    def test_extract_department_and_breadcrumbs(self):
        dept, crumbs = extract_department_and_breadcrumbs(
            "handbook/content/handbook/engineering/development/architecture.md"
        )
        self.assertEqual(dept, "engineering")
        self.assertEqual(crumbs, ["Handbook", "Engineering", "Development", "Architecture"])

    def test_parse_text(self):
        raw = """---
title: "Remote Work Guide"
description: "How GitLab works remotely"
canonical_path: "/handbook/company/remote/"
---

## Introduction
All-remote means everyone works wherever they are happiest.
"""
        doc = self.parser.parse_text(raw, file_path="handbook/content/handbook/company/remote.md")
        self.assertEqual(doc.metadata["title"], "Remote Work Guide")
        self.assertEqual(doc.metadata["department"], "company")
        self.assertEqual(doc.metadata["url"], "https://handbook.gitlab.com/handbook/company/remote/")
        self.assertIn("All-remote means", doc.page_content)
        self.assertTrue(len(doc.metadata["doc_hash"]) == 64)


class TestHandbookPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.handbook_dir = self.root / "handbook" / "content" / "handbook"
        self.handbook_dir.mkdir(parents=True, exist_ok=True)

        self.values_dir = self.handbook_dir / "values"
        self.values_dir.mkdir(parents=True, exist_ok=True)

        # Create sample files
        (self.values_dir / "index.md").write_text(
            """---
title: "GitLab Values"
description: "Our core values"
---

# CREDIT Values
GitLab has six core values: Collaboration, Results, Efficiency, Diversity, Iteration, Transparency.

## Collaboration
We work together asynchronously across all time zones.
""",
            encoding="utf-8",
        )

        (self.values_dir / "transparency.md").write_text(
            """---
title: "Transparency Value"
description: "Transparency sub-value"
---

# Transparency
Be open about everything by default. Public by default.
""",
            encoding="utf-8",
        )

        self.persist_dir = self.root / "chroma_db"
        self.manifest_path = self.root / "manifest.json"
        self.bm25_path = self.root / "bm25.json"

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_sanitize_metadata_for_chroma(self):
        meta = {
            "title": "Test Title",
            "count": 42,
            "score": 0.95,
            "is_active": True,
            "tags": ["values", "remote"],
            "none_val": None,
            "dict_val": {"nested": "value"},
        }
        clean = sanitize_metadata_for_chroma(meta)
        self.assertEqual(clean["title"], "Test Title")
        self.assertEqual(clean["count"], 42)
        self.assertEqual(clean["score"], 0.95)
        self.assertEqual(clean["is_active"], True)
        self.assertEqual(clean["tags"], "values, remote")
        self.assertEqual(clean["none_val"], "")
        self.assertIsInstance(clean["dict_val"], str)

    def test_pipeline_dry_run(self):
        pipeline = HandbookIngestionPipeline(
            handbook_dir=self.root / "handbook",
            persist_directory=self.persist_dir,
            manifest_path=self.manifest_path,
            bm25_path=self.bm25_path,
            max_chunk_chars=500,
        )
        summary = pipeline.run(dry_run=True)
        self.assertEqual(summary["total_scanned"], 2)
        self.assertEqual(summary["processed_files"], 2)
        self.assertGreater(summary["total_chunks_generated"], 0)
        self.assertFalse(self.manifest_path.exists())

    def test_pipeline_persistent_and_incremental(self):
        pipeline = HandbookIngestionPipeline(
            handbook_dir=self.root / "handbook",
            persist_directory=self.persist_dir,
            manifest_path=self.manifest_path,
            bm25_path=self.bm25_path,
            max_chunk_chars=500,
        )

        # 1. First run - should process both files
        summary1 = pipeline.run()
        self.assertEqual(summary1["total_scanned"], 2)
        self.assertEqual(summary1["processed_files"], 2)
        self.assertEqual(summary1["skipped_unchanged"], 0)
        self.assertTrue(self.manifest_path.exists())
        self.assertTrue(self.bm25_path.exists())

        # 2. Second run without changes - should skip all files
        pipeline2 = HandbookIngestionPipeline(
            handbook_dir=self.root / "handbook",
            persist_directory=self.persist_dir,
            manifest_path=self.manifest_path,
            bm25_path=self.bm25_path,
            max_chunk_chars=500,
        )
        summary2 = pipeline2.run()
        self.assertEqual(summary2["total_scanned"], 2)
        self.assertEqual(summary2["processed_files"], 0)
        self.assertEqual(summary2["skipped_unchanged"], 2)

        # 3. Third run with force=True - should re-process all files
        summary3 = pipeline2.run(force=True)
        self.assertEqual(summary3["processed_files"], 2)
        self.assertEqual(summary3["skipped_unchanged"], 0)

    def test_pipeline_query(self):
        pipeline = HandbookIngestionPipeline(
            handbook_dir=self.root / "handbook",
            persist_directory=self.persist_dir,
            manifest_path=self.manifest_path,
            bm25_path=self.bm25_path,
            max_chunk_chars=500,
        )
        pipeline.run()

        # Hybrid query
        results = pipeline.query("What are the CREDIT values?", top_k=2, search_type="hybrid")
        self.assertGreater(len(results), 0)
        self.assertIn("Collaboration", results[0]["text"])

        # Dense query
        dense_results = pipeline.query("transparency public by default", top_k=1, search_type="dense")
        self.assertEqual(len(dense_results), 1)

        # Sparse BM25 query
        sparse_results = pipeline.query("asynchronously across all time zones", top_k=1, search_type="sparse")
        self.assertEqual(len(sparse_results), 1)


if __name__ == "__main__":
    unittest.main()
