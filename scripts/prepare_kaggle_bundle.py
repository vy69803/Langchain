#!/usr/bin/env python
"""Prepares zip archives for offloading batch ingestion to Kaggle.

Creates:
1. `kaggle_bundle/financebench_dataset.zip` (368 PDFs + JSONL metadata)
2. `kaggle_bundle/financial_intelligence_code.zip` (src/ and scripts/)
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT_DIR / "kaggle_bundle"
DATA_DIR = ROOT_DIR / "data" / "financebench"
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"


def create_dataset_bundle() -> Path:
    """Zip 368 PDFs and metadata files into financebench_dataset.zip."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = BUNDLE_DIR / "financebench_dataset.zip"

    print(f"\n[1/2] Creating Dataset Bundle: {out_zip}...")
    pdf_files = list((DATA_DIR / "pdfs").glob("*.pdf"))
    print(f"      Found {len(pdf_files)} PDF files in {DATA_DIR / 'pdfs'}")

    meta_files = [
        DATA_DIR / "financebench_document_information.jsonl",
        DATA_DIR / "financebench_open_source.jsonl",
    ]

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add metadata files
        for mf in meta_files:
            if mf.exists():
                arcname = mf.name
                zf.write(mf, arcname=arcname)
                print(f"      + Added metadata: {arcname}")

        # Add PDFs
        for idx, pdf in enumerate(pdf_files, start=1):
            arcname = f"pdfs/{pdf.name}"
            zf.write(pdf, arcname=arcname)
            if idx % 50 == 0 or idx == len(pdf_files):
                print(f"      + Added {idx}/{len(pdf_files)} PDFs...")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"      Done! Created {out_zip.name} ({size_mb:.1f} MB)\n")
    return out_zip


def create_code_bundle() -> Path:
    """Zip src/ and scripts/ directories into financial_intelligence_code.zip."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = BUNDLE_DIR / "financial_intelligence_code.zip"

    print(f"[2/2] Creating Code Bundle: {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add src/
        for root, _, files in os.walk(SRC_DIR):
            for file in files:
                if file.endswith((".py", ".json", ".txt")) and "__pycache__" not in root:
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(ROOT_DIR)
                    zf.write(full_path, arcname=str(rel_path))

        # Add scripts/
        for root, _, files in os.walk(SCRIPTS_DIR):
            for file in files:
                if file.endswith((".py", ".sh", ".json")) and "__pycache__" not in root and file != "prepare_kaggle_bundle.py":
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(ROOT_DIR)
                    zf.write(full_path, arcname=str(rel_path))

    size_kb = out_zip.stat().st_size / 1024
    print(f"      Done! Created {out_zip.name} ({size_kb:.1f} KB)\n")
    return out_zip


def main() -> None:
    print("=" * 60)
    print("   PREPARING KAGGLE BUNDLES FOR OFFLOADED INGESTION")
    print("=" * 60)
    dataset_zip = create_dataset_bundle()
    code_zip = create_code_bundle()
    print("=" * 60)
    print(" Bundles ready in folder: kaggle_bundle/")
    print(f" 1. {dataset_zip.name} -> Upload as Kaggle Dataset: 'financebench-dataset'")
    print(f" 2. {code_zip.name}    -> Unpacked automatically by Kaggle notebook")
    print("=" * 60)


if __name__ == "__main__":
    main()
