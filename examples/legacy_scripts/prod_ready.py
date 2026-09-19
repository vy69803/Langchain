#!/usr/bin/env python3
"""Root alias for Production-Ready Chunking Architecture & Benchmark Suite."""

import sys
from pathlib import Path

# Add Chunking directory and src to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir / "Chunking"))

from Chunking.prod_ready import main

if __name__ == "__main__":
    main()
