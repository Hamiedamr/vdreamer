"""Shared pytest setup: make the smoke example importable as a top-level module.

CI does the same with PYTHONPATH=examples/smoke for the plain-pip job.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = str(REPO_ROOT / "examples" / "smoke")
if SMOKE_DIR not in sys.path:
    sys.path.insert(0, SMOKE_DIR)
