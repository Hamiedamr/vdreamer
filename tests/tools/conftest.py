"""Tool-governance setup: make the streaming kit (``tests/streaming/_kit.py``) importable."""

import sys
from pathlib import Path

STREAMING_DIR = str(Path(__file__).resolve().parent.parent / "streaming")
if STREAMING_DIR not in sys.path:
    sys.path.insert(0, STREAMING_DIR)
