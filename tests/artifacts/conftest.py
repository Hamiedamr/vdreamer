"""Artifact/files/sandbox setup: make the streaming and tool kits importable."""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent
for kit in ("streaming", "tools"):
    path = str(TESTS / kit)
    if path not in sys.path:
        sys.path.insert(0, path)
