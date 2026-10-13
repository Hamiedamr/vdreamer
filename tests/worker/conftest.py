"""Worker-suite setup: make the WP2 kit (``tests/streaming/_kit.py``) importable.

Same bridge as ``tests/golden/conftest.py`` — ``tests/`` uses no ``__init__.py``
files, so ``import _kit`` only works once ``tests/streaming`` is on ``sys.path``.
"""

import sys
from pathlib import Path

STREAMING_DIR = str(Path(__file__).resolve().parent.parent / "streaming")
if STREAMING_DIR not in sys.path:
    sys.path.insert(0, STREAMING_DIR)
