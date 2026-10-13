"""Golden-suite setup: make the WP2 kit (``tests/streaming/_kit.py``) importable.

``tests/`` uses no ``__init__.py`` files; pytest puts each test directory on
``sys.path``, so ``tests/golden`` can only ``import _kit`` once
``tests/streaming`` is importable — the same bridge ``tests/conftest.py``
already applies for ``examples/smoke``.
"""

import sys
from pathlib import Path

STREAMING_DIR = str(Path(__file__).resolve().parent.parent / "streaming")
if STREAMING_DIR not in sys.path:
    sys.path.insert(0, STREAMING_DIR)
