"""The vendored static assets and their integrity check (PRD §5.2, A13).

No CDN, ever (§5.2): the gateway serves third-party browser assets from the
wheel itself, under ``/static/vendor/``. Each file is pinned here by sha256 —
``vdreamer doctor --offline`` fails when one is missing or modified, which is
what "vendored assets are covered by the zero-egress checks" means in
practice. Milestone 4 vendors Swagger UI; Milestone 6 adds the chart JS
(plotly.js, at ``/static/vendor/plotly/plotly.min.js``).

The same check asserts the gateway's own docs shell (``docs.html`` and
``swagger-init.js``, under ``static/vdreamer/``) references no external
origin: every ``src``/``href``/``url`` there must be same-origin.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

#: The gateway's static root inside the package (served at ``/static``).
STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "gateway" / "static"

#: Third-party files under ``STATIC_DIR / "vendor"`` → sha256 (see each dir's README).
VENDORED_ASSETS: Final[dict[str, str]] = {
    "plotly/LICENSE": "ba712274f8336a5f0828c8e2bf639ab8a632c623800a35f96c8b0a67b5f2ab40",
    "plotly/plotly.min.js": "3b6e15d45dbb7fca5bd2094291e961ddc5472cd887009e6009a56dab668d721f",
    "swagger-ui/LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "swagger-ui/favicon-32x32.png": (
        "3ed612f41e050ca5e7000cad6f1cbe7e7da39f65fca99c02e99e6591056e5837"
    ),
    "swagger-ui/swagger-ui-bundle.js": (
        "050bc415ee7048dcd881682678f720264e7da5e373f7461d7c58c755305255f7"
    ),
    "swagger-ui/swagger-ui.css": (
        "23ae95db3e810a3e1eb7d8995ddb587dd463c25d0c06bbd768d58ec8b08f6981"
    ),
}

#: vdreamer's own same-origin docs shell, checked for external references.
FIRST_PARTY_SHELL: Final[tuple[str, ...]] = ("vdreamer/docs.html", "vdreamer/swagger-init.js")

#: Any absolute or protocol-relative URL — the shell must contain none.
_EXTERNAL_URL: Final[re.Pattern[str]] = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+", re.IGNORECASE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_urls(text: str) -> list[str]:
    """Every absolute (``https://…``) or protocol-relative (``//host``) URL in ``text``."""
    return _EXTERNAL_URL.findall(text)


def check_vendored_assets(static_dir: Path = STATIC_DIR) -> list[str]:
    """Problems with the vendored assets and the docs shell; empty when all is well."""
    problems: list[str] = []
    for relative, expected in VENDORED_ASSETS.items():
        path = static_dir / "vendor" / relative
        if not path.is_file():
            problems.append(f"vendored asset missing: vendor/{relative}")
        elif _sha256(path) != expected:
            problems.append(f"vendored asset modified (sha256 mismatch): vendor/{relative}")
    for relative in FIRST_PARTY_SHELL:
        path = static_dir / relative
        if not path.is_file():
            problems.append(f"docs shell missing: {relative}")
            continue
        found = external_urls(path.read_text(encoding="utf-8"))
        if found:
            problems.append(f"docs shell references external origins: {relative}: {found}")
    return problems
