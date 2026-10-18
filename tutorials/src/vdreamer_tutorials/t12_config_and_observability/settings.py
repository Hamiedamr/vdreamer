"""Settings loaded once from the environment and validated (``.env`` works too).

Every vdreamer command reads ``.env`` from its working directory into the
environment, so ``T12_*`` variables can live there next to the token. Secrets
follow the same rule as ``models.yaml``: configuration names an environment
variable; the value never appears in a file under version control.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

PREFIX = "T12_"


class SettingsError(ValueError):
    """The environment holds an invalid configuration."""


class ObservabilitySettings(BaseModel, frozen=True):
    shop_name: str = "Trailhead Outfitters"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    slow_tool_ms: int = Field(default=500, gt=0)
    #: The *name* of the env var holding the order-system key, never the key itself.
    orders_api_key_env: str = "T12_ORDERS_API_KEY"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ObservabilitySettings:
        """Read ``T12_<FIELD>`` variables; fail with every problem listed at once."""
        env = os.environ if environ is None else environ
        values = {
            name: env[PREFIX + name.upper()]
            for name in cls.model_fields
            if PREFIX + name.upper() in env
        }
        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            problems = "; ".join(
                f"{PREFIX}{'.'.join(map(str, e['loc'])).upper()}: {e['msg']}"
                for e in exc.errors(include_url=False)
            )
            raise SettingsError(f"invalid configuration: {problems}") from None

    def orders_api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        env = os.environ if environ is None else environ
        return env.get(self.orders_api_key_env)
