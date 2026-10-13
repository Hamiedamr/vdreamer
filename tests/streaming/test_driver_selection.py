"""Driver selection: ``VDREAMER_STREAM_DRIVER`` (PRD §2.4, A43).

Defaults to ``v2`` (the stable stream is the default driver); any other
value is a clear ``ValueError`` **at run start**, never at import.
"""

from __future__ import annotations

import pytest

from vdreamer.streaming import (
    STREAM_DRIVER_ENV_VAR,
    VALID_STREAM_DRIVERS,
    V2StreamDriver,
    V3StreamDriver,
    make_stream_driver,
    stream_driver_from_env,
)


def test_default_is_v2_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STREAM_DRIVER_ENV_VAR, raising=False)
    assert stream_driver_from_env() == "v2"


def test_env_override_selects_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STREAM_DRIVER_ENV_VAR, "v3")
    assert stream_driver_from_env() == "v3"
    assert isinstance(make_stream_driver(stream_driver_from_env()), V3StreamDriver)


def test_env_override_selects_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STREAM_DRIVER_ENV_VAR, "v2")
    assert stream_driver_from_env() == "v2"
    assert isinstance(make_stream_driver("v2"), V2StreamDriver)


def test_invalid_env_value_is_a_clear_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STREAM_DRIVER_ENV_VAR, "v4")
    with pytest.raises(ValueError, match=STREAM_DRIVER_ENV_VAR) as excinfo:
        stream_driver_from_env()
    # The message names the valid values.
    assert "v2" in str(excinfo.value)
    assert "v3" in str(excinfo.value)


def test_invalid_env_value_raises_at_run_start_not_import() -> None:
    """Importing the module never reads the environment (doctor --offline safe)."""
    import vdreamer.streaming  # noqa: F401  (import must not raise)

    assert VALID_STREAM_DRIVERS == ("v2", "v3")


def test_make_stream_driver_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="stream driver"):
        make_stream_driver("v9")


def test_explicit_env_mapping_beats_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STREAM_DRIVER_ENV_VAR, "v3")
    assert stream_driver_from_env({"VDREAMER_STREAM_DRIVER": "v2"}) == "v2"
