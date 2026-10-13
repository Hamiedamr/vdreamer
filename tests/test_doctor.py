"""``doctor --offline`` and egress-guard tests."""

import os
import socket
from collections.abc import Iterator

import pytest

from vdreamer.doctor import OFFLINE_ENV, EgressGuard, run_offline


@pytest.fixture
def _clean_offline_env() -> Iterator[None]:
    """Snapshot the env vars doctor mutates, so tests stay isolated."""
    saved = {key: os.environ.get(key) for key in OFFLINE_ENV}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_egress_guard_blocks_and_records() -> None:
    guard = EgressGuard()
    original = socket.create_connection
    guard.install()
    try:
        raised = False
        try:
            socket.create_connection(("example.invalid", 443), timeout=30)
        except OSError:
            raised = True
        assert raised, "guard must block socket.create_connection"
        assert len(guard.violations) == 1
        assert "socket.create_connection" in guard.violations[0]
    finally:
        guard.uninstall()
    assert socket.create_connection is original


def test_guard_blocks_raw_socket_connect() -> None:
    guard = EgressGuard()
    guard.install()
    try:
        raised = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.connect(("10.255.255.1", 65000))
            except OSError:
                raised = True
        assert raised
        assert any("socket.socket.connect" in violation for violation in guard.violations)
    finally:
        guard.uninstall()


@pytest.mark.usefixtures("_clean_offline_env")
def test_doctor_offline_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_offline() == 0
    out = capsys.readouterr().out
    assert "doctor: PASS" in out
    assert "zero outbound socket calls" in out
