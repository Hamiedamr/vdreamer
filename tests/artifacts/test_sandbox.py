"""Milestone 6 exit: the sandbox runs untrusted code with its rlimits verified (D19/A19).

POSIX only — skipped on Windows dev hosts. On Linux CI (or in a container) the
tests run as is; run as root with ``VDREAMER_TEST_SANDBOX_UID`` set to an
unprivileged uid and the dedicated-uid assertions run too.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from _tools_kit import make_ctx

from vdreamer.artifacts import InMemoryArtifactStore, bind_artifact_store
from vdreamer.context import bind_context
from vdreamer.sandbox import (
    SANDBOX_ROOT_ENV,
    SANDBOX_UID_ENV,
    Sandbox,
    SandboxLimits,
    SandboxResult,
    SandboxUnavailable,
)

posix = pytest.mark.skipif(sys.platform == "win32", reason="the sandbox is POSIX-only (D19)")
DEDICATED_UID = os.environ.get("VDREAMER_TEST_SANDBOX_UID", "")


@pytest.fixture(autouse=True)
def _sandbox_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Directly under the system temp dir: pytest's own tmp dirs are 0700, which a
    # dedicated sandbox uid could not traverse.
    root = Path(tempfile.mkdtemp(prefix="vdreamer-sandbox-test-"))
    root.chmod(0o755)
    monkeypatch.setenv(SANDBOX_ROOT_ENV, str(root))
    if DEDICATED_UID:
        monkeypatch.setenv(SANDBOX_UID_ENV, DEDICATED_UID)
    else:
        monkeypatch.delenv(SANDBOX_UID_ENV, raising=False)
    monkeypatch.setenv("VDREAMER_SECRET_FOR_TEST", "must-not-leak")
    yield
    shutil.rmtree(root, ignore_errors=True)


async def _run(code: str, **kwargs: Any) -> tuple[SandboxResult, InMemoryArtifactStore]:
    store = InMemoryArtifactStore()
    ctx = make_ctx()
    with bind_context(ctx), bind_artifact_store(store):
        result = await Sandbox(ctx).arun(code, **kwargs)
    return result, store


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only refusal")
async def test_windows_hosts_refuse() -> None:
    with pytest.raises(SandboxUnavailable, match="POSIX"):
        await _run("print(1)")


@posix
async def test_runs_code_with_inputs_and_an_empty_env() -> None:
    code = (
        "import os, json\n"
        "print(open('data.csv').read().strip())\n"
        "print(json.dumps(sorted(os.environ)))\n"
        "print(os.getcwd())\n"
    )
    result, _ = await _run(code, inputs={"data.csv": "a,b\n1,2\n"})
    assert result.ok, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["a,b", "1,2"]
    env = __import__("json").loads(lines[-2])
    assert "VDREAMER_SECRET_FOR_TEST" not in env and "PATH" not in env
    assert set(env) <= {
        "HOME", "TMPDIR", "MPLCONFIGDIR", "MPLBACKEND", "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
        "PYTHONIOENCODING", "LANG", "LC_CTYPE",
    }
    scratch = Path(lines[-1])
    assert not scratch.exists(), "the scratch dir is wiped"


@posix
async def test_rlimits_are_applied() -> None:
    code = (
        "import resource\n"
        "for name in ['CPU', 'AS', 'FSIZE', 'NOFILE', 'CORE', 'NPROC']:\n"
        "    limit = resource.getrlimit(getattr(resource, 'RLIMIT_' + name))[0]\n"
        "    print('RLIMIT_' + name, limit)\n"
    )
    limits = SandboxLimits(memory_bytes=1536 * 1024 * 1024, file_size_bytes=8 * 1024 * 1024)
    result, _ = await _run(code, timeout_s=10, limits=limits)
    assert result.ok, result.stderr
    got = dict(line.split() for line in result.stdout.splitlines())
    assert int(got["RLIMIT_CPU"]) == 12
    assert int(got["RLIMIT_AS"]) == 1536 * 1024 * 1024
    assert int(got["RLIMIT_FSIZE"]) == 8 * 1024 * 1024
    assert int(got["RLIMIT_NOFILE"]) == 256
    assert int(got["RLIMIT_CORE"]) == 0
    if DEDICATED_UID:
        assert int(got["RLIMIT_NPROC"]) == 64


@posix
async def test_the_memory_cap_stops_a_hog() -> None:
    code = "x = bytearray(3 * 1024 * 1024 * 1024)\nprint('allocated')\n"
    result, _ = await _run(code, timeout_s=20)
    assert not result.ok and "allocated" not in result.stdout
    assert "MemoryError" in result.stderr


@posix
async def test_the_file_size_cap_stops_a_writer() -> None:
    code = "open('out/big.bin','wb').write(b'x' * (4 * 1024 * 1024))\nprint('wrote')\n"
    result, store = await _run(code, limits=SandboxLimits(file_size_bytes=1024 * 1024))
    assert not result.ok and "wrote" not in result.stdout


@posix
async def test_a_spinning_child_is_killed_at_the_timeout() -> None:
    code = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'while True: pass'])\n"
        "while True: pass\n"
    )
    result, _ = await _run(code, timeout_s=2)
    assert result.timed_out and result.exit_code is None
    assert result.duration_ms < 10_000


@posix
async def test_out_files_become_artifacts_but_links_do_not() -> None:
    code = (
        "import json, os\n"
        "open('out/chart.svg','w').write('<svg width=\"4\" height=\"3\"></svg>')\n"
        "open('out/report.html','w').write('<title>R</title>')\n"
        "json.dump({'ok': True}, open('out/verdict.json','w'))\n"
        "open('out/rows.csv','w').write('a,b\\n1,2\\n3,4\\n')\n"
        "open('out/notes.bin','wb').write(b'\\x00\\x01')\n"
        "os.symlink('/etc/passwd', 'out/passwd')\n"
        "open('secret.txt','w').write('s')\n"
        "os.link('secret.txt', 'out/hard.txt')\n"
    )
    result, store = await _run(code)
    assert result.ok, result.stderr
    kinds = {ref.name: ref.kind.value for ref in result.artifacts}
    assert kinds == {
        "chart.svg": "image",
        "report.html": "html",
        "verdict.json": "json",
        "rows.csv": "table",
        "notes.bin": "file",
    }
    assert sorted(result.skipped_outputs) == ["hard.txt", "passwd"]
    table = next(ref for ref in result.artifacts if ref.name == "rows.csv")
    record = await store.get(table.artifact_id)
    assert record is not None and record.preview["row_count"] == 2


@posix
async def test_network_is_off_where_namespaces_permit() -> None:
    code = (
        "import socket\n"
        "s = socket.socket()\n"
        "s.settimeout(2)\n"
        "try:\n"
        "    s.connect(('1.1.1.1', 53)); print('connected')\n"
        "except OSError as e:\n"
        "    print('blocked', e.errno)\n"
    )
    result, _ = await _run(code, timeout_s=10)
    assert result.ok, result.stderr
    if result.network_isolated:
        assert result.stdout.startswith("blocked")
    else:
        pytest.skip("this kernel/container does not allow a new network namespace")


@posix
@pytest.mark.skipif(not DEDICATED_UID, reason="VDREAMER_TEST_SANDBOX_UID not set")
async def test_the_dedicated_uid_cannot_read_the_workers_files(tmp_path: Path) -> None:
    secret = tmp_path / "worker-secret"
    secret.write_text("top")
    secret.chmod(0o600)
    code = f"import os\nprint(os.getuid())\nprint(open({str(secret)!r}).read())\n"
    result, _ = await _run(code)
    assert result.dedicated_uid and result.uid == int(DEDICATED_UID)
    assert result.stdout.splitlines()[0] == DEDICATED_UID
    assert "PermissionError" in result.stderr
