"""CLI entry-point tests."""

import subprocess
import sys
from importlib import metadata as importlib_metadata

import vdreamer


def test_version_flag_prints_package_version() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "vdreamer", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"vdreamer {vdreamer.__version__}"


def test_console_script_is_registered() -> None:
    # The `vdreamer` console script is the product's single entry point;
    # the uv-independence CI job runs it from a plain `python -m venv` install.
    entry_points = importlib_metadata.entry_points(group="console_scripts")
    assert any(ep.name == "vdreamer" and ep.value.startswith("vdreamer") for ep in entry_points)
