"""``.env`` loading: the parser, precedence, and the CLI wiring."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vdreamer.dotenv import EnvFileError, load_env_file, parse_env_file


def _write(tmp_path: Path, text: str, name: str = ".env") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_env_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "export EXPORTED=yes\n"
        "SPACED = padded value   \n"
        "INLINE=value # trailing comment\n"
        "HASH=abc#def\n"
        "SINGLE='keep $HOME and \\n as is'\n"
        'DOUBLE="line1\\nline2 \\"quoted\\""\n'
        "EMPTY=\n"
        "PLAIN=later wins\n",
    )
    assert parse_env_file(path) == {
        "PLAIN": "later wins",
        "EXPORTED": "yes",
        "SPACED": "padded value",
        "INLINE": "value",
        "HASH": "abc#def",
        "SINGLE": "keep $HOME and \\n as is",
        "DOUBLE": 'line1\nline2 "quoted"',
        "EMPTY": "",
    }


@pytest.mark.parametrize("line", ["NOT A PAIR", "1BAD=x", 'OPEN="never closed', "OPEN='x"])
def test_malformed_lines_name_the_line(tmp_path: Path, line: str) -> None:
    path = _write(tmp_path, f"OK=1\n{line}\n")
    with pytest.raises(EnvFileError, match=r"\.env:2"):
        parse_env_file(path)


def test_the_environment_wins_over_the_file(tmp_path: Path) -> None:
    _write(tmp_path, "TOKEN=from-file\nNEW=from-file\n")
    env = {"TOKEN": "from-shell"}
    loaded = load_env_file(environ=env, cwd=tmp_path)
    assert loaded is not None and loaded.applied == ["NEW"]
    assert env == {"TOKEN": "from-shell", "NEW": "from-file"}


def test_no_default_file_is_fine_but_an_explicit_one_must_exist(tmp_path: Path) -> None:
    assert load_env_file(environ={}, cwd=tmp_path) is None
    with pytest.raises(FileNotFoundError):
        load_env_file(tmp_path / "missing.env", environ={})


def test_the_cli_reads_dotenv_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from vdreamer.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VDREAMER_TEST_DOTENV", raising=False)
    _write(tmp_path, "VDREAMER_TEST_DOTENV=from-dotenv\nLANGSMITH_TRACING=true\n")
    assert main(["doctor", "--offline"]) == 0
    assert os.environ["VDREAMER_TEST_DOTENV"] == "from-dotenv"
    # Zero egress is forced after the file: a .env cannot turn tracing back on.
    assert os.environ["LANGSMITH_TRACING"] == "false"
    err = capsys.readouterr().err
    assert "VDREAMER_TEST_DOTENV" in err and "from-dotenv" not in err  # names, never values
    monkeypatch.delenv("VDREAMER_TEST_DOTENV")


@pytest.mark.parametrize("position", ["before", "after"])
def test_env_file_flag_before_or_after_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, position: str
) -> None:
    from vdreamer.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VDREAMER_TEST_ENVFILE", raising=False)
    path = _write(tmp_path, "VDREAMER_TEST_ENVFILE=1\n", name="local.env")
    flag = ["--env-file", str(path)]
    before = [*flag, "doctor", "--offline"]
    argv = before if position == "before" else ["doctor", *flag, "--offline"]
    assert main(argv) == 0
    assert os.environ["VDREAMER_TEST_ENVFILE"] == "1"
    monkeypatch.delenv("VDREAMER_TEST_ENVFILE")


def test_a_missing_env_file_stops_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from vdreamer.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["--env-file", "nope.env", "doctor", "--offline"]) == 1
    assert "env file not found" in capsys.readouterr().err
