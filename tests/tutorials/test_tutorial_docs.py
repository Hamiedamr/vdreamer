"""The tutorial READMEs: complete, linked, and their code excerpts match the code.

Each lesson README has the same sections in the same order. A fenced ``python``
block whose first line is ``# file: <path>`` is an *excerpt*: the rest of the
block must appear verbatim in that file (lines ``...`` or ``# ...`` elide code
between parts, which must appear in order). Paths starting with ``tests/`` or
``tutorials/`` are relative to the repository, others to the lesson directory.
So every snippet a reader copies is code the test suite executed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _tutorial_kit import LESSONS, REPO, TUTORIALS

SECTIONS = [
    "## Learning objectives",
    "## Prerequisites",
    "## Concepts",
    "## What comes from where",
    "## How it works",
    "## Step by step",
    "## The complete example",
    "## Run it",
    "## Expected behavior",
    "## Test it",
    "## Common mistakes and debugging",
    "## Design trade-offs",
    "## Exercises",
]

LESSON_DIRS = sorted(p for p in LESSONS.iterdir() if p.is_dir() and re.match(r"t\d\d_", p.name))
FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def _readme(lesson: Path) -> str:
    return (lesson / "README.md").read_text(encoding="utf-8")


def test_lessons_are_numbered_without_gaps() -> None:
    numbers = [int(p.name[1:3]) for p in LESSON_DIRS]
    assert numbers == list(range(1, len(numbers) + 1)), numbers


@pytest.mark.parametrize("lesson", LESSON_DIRS, ids=lambda p: p.name)
def test_each_readme_has_every_section_in_order(lesson: Path) -> None:
    text = _readme(lesson)
    assert text.startswith(f"# Lesson {int(lesson.name[1:3])} — "), text[:80]
    positions = [text.find(f"\n{heading}\n") for heading in SECTIONS]
    missing = [h for h, pos in zip(SECTIONS, positions, strict=True) if pos < 0]
    assert missing == []
    assert positions == sorted(positions), "sections out of order"


def _excerpts(lesson: Path) -> list[tuple[Path, list[str]]]:
    found = []
    for language, body in FENCE.findall(_readme(lesson)):
        lines = body.splitlines()
        if language != "python" or not lines or not lines[0].startswith("# file: "):
            continue
        target = lines[0].removeprefix("# file: ").strip()
        root = REPO if target.startswith(("tests/", "tutorials/")) else lesson
        parts: list[list[str]] = [[]]
        for line in lines[1:]:
            if line.strip() in ("...", "# ..."):
                parts.append([])
            else:
                parts[-1].append(line)
        chunks = ["\n".join(part).strip("\n") for part in parts]
        found.append((root / target, [chunk for chunk in chunks if chunk.strip()]))
    return found


@pytest.mark.parametrize("lesson", LESSON_DIRS, ids=lambda p: p.name)
def test_code_excerpts_match_the_source(lesson: Path) -> None:
    for path, chunks in _excerpts(lesson):
        assert path.is_file(), f"{lesson.name}: excerpt names a missing file {path}"
        source = path.read_text(encoding="utf-8")
        position = 0
        for chunk in chunks:
            found = source.find(chunk, position)
            assert found >= 0, f"{lesson.name}: not in {path.name} (in order):\n{chunk}"
            position = found + len(chunk)


@pytest.mark.parametrize("lesson", LESSON_DIRS, ids=lambda p: p.name)
def test_relative_links_resolve(lesson: Path) -> None:
    for target in LINK.findall(_readme(lesson)):
        if "://" in target:
            continue
        assert (lesson / target).resolve().exists(), f"{lesson.name}: broken link {target}"


def test_the_index_lists_every_lesson_in_order() -> None:
    index = (TUTORIALS / "README.md").read_text(encoding="utf-8")
    links = [t for t in LINK.findall(index) if t.startswith("src/vdreamer_tutorials/t")]
    lessons = [t.split("/")[2] for t in links if t.endswith("README.md")]
    assert lessons == [p.name for p in LESSON_DIRS]
    for target in LINK.findall(index):
        if "://" not in target:
            assert (TUTORIALS / target).resolve().exists(), f"index: broken link {target}"
