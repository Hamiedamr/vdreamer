"""fastembed aliases: native objects, vendored cache only, CPU provider (A36, §5.2).

The offline tests swap the fastembed classes for recorders (no model files
ship with the repo). The real classes run once against an empty cache to
prove "missing model files" fails without touching the network.
``VDREAMER_TEST_FASTEMBED_CACHE`` (a cache holding ``BAAI/bge-small-en-v1.5``,
``Qdrant/bm25`` and ``Xenova/ms-marco-MiniLM-L-6-v2``) runs the live load.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from vdreamer import models as models_module
from vdreamer.doctor import EgressGuard
from vdreamer.models import (
    FastembedModelConfig,
    FastembedSettings,
    ModelUnavailable,
    build_models,
    check_fastembed_files,
    load_fastembed,
    parse_models,
)


class Recorder:
    calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        Recorder.calls.append((type(self).__name__, kwargs))


class TextEmbedding(Recorder):
    pass


class SparseTextEmbedding(Recorder):
    pass


class TextCrossEncoder(Recorder):
    pass


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    Recorder.calls = []
    classes = {"embedding": TextEmbedding, "sparse": SparseTextEmbedding,
               "rerank": TextCrossEncoder}
    monkeypatch.setattr(models_module, "_fastembed_class", lambda kind: classes[kind])
    return Recorder.calls


CONFIG = parse_models(
    {
        "aliases": {
            "dense": {"provider": "fastembed", "kind": "embedding",
                      "model": "BAAI/bge-small-en-v1.5"},
            "sparse": {"provider": "fastembed", "kind": "sparse", "model": "Qdrant/bm25"},
            "rerank": {"provider": "fastembed", "kind": "rerank",
                       "model": "Xenova/ms-marco-MiniLM-L-6-v2"},
        },
        "fastembed": {"cache_dir": "/opt/models/fastembed", "threads": 2},
    }
)


def test_each_kind_is_its_native_class(fake_fastembed: list[Any]) -> None:
    built = build_models(CONFIG, ["dense", "sparse", "rerank"])
    assert [type(built[a]).__name__ for a in ("dense", "sparse", "rerank")] == [
        "TextEmbedding",
        "SparseTextEmbedding",
        "TextCrossEncoder",
    ]
    for _name, kwargs in fake_fastembed:
        assert kwargs["cache_dir"] == "/opt/models/fastembed"
        assert kwargs["local_files_only"] is True
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        assert kwargs["threads"] == 2
        assert kwargs["lazy_load"] is False
    assert [kw["model_name"] for _n, kw in fake_fastembed] == [
        "BAAI/bge-small-en-v1.5",
        "Qdrant/bm25",
        "Xenova/ms-marco-MiniLM-L-6-v2",
    ]


def test_the_file_check_locates_without_loading(fake_fastembed: list[Any]) -> None:
    config = CONFIG.aliases["dense"]
    assert isinstance(config, FastembedModelConfig) and CONFIG.fastembed is not None
    assert "in /opt/models/fastembed" in check_fastembed_files(config, CONFIG.fastembed)
    assert fake_fastembed[-1][1]["lazy_load"] is True


def test_loading_forces_the_hub_offline(
    fake_fastembed: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    models_module._force_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_missing_model_files_fail_without_egress(tmp_path: Path) -> None:
    pytest.importorskip("fastembed")
    config = FastembedModelConfig(alias="dense", kind="embedding", model="BAAI/bge-small-en-v1.5")
    settings = FastembedSettings(cache_dir=str(tmp_path))
    guard = EgressGuard()
    guard.install()
    try:
        with pytest.raises(ModelUnavailable, match="cannot load embedding model"):
            load_fastembed(config, settings, lazy=True)
    finally:
        guard.uninstall()
    assert guard.violations == []


def test_an_unknown_model_name_fails(tmp_path: Path) -> None:
    pytest.importorskip("fastembed")
    config = FastembedModelConfig(alias="dense", kind="embedding", model="no/such-model")
    with pytest.raises(ModelUnavailable, match="no/such-model"):
        load_fastembed(config, FastembedSettings(cache_dir=str(tmp_path)), lazy=True)


@pytest.mark.skipif(
    not os.environ.get("VDREAMER_TEST_FASTEMBED_CACHE"),
    reason="set VDREAMER_TEST_FASTEMBED_CACHE to a cache holding the three test models",
)
def test_live_models_load_and_run() -> None:
    cache = os.environ["VDREAMER_TEST_FASTEMBED_CACHE"]
    config = parse_models(
        {
            "aliases": {
                alias: entry
                for alias, entry in (
                    ("dense", {"provider": "fastembed", "kind": "embedding",
                               "model": "BAAI/bge-small-en-v1.5"}),
                    ("sparse", {"provider": "fastembed", "kind": "sparse", "model": "Qdrant/bm25"}),
                    ("rerank", {"provider": "fastembed", "kind": "rerank",
                                "model": "Xenova/ms-marco-MiniLM-L-6-v2"}),
                )
            },
            "fastembed": {"cache_dir": cache},
        }
    )
    guard = EgressGuard()
    guard.install()
    try:
        built = build_models(config, ["dense", "sparse", "rerank"])
        vector = next(iter(built["dense"].embed(["hello"])))
        sparse = next(iter(built["sparse"].embed(["hello world"])))
        scores = list(built["rerank"].rerank("what is 4?", ["four", "a cat"]))
    finally:
        guard.uninstall()
    assert guard.violations == []
    assert len(vector) == 384
    assert len(sparse.indices) > 0
    assert scores[0] > scores[1]
