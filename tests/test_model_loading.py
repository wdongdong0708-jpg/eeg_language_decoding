from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from features.audio_features import AudioFeatureConfig, Wav2VecFrameExtractor
from features.model_loading import (
    STRICT_OFFLINE_ENVIRONMENT,
    resolve_model_source,
)
from features.text_features import TextEmbeddingExtractor, TextFeatureConfig


def test_online_mode_keeps_repo_id_and_does_not_force_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in STRICT_OFFLINE_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)
    assert (
        resolve_model_source(
            "organization/model",
            local_files_only=False,
        )
        == "organization/model"
    )
    for variable in STRICT_OFFLINE_ENVIRONMENT:
        assert variable not in os.environ


def test_local_directory_enables_strict_offline_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for variable in STRICT_OFFLINE_ENVIRONMENT:
        monkeypatch.delenv(variable, raising=False)

    resolved = resolve_model_source(
        str(model_dir),
        local_files_only=True,
    )

    assert resolved == str(model_dir.resolve())
    for variable, value in STRICT_OFFLINE_ENVIRONMENT.items():
        assert os.environ[variable] == value


def test_repo_id_is_resolved_without_network_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[tuple[str, bool]] = []

    def fake_snapshot_download(
        *,
        repo_id: str,
        local_files_only: bool,
    ) -> str:
        calls.append((repo_id, local_files_only))
        return str(snapshot)

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        fake_snapshot_download,
    )
    resolved = resolve_model_source(
        "organization/model",
        local_files_only=True,
    )
    assert resolved == str(snapshot.resolve())
    assert calls == [("organization/model", True)]


def test_missing_cached_snapshot_fails_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    def missing_snapshot(**_: object) -> str:
        raise RuntimeError("not cached")

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        missing_snapshot,
    )
    with pytest.raises(FileNotFoundError, match="strict offline mode"):
        resolve_model_source(
            "organization/missing-model",
            local_files_only=True,
        )


class _FakeModel:
    def to(self, _: str) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self


class _FakeTokenizer:
    is_fast = True


def test_text_extractor_loads_resolved_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str, **_: object) -> _FakeTokenizer:
            calls.append(("tokenizer", path))
            return _FakeTokenizer()

    class AutoModel:
        @classmethod
        def from_pretrained(cls, path: str, **_: object) -> _FakeModel:
            calls.append(("model", path))
            return _FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=AutoModel, AutoTokenizer=AutoTokenizer),
    )
    monkeypatch.setattr(
        "features.text_features.resolve_model_source",
        lambda *_args, **_kwargs: "C:/cache/text-snapshot",
    )
    extractor = TextEmbeddingExtractor.from_pretrained(
        TextFeatureConfig(),
        device="cuda",
        local_files_only=True,
    )
    assert extractor.resolved_model_path == "C:/cache/text-snapshot"
    assert calls == [
        ("tokenizer", "C:/cache/text-snapshot"),
        ("model", "C:/cache/text-snapshot"),
    ]


def test_audio_extractor_loads_resolved_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, path: str, **_: object) -> object:
            calls.append(("processor", path))
            return object()

    class AutoModel:
        @classmethod
        def from_pretrained(cls, path: str, **_: object) -> _FakeModel:
            calls.append(("model", path))
            return _FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=AutoModel, AutoProcessor=AutoProcessor),
    )
    monkeypatch.setattr(
        "features.audio_features.resolve_model_source",
        lambda *_args, **_kwargs: "C:/cache/audio-snapshot",
    )
    extractor = Wav2VecFrameExtractor.from_pretrained(
        AudioFeatureConfig(),
        device="cuda",
        local_files_only=True,
    )
    assert extractor.resolved_model_path == "C:/cache/audio-snapshot"
    assert calls == [
        ("processor", "C:/cache/audio-snapshot"),
        ("model", "C:/cache/audio-snapshot"),
    ]
