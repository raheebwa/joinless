# SPDX-License-Identifier: MIT
"""The fp32 embedding arm: pooling, similarity, batching and fail-closed behaviour.

Every test here runs against a tokenizer/session double, never the real 90 MB artefact
(see ``joinless/embedding.py``'s module docstring for why that needs no NumPy and no real
``onnxruntime``/``tokenizers`` package at all for the arm's own pooling and batching
logic). Where a test does need to prove something about the *real* ``onnxruntime`` and
``tokenizers`` packages' import shape — the fail-closed checksum path and the no-network
property — it exercises the real ``load_fp32_scorer``/``probe_fp32`` code with a small,
real, on-disk fixture file standing in for the 90 MB production artefact, its checksum
patched onto the module's own constant, rather than a double for the checksum
verification itself (ADR-0016 rule 3: a real file and a real hash, not a stand-in for
either).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from joinless import embedding
from joinless.embedding import (
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    CacheDirNotSetError,
    EmbeddingScorer,
)


def test_model_identity_is_recorded_alongside_the_artefact() -> None:
    """Issue #59: "the model card's licence is recorded alongside the artefact
    identity." Pinned as literals matching
    ``benchmarks/20260812T181752Z-quantization-spike.json`` - the record this arm's
    artefact was produced against - so a change to any of the three is a visible diff
    here, not a silent drift between what is recorded and what this module assumes."""
    assert MODEL_ID == "sentence-transformers/all-MiniLM-L6-v2"
    assert MODEL_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert MODEL_LICENSE == "apache-2.0"


# --- resolve_model_paths / CacheDirNotSetError --------------------------------------


def test_resolve_model_paths_raises_naming_the_env_var_and_the_setup_command_when_unset() -> (
    None
):
    with pytest.raises(CacheDirNotSetError) as excinfo:
        embedding.resolve_model_paths({})

    message = str(excinfo.value)
    assert "JOINLESS_MODEL_CACHE_DIR" in message
    assert "spikes/quantization" in message


def test_resolve_model_paths_raises_when_the_env_var_is_set_but_empty() -> None:
    with pytest.raises(CacheDirNotSetError):
        embedding.resolve_model_paths({"JOINLESS_MODEL_CACHE_DIR": ""})


def test_resolve_model_paths_returns_the_fp32_model_and_tokenizer_paths_under_the_cache_dir() -> (
    None
):
    model_path, tokenizer_path = embedding.resolve_model_paths(
        {"JOINLESS_MODEL_CACHE_DIR": "/some/cache"}
    )

    assert model_path == Path("/some/cache/fp32/model.onnx")
    assert tokenizer_path == Path("/some/cache/fp32/tokenizer.json")


# --- artifact_requirements_fp32: exposed for artefact-size measurement (issue #63) --


def test_artifact_requirements_fp32_names_the_model_and_tokenizer_paths_and_checksums() -> (
    None
):
    requirements = embedding.artifact_requirements_fp32(
        {"JOINLESS_MODEL_CACHE_DIR": "/some/cache"}
    )

    model_requirement, tokenizer_requirement = requirements
    assert model_requirement.path == Path("/some/cache/fp32/model.onnx")
    assert model_requirement.sha256 == embedding.FP32_MODEL_SHA256
    assert tokenizer_requirement.path == Path("/some/cache/fp32/tokenizer.json")
    assert tokenizer_requirement.sha256 == embedding.FP32_TOKENIZER_SHA256


def test_model_identity_fp32_carries_the_pinned_identity_checksum_and_licence() -> None:
    """Issue #59: model identity, revision, checksum and licence belong in the
    one place already pinning them as literals (module docstring), so a
    caller wanting to record what this arm loaded reads them from here rather
    than re-deriving them somewhere they could drift out of step with
    :func:`artifact_requirements_fp32`, which verifies the same
    ``FP32_MODEL_SHA256`` this returns."""
    identity = embedding.model_identity_fp32()

    assert identity.model_id == embedding.MODEL_ID
    assert identity.revision == embedding.MODEL_REVISION
    assert identity.checksum_sha256 == embedding.FP32_MODEL_SHA256
    assert identity.license == embedding.MODEL_LICENSE


def test_probe_fp32_verifies_the_same_requirements_artifact_requirements_fp32_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``probe_fp32`` must not re-derive path/checksum pairing a second way
    (issue #63's DRY bullet, applied here to artefact identity rather than
    checksum verification itself) - proven by breaking only the checksum
    ``artifact_requirements_fp32`` names and watching ``probe_fp32`` fail on
    exactly that file."""
    _fake_dependencies_present(monkeypatch)
    _write_fixture(tmp_path / "fp32" / "model.onnx", b"a fake fp32 graph")
    _write_fixture(tmp_path / "fp32" / "tokenizer.json", b"a fake tokenizer config")
    monkeypatch.setattr(embedding, "FP32_MODEL_SHA256", "0" * 64)

    reason = embedding.probe_fp32({"JOINLESS_MODEL_CACHE_DIR": str(tmp_path)})

    assert reason is not None
    assert "model.onnx" in reason


# --- probe_fp32: dependency, configuration and artefact checks, cheapest-first ------


def test_probe_fp32_reports_a_missing_onnxruntime_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    reason = embedding.probe_fp32({})

    assert reason is not None
    assert "onnxruntime" in reason
    assert "joinless[neural]" in reason


def test_probe_fp32_reports_a_missing_tokenizers_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forces onnxruntime present via a fake module, exactly as it forces tokenizers
    absent via ``sys.modules["tokenizers"] = None`` - both directions patched, so this
    test's outcome does not depend on which install profile happens to be running it."""
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "tokenizers", None)

    reason = embedding.probe_fp32({})

    assert reason is not None
    assert "tokenizers" in reason
    assert "joinless[neural]" in reason


def _fake_dependencies_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force both ``onnxruntime`` and ``tokenizers`` to appear importable, without
    needing either real package - so every check downstream of dependency
    importability can be exercised regardless of install profile."""
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "tokenizers", types.ModuleType("tokenizers"))


def test_probe_fp32_reports_the_cache_dir_not_set_reason_once_dependencies_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_dependencies_present(monkeypatch)

    reason = embedding.probe_fp32({})

    assert reason is not None
    assert "JOINLESS_MODEL_CACHE_DIR" in reason


def test_probe_fp32_reports_a_missing_model_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_dependencies_present(monkeypatch)

    reason = embedding.probe_fp32({"JOINLESS_MODEL_CACHE_DIR": str(tmp_path)})

    assert reason is not None
    assert "artifact missing" in reason
    assert str(tmp_path / "fp32" / "model.onnx") in reason


def _write_fixture(path: Path, content: bytes) -> str:
    """Write ``content`` to ``path`` and return its sha256 hex digest - the digest a
    test then patches onto the module's own expected-checksum constant, so the real
    ``verify_artifact`` path is exercised against a real, small file rather than either
    a mocked hash function or the 90 MB production artefact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_probe_fp32_reports_a_model_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_dependencies_present(monkeypatch)
    _write_fixture(tmp_path / "fp32" / "model.onnx", b"not the real graph")
    # FP32_MODEL_SHA256 is left at its real, pinned value - which cannot match
    # arbitrary fixture bytes - so this is a genuine mismatch, not a scripted one.

    reason = embedding.probe_fp32({"JOINLESS_MODEL_CACHE_DIR": str(tmp_path)})

    assert reason is not None
    assert "checksum mismatch" in reason


def test_probe_fp32_reports_a_missing_tokenizer_artifact_once_the_model_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_dependencies_present(monkeypatch)
    model_digest = _write_fixture(
        tmp_path / "fp32" / "model.onnx", b"a fake fp32 graph"
    )
    monkeypatch.setattr(embedding, "FP32_MODEL_SHA256", model_digest)

    reason = embedding.probe_fp32({"JOINLESS_MODEL_CACHE_DIR": str(tmp_path)})

    assert reason is not None
    assert "artifact missing" in reason
    assert str(tmp_path / "fp32" / "tokenizer.json") in reason


def test_probe_fp32_returns_none_when_every_check_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_dependencies_present(monkeypatch)
    model_digest = _write_fixture(
        tmp_path / "fp32" / "model.onnx", b"a fake fp32 graph"
    )
    tokenizer_digest = _write_fixture(
        tmp_path / "fp32" / "tokenizer.json", b"a fake tokenizer config"
    )
    monkeypatch.setattr(embedding, "FP32_MODEL_SHA256", model_digest)
    monkeypatch.setattr(embedding, "FP32_TOKENIZER_SHA256", tokenizer_digest)

    reason = embedding.probe_fp32({"JOINLESS_MODEL_CACHE_DIR": str(tmp_path)})

    assert reason is None


# --- load_fp32_scorer: real construction path, fake tokenizer/session --------------


def _build_fake_tokenizers_module() -> tuple[types.ModuleType, list[str], list[Any]]:
    """A fake ``tokenizers`` module: ``Tokenizer.from_file`` records the path it was
    given and returns a fresh, independently-configurable fake tokenizer each call -
    standing in for the real package everywhere ``load_fp32_scorer`` touches it
    (``from tokenizers import Tokenizer``, ``Tokenizer.from_file``,
    ``enable_padding``, ``enable_truncation``)."""
    from_file_paths: list[str] = []
    created: list[Any] = []

    class _FakeTokenizer:
        def __init__(self) -> None:
            self.padding: dict[str, object] | None = None
            self.truncation: dict[str, object] | None = None

        def enable_padding(self, *, pad_token: str, pad_id: int) -> None:
            self.padding = {"pad_token": pad_token, "pad_id": pad_id}

        def enable_truncation(self, *, max_length: int) -> None:
            self.truncation = {"max_length": max_length}

        def encode_batch(self, texts: Sequence[str]) -> list[_Encoding]:
            return [
                _Encoding(ids=[len(text) + 1], attention_mask=[1], type_ids=[0])
                for text in texts
            ]

    class _TokenizerNamespace:
        @staticmethod
        def from_file(path: str) -> _FakeTokenizer:
            from_file_paths.append(path)
            tokenizer = _FakeTokenizer()
            created.append(tokenizer)
            return tokenizer

    module = types.ModuleType("tokenizers")
    module.Tokenizer = _TokenizerNamespace  # type: ignore[attr-defined]
    return module, from_file_paths, created


def _build_fake_onnxruntime_module() -> tuple[
    types.ModuleType, list[dict[str, object]]
]:
    """A fake ``onnxruntime`` module: ``InferenceSession(path, providers=...)`` records
    what it was constructed with and returns a session whose ``run`` produces a
    trivial, well-shaped hidden state - enough for ``EmbeddingScorer`` to pool and
    normalise without erroring, not a claim about matching the real graph's output."""
    sessions_created: list[dict[str, object]] = []

    class _FakeSession:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            sessions_created.append({"path": path, "providers": providers})

        def run(
            self, output_names: list[str] | None, input_feed: Mapping[str, object]
        ) -> list[list[list[list[float]]]]:
            del output_names
            rows = cast(list[list[int]], input_feed["input_ids"])
            hidden = [[[1.0, 0.0] for _ in row] for row in rows]
            return [hidden]

    module = types.ModuleType("onnxruntime")
    module.InferenceSession = _FakeSession  # type: ignore[attr-defined]
    return module, sessions_created


def test_load_fp32_scorer_wires_a_real_tokenizer_and_session_from_the_resolved_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_onnxruntime, sessions_created = _build_fake_onnxruntime_module()
    fake_tokenizers, from_file_paths, tokenizers_created = (
        _build_fake_tokenizers_module()
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)

    fp32_dir = tmp_path / "fp32"
    model_digest = _write_fixture(fp32_dir / "model.onnx", b"a fake fp32 graph")
    tokenizer_digest = _write_fixture(
        fp32_dir / "tokenizer.json", b"a fake tokenizer config"
    )
    monkeypatch.setattr(embedding, "FP32_MODEL_SHA256", model_digest)
    monkeypatch.setattr(embedding, "FP32_TOKENIZER_SHA256", tokenizer_digest)

    scorer = embedding.load_fp32_scorer({"JOINLESS_MODEL_CACHE_DIR": str(tmp_path)})

    assert isinstance(scorer, EmbeddingScorer)
    assert scorer.name == "embed-fp32"
    assert from_file_paths == [str(fp32_dir / "tokenizer.json")]
    assert sessions_created == [
        {"path": str(fp32_dir / "model.onnx"), "providers": ["CPUExecutionProvider"]}
    ]
    # Pinned as literals, not the module's own private constants: a test that
    # compared against the same names the implementation uses could not see a
    # content change to either value.
    [tokenizer] = tokenizers_created
    assert tokenizer.padding == {"pad_token": "[PAD]", "pad_id": 0}
    assert tokenizer.truncation == {"max_length": 128}

    prepared = scorer.prepare_all(["hi", "hello"])
    assert len(prepared) == 2
    assert all(value is not None for value in prepared)


_NO_NETWORK_PROBE = """
import sys
import types

class _FakeEncoding:
    def __init__(self, ids, attention_mask, type_ids):
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids

class _FakeTokenizer:
    def enable_padding(self, *, pad_token, pad_id):
        pass
    def enable_truncation(self, *, max_length):
        pass
    def encode_batch(self, texts):
        return [_FakeEncoding(ids=[1], attention_mask=[1], type_ids=[0]) for _ in texts]

class _TokenizerNamespace:
    @staticmethod
    def from_file(path):
        return _FakeTokenizer()

fake_tokenizers = types.ModuleType("tokenizers")
fake_tokenizers.Tokenizer = _TokenizerNamespace
sys.modules["tokenizers"] = fake_tokenizers

class _FakeSession:
    def __init__(self, path, providers=None):
        pass
    def run(self, output_names, input_feed):
        rows = input_feed["input_ids"]
        return [[[[1.0, 0.0] for _ in row] for row in rows]]

fake_onnxruntime = types.ModuleType("onnxruntime")
fake_onnxruntime.InferenceSession = _FakeSession
sys.modules["onnxruntime"] = fake_onnxruntime

import hashlib
from pathlib import Path

fp32_dir = Path(sys.argv[1]) / "fp32"
fp32_dir.mkdir(parents=True, exist_ok=True)
model_bytes = b"a fake fp32 graph"
tokenizer_bytes = b"a fake tokenizer config"
(fp32_dir / "model.onnx").write_bytes(model_bytes)
(fp32_dir / "tokenizer.json").write_bytes(tokenizer_bytes)

import joinless.embedding as embedding
embedding.FP32_MODEL_SHA256 = hashlib.sha256(model_bytes).hexdigest()
embedding.FP32_TOKENIZER_SHA256 = hashlib.sha256(tokenizer_bytes).hexdigest()

scorer = embedding.load_fp32_scorer({"JOINLESS_MODEL_CACHE_DIR": sys.argv[1]})
scorer.prepare_all(["Acme Traders", "Acme Trading Co"])

offenders = sorted(
    m for m in sys.modules
    if m == "huggingface_hub" or m.startswith("huggingface_hub.")
    or m == "requests" or m.startswith("requests.")
    or m == "urllib3" or m.startswith("urllib3.")
)
print("\\n".join(offenders))
sys.exit(1 if offenders else 0)
"""


def test_loading_and_scoring_with_the_fp32_arm_never_touches_a_hub_or_http_client(
    tmp_path: Path,
) -> None:
    """Issue #59: "a test asserts no network call occurs on the match path."

    Structural, not empirical against the real 90 MB artefact: this module's own code
    only ever calls ``Tokenizer.from_file`` (never ``Tokenizer.from_pretrained``,
    ADR-0017's load-bearing clause), so a fake tokenizer/session pair that has no
    network client to reach for is enough to prove the *code path* this module takes
    never imports one - the same distinction ADR-0017's own consequences draw: the
    packages may be present in the environment, what matters is that nothing on this
    path imports or exercises them. A child interpreter is required for the same
    reason ``tests/test_import_boundary.py`` needs one: ``sys.modules`` is only
    meaningful about an import this process actually performed.
    """
    result = subprocess.run(
        [sys.executable, "-c", _NO_NETWORK_PROBE, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "loading or scoring with the fp32 embedding arm pulled in a hub/HTTP client "
        f"module: {result.stdout.strip() or result.stderr.strip()}"
    )


# --- EmbeddingScorer: pooling, similarity, batching, empty handling ----------------


class _Encoding:
    def __init__(
        self, ids: list[int], attention_mask: list[int], type_ids: list[int]
    ) -> None:
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids


class _StubTokenizer:
    """A minimal double for ``tokenizers.Tokenizer``: ``encode_batch`` maps each text
    through a caller-supplied function to one encoding, and records every batch it was
    asked to encode - the observation issue #61's batching claim needs a test to make,
    not merely infer from reading the code."""

    def __init__(
        self, encode: Callable[[str], tuple[list[int], list[int], list[int]]]
    ) -> None:
        self._encode = encode
        self.encode_batch_calls: list[list[str]] = []

    def encode_batch(self, texts: Sequence[str]) -> list[_Encoding]:
        batch = list(texts)
        self.encode_batch_calls.append(batch)
        return [_Encoding(*self._encode(text)) for text in batch]


class _StubSession:
    """A minimal double for ``onnxruntime.InferenceSession``: ``run`` maps each row's
    ids and mask through a caller-supplied function to that row's per-token
    embeddings, and records every call - the observation issue #61's "exactly once per
    run" claim needs."""

    def __init__(
        self, embed_row: Callable[[list[int], list[int]], list[list[float]]]
    ) -> None:
        self._embed_row = embed_row
        self.run_calls: list[dict[str, list[list[int]]]] = []

    def run(
        self, output_names: list[str] | None, input_feed: Mapping[str, object]
    ) -> list[list[list[list[float]]]]:
        del output_names
        feed = cast(dict[str, list[list[int]]], input_feed)
        self.run_calls.append(
            {key: [list(row) for row in value] for key, value in feed.items()}
        )
        hidden = [
            self._embed_row(ids, mask)
            for ids, mask in zip(feed["input_ids"], feed["attention_mask"], strict=True)
        ]
        return [hidden]


def _single_token_scorer(
    vectors: Mapping[str, Sequence[float]],
) -> tuple[EmbeddingScorer, _StubTokenizer, _StubSession]:
    """A scorer where every text named in ``vectors`` is a single non-padded token
    whose own embedding is exactly the vector given for it - so ``prepare(text)``
    resolves to the L2-normalised form of that vector directly, with no pooling
    arithmetic a test has to reason through by hand to predict it."""
    ids_by_text = {text: index for index, text in enumerate(vectors)}
    vector_by_id = {ids_by_text[text]: list(vector) for text, vector in vectors.items()}

    tokenizer = _StubTokenizer(lambda text: ([ids_by_text[text]], [1], [0]))
    session = _StubSession(lambda ids, mask: [vector_by_id[ids[0]]])
    scorer = EmbeddingScorer(name="embed-fp32", session=session, tokenizer=tokenizer)
    return scorer, tokenizer, session


def test_name_property_returns_whichever_name_the_scorer_was_constructed_with() -> None:
    """RFC-0001: "the fp32 and int8 arms are the same class with different model
    artefacts" - nothing about ``EmbeddingScorer`` itself should hardcode
    "embed-fp32", so a different name is handed in here specifically to prove that."""
    scorer = EmbeddingScorer(
        name="embed-int8",
        session=_StubSession(lambda ids, mask: [[1.0, 0.0] for _ in ids]),
        tokenizer=_StubTokenizer(lambda text: ([0], [1], [0])),
    )

    assert scorer.name == "embed-int8"


def test_prepare_returns_none_for_an_absent_name() -> None:
    scorer, _, session = _single_token_scorer({})

    assert scorer.prepare(None) is None
    assert session.run_calls == []


def test_prepare_returns_none_for_a_whitespace_only_name() -> None:
    scorer, _, session = _single_token_scorer({})

    assert scorer.prepare("   ") is None
    assert session.run_calls == []


def test_score_of_two_unnamed_prepared_values_is_zero() -> None:
    scorer, _, _ = _single_token_scorer({})

    assert scorer.score(None, None) == 0.0


def test_score_of_an_unnamed_and_a_named_prepared_value_is_zero_either_order() -> None:
    scorer, _, _ = _single_token_scorer({"Acme": [1.0, 0.0]})
    prepared = scorer.prepare("Acme")

    assert scorer.score(None, prepared) == 0.0
    assert scorer.score(prepared, None) == 0.0


def test_a_name_scored_against_itself_is_an_exact_match() -> None:
    scorer, _, _ = _single_token_scorer({"Acme": [1.0, 0.0]})
    prepared = scorer.prepare("Acme")

    assert scorer.score(prepared, prepared) == 1.0


def test_score_of_orthogonal_embeddings_is_the_midpoint() -> None:
    scorer, _, _ = _single_token_scorer({"Acme": [1.0, 0.0], "Beta": [0.0, 1.0]})

    score = scorer.score(scorer.prepare("Acme"), scorer.prepare("Beta"))

    assert score == 0.5


def test_score_of_opposite_embeddings_is_zero() -> None:
    scorer, _, _ = _single_token_scorer({"Acme": [1.0, 0.0], "Zeta": [-1.0, 0.0]})

    score = scorer.score(scorer.prepare("Acme"), scorer.prepare("Zeta"))

    assert score == 0.0


def test_prepare_l2_normalises_a_non_unit_embedding() -> None:
    scorer, _, _ = _single_token_scorer({"Acme": [3.0, 4.0]})

    assert scorer.prepare("Acme") == (0.6, 0.8)


def test_prepare_mean_pools_only_the_unmasked_token_positions() -> None:
    """A padded second token carries an implausible embedding on purpose - if pooling
    ever included it, the pooled vector would visibly diverge from the unpadded
    result asserted here."""
    tokenizer = _StubTokenizer(lambda text: ([1, 2], [1, 0], [0, 0]))
    session = _StubSession(lambda ids, mask: [[2.0, 0.0], [999.0, 999.0]])
    scorer = EmbeddingScorer(name="embed-fp32", session=session, tokenizer=tokenizer)

    assert scorer.prepare("Acme") == (1.0, 0.0)


def test_prepare_raises_when_the_tokenizer_returns_an_encoding_with_no_unmasked_tokens() -> (
    None
):
    """Unreachable through any real tokenizer output - every encoding this arm builds
    carries at least a [CLS]/[SEP] pair with ``attention_mask=1`` - but a defensive
    guard against a malformed encoding all the same, reachable here through a double
    that returns exactly the malformed shape the guard exists for."""
    tokenizer = _StubTokenizer(lambda text: ([5], [0], [0]))
    session = _StubSession(lambda ids, mask: [[1.0, 2.0]])
    scorer = EmbeddingScorer(name="embed-fp32", session=session, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="attention mask selects no tokens"):
        scorer.prepare("odd")


def test_prepare_of_a_zero_embedding_is_not_normalised_by_dividing_by_zero() -> None:
    tokenizer = _StubTokenizer(lambda text: ([1], [1], [0]))
    session = _StubSession(lambda ids, mask: [[0.0, 0.0]])
    scorer = EmbeddingScorer(name="embed-fp32", session=session, tokenizer=tokenizer)

    assert scorer.prepare("odd") == (0.0, 0.0)


def test_score_of_a_zero_embedding_against_a_real_one_is_zero_not_a_division_error() -> (
    None
):
    scorer, _, _ = _single_token_scorer({})

    assert scorer.score((0.0, 0.0), (1.0, 0.0)) == 0.0


def test_prepare_all_of_an_empty_or_entirely_blank_batch_makes_no_model_call() -> None:
    scorer, tokenizer, session = _single_token_scorer({})

    assert scorer.prepare_all([]) == []
    assert scorer.prepare_all([None, "   ", ""]) == [None, None, None]
    assert session.run_calls == []
    assert tokenizer.encode_batch_calls == []


def test_prepare_all_embeds_each_unique_name_exactly_once_in_a_single_batched_call() -> (
    None
):
    """Issue #61's core claim, asserted by observing the model invocation itself
    (batch count and batch contents), not inferred from reading ``prepare_all``'s
    source: "Acme" recurs three times and must still cost one embedding, in the one
    batched call this whole preparation makes."""
    scorer, tokenizer, session = _single_token_scorer(
        {"Acme": [1.0, 0.0], "Beta": [0.0, 1.0]}
    )

    results = scorer.prepare_all(["Acme", "Acme", "Beta", None, "  ", "Acme"])

    assert results == [
        (1.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        None,
        None,
        (1.0, 0.0),
    ]
    assert len(session.run_calls) == 1
    assert len(session.run_calls[0]["input_ids"]) == 2
    assert tokenizer.encode_batch_calls == [["Acme", "Beta"]]


def test_prepare_is_the_naive_control_and_embeds_once_per_call() -> None:
    """The contrast case to the batched test above: the same name prepared three
    separate times through ``prepare`` costs three separate model calls, each over a
    batch of one - this is the per-comparison cost ADR-0009's hoist removes, and it
    has to actually cost what it claims to for the hoist to be measurable at all."""
    scorer, tokenizer, session = _single_token_scorer({"Acme": [1.0, 0.0]})

    scorer.prepare("Acme")
    scorer.prepare("Acme")
    scorer.prepare("Acme")

    assert len(session.run_calls) == 3
    assert all(len(call["input_ids"]) == 1 for call in session.run_calls)
    assert tokenizer.encode_batch_calls == [["Acme"], ["Acme"], ["Acme"]]


def test_batched_and_unbatched_preparation_agree() -> None:
    """Mirrors ``tests/test_scoring.py``'s classical-arm parity test, which this arm
    cannot share: its batched path is not the one-liner that test's own docstring
    anticipated an embedding scorer would eventually break out of."""
    scorer, _, _ = _single_token_scorer(
        {"Acme": [1.0, 0.0], "Beta": [0.0, 1.0], "Zeta": [-1.0, 0.0]}
    )
    names = ["Acme", None, "Beta", "Acme", "  ", "Zeta"]

    assert scorer.prepare_all(names) == [scorer.prepare(n) for n in names]
