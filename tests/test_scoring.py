# SPDX-License-Identifier: MIT
"""The scorer seam: similarity, thresholding, and selection by configuration."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from joinless import embedding
from joinless.scoring import (
    FuzzyScorer,
    OverlapScorer,
    Scorer,
    ScorerUnavailable,
    ThresholdMatcher,
    get_artifact_paths,
    get_scorer,
)

_SCORERS = [
    pytest.param(OverlapScorer(), id="overlap"),
    pytest.param(FuzzyScorer(), id="fuzzy"),
]

# An invented vocabulary that reads like the business names the corpus module
# builds fixtures from, so hypothesis draws mostly look like real input rather
# than degenerating to strings normalisation folds to empty.
_NAME_WORDS = (
    "acme",
    "baobab",
    "riverside",
    "highland",
    "trading",
    "holdings",
    "traders",
    "supply",
    "logistics",
    "grove",
    "ridge",
    "mercantile",
)


def _business_names() -> st.SearchStrategy[str]:
    return st.lists(st.sampled_from(_NAME_WORDS), min_size=1, max_size=4).map(
        lambda words: " ".join(words).title()
    )


# Business-shaped names most of the time - repeating the strategy biases
# ``one_of`` toward it - with ``None``, empty, and arbitrary text kept in
# reach so the properties below still hold at the edges. Previously this
# strategy was ~92% empty or near-empty, which made both properties below
# trivially true on all but a handful of draws; the degenerate cases are
# still covered, just no longer the overwhelming majority of them.
_optional_names = st.one_of(
    _business_names(),
    _business_names(),
    _business_names(),
    st.none(),
    st.text(max_size=30),
)

# Restricted to letters and digits so normalisation cannot fold a name down to
# the empty string - that degenerate case is covered separately, on purpose,
# by the empty-input tests below rather than blurred into this property.
_non_degenerate_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Lo", "Nd")),
    min_size=1,
    max_size=30,
)


@pytest.mark.parametrize("scorer", _SCORERS)
@given(a=_optional_names, b=_optional_names)
def test_score_is_always_within_the_unit_interval(
    scorer: Scorer[Any], a: str | None, b: str | None
) -> None:
    score = scorer.score(scorer.prepare(a), scorer.prepare(b))
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("scorer", _SCORERS)
@given(a=_optional_names, b=_optional_names)
def test_score_does_not_depend_on_argument_order(
    scorer: Scorer[Any], a: str | None, b: str | None
) -> None:
    prepared_a, prepared_b = scorer.prepare(a), scorer.prepare(b)
    assert scorer.score(prepared_a, prepared_b) == scorer.score(prepared_b, prepared_a)


@pytest.mark.parametrize("scorer", _SCORERS)
@given(name=_non_degenerate_names)
def test_a_name_scored_against_itself_is_an_exact_match(
    scorer: Scorer[Any], name: str
) -> None:
    prepared = scorer.prepare(name)
    assert scorer.score(prepared, prepared) == 1.0


@pytest.mark.parametrize("scorer", _SCORERS)
@given(names=st.lists(_optional_names, max_size=10))
def test_batched_preparation_matches_preparing_one_name_at_a_time(
    scorer: Scorer[Any], names: list[str | None]
) -> None:
    """A guard placed ahead of a future batched arm (ADR-0009): both arms
    shipped today implement ``prepare_all`` as ``[prepare(n) for n in
    names]``, so this cannot currently fail on either of them. It earns its
    keep the day an arm's batched path stops being that one-liner - e.g. an
    embedding scorer that prepares a whole batch in one model call - by
    already being in place to catch that path diverging from the unbatched
    control."""
    assert scorer.prepare_all(names) == [scorer.prepare(n) for n in names]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param("BRIGHTWATR", "BRIGHTWATER", id="single-character-deletion"),
        pytest.param("MERIDIAN", "MEIRDIAN", id="adjacent-transposition"),
        pytest.param("Acme Trading", "AcmeTrading", id="concatenation"),
    ],
)
def test_overlap_is_blind_to_character_level_near_misses_fuzzy_catches(
    a: str, b: str
) -> None:
    """ADR-0008: 'Every single-character typo, transposition, and
    concatenation is invisible to [overlap].' Three failure shapes, one
    per case - overlap sees zero shared tokens in all three, fuzzy sees a
    near match in all three, which is the whole reason both arms exist."""
    overlap, fuzzy = OverlapScorer(), FuzzyScorer()

    assert overlap.score(overlap.prepare(a), overlap.prepare(b)) == 0.0
    assert fuzzy.score(fuzzy.prepare(a), fuzzy.prepare(b)) > 0.9


def test_fuzzy_scores_a_reordered_legal_form_above_what_jaro_winkler_alone_gives() -> (
    None
):
    """Pins the token-set half of the metric specifically: Jaro-Winkler on
    the whole string alone scores this pair 0.70 (reordering confuses a
    position-sensitive comparison), so a floor above that can only be
    cleared by the token_set_ratio component ADR-0008 also names."""
    fuzzy = FuzzyScorer()
    a, b = fuzzy.prepare("Acme Trading Co"), fuzzy.prepare("Trading Co Acme")
    assert fuzzy.score(a, b) > 0.85


def test_fuzzy_scores_a_truncated_name_above_what_jaro_winkler_alone_gives() -> None:
    """Same component, the other failure mode ADR-0008 names: a name that is
    a strict subset of a much longer one. Jaro-Winkler alone scores this
    pair 0.847 - below the floor asserted here - because it penalises the
    large length difference; token_set_ratio does not."""
    fuzzy = FuzzyScorer()
    a = fuzzy.prepare("Acme Trading")
    b = fuzzy.prepare("Acme Trading Company International Holdings Limited")
    assert fuzzy.score(a, b) > 0.85


@pytest.mark.parametrize("scorer", _SCORERS)
def test_an_unnamed_record_shares_nothing_with_another_unnamed_record(
    scorer: Scorer[Any],
) -> None:
    """Documented choice: an empty prepared value overlaps with nothing,
    including another empty value - two unnamed records are not evidence that
    they are the same entity."""
    assert scorer.score(scorer.prepare(None), scorer.prepare(None)) == 0.0


@pytest.mark.parametrize("scorer", _SCORERS)
def test_an_unnamed_record_shares_nothing_with_a_named_one(scorer: Scorer[Any]) -> None:
    assert scorer.score(scorer.prepare(None), scorer.prepare("Acme Traders")) == 0.0
    assert scorer.score(scorer.prepare("Acme Traders"), scorer.prepare(None)) == 0.0


def test_fuzzy_empty_guard_holds_even_if_the_underlying_metrics_would_not_return_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty-input guard is meant to remove any dependence on what
    rapidfuzz happens to do with empty strings (class docstring). Today's
    rapidfuzz version happens to agree with the guard, which means a test
    that only calls ``score("", "")`` cannot tell whether the guard exists
    or is merely redundant (ADR-0016 rule 2: a double, or here a patched
    real dependency, must be at least as awkward as the thing it stands in
    for). Patching both underlying metrics to return a perfect match on
    empty input, then asserting the guard still returns 0.0, is the only way
    to observe that the guard - and not the library - is what is deciding."""
    import rapidfuzz.distance.JaroWinkler as jaro_winkler_module
    import rapidfuzz.fuzz as fuzz_module

    monkeypatch.setattr(fuzz_module, "token_set_ratio", lambda a, b: 100.0)
    monkeypatch.setattr(jaro_winkler_module, "normalized_similarity", lambda a, b: 1.0)

    fuzzy = FuzzyScorer()
    assert fuzzy.score(fuzzy.prepare(""), fuzzy.prepare("")) == 0.0
    assert fuzzy.score(fuzzy.prepare(""), fuzzy.prepare("Acme Traders")) == 0.0
    assert fuzzy.score(fuzzy.prepare("Acme Traders"), fuzzy.prepare("")) == 0.0


@pytest.mark.parametrize("scorer", _SCORERS)
def test_scoring_ignores_case_and_punctuation(scorer: Scorer[Any]) -> None:
    a = scorer.prepare("Acme, Trading & Co.")
    b = scorer.prepare("ACME TRADING CO")
    assert scorer.score(a, b) == 1.0


@pytest.mark.parametrize("scorer", _SCORERS)
def test_punctuation_between_two_word_characters_becomes_a_space_not_nothing(
    scorer: Scorer[Any],
) -> None:
    """Deleting punctuation instead of replacing it with a space would fuse
    'Smith-Jones' into the single token 'smithjones', which shares nothing
    with the two-token 'Smith Jones' - exactly the fusion the module
    docstring names as the reason the rule is a space, not a deletion. Every
    punctuation mark in the pre-existing case/punctuation fixture already
    sits next to whitespace, so it cannot tell the two rules apart; this
    fixture places a hyphen between two word characters, where they diverge.
    """
    joined = scorer.prepare("Smith-Jones")
    split = scorer.prepare("Smith Jones")
    assert scorer.score(joined, split) == 1.0


@pytest.mark.parametrize("scorer", _SCORERS)
def test_an_underscore_is_treated_as_punctuation_not_a_word_character(
    scorer: Scorer[Any],
) -> None:
    """``\\w`` includes the underscore, so without the explicit ``|_``
    alternative in the punctuation pattern an underscore would survive
    normalisation as part of a token instead of becoming a space."""
    underscored = scorer.prepare("Acme_Trading")
    spaced = scorer.prepare("Acme Trading")
    assert scorer.score(underscored, spaced) == 1.0


@pytest.mark.parametrize("scorer", _SCORERS)
def test_casefold_handles_a_character_whose_casefold_differs_from_its_lowercase(
    scorer: Scorer[Any],
) -> None:
    """``casefold()``, not ``lower()``: ``'straße'.lower()`` is still
    ``'straße'``, but ``'straße'.casefold()`` is ``'strasse'``. Every other
    case fixture in this file is ASCII, where the two functions agree, so
    none of them can tell a casefold from a lower - this one can."""
    assert scorer.score(scorer.prepare("STRASSE"), scorer.prepare("straße")) == 1.0


def test_threshold_matcher_matches_when_the_score_meets_the_threshold() -> None:
    scorer = OverlapScorer()
    a = scorer.prepare("Acme Trading Company")
    b = scorer.prepare("Acme Holdings")
    # {acme,trading,company} vs {acme,holdings}: intersection 1, min size 2.
    assert scorer.score(a, b) == 0.5

    matcher = ThresholdMatcher(scorer=scorer, threshold=0.5)
    assert matcher.matches(a, b) is True


def test_threshold_matcher_rejects_a_score_below_the_threshold() -> None:
    scorer = OverlapScorer()
    a = scorer.prepare("Acme Trading Company")
    b = scorer.prepare("Acme Holdings")

    matcher = ThresholdMatcher(scorer=scorer, threshold=0.5 + 1e-9)
    assert matcher.matches(a, b) is False


@dataclass(frozen=True)
class _StubScorer:
    """A scorer whose ``score`` returns a fixed value regardless of its
    arguments, so :class:`ThresholdMatcher`'s decision can be checked as a
    pure function of ``scorer.score`` and ``threshold`` without depending on
    any real scorer's arithmetic. The resolver issue #32 bullet 4 asks for
    does not exist yet - this pins the part of RFC-0001's substitution
    invariant that a scorer-agnostic matcher can pin today."""

    fixed_score: float

    @property
    def name(self) -> str:
        return "stub"

    def prepare_all(self, names: Sequence[str | None]) -> list[str | None]:
        return list(names)

    def prepare(self, name: str | None) -> str | None:
        return name

    def score(self, a: str | None, b: str | None) -> float:
        return self.fixed_score


@pytest.mark.parametrize(
    ("fixed_score", "threshold", "expected"),
    [
        pytest.param(0.0, 0.5, False, id="below-threshold"),
        pytest.param(0.5, 0.5, True, id="exactly-at-threshold"),
        pytest.param(0.5, 0.5 + 1e-9, False, id="just-below-threshold"),
        pytest.param(1.0, 0.0, True, id="well-above-threshold"),
        pytest.param(0.0, 0.0, True, id="both-zero"),
    ],
)
def test_threshold_matcher_decides_from_only_score_and_threshold(
    fixed_score: float, threshold: float, expected: bool
) -> None:
    """A scripted scorer isolates the claim from any real scorer's
    arithmetic: whatever the score turns out to be, ``matches`` is exactly
    ``score >= threshold`` and nothing else about the arguments matters -
    unlike comparing two calls to the same method on the same object, which
    can never be false regardless of what the method does."""
    matcher = ThresholdMatcher(scorer=_StubScorer(fixed_score), threshold=threshold)
    assert matcher.matches("left", "right") is expected


def test_get_scorer_selects_the_overlap_arm_by_name() -> None:
    scorer = get_scorer("overlap")
    assert isinstance(scorer, OverlapScorer)
    assert scorer.name == "overlap"


def test_get_scorer_selects_the_fuzzy_arm_by_name() -> None:
    scorer = get_scorer("fuzzy")
    assert isinstance(scorer, FuzzyScorer)
    assert scorer.name == "fuzzy"


def test_get_scorer_rejects_a_genuinely_unknown_name_and_lists_the_known_ones() -> None:
    """'nonesuch' names nothing this module ever defines - unlike ``embed-fp32``,
    which is registered (below) and raises :class:`ScorerUnavailable` instead,
    carrying a reason, when it cannot initialise."""
    with pytest.raises(ValueError, match="Unknown scorer") as excinfo:
        get_scorer("nonesuch")

    message = str(excinfo.value)
    assert "nonesuch" in message
    assert "overlap" in message
    assert "fuzzy" in message


def test_get_artifact_paths_is_empty_for_overlap() -> None:
    """Overlap carries no model artefact (ADR-0003), so its size on disk is
    explicitly ``()``, not merely absent from a mapping (issue #63)."""
    assert get_artifact_paths("overlap") == ()


def test_get_artifact_paths_is_empty_for_fuzzy() -> None:
    assert get_artifact_paths("fuzzy") == ()


def test_get_artifact_paths_names_the_fp32_model_and_tokenizer_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOINLESS_MODEL_CACHE_DIR", "/some/cache")

    paths = get_artifact_paths("embed-fp32")

    assert paths == (
        Path("/some/cache/fp32/model.onnx"),
        Path("/some/cache/fp32/tokenizer.json"),
    )


def test_get_artifact_paths_rejects_a_genuinely_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown scorer"):
        get_artifact_paths("nonesuch")


_ARTIFACT_PATHS_CLASSICAL_ONLY_PROBE = """
import sys
from joinless.scoring import get_artifact_paths
assert get_artifact_paths("overlap") == ()
assert get_artifact_paths("fuzzy") == ()
offenders = sorted(m for m in sys.modules if m == "joinless.embedding")
sys.exit(1 if offenders else 0)
"""


def test_asking_for_a_classical_arms_artifact_paths_never_reaches_the_embedding_module() -> (
    None
):
    """Mirrors :mod:`joinless.scoring`'s module-docstring invariant, "a
    classical-only run never reaches joinless.embedding", applied to
    :func:`get_artifact_paths` (issue #63): the empty-tuple case for
    ``overlap``/``fuzzy`` must be a literal, not a call into the embedding
    module that happens to return nothing."""
    result = subprocess.run(
        [sys.executable, "-c", _ARTIFACT_PATHS_CLASSICAL_ONLY_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "asking for a classical arm's artifact paths imported joinless.embedding: "
        f"{result.stdout.strip() or result.stderr.strip()}"
    )


def test_get_scorer_reports_a_missing_dependency_as_unavailable_with_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0013: a configured arm whose dependency cannot be imported is
    reported unavailable with a reason, never silently omitted from the
    registry and never left to surface as a raw ``ImportError`` from inside
    ``FuzzyScorer.__init__``."""
    monkeypatch.setitem(sys.modules, "rapidfuzz", None)

    with pytest.raises(ScorerUnavailable) as excinfo:
        get_scorer("fuzzy")

    assert excinfo.value.scorer_name == "fuzzy"
    assert "rapidfuzz" in excinfo.value.reason


def test_get_scorer_selects_overlap_even_when_rapidfuzz_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The overlap arm needs nothing beyond the standard library (ADR-0008
    consequence: 'the zero-dependency property now belongs to overlap
    alone') - a missing rapidfuzz must not take this arm down too."""
    monkeypatch.setitem(sys.modules, "rapidfuzz", None)

    scorer = get_scorer("overlap")
    prepared = scorer.prepare("Acme Traders")
    assert scorer.score(prepared, prepared) == 1.0


def test_get_scorer_reports_embed_fp32_unavailable_when_onnxruntime_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors ``test_get_scorer_reports_a_missing_dependency_as_unavailable_with_a_reason``
    for the fp32 embedding arm: the registry's own probe (``_embed_fp32_probe``)
    forwards :func:`joinless.embedding.probe_fp32`'s reason without altering it."""
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(ScorerUnavailable) as excinfo:
        get_scorer("embed-fp32")

    assert excinfo.value.scorer_name == "embed-fp32"
    assert "onnxruntime" in excinfo.value.reason


def test_get_scorer_reports_embed_fp32_unavailable_without_a_configured_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with both dependencies present, requesting the fp32 arm on a machine
    that has not run setup fails closed (ADR-0013) - via the same registry seam
    every other arm goes through, not a special case in ``get_scorer`` itself."""
    monkeypatch.setitem(sys.modules, "onnxruntime", types.ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "tokenizers", types.ModuleType("tokenizers"))
    monkeypatch.delenv("JOINLESS_MODEL_CACHE_DIR", raising=False)

    with pytest.raises(ScorerUnavailable) as excinfo:
        get_scorer("embed-fp32")

    assert excinfo.value.scorer_name == "embed-fp32"
    assert "JOINLESS_MODEL_CACHE_DIR" in excinfo.value.reason


def _fake_onnxruntime_and_tokenizers_modules() -> tuple[
    types.ModuleType, types.ModuleType
]:
    """A fake ``onnxruntime``/``tokenizers`` pair minimal enough to let
    ``embedding.load_fp32_scorer`` run to completion and produce a scorer that can
    actually be asked to prepare and score - not just construct without raising."""

    class _FakeEncoding:
        def __init__(self, text: str) -> None:
            self.ids = [len(text) + 1]
            self.attention_mask = [1]
            self.type_ids = [0]

    class _FakeTokenizer:
        def enable_padding(self, *, pad_token: str, pad_id: int) -> None:
            pass

        def enable_truncation(self, *, max_length: int) -> None:
            pass

        def encode_batch(self, texts: Sequence[str]) -> list[_FakeEncoding]:
            return [_FakeEncoding(text) for text in texts]

    class _TokenizerNamespace:
        @staticmethod
        def from_file(path: str) -> _FakeTokenizer:
            return _FakeTokenizer()

    class _FakeSession:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            pass

        def run(
            self, output_names: list[str] | None, input_feed: dict[str, list[list[int]]]
        ) -> list[list[list[list[float]]]]:
            rows = input_feed["input_ids"]
            return [[[[1.0, 0.0] for _ in row] for row in rows]]

    fake_tokenizers = types.ModuleType("tokenizers")
    fake_tokenizers.Tokenizer = _TokenizerNamespace  # type: ignore[attr-defined]
    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.InferenceSession = _FakeSession  # type: ignore[attr-defined]
    return fake_onnxruntime, fake_tokenizers


def test_get_scorer_selects_the_embed_fp32_arm_through_the_full_registry_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registration wiring itself (``_embed_fp32_probe``/``_embed_fp32_factory``),
    not ``joinless.embedding``'s own functions called directly: proves ``get_scorer``
    reaches a real, working :class:`~joinless.embedding.EmbeddingScorer` for
    ``"embed-fp32"`` exactly as it does for ``"overlap"`` and ``"fuzzy"`` above."""
    fake_onnxruntime, fake_tokenizers = _fake_onnxruntime_and_tokenizers_modules()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)

    fp32_dir = tmp_path / "fp32"
    fp32_dir.mkdir(parents=True)
    model_bytes = b"a fake fp32 graph"
    tokenizer_bytes = b"a fake tokenizer config"
    (fp32_dir / "model.onnx").write_bytes(model_bytes)
    (fp32_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    monkeypatch.setattr(
        embedding, "FP32_MODEL_SHA256", hashlib.sha256(model_bytes).hexdigest()
    )
    monkeypatch.setattr(
        embedding, "FP32_TOKENIZER_SHA256", hashlib.sha256(tokenizer_bytes).hexdigest()
    )
    monkeypatch.setenv("JOINLESS_MODEL_CACHE_DIR", str(tmp_path))

    scorer = get_scorer("embed-fp32")

    assert scorer.name == "embed-fp32"
    prepared = scorer.prepare("Acme Traders")
    assert scorer.score(prepared, prepared) == 1.0


_EMBED_FP32_LAZY_IMPORT_PROBE = """
import sys
sys.modules["onnxruntime"] = None
sys.modules["tokenizers"] = None
from joinless.scoring import ScorerUnavailable, get_scorer
try:
    get_scorer("embed-fp32")
except ScorerUnavailable as exc:
    assert exc.scorer_name == "embed-fp32"
    assert "onnxruntime" in exc.reason
else:
    raise AssertionError("expected ScorerUnavailable")
"""


def test_importing_scoring_never_imports_the_embedding_module_or_its_dependencies() -> (
    None
):
    """Mirrors ``test_importing_scoring_and_scoring_with_overlap_needs_only_the_standard_library``
    for the neural arm: blocking ``onnxruntime`` and ``tokenizers`` *before*
    ``joinless.scoring`` is ever imported, then merely importing it and requesting
    ``"embed-fp32"``, is the only way to observe that neither dependency is reachable
    from module import alone - patching ``sys.modules`` in this already-running
    process (as the tests above do) proves nothing about that."""
    result = subprocess.run(
        [sys.executable, "-c", _EMBED_FP32_LAZY_IMPORT_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "requesting the fp32 embedding arm did not fail the way expected: "
        f"{result.stdout.strip() or result.stderr.strip()}"
    )


_OVERLAP_ONLY_PROBE = """
import sys
sys.modules["rapidfuzz"] = None
from joinless.scoring import get_scorer
scorer = get_scorer("overlap")
prepared = scorer.prepare("Acme Traders")
assert scorer.score(prepared, prepared) == 1.0
"""


def test_importing_scoring_and_scoring_with_overlap_needs_only_the_standard_library() -> (
    None
):
    """Issue #33 bullet 1: overlap's zero-dependency property has to hold of
    the module, not just of its arithmetic. Blocking rapidfuzz in a fresh
    interpreter *before* ``joinless.scoring`` is ever imported is the only
    way to observe whether a module-level ``from rapidfuzz import ...``
    would have failed - patching ``sys.modules`` in this process (as the
    tests above do) proves nothing about that, because this process already
    imported and cached the module before the patch was applied. Mirrors
    tests/test_import_boundary.py's reasoning for the same class of claim.
    """
    result = subprocess.run(
        [sys.executable, "-c", _OVERLAP_ONLY_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "scoring with the overlap arm pulled in rapidfuzz at module level: "
        f"{result.stdout.strip() or result.stderr.strip()}"
    )


def test_the_module_imports_under_an_optimised_interpreter() -> None:
    """``python -OO`` discards docstrings, so anything that rewrites ``__doc__`` at
    import time meets ``None`` there. A child interpreter is the only way to observe
    it: ``-OO`` is decided when the process starts and cannot be turned on later."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-OO", "-c", "import joinless.scoring"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.strip()
