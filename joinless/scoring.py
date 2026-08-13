# SPDX-License-Identifier: MIT
"""The scorer seam: how two names are compared, kept apart from the decision.

RFC-0001 splits "how alike are these two names" from "are they the same entity"
into two types, on purpose. A :class:`Scorer` answers the first question and
returns a similarity in ``[0.0, 1.0]``; it never decides anything.
:class:`ThresholdMatcher` answers the second, and it is the **only** place a
threshold is applied. Folding a comparison into each scorer would give every
arm its own ``>=``, and "substituting a scorer changes nothing but the score"
would be a convention each one had to keep rather than a structural fact.

``Prepared`` is generic and opaque: each scorer chooses its own representation
(a token set here, a normalised string there) and nothing outside the scorer
that produced a value is entitled to look inside it.

Preparation is split from scoring, and batched preparation
(:meth:`Scorer.prepare_all`) is the contractual, production call pattern
(ADR-0009): under grid blocking a single name participates in many
comparisons, so preparing it once per record rather than once per comparison
is the optimisation the benchmark's headline rests on.
:meth:`Scorer.prepare` remains as the unbatched control that hoist is measured
against, not as a second production path — both call patterns must produce
identical prepared values, or the hoist would be measuring two different
computations rather than two ways of doing one.

Two classical arms are implemented here. :class:`OverlapScorer` is the
token-overlap coefficient inherited from the prior-art resolver: standard
library only, and character-blind by construction (ADR-0003).
:class:`FuzzyScorer` is the character-aware counterpart added because a
transformer measured only against a character-blind heuristic proves nothing
about the transformer (ADR-0008). Its dependency, ``rapidfuzz``, is imported
only inside :class:`FuzzyScorer`, never at module scope, so that importing
this module — and scoring with :class:`OverlapScorer` — needs nothing beyond
the standard library. That is the same boundary ADR-0014 draws around the
inference runtime, applied to the one arm here that has a dependency to keep
out of the other's way.

``embed-fp32`` and ``embed-int8`` are both :class:`joinless.embedding.EmbeddingScorer`
— the same class, over different model artefacts (RFC-0001) — not defined here — their
shared dependencies (``onnxruntime``, ``tokenizers``) and their artefacts are heavier
than a lazy ``import`` inside one constructor can hide, so they get their own module
(ADR-0002, ADR-0007, ADR-0014, ADR-0017). What lives here is the seam: each arm's own
probe/factory/artefact-paths trio — :func:`_embed_fp32_probe`, :func:`_embed_fp32_factory`
and :func:`_embed_fp32_artifact_paths` for fp32; :func:`_embed_int8_probe`,
:func:`_embed_int8_factory` and :func:`_embed_int8_artifact_paths` for int8 — each
importing :mod:`joinless.embedding` lazily, at call time, exactly like
:func:`_fuzzy_probe` imports ``rapidfuzz`` lazily — so a classical-only run
never reaches :mod:`joinless.embedding`, let alone the inference runtime it
eventually imports, and :mod:`joinless.embedding` is free to import
``onnxruntime``/``tokenizers`` inside its own functions without either
import ever becoming reachable from this module's top level.

Both arms share one normalisation: casefold, then punctuation and
underscores collapsed to a single space (not deleted — deleting would fuse
"Smith-Jones" into "smithjones" instead of separating it into two tokens),
then runs of whitespace collapsed and the ends trimmed. Two arms normalising
differently would be a confound the benchmark could not separate from a real
difference in matching strategy, so the rule lives in exactly one function
and both scorers call it. A record without a name (``name=None``) normalises
to the empty string, never raises.

An empty prepared value carries no information about identity, so it is
defined to overlap with nothing, including another empty value — two unnamed
records are not evidence that they are the same entity. Both scorers apply
that rule identically rather than letting it fall out of whatever their
underlying computation happens to do with empty input.

A scorer that is configured but cannot run — its dependency is not installed
— is a different failure from an unrecognised name, and :func:`get_scorer`
keeps the two apart (ADR-0013): an unknown name is :class:`ValueError`, a
known arm with a missing dependency is :class:`ScorerUnavailable` carrying a
reason. Neither is silently omitted or allowed to surface as a raw
``ImportError`` from deep inside a scorer's constructor.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

Prepared = TypeVar("Prepared")

# Punctuation and underscores become a space, not nothing, so that two words
# joined only by punctuation stay two tokens rather than merging into one.
_PUNCTUATION = re.compile(r"[^\w\s]|_", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _normalise(name: str | None) -> str:
    """Case-fold, strip punctuation, and collapse whitespace.

    Shared by every scorer in this module so that a difference in behaviour
    between arms can only ever be a difference in matching strategy, never a
    difference in how the input text was cleaned up first.
    """
    if name is None:
        return ""
    folded = name.casefold()
    despaced = _PUNCTUATION.sub(" ", folded)
    return _WHITESPACE.sub(" ", despaced).strip()


class Scorer(Protocol[Prepared]):
    """How alike two names are. Never decides whether they match (RFC-0001)."""

    @property
    def name(self) -> str:
        """Stable identifier used in benchmark records."""
        ...

    def prepare_all(self, names: Sequence[str | None]) -> list[Prepared]:
        """Batch preparation: the production call pattern (ADR-0009)."""
        ...

    def prepare(self, name: str | None) -> Prepared:
        """Single-record preparation: the naive control the hoist is measured
        against, not a second production path."""
        ...

    def score(self, a: Prepared, b: Prepared) -> float:
        """Similarity in ``[0.0, 1.0]``. Higher means more likely the same
        entity. Never a decision — see :class:`ThresholdMatcher`."""
        ...


@dataclass(frozen=True)
class ThresholdMatcher(Generic[Prepared]):
    """Turns any scorer into a decision. The only place a threshold is applied.

    This is the entire point of splitting :class:`Scorer` from the decision
    (RFC-0001): with the comparison written once, here, substituting a scorer
    cannot change anything about how a decision is reached — only what
    similarity value feeds into it.
    """

    scorer: Scorer[Prepared]
    threshold: float

    def matches(self, a: Prepared, b: Prepared) -> bool:
        return self.scorer.score(a, b) >= self.threshold


class OverlapScorer:
    """Token-overlap coefficient. Standard library only (ADR-0003).

    ``prepare`` returns the frozenset of normalised tokens — casefold,
    punctuation and underscores to a space, whitespace collapsed. ``score`` is
    the overlap coefficient ``|A ∩ B| / min(|A|, |B|)``: the fraction of the
    smaller name's tokens that also appear in the larger one.

    This arm is character-blind: it compares whole tokens, so a single typo,
    transposition, or word concatenation that changes a token entirely
    (``BRIGHTWATR`` vs ``BRIGHTWATER``) removes any overlap. That is its
    documented weakness, not a bug — it is the reason a character-aware arm
    (:class:`FuzzyScorer`) exists alongside it.

    If either name's token set is empty, the coefficient's denominator would
    be zero. Rather than raise or treat that as a special match, an empty set
    is defined to overlap with nothing: ``score`` returns ``0.0``.
    """

    @property
    def name(self) -> str:
        return "overlap"

    def prepare_all(self, names: Sequence[str | None]) -> list[frozenset[str]]:
        return [self.prepare(n) for n in names]

    def prepare(self, name: str | None) -> frozenset[str]:
        return frozenset(_normalise(name).split())

    def score(self, a: frozenset[str], b: frozenset[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / min(len(a), len(b))


class FuzzyScorer:
    """Character-aware similarity via ``rapidfuzz`` (ADR-0008).

    ``prepare`` returns the normalised name as a single string — casefold,
    punctuation and underscores to a space, whitespace collapsed, the same rule
    the overlap arm applies, so a difference between the two arms can only be a
    scoring difference. ``score`` is the larger of two ``rapidfuzz`` metrics
    computed on that string, matching ADR-0008's decision table entry,
    "Jaro-Winkler / token-set ratio":

    - ``rapidfuzz.distance.JaroWinkler.normalized_similarity`` — the
      canonical character-aware metric for census and administrative name
      linkage, and the one that catches a single-character typo,
      transposition, or word concatenation (``BRIGHTWATR`` vs
      ``BRIGHTWATER``) that :class:`OverlapScorer` cannot see at all.
    - ``rapidfuzz.fuzz.token_set_ratio`` — compares token sets rather than
      the raw strings, so a reordered legal form (``Acme Trading Co`` vs
      ``Trading Co Acme``) or a name that is a strict subset of a longer one
      (``Acme Trading`` vs ``Acme Trading Company International Holdings
      Limited``) still scores near the top even though a direct
      character-by-character comparison of the whole string would not.

    Neither metric alone covers both failure shapes — Jaro-Winkler alone
    scores the reordered pair above at 0.70 and the subset pair at 0.85,
    both well below what either pair deserves — so ``score`` takes whichever
    is higher rather than committing to one and missing the other.

    ``token_set_ratio``'s ``[0, 100]`` range is divided by 100 so every
    arm's range stays fixed at ``[0.0, 1.0]`` (RFC-0001, "Comparability");
    ``normalized_similarity`` is already ``[0.0, 1.0]``. Preprocessing is
    left to this module's own ``_normalise`` rather than either function's
    own default processor, so that normalisation stays the one rule both
    arms share, not two similar but independently maintained rules.

    ``rapidfuzz`` is imported here, inside ``__init__``, rather than at
    module scope — see the module docstring for why that boundary matters.

    An empty prepared string carries no information, so — matching
    :class:`OverlapScorer` — ``score`` returns ``0.0`` whenever either input
    is empty rather than whatever the underlying metrics happen to do with
    it.
    """

    def __init__(self) -> None:
        from rapidfuzz import fuzz
        from rapidfuzz.distance import JaroWinkler

        # Stored as module references, not bound functions, so that a test
        # patching rapidfuzz's own attributes (to make an inconvenient
        # double, ADR-0016 rule 2) affects this scorer too: attribute lookup
        # on ``self._fuzz`` happens at call time, not at construction time.
        self._fuzz = fuzz
        self._jaro_winkler = JaroWinkler

    @property
    def name(self) -> str:
        return "fuzzy"

    def prepare_all(self, names: Sequence[str | None]) -> list[str]:
        return [self.prepare(n) for n in names]

    def prepare(self, name: str | None) -> str:
        return _normalise(name)

    def score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        direct = self._jaro_winkler.normalized_similarity(a, b)
        token_set = self._fuzz.token_set_ratio(a, b) / 100.0
        return max(direct, token_set)


class ScorerUnavailable(RuntimeError):
    """A known arm whose dependency could not be loaded.

    Deliberately not a :class:`ValueError`: an unknown scorer name is a
    configuration mistake, but a known arm with a missing dependency is a
    real arm that simply cannot run here (ADR-0013) — the two must be
    distinguishable to a caller deciding what to report.
    """

    def __init__(self, scorer_name: str, reason: str) -> None:
        self.scorer_name = scorer_name
        self.reason = reason
        super().__init__(f"Scorer {scorer_name!r} is unavailable: {reason}")


def _overlap_probe() -> str | None:
    """Standard library only (ADR-0003): always available."""
    return None


def _fuzzy_probe() -> str | None:
    """Import ``rapidfuzz`` here, at call time, not at module scope.

    This is the check :func:`get_scorer` runs before it will construct
    :class:`FuzzyScorer`, so a missing dependency is reported with a reason
    rather than raising a bare ``ImportError`` out of the scorer's own
    constructor.
    """
    try:
        import rapidfuzz  # noqa: F401
    except ImportError as exc:
        return f"the 'rapidfuzz' package is not installed ({exc})"
    return None


def _embed_fp32_probe() -> str | None:
    """Import :mod:`joinless.embedding` here, at call time, not at module scope.

    Mirrors :func:`_fuzzy_probe` exactly: a classical-only run never executes
    this function's body (:func:`get_scorer` only calls a name's registered
    probe when that name is requested), so a classical-only run never imports
    :mod:`joinless.embedding` either — and that module, in turn, never
    imports ``onnxruntime`` or ``tokenizers`` at its own module scope
    (ADR-0014, ADR-0017). This is the check :func:`get_scorer` runs before it
    will construct the fp32 embedding arm.
    """
    from joinless import embedding

    return embedding.probe_fp32()


def _embed_fp32_factory() -> Scorer[Any]:
    """Constructed only after :func:`_embed_fp32_probe` has already returned
    ``None`` for this same process (:func:`get_scorer`'s probe-then-factory
    order) — artefact and dependency checks are not repeated here."""
    from joinless import embedding

    return embedding.load_fp32_scorer()


def _embed_int8_probe() -> str | None:
    """Mirrors :func:`_embed_fp32_probe` exactly, over
    :func:`joinless.embedding.probe_int8` instead — this is the check
    :func:`get_scorer` runs before it will construct the int8 embedding arm."""
    from joinless import embedding

    return embedding.probe_int8()


def _embed_int8_factory() -> Scorer[Any]:
    """Mirrors :func:`_embed_fp32_factory` exactly, over
    :func:`joinless.embedding.load_int8_scorer` instead."""
    from joinless import embedding

    return embedding.load_int8_scorer()


def _no_artifact_paths() -> tuple[Path, ...]:
    """Overlap and fuzzy carry no model artefact (ADR-0003): an explicit empty
    tuple, not a call into :mod:`joinless.embedding` that happens to return
    nothing — the two must stay distinguishable in what they *do*, not only
    in what they return (issue #63)."""
    return ()


def _embed_fp32_artifact_paths() -> tuple[Path, ...]:
    """Import :mod:`joinless.embedding` here, at call time, mirroring
    :func:`_embed_fp32_probe` and :func:`_embed_fp32_factory` exactly — a
    caller reaches this only through :func:`get_artifact_paths`, for the same
    arm whose :func:`_embed_fp32_probe` has already run, so the lazy-import
    boundary these two mirror is never bypassed here either (issue #63).
    """
    from joinless import embedding

    return tuple(
        requirement.path for requirement in embedding.artifact_requirements_fp32()
    )


def _embed_int8_artifact_paths() -> tuple[Path, ...]:
    """Mirrors :func:`_embed_fp32_artifact_paths` exactly, over
    :func:`joinless.embedding.artifact_requirements_int8` instead."""
    from joinless import embedding

    return tuple(
        requirement.path for requirement in embedding.artifact_requirements_int8()
    )


@dataclass(frozen=True)
class _Registration:
    """One arm's constructor, availability check, and artefact-file list
    (issue #63) — everything :func:`get_scorer` and :func:`get_artifact_paths`
    need, kept in the same one place per arm rather than a second registry a
    future arm could add to without the first."""

    factory: Callable[[], Scorer[Any]]
    probe: Callable[[], str | None]
    artifact_paths: Callable[[], tuple[Path, ...]]


_SCORERS: Mapping[str, _Registration] = {
    "overlap": _Registration(
        factory=OverlapScorer, probe=_overlap_probe, artifact_paths=_no_artifact_paths
    ),
    "fuzzy": _Registration(
        factory=FuzzyScorer, probe=_fuzzy_probe, artifact_paths=_no_artifact_paths
    ),
    "embed-fp32": _Registration(
        factory=_embed_fp32_factory,
        probe=_embed_fp32_probe,
        artifact_paths=_embed_fp32_artifact_paths,
    ),
    "embed-int8": _Registration(
        factory=_embed_int8_factory,
        probe=_embed_int8_probe,
        artifact_paths=_embed_int8_artifact_paths,
    ),
}


def get_scorer(name: str) -> Scorer[Any]:
    """Select an arm by configuration rather than by import (PRD FR-10).

    Choosing a matching strategy is meant to be a configuration value, not a
    code change — a benchmark that has to be edited between arms is a
    benchmark whose arms are not running the same code. Selection fails
    closed in two distinct ways: an unrecognised name is :class:`ValueError`
    naming both the value that was not recognised and the names that were;
    a recognised name whose dependency cannot be imported is
    :class:`ScorerUnavailable` naming the reason. Neither case falls back to
    a different arm under another name.
    """
    try:
        registration = _SCORERS[name]
    except KeyError:
        available = ", ".join(sorted(_SCORERS))
        raise ValueError(
            f"Unknown scorer {name!r}. Available scorers: {available}."
        ) from None

    reason = registration.probe()
    if reason is not None:
        raise ScorerUnavailable(name, reason)
    return registration.factory()


def get_artifact_paths(name: str) -> tuple[Path, ...]:
    """The on-disk artefact files ``name``'s scorer depends on — an empty
    tuple for an arm with no model, such as ``overlap`` and ``fuzzy``
    (ADR-0013: the classical arms' zero footprint is a fact this returns
    explicitly, not an absence a caller has to infer).

    Looked up through :data:`_SCORERS`, the same registry :func:`get_scorer`
    uses, so artefact-path knowledge for an arm lives in the one place its
    construction knowledge already does. Intended for a caller whose
    ``get_scorer(name)`` call already succeeded — an unrecognised name is
    still a :class:`ValueError`, matching :func:`get_scorer`, but this
    function does not itself check whether the arm can initialise; that is
    what ``get_scorer`` is for.
    """
    try:
        registration = _SCORERS[name]
    except KeyError:
        available = ", ".join(sorted(_SCORERS))
        raise ValueError(
            f"Unknown scorer {name!r}. Available scorers: {available}."
        ) from None
    return registration.artifact_paths()
