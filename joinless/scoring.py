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

Two arms are implemented here. :class:`OverlapScorer` is the token-overlap
coefficient inherited from the prior-art resolver: standard library only, and
character-blind by construction (ADR-0003). :class:`FuzzyScorer` is the
character-aware counterpart added because a transformer measured only against
a character-blind heuristic proves nothing about the transformer (ADR-0008).
Its dependency, ``rapidfuzz``, is imported only inside :class:`FuzzyScorer`,
never at module scope, so that importing this module — and scoring with
:class:`OverlapScorer` — needs nothing beyond the standard library. That is
the same boundary ADR-0014 draws around the inference runtime, applied to the
one arm here that has a dependency to keep out of the other's way.

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

    ``prepare`` returns the frozenset of normalised tokens (see the module
    docstring for the exact normalisation rule: casefold; punctuation and
    underscores to a space; whitespace collapsed). ``score`` is the overlap
    coefficient ``|A ∩ B| / min(|A|, |B|)`` — the fraction of the smaller
    name's tokens that also appear in the larger one.

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

    ``prepare`` returns the normalised name (see the module docstring for the
    exact rule) as a single string. ``score`` is the larger of two
    ``rapidfuzz`` metrics computed on that string, matching ADR-0008's
    decision table entry, "Jaro-Winkler / token-set ratio":

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


@dataclass(frozen=True)
class _Registration:
    """One arm's constructor paired with the availability check that must
    pass before :func:`get_scorer` will call it."""

    factory: Callable[[], Scorer[Any]]
    probe: Callable[[], str | None]


_SCORERS: Mapping[str, _Registration] = {
    "overlap": _Registration(factory=OverlapScorer, probe=_overlap_probe),
    "fuzzy": _Registration(factory=FuzzyScorer, probe=_fuzzy_probe),
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
