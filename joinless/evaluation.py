# SPDX-License-Identifier: MIT
"""Per-family evaluation, threshold selection, and pre-registered expectations
(ADR-0011, ADR-0013, RFC-0002 "Metrics").

Every function here is pure: labelled pairs and an already-constructed scorer or
matcher go in, figures come out. Nothing reads a file, spawns a process, reads the
clock, or draws from an unseeded source of randomness.

That is a deliberate boundary rather than a coincidence. RFC-0002 requires every
resource measurement to be taken in a fresh child process per arm, so a module that
timed or measured anything here would be reporting a figure contaminated by whatever
else this process had already done. Keeping the figures a pure function of the pairs
also makes them reproducible from the corpus seed alone, which is what lets a run
record be checked by recomputing it rather than trusted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from joinless.corpus import Corpus, LabelledPair, Role
from joinless.scoring import Scorer, ThresholdMatcher


@dataclass(frozen=True, slots=True)
class Metric:
    """A figure that may be undefined (ADR-0013).

    Precision with no predicted positives, and recall with no actual positives,
    are not zero — collapsing them into ``0.0`` makes a broken run look like a bad
    arm (ADR-0013's motivating example). ``value`` is ``None`` exactly when
    ``undefined_reason`` is set, so the two can never drift into a state that is
    both "here is a number" and "here is why there isn't one" — or neither.
    """

    value: float | None
    undefined_reason: str | None

    def __post_init__(self) -> None:
        if (self.value is None) != (self.undefined_reason is not None):
            raise ValueError(
                "Metric.value must be None exactly when undefined_reason is set "
                f"(got value={self.value!r}, undefined_reason={self.undefined_reason!r})"
            )


def _ratio(numerator: int, denominator: int, reason_when_zero: str) -> Metric:
    """A metric that is undefined, not ``0.0``, when its denominator is empty
    (ADR-0013) — the shared arithmetic behind both precision and recall, so the two
    can never disagree about what an empty denominator means."""
    if denominator == 0:
        return Metric(value=None, undefined_reason=reason_when_zero)
    return Metric(value=numerator / denominator, undefined_reason=None)


def _f1(precision: Metric, recall: Metric) -> Metric:
    """The harmonic mean of precision and recall, undefined whenever either input
    is undefined, or when both are defined but zero — ``2PR/(P+R)`` has a zero
    denominator exactly there, and that is the same "nothing to divide by" fact
    ``_ratio`` already treats as undefined rather than ``0.0``."""
    if precision.value is None:
        return Metric(
            value=None,
            undefined_reason=f"precision is undefined: {precision.undefined_reason}",
        )
    if recall.value is None:
        return Metric(
            value=None,
            undefined_reason=f"recall is undefined: {recall.undefined_reason}",
        )
    denominator = precision.value + recall.value
    if denominator == 0:
        return Metric(value=None, undefined_reason="precision and recall are both zero")
    return Metric(
        value=2 * precision.value * recall.value / denominator, undefined_reason=None
    )


@dataclass(frozen=True, slots=True)
class FamilyResult:
    """Precision, recall and F1 for one perturbation family (issue #48), plus the
    raw counts behind them. The counts travel with the metrics — not just the
    ratios — so that :func:`_aggregate` can pool a mixture of families without
    re-scanning the pairs that produced this row (see its docstring)."""

    family: str
    precision: Metric
    recall: Metric
    f1: Metric
    true_positives: int
    predicted_positives: int
    actual_positives: int


# ADR-0011 rule 4 / issue #48 rule 3: "any aggregate is labelled as derived".
# Naming it once, here, is what stops that label from drifting out of sync with
# what :func:`_aggregate` actually does — the same reason ADR-0011 rule 2 names
# _SELECTION_PROCEDURE once above rather than inlining its text at each call site.
_AGGREGATE_DERIVATION = (
    "sum true positives, predicted positives and actual positives across "
    "every family in the per-family table, then compute precision, recall "
    "and F1 from those pooled counts — never computed independently of the "
    "table"
)


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """A summary of a per-family table, never an independent measurement — see
    :func:`_aggregate`, the only place one of these is built.

    ``derivation`` is the runtime form of "any aggregate is labelled as
    derived" (issue #48): a consumer holding a bare ``AggregateResult``, with
    no view of the ``EvaluationReport`` it came from, can still tell it apart
    from a ninth family by reading this field, rather than having to infer it
    from field order on a type it may not even hold.
    """

    precision: Metric
    recall: Metric
    f1: Metric
    derivation: str


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """The per-family table, and the aggregate derived from it (issue #48).

    Field order is half of that bullet: it is what "per-family results first"
    means for a type with no serialiser of its own yet (issue #46 owns that) —
    ``per_family`` is declared before ``aggregate``. The other half, "any
    aggregate is labelled as derived", is satisfied separately, by
    :attr:`AggregateResult.derivation` — a fact a reader can see on the
    aggregate itself, not one that depends on where it sits in this type.
    ``aggregate`` is never built from anything but ``per_family`` — see
    :func:`_aggregate`.
    """

    per_family: tuple[FamilyResult, ...]
    aggregate: AggregateResult


def _predict(
    pairs: Sequence[LabelledPair], matcher: ThresholdMatcher[Any]
) -> list[bool]:
    """Score every pair with ``matcher.scorer``, prepared in batch (ADR-0009's
    production call pattern — see the module docstring of :mod:`joinless.scoring`
    for why the batched and unbatched paths must agree, and why production code
    uses the batched one)."""
    left_prepared = matcher.scorer.prepare_all([pair.left_name for pair in pairs])
    right_prepared = matcher.scorer.prepare_all([pair.right_name for pair in pairs])
    return [
        matcher.matches(left, right)
        for left, right in zip(left_prepared, right_prepared, strict=True)
    ]


def _counts(
    pairs: Sequence[LabelledPair], predictions: Sequence[bool]
) -> tuple[int, int, int]:
    true_positives = sum(
        1
        for pair, predicted in zip(pairs, predictions, strict=True)
        if predicted and pair.label == 1
    )
    predicted_positives = sum(1 for predicted in predictions if predicted)
    actual_positives = sum(1 for pair in pairs if pair.label == 1)
    return true_positives, predicted_positives, actual_positives


def _family_result(
    family: str, pairs: Sequence[LabelledPair], predictions: Sequence[bool]
) -> FamilyResult:
    true_positives, predicted_positives, actual_positives = _counts(pairs, predictions)
    precision = _ratio(true_positives, predicted_positives, "no predicted positives")
    recall = _ratio(true_positives, actual_positives, "no actual positives")
    return FamilyResult(
        family=family,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        true_positives=true_positives,
        predicted_positives=predicted_positives,
        actual_positives=actual_positives,
    )


def _aggregate(per_family: Sequence[FamilyResult]) -> AggregateResult:
    """Pool the per-family counts into one precision, recall and F1 (issue #48:
    "the aggregate is computed from the per-family table, not independently").

    Summing counts already sitting on ``per_family`` — rather than re-deriving them
    from the underlying pairs — is what makes the aggregate a *function of the
    table a reader already sees*, not a second, potentially disagreeing pass over
    the raw data. It also gives the aggregate its one real property: drawing the
    same families in a different mixture (more pairs from one family, fewer from
    another) changes which family's counts dominate the sum, and therefore the
    aggregate, while every individual family's own ratios stay exactly what they
    were — a per-family figure depends only on that family's own pairs.
    """
    true_positives = sum(family.true_positives for family in per_family)
    predicted_positives = sum(family.predicted_positives for family in per_family)
    actual_positives = sum(family.actual_positives for family in per_family)
    precision = _ratio(true_positives, predicted_positives, "no predicted positives")
    recall = _ratio(true_positives, actual_positives, "no actual positives")
    return AggregateResult(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        derivation=_AGGREGATE_DERIVATION,
    )


def evaluate(
    pairs: Sequence[LabelledPair], matcher: ThresholdMatcher[Any]
) -> EvaluationReport:
    """Score every pair with ``matcher`` and report precision, recall and F1 per
    family (issue #48), with an aggregate derived from that table rather than
    computed separately.

    ``pairs`` must be non-empty and every pair must carry a ``category`` — an
    empty split or an uncategorised pair means the caller mis-selected which
    pairs to evaluate, and reporting an empty or partial table would look like a
    sound result of zero families rather than the caller error it is.
    """
    if not pairs:
        raise ValueError("evaluate requires at least one pair")

    predictions = _predict(pairs, matcher)

    indices_by_family: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        if pair.category is None:
            raise ValueError(
                f"pair {pair.pair_id!r} has no category; per-family evaluation "
                "requires one"
            )
        indices_by_family.setdefault(pair.category, []).append(index)

    per_family = tuple(
        _family_result(
            family,
            [pairs[i] for i in indices],
            [predictions[i] for i in indices],
        )
        for family, indices in indices_by_family.items()
    )
    return EvaluationReport(per_family=per_family, aggregate=_aggregate(per_family))


def _pairs_for_role(corpus: Corpus, role: Role) -> list[LabelledPair]:
    """Every pair ``corpus`` assigns to ``role``. This is the one place a role
    boundary is drawn — :func:`select_threshold` calls it with ``"calibration"``
    hard-coded, never a role a caller supplies, which is what makes "selection
    reads calibration data only" enforced rather than a convention a caller has
    to remember to uphold (issue #49)."""
    # pair_id is str | None on LabelledPair (RFC-0005 leaves it open for a
    # supplied file with none), but Corpus's own validation ties roles' keys to
    # exactly the corpus's pair ids - the same cast joinless.corpus._split_into_roles
    # makes, on the same invariant.
    return [
        pair for pair in corpus.pairs if corpus.roles[cast(str, pair.pair_id)] == role
    ]


# ADR-0011 rule 2: "a documented and identical procedure", the same for every arm.
# Naming it once, here, is what keeps two call sites from drifting into two
# different sweeps that happen to look similar.
_SELECTION_PROCEDURE = (
    "sweep every similarity score the scorer produces on the calibration split as "
    "a threshold candidate; keep whichever candidate maximises calibration F1, "
    "breaking a tie toward the lowest candidate that achieves it"
)


@dataclass(frozen=True, slots=True)
class SelectedThreshold:
    """One arm's threshold, chosen by the one procedure every arm shares
    (issue #49), plus the audit trail: which role's pairs it was computed from,
    and how many. ``role`` is what lets a later step recognise contamination —
    see :class:`FrozenThreshold` and :func:`evaluate_sealed_test`."""

    scorer_name: str
    value: float
    procedure: str
    role: Role
    n_pairs: int


def select_threshold(corpus: Corpus, scorer: Scorer[Any]) -> SelectedThreshold:
    """Select a threshold for ``scorer`` from ``corpus``'s calibration split alone
    (ADR-0011 rule 2, issue #49).

    The procedure takes no argument that varies per arm beyond the scorer itself:
    every score the scorer actually produces on the calibration pairs becomes a
    threshold candidate, and whichever maximises calibration F1 is kept. Iterating
    candidates in ascending order and only replacing the best on a strict
    improvement (``>``, not ``>=``) means a tie resolves to the lowest candidate
    that reached the maximum — the most permissive threshold among equally good
    ones, rather than an arbitrary one that happens to sort last.

    Reads ``corpus``'s calibration pairs only: there is no parameter through which
    a caller could pass development or sealed-test pairs into this function, so
    the "calibration data only" rule is a fact about what this function is able to
    read, not a convention about what a caller happens to pass it.
    """
    calibration_pairs = _pairs_for_role(corpus, "calibration")
    if not calibration_pairs:
        raise ValueError("corpus has no calibration pairs to select a threshold from")

    left_prepared = scorer.prepare_all([pair.left_name for pair in calibration_pairs])
    right_prepared = scorer.prepare_all([pair.right_name for pair in calibration_pairs])
    scores = [
        scorer.score(left, right)
        for left, right in zip(left_prepared, right_prepared, strict=True)
    ]
    labels = [pair.label for pair in calibration_pairs]

    best_value = min(scores)
    best_f1 = -1.0
    for candidate in sorted(set(scores)):
        predictions = [score >= candidate for score in scores]
        true_positives = sum(
            1
            for predicted, label in zip(predictions, labels, strict=True)
            if predicted and label == 1
        )
        precision = _ratio(true_positives, sum(predictions), "no predicted positives")
        recall = _ratio(true_positives, sum(labels), "no actual positives")
        f1 = _f1(precision, recall)
        f1_value = f1.value if f1.value is not None else -1.0
        if f1_value > best_f1:
            best_f1 = f1_value
            best_value = candidate

    return SelectedThreshold(
        scorer_name=scorer.name,
        value=best_value,
        procedure=_SELECTION_PROCEDURE,
        role="calibration",
        n_pairs=len(calibration_pairs),
    )


@dataclass(frozen=True, slots=True)
class FrozenThreshold:
    """The freeze point ADR-0011 rule 2 requires as "a distinct step, not a
    convention" (issue #49). Wrapping a :class:`SelectedThreshold` in its own type
    — rather than reusing ``SelectedThreshold`` for both "just selected" and
    "safe to score the sealed test with" — is what makes
    :func:`evaluate_sealed_test`'s parameter type reject a threshold that has not
    been through :func:`freeze_threshold`: passing a bare ``SelectedThreshold`` is
    a type error, not a naming convention a caller has to remember to respect.
    """

    selected: SelectedThreshold

    @property
    def scorer_name(self) -> str:
        return self.selected.scorer_name

    @property
    def value(self) -> float:
        return self.selected.value

    @property
    def role(self) -> Role:
        return self.selected.role


def freeze_threshold(selected: SelectedThreshold) -> FrozenThreshold:
    """The freeze step itself (issue #49) — a distinct function call producing a
    distinct type, so "has this threshold been frozen yet" is a fact the type
    checker can see rather than something only a docstring asserts."""
    return FrozenThreshold(selected=selected)


@dataclass(frozen=True, slots=True)
class InvalidRun:
    """ADR-0013: a run whose threshold selection touched sealed-test data is
    ``invalid``, not warned. This type carries no metric at all — there is no
    field on it a caller could mistakenly read and report as a real figure."""

    reason: str


def evaluate_sealed_test(
    corpus: Corpus, scorer: Scorer[Any], frozen: FrozenThreshold
) -> EvaluationReport | InvalidRun:
    """Evaluate ``corpus``'s sealed-test split with a threshold that has already
    been through :func:`freeze_threshold` (ADR-0011 rule 2, issue #49).

    ``frozen`` carries the role its underlying :class:`SelectedThreshold` was
    computed from. :func:`select_threshold` only ever produces ``"calibration"``,
    but this check exists for the ``SelectedThreshold`` values that function does
    not produce — one built directly, naming a different role — which is exactly
    the shape "threshold selection touched the sealed test" takes (issue #51).
    Rather than a number with a warning attached, the run is reported as
    :class:`InvalidRun` and no metric is computed at all (ADR-0013).
    """
    if frozen.role != "calibration":
        return InvalidRun(
            reason=(
                f"threshold selection read role {frozen.role!r}, not "
                "'calibration' — the sealed test is not scored with a threshold "
                "that may have seen it"
            )
        )

    sealed_test_pairs = _pairs_for_role(corpus, "sealed test")
    if not sealed_test_pairs:
        raise ValueError("corpus has no sealed test pairs to evaluate")

    matcher = ThresholdMatcher(scorer=scorer, threshold=frozen.value)
    return evaluate(sealed_test_pairs, matcher)


@dataclass(frozen=True, slots=True)
class ExpectedWinners:
    """Which arm is expected to win each family (ADR-0011 rule 4, issue #50).

    What this type guarantees: once built, it cannot be edited through any code
    path. ``winners`` is copied into a :class:`~types.MappingProxyType` in
    ``__post_init__`` rather than trusted as given, so that neither the caller's
    own dict, mutated after construction, nor an attempt to write through
    ``expected.winners[...]`` can change what this instance reports — combined
    with the dataclass being frozen (so ``expected.winners = ...`` also fails),
    there is no code path, through this type or the caller's own reference, that
    edits an ``ExpectedWinners`` after it is built. Reporting a different
    expectation means building a new one, which is a different fact about a
    different run, not an edit to this run's.

    What this type does not guarantee: that the expectation was recorded
    before the run it is compared against. That ordering is a property of how
    a run is assembled — the expectation existing, untouched, before a report
    is computed — not a property this type can enforce or this module can
    observe. It belongs to the run record (``joinless/runrecord.py``,
    issue #57), which does not exist yet.
    """

    winners: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "winners", MappingProxyType(dict(self.winners)))


@dataclass(frozen=True, slots=True)
class Contradiction:
    """One family whose actual best-F1 arm was not the pre-registered expectation
    (ADR-0011 rule 4) — "a finding... not a footnote": a first-class value in a
    tuple :func:`find_contradictions` returns, never text folded into a log line."""

    family: str
    expected_winner: str
    actual_winner: str


def find_contradictions(
    expected: ExpectedWinners, reports: Mapping[str, EvaluationReport]
) -> tuple[Contradiction, ...]:
    """Every family in ``expected`` whose actual best-scoring arm, among
    ``reports``, was not the one pre-registered (issue #50), in the order
    ``expected`` records them.

    "Actual best-scoring" compares each arm's F1 for that family, skipping an
    arm that did not report the family at all or reported an undefined F1 there
    — an undefined figure is not evidence the arm lost, only that no comparison
    can be drawn from it (ADR-0013). A family fewer than two arms can be compared
    on is skipped entirely: winning is only meaningful against a competitor, so
    this is neither a contradiction nor a confirmation, just nothing to report.
    """
    contradictions: list[Contradiction] = []
    for family, expected_winner in expected.winners.items():
        scores: dict[str, float] = {}
        for arm, report in reports.items():
            row = next((f for f in report.per_family if f.family == family), None)
            if row is not None and row.f1.value is not None:
                scores[arm] = row.f1.value
        if len(scores) < 2:
            continue
        actual_winner = max(scores, key=scores.__getitem__)
        if actual_winner != expected_winner:
            contradictions.append(
                Contradiction(
                    family=family,
                    expected_winner=expected_winner,
                    actual_winner=actual_winner,
                )
            )
    return tuple(contradictions)
