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

import statistics
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

    ``n_pairs`` is how many pairs this report evaluated — declared last
    because, unlike ``per_family``/``aggregate``'s ordering rule, no issue ties
    its position to anything a reader parses positionally. It is also the
    batch size issue #61's third bullet asks recorded: :func:`_predict` calls
    ``matcher.scorer.prepare_all`` exactly twice per evaluation, once with
    every pair's left name and once with every pair's right name, so
    ``len(pairs)`` **is** the size of each of those two calls under this
    arm's batching contract (:mod:`joinless.scoring`'s module docstring,
    ADR-0009) — not a second figure computed independently of it, which is
    what would let this field and the batch ``prepare_all`` actually saw
    drift apart.
    """

    per_family: tuple[FamilyResult, ...]
    aggregate: AggregateResult
    n_pairs: int


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
    return EvaluationReport(
        per_family=per_family, aggregate=_aggregate(per_family), n_pairs=len(pairs)
    )


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


# Issue #97 / ADR-0011 rule 3: "results are reported per perturbation family...
# the corpus is generated under several deterministic seeds and variation across
# seeds is reported." A single defined value has no spread to report — the same
# "undefined propagates, never collapses to a number" rule ADR-0013 states for a
# ratio applies here to a spread: reporting ``0.0`` for one seed would claim a
# stability nothing measured.
_MIN_SEEDS_FOR_VARIATION = 2


def _variation_metric(values: Sequence[Metric], metric_name: str) -> Metric:
    """The sample standard deviation of ``values``'s defined figures, or
    undefined naming how many of ``values`` actually had one — the shared
    arithmetic behind every field on :class:`FamilyVariation`, so precision,
    recall and F1 can never disagree about what "not enough seeds" means."""
    defined = [value.value for value in values if value.value is not None]
    if len(defined) < _MIN_SEEDS_FOR_VARIATION:
        return Metric(
            value=None,
            undefined_reason=(
                f"only {len(defined)} of {len(values)} seed(s) produced a "
                f"defined {metric_name} for this family"
            ),
        )
    return Metric(value=statistics.stdev(defined), undefined_reason=None)


@dataclass(frozen=True, slots=True)
class FamilyVariation:
    """Seed-to-seed spread of one family's precision, recall and F1
    (ADR-0011 rule 3, issue #97) — the sample standard deviation across
    whichever of :func:`compute_family_variation`'s seeds produced a defined
    value for that metric and family, never estimated and never a second
    measurement: pure arithmetic over the per-seed
    :class:`EvaluationReport` values a caller already has, mirroring how
    :class:`AggregateResult` is derived from :class:`FamilyResult` rows
    rather than computed independently of them.
    """

    family: str
    precision: Metric
    recall: Metric
    f1: Metric


def compute_family_variation(
    by_seed: Mapping[int, EvaluationReport],
) -> tuple[FamilyVariation, ...]:
    """One :class:`FamilyVariation` per family in ``by_seed``'s reports, in
    the lowest-numbered seed's own family order (ADR-0011 rule 3, issue
    #97) — the figure that lets a reader see whether a pooled per-family
    result depends on one draw, without computing it from ``by_seed`` by
    hand.

    Every corpus :func:`joinless.corpus.generate_corpus` builds carries the
    same eight families in the same order (that module's docstring), so in
    the one call path this module ships — every entry in ``by_seed`` coming
    from :func:`evaluate_sealed_test_with_variation` — the reports already
    agree. This function still checks rather than assumes it: two reports
    naming different families is a caller error (an evaluation set that
    changed shape between seeds is not something a spread figure can paper
    over), so it raises rather than silently comparing family "exact" in one
    seed against family "exact" in another as if they meant the same
    evaluation.
    """
    if not by_seed:
        raise ValueError("compute_family_variation requires at least one seed's report")

    ordered_seeds = sorted(by_seed)
    reference_families = tuple(
        row.family for row in by_seed[ordered_seeds[0]].per_family
    )
    rows_by_seed = {
        seed: {row.family: row for row in report.per_family}
        for seed, report in by_seed.items()
    }
    for seed in ordered_seeds:
        families = set(rows_by_seed[seed])
        if families != set(reference_families):
            raise ValueError(
                f"seed {seed}'s report families {sorted(families)} do not "
                f"match seed {ordered_seeds[0]}'s {sorted(reference_families)}"
            )

    variations = []
    for family in reference_families:
        rows = [rows_by_seed[seed][family] for seed in ordered_seeds]
        variations.append(
            FamilyVariation(
                family=family,
                precision=_variation_metric(
                    [row.precision for row in rows], "precision"
                ),
                recall=_variation_metric([row.recall for row in rows], "recall"),
                f1=_variation_metric([row.f1 for row in rows], "F1"),
            )
        )
    return tuple(variations)


# Issue #97's third bullet: pooling and per-seed reporting answer different
# questions, and the record states which one each number answers — named once
# here, exactly as `_SELECTION_PROCEDURE` and `_AGGREGATE_DERIVATION` above are
# each named once, so the run record and this module's own reasoning about it
# cannot drift apart.
_POOLED_ACCURACY_ANSWERS = (
    "what this arm scores across every seed's sealed-test pairs, pooled into "
    "one split and scored together under the one threshold ADR-0011 rule 2 "
    "selects from pooled calibration data — the reported headline RFC-0002's "
    "splits table calls the sealed test's result"
)
_BY_SEED_ACCURACY_ANSWERS = (
    "what this arm scores on each seed's own sealed-test split alone, under "
    "that identical threshold and procedure — whether the pooled figure "
    "above depends on one draw or holds across the corpus's several "
    "deterministic seeds (ADR-0011 rule 3)"
)


@dataclass(frozen=True, slots=True)
class SealedTestAccuracy:
    """One arm's sealed-test accuracy, structured so a pooled figure can
    never be read without the seed-to-seed variation that explains what it
    hides (ADR-0011 rule 3, issue #97).

    ``pooled`` and ``by_seed``/``variation`` answer different questions —
    ``pooled_answers`` and ``by_seed_answers`` say which, in the record
    itself, not only in this docstring (issue #97's third bullet: "the
    record should say which one each number answers"). Both are pooled and
    per-seed evaluations of the exact same sealed-test pairs under the exact
    same frozen threshold (:func:`evaluate_sealed_test_with_variation`) —
    ADR-0011 rule 2's "identical procedure" holds seed to seed here, the
    same way it already holds arm to arm, or the variation reported would be
    an artefact of the procedure rather than of the draw.

    ``variation`` is *derived* from ``by_seed`` — mirroring
    :attr:`AggregateResult.derivation` — never computed independently of it,
    and :meth:`__post_init__` makes that a structural fact rather than a
    convention a caller has to remember: a ``SealedTestAccuracy`` naming a
    pooled family with no corresponding row in ``variation`` cannot be
    constructed at all.
    """

    pooled: EvaluationReport
    pooled_answers: str
    by_seed: Mapping[int, EvaluationReport]
    variation: tuple[FamilyVariation, ...]
    by_seed_answers: str

    def __post_init__(self) -> None:
        if not self.by_seed:
            raise ValueError("SealedTestAccuracy requires at least one seed's report")
        pooled_families = {row.family for row in self.pooled.per_family}
        variation_families = {row.family for row in self.variation}
        if variation_families != pooled_families:
            raise ValueError(
                "a pooled accuracy figure must never be reported without the "
                "seed-to-seed variation behind it (issue #97): variation must "
                "cover exactly the pooled report's families (missing "
                f"{sorted(pooled_families - variation_families)}, extra "
                f"{sorted(variation_families - pooled_families)})"
            )


def evaluate_sealed_test_with_variation(
    corpora: Sequence[Corpus],
    pooled: Corpus,
    scorer: Scorer[Any],
    frozen: FrozenThreshold,
) -> SealedTestAccuracy | InvalidRun:
    """Evaluate ``scorer``'s sealed-test accuracy both pooled across
    ``corpora`` and per seed (ADR-0011 rule 3, issue #97), under the one
    ``frozen`` threshold every seed and every arm shares (ADR-0011 rule 2).

    ``pooled`` is evaluated first, and its outcome decides the whole call:
    if the threshold turns out to have read the sealed test
    (:func:`evaluate_sealed_test` returning :class:`InvalidRun`), that
    reason is returned as-is and no per-seed evaluation runs at all — a
    threshold the protocol has already rejected is not a threshold worth
    scoring five more times (ADR-0013). ``pooled`` is not required to be
    ``corpora`` pooled together by this function; it is whatever the caller
    already built for that purpose (``joinless.cli``'s own ``_pool_corpora``,
    used for threshold selection too), so pooling stays defined in exactly
    one place rather than duplicated here.
    """
    pooled_report = evaluate_sealed_test(pooled, scorer, frozen)
    if isinstance(pooled_report, InvalidRun):
        return pooled_report

    by_seed: dict[int, EvaluationReport] = {}
    for one_corpus in corpora:
        seed_report = evaluate_sealed_test(one_corpus, scorer, frozen)
        # evaluate_sealed_test's only failure mode is `frozen.role !=
        # "calibration"` (its own docstring) - a fact about `frozen`, not
        # about which corpus it scores. `pooled_report` above already proved
        # that check passes for this exact `frozen`, so every per-seed call
        # with the same `frozen` passes it too - not a second, independent
        # possibility this loop needs to guard against.
        assert isinstance(seed_report, EvaluationReport)
        by_seed[one_corpus.seed] = seed_report

    return SealedTestAccuracy(
        pooled=pooled_report,
        pooled_answers=_POOLED_ACCURACY_ANSWERS,
        by_seed=MappingProxyType(by_seed),
        variation=compute_family_variation(by_seed),
        by_seed_answers=_BY_SEED_ACCURACY_ANSWERS,
    )


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
    """One family where the pre-registered arm did not reach the top F1 score
    (ADR-0011 rule 4) — "a finding... not a footnote": a first-class value in a
    tuple :func:`find_contradictions` returns, never text folded into a log line.

    ``actual_winners`` holds every arm that reached that top score, sorted
    ascending by name. A single-element tuple means one arm beat the
    expectation outright; more than one means the arms that beat it also tied
    with each other — reported as the tie it is, rather than collapsed to
    whichever of them a mapping happens to iterate first.
    """

    family: str
    expected_winner: str
    actual_winners: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AccuracyDivergence:
    """How much one arm's per-family F1 differs from a baseline arm's,
    computed from each arm's own :class:`EvaluationReport` in the same run
    (ADR-0007: "quantization may hurt short strings differently from long
    ones" — issue #67's third bullet). Never estimated, and never a second
    evaluation pass: both F1 figures are read straight off the two reports
    :func:`compute_accuracy_divergence`'s caller already has.

    ``delta_f1`` is ``candidate_f1 - baseline_f1``, defined only when both
    inputs are — the same "undefined propagates, never collapses to zero"
    rule :func:`_f1` already applies to precision and recall (ADR-0013): a
    delta computed against an undefined F1 would be a comparison against
    nothing, not a real gap.
    """

    family: str
    baseline_f1: Metric
    candidate_f1: Metric
    delta_f1: Metric


def _delta_metric(baseline: Metric, candidate: Metric) -> Metric:
    """``candidate.value - baseline.value``, or undefined with a reason
    naming which side was undefined — mirrors :func:`_f1`'s own
    either-side-undefined handling, applied to a difference instead of a
    harmonic mean."""
    if baseline.value is None:
        return Metric(
            value=None,
            undefined_reason=f"baseline F1 is undefined: {baseline.undefined_reason}",
        )
    if candidate.value is None:
        return Metric(
            value=None,
            undefined_reason=f"candidate F1 is undefined: {candidate.undefined_reason}",
        )
    return Metric(value=candidate.value - baseline.value, undefined_reason=None)


def compute_accuracy_divergence(
    *, baseline: EvaluationReport, candidate: EvaluationReport
) -> tuple[AccuracyDivergence, ...]:
    """Per-family F1 divergence of ``candidate`` from ``baseline``, in
    ``baseline``'s own family order (issue #67's third bullet).

    Both reports come from the same pooled corpus (ADR-0011 rule 3), so
    ``candidate`` is expected to report every family ``baseline`` does. A
    family ``candidate`` did not report is not silently dropped from the
    table (ADR-0013): it is included with an explicit undefined
    ``candidate_f1`` naming the gap, so a reader sees that the comparison
    could not be drawn rather than seeing one fewer row with no explanation.
    """
    candidate_by_family = {row.family: row for row in candidate.per_family}
    divergences = []
    for baseline_row in baseline.per_family:
        candidate_row = candidate_by_family.get(baseline_row.family)
        if candidate_row is None:
            candidate_f1 = Metric(
                value=None,
                undefined_reason=(
                    f"{baseline_row.family!r} was not reported by the candidate arm"
                ),
            )
        else:
            candidate_f1 = candidate_row.f1
        divergences.append(
            AccuracyDivergence(
                family=baseline_row.family,
                baseline_f1=baseline_row.f1,
                candidate_f1=candidate_f1,
                delta_f1=_delta_metric(baseline_row.f1, candidate_f1),
            )
        )
    return tuple(divergences)


def find_contradictions(
    expected: ExpectedWinners, reports: Mapping[str, EvaluationReport]
) -> tuple[Contradiction, ...]:
    """Every family in ``expected`` where the pre-registered arm did not reach
    the top F1 score, among ``reports`` (issue #50), in the order ``expected``
    records them.

    A contradiction is the pre-registered arm failing to reach the top score for
    that family — however many arms are above it, and whether or not those arms
    tie with each other. The expected arm reaching the top, alone or tied with
    one or more other arms, is the expectation holding, not a case reported here.

    Comparing requires a real figure on both sides. An arm that did not report
    the family at all, or reported an undefined F1 there, contributes no
    comparable score — an undefined figure is not evidence the arm lost, only
    that no comparison can be drawn from it (ADR-0013). A family with fewer than
    two comparable arms is skipped entirely, and so is one where the
    pre-registered arm itself has no comparable score there: winning is only
    meaningful against a competitor, and against a figure the expected arm
    actually has.
    """
    contradictions: list[Contradiction] = []
    for family, expected_winner in expected.winners.items():
        scores: dict[str, float] = {}
        for arm, report in reports.items():
            row = next((f for f in report.per_family if f.family == family), None)
            if row is not None and row.f1.value is not None:
                scores[arm] = row.f1.value
        if len(scores) < 2 or expected_winner not in scores:
            continue
        best_score = max(scores.values())
        if scores[expected_winner] == best_score:
            continue
        actual_winners = tuple(
            sorted(arm for arm, score in scores.items() if score == best_score)
        )
        contradictions.append(
            Contradiction(
                family=family,
                expected_winner=expected_winner,
                actual_winners=actual_winners,
            )
        )
    return tuple(contradictions)
