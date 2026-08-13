# SPDX-License-Identifier: MIT
"""Per-family evaluation, threshold selection, and pre-registered expectations."""

from __future__ import annotations

import pytest

from joinless.evaluation import Metric


def test_a_defined_metric_carries_no_undefined_reason() -> None:
    metric = Metric(value=0.5, undefined_reason=None)
    assert metric.value == 0.5
    assert metric.undefined_reason is None


def test_an_undefined_metric_carries_a_reason_instead_of_a_value() -> None:
    metric = Metric(value=None, undefined_reason="no predicted positives")
    assert metric.value is None
    assert metric.undefined_reason == "no predicted positives"


def test_a_metric_cannot_carry_both_a_value_and_an_undefined_reason() -> None:
    with pytest.raises(ValueError, match="must be None exactly when"):
        Metric(value=0.5, undefined_reason="no predicted positives")


def test_a_metric_cannot_omit_both_a_value_and_an_undefined_reason() -> None:
    with pytest.raises(ValueError, match="must be None exactly when"):
        Metric(value=None, undefined_reason=None)


# --- evaluate(): per-family precision, recall, F1 (issue #48) ----------------------

from dataclasses import fields

from joinless.corpus import LabelledPair
from joinless.scoring import OverlapScorer, ThresholdMatcher


def _pair(pair_id: str, left: str, right: str, label: int, family: str) -> LabelledPair:
    return LabelledPair(
        pair_id=pair_id, left_name=left, right_name=right, label=label, category=family
    )


def test_evaluate_computes_precision_recall_f1_per_family() -> None:
    from joinless.evaluation import evaluate

    pairs = [
        _pair("1", "Acme Traders", "Acme Traders", 1, "exact"),
        _pair("2", "Acme Traders", "Zeta Motors", 0, "near-miss negative"),
    ]
    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.5)

    report = evaluate(pairs, matcher)

    families = {f.family: f for f in report.per_family}
    assert families["exact"].precision.value == 1.0
    assert families["exact"].recall.value == 1.0
    assert families["exact"].f1.value == 1.0
    assert families["near-miss negative"].precision.value is None
    assert families["near-miss negative"].recall.value is None


def test_evaluation_report_presents_per_family_before_aggregate() -> None:
    from joinless.evaluation import EvaluationReport

    names = [f.name for f in fields(EvaluationReport)]
    assert names == ["per_family", "aggregate", "n_pairs"]


def test_evaluate_records_n_pairs_as_the_batch_size_of_its_prepare_all_calls() -> None:
    """Issue #61's third bullet: "batch size is a recorded parameter, since it
    moves the result." ``evaluate`` calls ``matcher.scorer.prepare_all`` exactly
    twice - once with every pair's left name, once with every pair's right name
    (``_predict``) - so the length of ``pairs`` *is* the size of each of those
    two calls. ``n_pairs`` is that same count, not a second figure that could
    disagree with it.
    """
    from joinless.evaluation import evaluate

    pairs = [
        _pair("1", "Acme Traders", "Acme Traders", 1, "exact"),
        _pair("2", "Acme Traders", "Zeta Motors", 0, "near-miss negative"),
        _pair("3", "Acme Traders", "Acme Trading Co", 1, "abbreviation"),
    ]
    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.5)

    report = evaluate(pairs, matcher)

    assert report.n_pairs == 3


# --- undefined precision/recall (issue #51) -----------------------------------------


def test_undefined_precision_carries_the_reason_no_predicted_positives() -> None:
    from joinless.evaluation import evaluate

    pairs = [_pair("1", "Acme Traders", "Zeta Motors", 0, "near-miss negative")]
    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.99)

    report = evaluate(pairs, matcher)

    family = report.per_family[0]
    assert family.precision.value is None
    assert family.precision.undefined_reason == "no predicted positives"
    assert family.f1.value is None
    assert (
        family.f1.undefined_reason == "precision is undefined: no predicted positives"
    )


def test_undefined_recall_carries_the_reason_no_actual_positives() -> None:
    from joinless.evaluation import evaluate

    pairs = [_pair("1", "Acme Traders", "Acme Traders", 0, "near-miss negative")]
    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.0)

    report = evaluate(pairs, matcher)

    family = report.per_family[0]
    assert family.recall.value is None
    assert family.recall.undefined_reason == "no actual positives"
    assert family.f1.value is None
    assert family.f1.undefined_reason == "recall is undefined: no actual positives"


def test_undefined_precision_and_recall_are_reported_as_null_never_zero() -> None:
    """Pins ADR-0013's literal distinction: a zero and an undefined are different
    facts. This test would pass even if ``_ratio`` substituted ``0.0`` for the
    value while still setting a reason, so it asserts on ``value`` directly rather
    than only on ``undefined_reason``."""
    from joinless.evaluation import evaluate

    pairs = [_pair("1", "Acme Traders", "Zeta Motors", 0, "near-miss negative")]
    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.99)

    report = evaluate(pairs, matcher)

    assert report.per_family[0].precision.value is None
    assert report.per_family[0].precision.value != 0.0


def test_evaluate_rejects_an_empty_split() -> None:
    from joinless.evaluation import evaluate

    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.5)
    with pytest.raises(ValueError, match="at least one pair"):
        evaluate([], matcher)


def test_evaluate_rejects_a_pair_with_no_category() -> None:
    from joinless.evaluation import evaluate

    pairs = [
        LabelledPair(pair_id="1", left_name="A", right_name="A", label=1, category=None)
    ]
    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.5)
    with pytest.raises(ValueError, match="has no category"):
        evaluate(pairs, matcher)


def test_f1_is_undefined_when_precision_and_recall_are_both_defined_zero() -> None:
    from joinless.evaluation import evaluate

    pairs = [
        _pair("1", "Acme Traders", "Wholly Different Co", 1, "near-miss negative"),
        _pair("2", "Acme Traders", "Acme Traders", 0, "near-miss negative"),
    ]
    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.5)

    report = evaluate(pairs, matcher)

    family = report.per_family[0]
    assert family.precision.value == 0.0
    assert family.recall.value == 0.0
    assert family.f1.value is None
    assert family.f1.undefined_reason == "precision and recall are both zero"


# --- aggregate is derived, and moves with the mixture (issue #48) -------------------


def test_aggregate_result_names_what_it_is_derived_from() -> None:
    """Issue #48, third bullet: a consumer holding a bare ``AggregateResult``
    must be able to tell it apart from a ninth family — it is labelled as
    derived, not just positioned after the per-family rows. The literal text
    is pinned by content, for the same reason as the selection procedure
    above: a comparison against the constant that produces it cannot fail
    when the constant's content changes."""
    from joinless.evaluation import evaluate

    pairs = [_pair("1", "Acme Traders", "Acme Traders", 1, "exact")]
    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.5)

    report = evaluate(pairs, matcher)

    assert report.aggregate.derivation == (
        "sum true positives, predicted positives and actual positives across "
        "every family in the per-family table, then compute precision, "
        "recall and F1 from those pooled counts — never computed "
        "independently of the table"
    )


def test_aggregate_is_built_from_the_per_family_table_not_raw_pairs() -> None:
    """The aggregate function's own signature is the guarantee: it takes
    FamilyResult rows, not pairs, so there is no code path back to the raw data
    for it to independently disagree with the table it summarises."""
    from joinless.evaluation import FamilyResult, Metric, _aggregate

    family = FamilyResult(
        family="exact",
        precision=Metric(value=1.0, undefined_reason=None),
        recall=Metric(value=1.0, undefined_reason=None),
        f1=Metric(value=1.0, undefined_reason=None),
        true_positives=10,
        predicted_positives=10,
        actual_positives=10,
    )

    aggregate = _aggregate([family])

    assert aggregate.precision.value == 1.0
    assert aggregate.recall.value == 1.0
    assert aggregate.f1.value == 1.0


def test_changing_the_family_mixture_moves_the_aggregate_but_not_any_per_family_figure() -> (
    None
):
    from joinless.evaluation import evaluate

    matcher = ThresholdMatcher(scorer=OverlapScorer(), threshold=0.5)

    # "exact" always matches (precision=recall=1); "near-miss negative" always
    # produces one false positive out of one proposal (precision=0, recall
    # undefined - no actual positives in the family at all). Weighting the mixture
    # toward the negative family should drag the aggregate down while leaving both
    # families' own ratios exactly where they were.
    exact_pairs = [_pair("e1", "Acme Traders", "Acme Traders", 1, "exact")]
    negative_pair = _pair("n1", "Acme Traders", "Acme Traders", 0, "near-miss negative")

    light_mixture = evaluate([*exact_pairs, negative_pair], matcher)
    heavy_mixture = evaluate(
        [*exact_pairs, negative_pair, negative_pair, negative_pair], matcher
    )

    def _ratios(report, family):  # type: ignore[no-untyped-def]
        row = next(f for f in report.per_family if f.family == family)
        return (row.precision, row.recall, row.f1)

    assert _ratios(light_mixture, "exact") == _ratios(heavy_mixture, "exact")
    assert _ratios(light_mixture, "near-miss negative") == _ratios(
        heavy_mixture, "near-miss negative"
    )
    assert (
        light_mixture.aggregate.precision.value
        != heavy_mixture.aggregate.precision.value
    )


# --- threshold selection reads calibration data only (issue #49) -------------------

from types import MappingProxyType

from joinless.corpus import Corpus


def _corpus(pairs_by_role: dict) -> Corpus:  # type: ignore[type-arg]
    pairs = tuple(pair for pairs in pairs_by_role.values() for pair in pairs)
    roles = {
        pair.pair_id: role
        for role, role_pairs in pairs_by_role.items()
        for pair in role_pairs
    }
    return Corpus(seed=1, pairs=pairs, roles=MappingProxyType(roles))


def test_select_threshold_reads_calibration_pairs_only() -> None:
    from joinless.evaluation import select_threshold

    # Calibration pairs alone are only ever positive at overlap >= 1.0 (identical
    # names); the sealed test pairs below are near-miss negatives with a high
    # overlap score that would pull an optimal threshold down if they leaked in.
    calibration = [
        _pair("c1", "Acme Traders", "Acme Traders", 1, "exact"),
        _pair("c2", "Acme Traders", "Zeta Motors", 0, "near-miss negative"),
    ]
    sealed_test = [
        _pair("s1", "Acme Traders Ltd", "Acme Traders Ltd", 0, "near-miss negative"),
    ]
    corpus = _corpus({"calibration": calibration, "sealed test": sealed_test})

    selected = select_threshold(corpus, OverlapScorer())

    assert selected.role == "calibration"
    assert selected.n_pairs == len(calibration)


def test_select_threshold_takes_no_per_arm_parameters() -> None:
    """Pins the signature issue #49 requires: one documented procedure, applied
    identically regardless of which arm is being calibrated - callable with just
    the corpus and the scorer, nothing tuned per arm."""
    import inspect

    from joinless.evaluation import select_threshold

    signature = inspect.signature(select_threshold)
    assert list(signature.parameters) == ["corpus", "scorer"]


def test_select_threshold_records_the_procedure_and_the_selected_value() -> None:
    from joinless.evaluation import select_threshold

    calibration = [
        _pair("c1", "Acme Traders", "Acme Traders", 1, "exact"),
        _pair("c2", "Acme Traders", "Zeta Motors", 0, "near-miss negative"),
    ]
    corpus = _corpus({"calibration": calibration})

    selected = select_threshold(corpus, OverlapScorer())

    assert selected.scorer_name == "overlap"
    assert isinstance(selected.procedure, str) and selected.procedure
    assert 0.0 <= selected.value <= 1.0


def test_select_threshold_records_the_literal_selection_procedure_text() -> None:
    """Pins ADR-0011 rule 2's procedure text by content (issue #49, third
    bullet), not by comparison against the constant that produces it — a test
    that imports ``_SELECTION_PROCEDURE`` and asserts equality to itself can
    never fail when the constant's content drifts, only when it disappears."""
    from joinless.evaluation import select_threshold

    calibration = [
        _pair("c1", "Acme Traders", "Acme Traders", 1, "exact"),
        _pair("c2", "Acme Traders", "Zeta Motors", 0, "near-miss negative"),
    ]
    corpus = _corpus({"calibration": calibration})

    selected = select_threshold(corpus, OverlapScorer())

    assert selected.procedure == (
        "sweep every similarity score the scorer produces on the calibration "
        "split as a threshold candidate; keep whichever candidate maximises "
        "calibration F1, breaking a tie toward the lowest candidate that "
        "achieves it"
    )


def test_select_threshold_breaks_a_tie_toward_the_lower_candidate() -> None:
    from joinless.evaluation import select_threshold

    # Both pairs score 1.0 under OverlapScorer (identical token sets); every
    # candidate threshold at or below 1.0 achieves the same F1, so the tie-break
    # picks the lowest candidate the scorer actually produced.
    calibration = [
        _pair("c1", "Acme Traders", "Acme Traders", 1, "exact"),
        _pair("c2", "Zeta Motors", "Zeta Motors", 1, "exact"),
    ]
    corpus = _corpus({"calibration": calibration})

    selected = select_threshold(corpus, OverlapScorer())

    assert selected.value == 1.0


def test_select_threshold_requires_calibration_pairs() -> None:
    from joinless.evaluation import select_threshold

    corpus = _corpus({"development": [_pair("d1", "A", "A", 1, "exact")]})
    with pytest.raises(ValueError, match="no calibration pairs"):
        select_threshold(corpus, OverlapScorer())


class _FixedScorer:
    """A minimal :class:`~joinless.scoring.Scorer` whose score for a given
    prepared pair is set directly by the test, rather than computed — used only
    where the test needs exact, engineered similarity values that a real string
    metric cannot be made to hit precisely (ADR-0016 rule 2: this stands in for
    nothing complicated, so being a thin double costs nothing)."""

    def __init__(self, scores: dict) -> None:  # type: ignore[type-arg]
        self._scores = scores

    @property
    def name(self) -> str:
        return "fixed"

    def prepare_all(self, names):  # type: ignore[no-untyped-def]
        return list(names)

    def prepare(self, name):  # type: ignore[no-untyped-def]
        return name

    def score(self, a, b):  # type: ignore[no-untyped-def]
        return self._scores[(a, b)]


def test_select_threshold_keeps_the_best_candidate_seen_so_far() -> None:
    """The sweep is not "last candidate wins" - a later, worse-scoring candidate
    must not overwrite an earlier, better one."""
    from joinless.evaluation import select_threshold

    scores = {("P1L", "P1R"): 0.9, ("P2L", "P2R"): 0.5, ("NL", "NR"): 0.6}
    calibration = [
        _pair("p1", "P1L", "P1R", 1, "exact"),
        _pair("p2", "P2L", "P2R", 1, "exact"),
        _pair("n1", "NL", "NR", 0, "near-miss negative"),
    ]
    corpus = _corpus({"calibration": calibration})

    selected = select_threshold(corpus, _FixedScorer(scores))

    assert selected.value == 0.5


# --- freeze before the sealed test is read (issue #49, issue #51) ------------------


def test_evaluate_sealed_test_requires_a_frozen_threshold_type() -> None:
    """The type system carries the ordering (issue #49): evaluate_sealed_test's
    signature names FrozenThreshold, not SelectedThreshold, so passing a bare
    SelectedThreshold — one that has not gone through the freeze step — is a
    mypy error, not a review comment. This test pins the annotation itself,
    since a runtime test cannot observe a type error."""
    from typing import get_type_hints

    from joinless.evaluation import FrozenThreshold, evaluate_sealed_test

    hints = get_type_hints(evaluate_sealed_test)
    assert hints["frozen"] is FrozenThreshold


def test_freeze_threshold_produces_a_distinct_type_carrying_the_same_facts() -> None:
    from joinless.evaluation import SelectedThreshold, freeze_threshold

    selected = SelectedThreshold(
        scorer_name="overlap",
        value=0.7,
        procedure="a procedure",
        role="calibration",
        n_pairs=10,
    )

    frozen = freeze_threshold(selected)

    assert frozen.value == 0.7
    assert frozen.scorer_name == "overlap"
    assert frozen.role == "calibration"
    assert type(frozen) is not type(selected)


def test_evaluate_sealed_test_scores_only_the_sealed_test_split() -> None:
    from joinless.evaluation import (
        evaluate_sealed_test,
        freeze_threshold,
        select_threshold,
    )

    calibration = [
        _pair("c1", "Acme Traders", "Acme Traders", 1, "exact"),
        _pair("c2", "Acme Traders", "Zeta Motors", 0, "near-miss negative"),
    ]
    sealed_test = [
        _pair("s1", "Acme Traders", "Acme Traders", 1, "exact"),
    ]
    corpus = _corpus({"calibration": calibration, "sealed test": sealed_test})
    scorer = OverlapScorer()
    frozen = freeze_threshold(select_threshold(corpus, scorer))

    result = evaluate_sealed_test(corpus, scorer, frozen)

    from joinless.evaluation import EvaluationReport

    assert isinstance(result, EvaluationReport)
    assert [f.family for f in result.per_family] == ["exact"]


def test_evaluate_sealed_test_rejects_an_empty_sealed_split() -> None:
    from joinless.evaluation import (
        evaluate_sealed_test,
        freeze_threshold,
        select_threshold,
    )

    calibration = [_pair("c1", "Acme Traders", "Acme Traders", 1, "exact")]
    corpus = _corpus({"calibration": calibration})
    scorer = OverlapScorer()
    frozen = freeze_threshold(select_threshold(corpus, scorer))

    with pytest.raises(ValueError, match="no sealed test pairs"):
        evaluate_sealed_test(corpus, scorer, frozen)


def test_a_run_whose_threshold_saw_sealed_test_data_is_marked_invalid_not_a_number() -> (
    None
):
    """Issue #51: contaminated threshold selection is marked invalid and reports
    no metrics, never a warning next to a number that is still there to be
    quoted (ADR-0013)."""
    from joinless.evaluation import (
        InvalidRun,
        SelectedThreshold,
        evaluate_sealed_test,
        freeze_threshold,
    )

    calibration = [_pair("c1", "Acme Traders", "Acme Traders", 1, "exact")]
    sealed_test = [_pair("s1", "Acme Traders", "Acme Traders", 1, "exact")]
    corpus = _corpus({"calibration": calibration, "sealed test": sealed_test})
    scorer = OverlapScorer()

    contaminated = SelectedThreshold(
        scorer_name="overlap",
        value=0.5,
        procedure="a procedure",
        role="sealed test",
        n_pairs=1,
    )
    frozen = freeze_threshold(contaminated)

    result = evaluate_sealed_test(corpus, scorer, frozen)

    assert isinstance(result, InvalidRun)
    assert "sealed test" in result.reason
    assert not hasattr(result, "per_family")


# --- pre-registered expectations (issue #50) ----------------------------------------


def _report_with_f1(family: str, value: float):  # type: ignore[no-untyped-def]
    from joinless.evaluation import (
        _AGGREGATE_DERIVATION,
        AggregateResult,
        EvaluationReport,
        FamilyResult,
    )

    row = FamilyResult(
        family=family,
        precision=Metric(value=value, undefined_reason=None),
        recall=Metric(value=value, undefined_reason=None),
        f1=Metric(value=value, undefined_reason=None),
        true_positives=1,
        predicted_positives=1,
        actual_positives=1,
    )
    aggregate = AggregateResult(
        precision=row.precision,
        recall=row.recall,
        f1=row.f1,
        derivation=_AGGREGATE_DERIVATION,
    )
    return EvaluationReport(per_family=(row,), aggregate=aggregate, n_pairs=1)


def test_expected_winners_cannot_be_edited_by_reassigning_the_mapping() -> None:
    import dataclasses

    from joinless.evaluation import ExpectedWinners

    expected = ExpectedWinners(winners={"exact": "overlap"})

    with pytest.raises(dataclasses.FrozenInstanceError):
        expected.winners = {"exact": "fuzzy"}  # type: ignore[misc]


def test_expected_winners_cannot_be_edited_by_mutating_the_mapping_in_place() -> None:
    from joinless.evaluation import ExpectedWinners

    expected = ExpectedWinners(winners={"exact": "overlap"})

    with pytest.raises(TypeError):
        expected.winners["exact"] = "fuzzy"  # type: ignore[index]


def test_expected_winners_is_immune_to_the_caller_mutating_their_own_dict() -> None:
    from joinless.evaluation import ExpectedWinners

    source = {"exact": "overlap"}
    expected = ExpectedWinners(winners=source)
    source["exact"] = "fuzzy"

    assert expected.winners["exact"] == "overlap"


def test_find_contradictions_names_every_family_that_contradicts_the_expectation() -> (
    None
):
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(
        winners={"abbreviation": "overlap", "semantic alias": "fuzzy"}
    )
    reports = {
        "overlap": _report_with_f1("abbreviation", 0.5),
        "fuzzy": _report_with_f1("abbreviation", 0.9),
    }
    # Merge in a second family so both entries in `expected` have something to
    # compare against.
    from joinless.evaluation import EvaluationReport, FamilyResult

    reports = {
        "overlap": EvaluationReport(
            per_family=(
                reports["overlap"].per_family[0],
                FamilyResult(
                    family="semantic alias",
                    precision=Metric(value=0.9, undefined_reason=None),
                    recall=Metric(value=0.9, undefined_reason=None),
                    f1=Metric(value=0.9, undefined_reason=None),
                    true_positives=1,
                    predicted_positives=1,
                    actual_positives=1,
                ),
            ),
            aggregate=reports["overlap"].aggregate,
            n_pairs=2,
        ),
        "fuzzy": EvaluationReport(
            per_family=(
                reports["fuzzy"].per_family[0],
                FamilyResult(
                    family="semantic alias",
                    precision=Metric(value=0.2, undefined_reason=None),
                    recall=Metric(value=0.2, undefined_reason=None),
                    f1=Metric(value=0.2, undefined_reason=None),
                    true_positives=1,
                    predicted_positives=1,
                    actual_positives=1,
                ),
            ),
            aggregate=reports["fuzzy"].aggregate,
            n_pairs=2,
        ),
    }

    contradictions = find_contradictions(expected, reports)

    # "abbreviation" expected overlap to win, but fuzzy scored higher: contradicted.
    # "semantic alias" expected fuzzy to win, and overlap scored higher: contradicted.
    families = {c.family for c in contradictions}
    assert families == {"abbreviation", "semantic alias"}


def test_find_contradictions_confirms_a_family_that_matches_the_expectation() -> None:
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(winners={"abbreviation": "fuzzy"})
    reports = {
        "overlap": _report_with_f1("abbreviation", 0.3),
        "fuzzy": _report_with_f1("abbreviation", 0.9),
    }

    contradictions = find_contradictions(expected, reports)

    assert contradictions == ()


def test_find_contradictions_confirms_a_family_where_the_expected_arm_ties_for_top() -> (
    None
):
    """A three-way tie at the top, with the expected arm among the tied scorers,
    is a no-contradiction case for the same reason a clean win is: the expected
    arm reached the top score. Nothing here turns on whether other arms also
    reached it or on the order ``reports`` happens to iterate in — this pins that
    an implementation reading ``max(scores, key=...)`` for an "actual winner"
    cannot silently promote one tied arm over the expected one just because it
    sorts first.
    """
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(winners={"exact": "overlap"})
    reports = {
        "fuzzy": _report_with_f1("exact", 1.0),
        "overlap": _report_with_f1("exact", 1.0),
        "embed-fp32": _report_with_f1("exact", 1.0),
    }

    contradictions = find_contradictions(expected, reports)

    assert contradictions == ()


def test_find_contradictions_confirms_a_family_where_the_expected_arm_wins_alone() -> (
    None
):
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(winners={"exact": "overlap"})
    reports = {
        "overlap": _report_with_f1("exact", 1.0),
        "fuzzy": _report_with_f1("exact", 0.4),
    }

    contradictions = find_contradictions(expected, reports)

    assert contradictions == ()


def test_find_contradictions_confirms_a_family_where_the_expected_arm_ties_with_one_other() -> (
    None
):
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(winners={"exact": "overlap"})
    reports = {
        "overlap": _report_with_f1("exact", 0.9),
        "fuzzy": _report_with_f1("exact", 0.9),
    }

    contradictions = find_contradictions(expected, reports)

    assert contradictions == ()


def test_find_contradictions_reports_the_expected_arm_beaten_by_exactly_one_arm() -> (
    None
):
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(winners={"transliteration": "fuzzy"})
    reports = {
        "fuzzy": _report_with_f1("transliteration", 0.961),
        "overlap": _report_with_f1("transliteration", 1.0),
    }

    contradictions = find_contradictions(expected, reports)

    assert len(contradictions) == 1
    contradiction = contradictions[0]
    assert contradiction.family == "transliteration"
    assert contradiction.expected_winner == "fuzzy"
    assert contradiction.actual_winners == ("overlap",)


def test_find_contradictions_reports_the_expected_arm_beaten_by_two_arms_tied_with_each_other() -> (
    None
):
    """The live case this rule exists for: the expected arm is strictly beaten,
    and the two arms that beat it happen to tie with each other. Whether those
    two tie says nothing about whether the expectation held — it did not, and
    both of the arms that beat it are named, in a deterministic order, rather
    than the tie between them being read as "no contradiction."
    """
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(winners={"transliteration": "fuzzy"})
    reports = {
        "overlap": _report_with_f1("transliteration", 1.0),
        "fuzzy": _report_with_f1("transliteration", 0.961),
        "embed-fp32": _report_with_f1("transliteration", 1.0),
    }

    contradictions = find_contradictions(expected, reports)

    assert len(contradictions) == 1
    contradiction = contradictions[0]
    assert contradiction.family == "transliteration"
    assert contradiction.expected_winner == "fuzzy"
    assert contradiction.actual_winners == ("embed-fp32", "overlap")


def test_find_contradictions_skips_a_family_with_fewer_than_two_defined_scores() -> (
    None
):
    """A family only one arm reported on has nothing to compare - not a
    contradiction, and not a confirmation either."""
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(winners={"abbreviation": "overlap"})
    reports = {"overlap": _report_with_f1("abbreviation", 0.3)}

    contradictions = find_contradictions(expected, reports)

    assert contradictions == ()


def test_find_contradictions_skips_a_family_neither_arm_reported() -> None:
    from joinless.evaluation import ExpectedWinners, find_contradictions

    expected = ExpectedWinners(winners={"transliteration": "overlap"})
    reports = {
        "overlap": _report_with_f1("abbreviation", 0.3),
        "fuzzy": _report_with_f1("abbreviation", 0.9),
    }

    contradictions = find_contradictions(expected, reports)

    assert contradictions == ()


def test_find_contradictions_skips_a_family_where_the_expected_arm_has_no_comparable_figure() -> (
    None
):
    """Two other arms have a real, comparable score for this family, but the
    expected arm's own F1 is undefined there (ADR-0013) - an undefined figure is
    not evidence the expected arm lost, so there is nothing to compare it
    against and the family is skipped, the same as a family fewer than two arms
    reported on."""
    from joinless.evaluation import (
        EvaluationReport,
        ExpectedWinners,
        FamilyResult,
        find_contradictions,
    )

    expected = ExpectedWinners(winners={"abbreviation": "fuzzy"})
    undefined_metric = Metric(value=None, undefined_reason="no predicted positives")
    fuzzy_row = FamilyResult(
        family="abbreviation",
        precision=undefined_metric,
        recall=undefined_metric,
        f1=undefined_metric,
        true_positives=0,
        predicted_positives=0,
        actual_positives=1,
    )
    reports = {
        "overlap": _report_with_f1("abbreviation", 0.5),
        "embed-fp32": _report_with_f1("abbreviation", 0.9),
        "fuzzy": EvaluationReport(
            per_family=(fuzzy_row,),
            aggregate=_report_with_f1("abbreviation", 0.5).aggregate,
            n_pairs=1,
        ),
    }

    contradictions = find_contradictions(expected, reports)

    assert contradictions == ()


# --- accuracy divergence: per-family F1 delta from a baseline arm (issue #67) -------


def _multi_family_report(families: dict[str, float | None]):  # type: ignore[no-untyped-def]
    """A report carrying one :class:`~joinless.evaluation.FamilyResult` per
    ``families`` entry - ``None`` produces an undefined F1 (no predicted
    positives), a float produces a defined one with matching precision and
    recall, mirroring :func:`_report_with_f1` but for more than one family at
    once, which every divergence test below needs."""
    from joinless.evaluation import AggregateResult, EvaluationReport, FamilyResult

    rows = []
    for family, value in families.items():
        if value is None:
            metric = Metric(value=None, undefined_reason="no predicted positives")
            rows.append(
                FamilyResult(
                    family=family,
                    precision=metric,
                    recall=metric,
                    f1=metric,
                    true_positives=0,
                    predicted_positives=0,
                    actual_positives=1,
                )
            )
        else:
            metric = Metric(value=value, undefined_reason=None)
            rows.append(
                FamilyResult(
                    family=family,
                    precision=metric,
                    recall=metric,
                    f1=metric,
                    true_positives=1,
                    predicted_positives=1,
                    actual_positives=1,
                )
            )
    aggregate = AggregateResult(
        precision=rows[0].precision,
        recall=rows[0].recall,
        f1=rows[0].f1,
        derivation="pooled",
    )
    return EvaluationReport(
        per_family=tuple(rows), aggregate=aggregate, n_pairs=len(rows)
    )


def test_accuracy_divergence_reports_a_delta_per_family() -> None:
    from joinless.evaluation import compute_accuracy_divergence

    baseline = _multi_family_report({"exact": 1.0, "character noise": 0.8})
    candidate = _multi_family_report({"exact": 0.9, "character noise": 0.85})

    divergence = compute_accuracy_divergence(baseline=baseline, candidate=candidate)

    by_family = {row.family: row for row in divergence}
    assert by_family["exact"].baseline_f1.value == 1.0
    assert by_family["exact"].candidate_f1.value == 0.9
    assert by_family["exact"].delta_f1.value == pytest.approx(-0.1)
    assert by_family["character noise"].delta_f1.value == pytest.approx(0.05)


def test_accuracy_divergence_preserves_the_baselines_family_order() -> None:
    from joinless.evaluation import compute_accuracy_divergence

    baseline = _multi_family_report(
        {"transliteration": 0.7, "exact": 1.0, "abbreviation": 0.6}
    )
    candidate = _multi_family_report(
        {"exact": 0.9, "abbreviation": 0.5, "transliteration": 0.65}
    )

    divergence = compute_accuracy_divergence(baseline=baseline, candidate=candidate)

    assert [row.family for row in divergence] == [
        "transliteration",
        "exact",
        "abbreviation",
    ]


def test_accuracy_divergence_is_undefined_when_the_baseline_f1_is_undefined() -> None:
    from joinless.evaluation import compute_accuracy_divergence

    baseline = _multi_family_report({"exact": None})
    candidate = _multi_family_report({"exact": 0.9})

    [row] = compute_accuracy_divergence(baseline=baseline, candidate=candidate)

    assert row.delta_f1.value is None
    assert row.delta_f1.undefined_reason is not None
    assert "baseline" in row.delta_f1.undefined_reason


def test_accuracy_divergence_is_undefined_when_the_candidate_f1_is_undefined() -> None:
    from joinless.evaluation import compute_accuracy_divergence

    baseline = _multi_family_report({"exact": 1.0})
    candidate = _multi_family_report({"exact": None})

    [row] = compute_accuracy_divergence(baseline=baseline, candidate=candidate)

    assert row.delta_f1.value is None
    assert row.delta_f1.undefined_reason is not None
    assert "candidate" in row.delta_f1.undefined_reason


def test_accuracy_divergence_is_undefined_when_the_candidate_never_reported_the_family() -> (
    None
):
    """A family the baseline reports but the candidate does not is not silently
    skipped from the table (ADR-0013): it appears with an explicit reason,
    naming the gap rather than letting a missing family look like it was never
    checked."""
    from joinless.evaluation import compute_accuracy_divergence

    baseline = _multi_family_report({"exact": 1.0, "near-miss negative": 0.8})
    candidate = _multi_family_report({"exact": 0.9})

    divergence = compute_accuracy_divergence(baseline=baseline, candidate=candidate)

    by_family = {row.family: row for row in divergence}
    missing = by_family["near-miss negative"]
    assert missing.candidate_f1.value is None
    assert missing.candidate_f1.undefined_reason is not None
    assert "near-miss negative" in missing.candidate_f1.undefined_reason
    assert missing.delta_f1.value is None
