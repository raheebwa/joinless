# SPDX-License-Identifier: MIT
"""The accuracy-cost frontier under stated constraints (issue #70, RFC-0002
"Decision output", issue #96).

Every fixture below is built at the plain, JSON-shaped record
:func:`joinless.runrecord.record_to_dict` produces and :func:`json.load` reads
back — the same convention ``tests/test_report.py`` uses, and for the same
reason: this module renders/computes over the record a reader actually has on
disk, not a second representation of it.
"""

from __future__ import annotations

from typing import Any

from joinless.frontier import (
    Constraints,
    FamilyFrontier,
    FrontierPoint,
    FrontierResult,
    NoArmQualifies,
    _dominates,
    compute_frontier,
)


def _metric(value: float | None, reason: str | None = None) -> dict[str, Any]:
    return {"value": value, "undefined_reason": reason}


def _family_row(
    family: str, f1: dict[str, Any], false_positives: int = 0
) -> dict[str, Any]:
    return {
        "family": family,
        "precision": _metric(1.0),
        "recall": _metric(1.0),
        "f1": f1,
        "true_positives": 1,
        "predicted_positives": 1,
        "actual_positives": 1,
        "false_positives": false_positives,
    }


_DERIVATION = "sum true positives... pooled counts"


def _ok_accuracy(per_family: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {
        "precision": _metric(1.0),
        "recall": _metric(1.0),
        "f1": _metric(1.0),
        "derivation": _DERIVATION,
    }
    pooled = {"per_family": per_family, "aggregate": aggregate, "n_pairs": 1}
    return {
        "status": "ok",
        "pooled": pooled,
        "pooled_answers": "pooled answers",
        "by_seed": {},
        "variation": [],
        "by_seed_answers": "by-seed answers",
    }


def _invalid_accuracy(reason: str) -> dict[str, Any]:
    return {"status": "invalid", "reason": reason}


def _unavailable_accuracy(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "arm": "some-arm", "reason": reason}


def _peak_memory(peak_rss_bytes: float) -> dict[str, Any]:
    return {
        "status": "ok",
        "arm": "some-arm",
        "peak_rss_bytes": peak_rss_bytes,
        "thread_count": 1,
        "power_mode": "ac",
    }


def _unavailable_peak_memory(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "arm": "some-arm", "reason": reason}


def _warm_latency(p50_seconds: float) -> dict[str, Any]:
    return {
        "status": "ok",
        "arm": "some-arm",
        "p50_seconds": p50_seconds,
        "p99_seconds": p50_seconds * 2,
        "warmup_count": 5,
        "repetition_count": 20,
        "scope": "score only",
    }


def _unavailable_warm_latency(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "arm": "some-arm", "reason": reason}


def _arm_result(
    *,
    families: list[str] | None = None,
    f1: float | None = 1.0,
    f1_reason: str | None = None,
    false_positives: int = 0,
    accuracy: dict[str, Any] | None = None,
    peak_rss_bytes: float = 1_000.0,
    peak_memory: dict[str, Any] | None = None,
    warm_p50_seconds: float = 0.001,
    warm_latency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if accuracy is None:
        families = families if families is not None else ["exact"]
        per_family = [
            _family_row(family, _metric(f1, f1_reason), false_positives)
            for family in families
        ]
        accuracy = _ok_accuracy(per_family)
    return {
        "accuracy": accuracy,
        "peak_memory": peak_memory
        if peak_memory is not None
        else _peak_memory(peak_rss_bytes),
        "warm_latency": warm_latency
        if warm_latency is not None
        else _warm_latency(warm_p50_seconds),
    }


def _record(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"results": results}


_NO_CONSTRAINTS = Constraints()


# --- computed per family, from a record, under caller-supplied constraints ---


def test_the_frontier_is_computed_per_family() -> None:
    record = _record({"overlap": _arm_result(families=["exact", "character noise"])})

    result = compute_frontier(record, _NO_CONSTRAINTS)

    assert [f.family for f in result.per_family] == ["exact", "character noise"]


def test_family_order_follows_the_records_own_per_family_order() -> None:
    record = _record(
        {"overlap": _arm_result(families=["word order", "abbreviation", "exact"])}
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    assert [f.family for f in result.per_family] == [
        "word order",
        "abbreviation",
        "exact",
    ]


def test_an_arm_meeting_every_constraint_appears_on_the_frontier() -> None:
    record = _record({"overlap": _arm_result(f1=0.9)})
    constraints = Constraints(
        max_peak_rss_bytes=10_000.0, max_warm_p50_seconds=1.0, min_f1=0.5
    )

    result = compute_frontier(record, constraints)

    frontier = result.per_family[0].frontier
    assert isinstance(frontier, tuple)
    assert frontier == (
        FrontierPoint(
            arm="overlap",
            f1=0.9,
            peak_rss_bytes=1_000.0,
            warm_p50_seconds=0.001,
            false_positives=0,
        ),
    )
    assert result.per_family[0].excluded == {}


def test_no_stated_constraint_never_excludes_an_arm() -> None:
    record = _record(
        {"overlap": _arm_result(f1=0.1, peak_rss_bytes=1e12, warm_p50_seconds=1e6)}
    )

    result = compute_frontier(record, Constraints())

    frontier = result.per_family[0].frontier
    assert isinstance(frontier, tuple)
    assert len(frontier) == 1


# --- constraints record with the output, for reproducibility -----------------


def test_constraints_are_recorded_with_the_output() -> None:
    record = _record({"overlap": _arm_result()})
    constraints = Constraints(
        max_peak_rss_bytes=5.0, max_warm_p50_seconds=6.0, min_f1=0.7
    )

    result = compute_frontier(record, constraints)

    assert result.constraints is constraints
    assert result.constraints.max_peak_rss_bytes == 5.0
    assert result.constraints.max_warm_p50_seconds == 6.0
    assert result.constraints.min_f1 == 0.7


# --- accuracy floor ------------------------------------------------------------


def test_an_arm_below_the_accuracy_floor_is_excluded_with_a_reason() -> None:
    record = _record({"overlap": _arm_result(f1=0.4)})
    constraints = Constraints(min_f1=0.5)

    result = compute_frontier(record, constraints)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert "below the stated floor" in family.excluded["overlap"]
    assert "0.400" in family.excluded["overlap"]
    assert "0.500" in family.excluded["overlap"]


def test_an_arm_at_exactly_the_accuracy_floor_qualifies() -> None:
    record = _record({"overlap": _arm_result(f1=0.5)})
    constraints = Constraints(min_f1=0.5)

    result = compute_frontier(record, constraints)

    frontier = result.per_family[0].frontier
    assert isinstance(frontier, tuple)
    assert len(frontier) == 1


def test_undefined_f1_excludes_the_arm_never_treated_as_zero_or_a_pass() -> None:
    record = _record({"overlap": _arm_result(f1=None, f1_reason="no actual positives")})
    constraints = Constraints(min_f1=0.0)

    result = compute_frontier(record, constraints)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert "F1 is undefined" in family.excluded["overlap"]
    assert "no actual positives" in family.excluded["overlap"]


def test_undefined_f1_places_the_arm_on_false_positives_when_no_floor_is_stated() -> (
    None
):
    """Issue #106: undefined F1 still excludes an arm from a stated accuracy
    floor (the test above) — it is not a valid comparison against a floor.
    But with no floor stated, the arm is not excluded for lacking F1 at all;
    the false-positives and cost axes still place it, so a family where F1 is
    undefined for every arm (``near-miss negative``, ``semantic alias``) no
    longer collapses to "no arm qualifies" for want of an axis."""
    record = _record(
        {
            "overlap": _arm_result(
                f1=None, f1_reason="no actual positives", false_positives=3
            )
        }
    )

    result = compute_frontier(record, Constraints())

    family = result.per_family[0]
    assert isinstance(family.frontier, tuple)
    assert family.frontier == (
        FrontierPoint(
            arm="overlap",
            f1=None,
            peak_rss_bytes=1_000.0,
            warm_p50_seconds=0.001,
            false_positives=3,
        ),
    )


# --- memory ceiling --------------------------------------------------------------


def test_an_arm_over_the_memory_ceiling_is_excluded_with_a_reason() -> None:
    record = _record({"overlap": _arm_result(peak_rss_bytes=2_000.0)})
    constraints = Constraints(max_peak_rss_bytes=1_000.0)

    result = compute_frontier(record, constraints)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert "peak RSS" in family.excluded["overlap"]
    assert "exceeds the stated ceiling" in family.excluded["overlap"]


def test_an_arm_at_exactly_the_memory_ceiling_qualifies() -> None:
    record = _record({"overlap": _arm_result(peak_rss_bytes=1_000.0)})
    constraints = Constraints(max_peak_rss_bytes=1_000.0)

    result = compute_frontier(record, constraints)

    frontier = result.per_family[0].frontier
    assert isinstance(frontier, tuple)
    assert len(frontier) == 1


# --- latency ceiling ---------------------------------------------------------


def test_an_arm_over_the_latency_ceiling_is_excluded_with_a_reason() -> None:
    record = _record({"overlap": _arm_result(warm_p50_seconds=2.0)})
    constraints = Constraints(max_warm_p50_seconds=1.0)

    result = compute_frontier(record, constraints)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert "warm p50" in family.excluded["overlap"]
    assert "exceeds the stated ceiling" in family.excluded["overlap"]


def test_an_arm_at_exactly_the_latency_ceiling_qualifies() -> None:
    record = _record({"overlap": _arm_result(warm_p50_seconds=1.0)})
    constraints = Constraints(max_warm_p50_seconds=1.0)

    result = compute_frontier(record, constraints)

    frontier = result.per_family[0].frontier
    assert isinstance(frontier, tuple)
    assert len(frontier) == 1


# --- false-positives ceiling (issue #106) -----------------------------------


def test_an_arm_over_the_false_positives_ceiling_is_excluded_with_a_reason() -> None:
    record = _record({"overlap": _arm_result(false_positives=10)})
    constraints = Constraints(max_false_positives=5)

    result = compute_frontier(record, constraints)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert "false positives" in family.excluded["overlap"]
    assert "exceeds the stated ceiling" in family.excluded["overlap"]


def test_an_arm_at_exactly_the_false_positives_ceiling_qualifies() -> None:
    record = _record({"overlap": _arm_result(false_positives=5)})
    constraints = Constraints(max_false_positives=5)

    result = compute_frontier(record, constraints)

    frontier = result.per_family[0].frontier
    assert isinstance(frontier, tuple)
    assert len(frontier) == 1


def test_false_positives_ceiling_is_recorded_with_the_output() -> None:
    record = _record({"overlap": _arm_result()})
    constraints = Constraints(max_false_positives=7)

    result = compute_frontier(record, constraints)

    assert result.constraints.max_false_positives == 7


# --- unavailable / invalid arms are excluded, never silently dropped --------


def test_an_invalid_accuracy_excludes_the_arm_with_its_reason() -> None:
    record = _record(
        {
            "overlap": _arm_result(
                accuracy=_invalid_accuracy("threshold selection read the sealed test")
            )
        }
    )
    # No other arm reports a family at all, so per_family is empty and there is
    # nothing to assert a frontier over directly — pair with a second, "ok" arm
    # so the family exists and the invalid arm's exclusion is visible.
    record["results"]["fuzzy"] = _arm_result()

    result = compute_frontier(record, _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert "threshold selection read the sealed test" in family.excluded["overlap"]


def test_an_unavailable_accuracy_excludes_the_arm_with_its_reason() -> None:
    record = _record(
        {
            "embed-fp32": _arm_result(
                accuracy=_unavailable_accuracy("JOINLESS_MODEL_CACHE_DIR is not set")
            ),
            "overlap": _arm_result(),
        }
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert "JOINLESS_MODEL_CACHE_DIR is not set" in family.excluded["embed-fp32"]


def test_unavailable_peak_memory_excludes_the_arm_with_its_reason() -> None:
    record = _record(
        {"overlap": _arm_result(peak_memory=_unavailable_peak_memory("worker crashed"))}
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert "worker crashed" in family.excluded["overlap"]
    assert "peak memory" in family.excluded["overlap"]


def test_unavailable_warm_latency_excludes_the_arm_with_its_reason() -> None:
    record = _record(
        {
            "overlap": _arm_result(
                warm_latency=_unavailable_warm_latency("worker crashed")
            )
        }
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert "worker crashed" in family.excluded["overlap"]
    assert "warm latency" in family.excluded["overlap"]


# --- "no arm qualifies" is a first-class result, not an empty table --------


def test_no_arm_qualifies_when_the_constraint_set_excludes_every_arm() -> None:
    record = _record(
        {
            "overlap": _arm_result(f1=0.4),
            "fuzzy": _arm_result(f1=0.3),
        }
    )
    constraints = Constraints(min_f1=0.5)

    result = compute_frontier(record, constraints)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert "overlap" in family.frontier.reason
    assert "fuzzy" in family.frontier.reason
    assert set(family.excluded) == {"overlap", "fuzzy"}


def test_no_arm_qualifies_is_a_distinct_type_not_an_empty_tuple() -> None:
    record = _record({"overlap": _arm_result(f1=0.1)})
    constraints = Constraints(min_f1=0.9)

    result = compute_frontier(record, constraints)

    frontier = result.per_family[0].frontier
    assert not isinstance(frontier, tuple)
    assert isinstance(frontier, NoArmQualifies)


def _all_negative_family_record() -> dict[str, Any]:
    """Four arms on an all-negative family (``near-miss negative`` /
    ``semantic alias``, :mod:`joinless.corpus`'s module docstring): F1 is
    undefined for every one of them, by design, always — and the four false-
    positive counts are the one figure that tells them apart (issue #106)."""
    return _record(
        {
            "overlap": _arm_result(
                f1=None,
                f1_reason="no actual positives",
                false_positives=115,
                peak_rss_bytes=22.4,
                warm_p50_seconds=0.21,
            ),
            "fuzzy": _arm_result(
                f1=None,
                f1_reason="no actual positives",
                false_positives=5,
                peak_rss_bytes=24.2,
                warm_p50_seconds=1.08,
            ),
            "embed-fp32": _arm_result(
                f1=None,
                f1_reason="no actual positives",
                false_positives=0,
                peak_rss_bytes=273.9,
                warm_p50_seconds=24.3,
            ),
            "embed-int8": _arm_result(
                f1=None,
                f1_reason="no actual positives",
                false_positives=1,
                peak_rss_bytes=208.4,
                warm_p50_seconds=25.0,
            ),
        }
    )


def test_an_all_negative_family_places_every_arm_under_no_constraints() -> None:
    """Issue #106's motivating case: under no constraints, a family where F1
    is undefined for every arm no longer reports "no arm qualifies" — each
    arm trades cost for false positives against every other, so none
    dominates and all four remain on the frontier."""
    result = compute_frontier(_all_negative_family_record(), _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert isinstance(family.frontier, tuple)
    assert {p.arm for p in family.frontier} == {
        "overlap",
        "fuzzy",
        "embed-fp32",
        "embed-int8",
    }
    assert family.excluded == {}


def test_a_stated_accuracy_floor_still_excludes_every_arm_on_an_all_negative_family() -> (
    None
):
    """The other half of issue #106's fourth bullet: "no arm qualifies" must
    still mean exactly that when the reader states a floor no arm can be
    compared against — undefined F1 cannot be measured against a floor
    (RFC-0002), so any stated floor at all excludes every arm here, exactly
    as it did before the false-positives axis existed."""
    constraints = Constraints(min_f1=0.0)

    result = compute_frontier(_all_negative_family_record(), constraints)

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)
    assert set(family.excluded) == {"overlap", "fuzzy", "embed-fp32", "embed-int8"}
    for arm in family.excluded:
        assert "F1 is undefined" in family.excluded[arm]


# --- no generic winner row: dominance, ties and genuine trade-offs ----------


def test_a_dominated_arm_is_excluded_and_names_its_dominator() -> None:
    record = _record(
        {
            "overlap": _arm_result(
                f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.001
            ),
            "fuzzy": _arm_result(f1=0.5, peak_rss_bytes=200.0, warm_p50_seconds=0.002),
        }
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert isinstance(family.frontier, tuple)
    assert [p.arm for p in family.frontier] == ["overlap"]
    assert "dominated by 'overlap'" in family.excluded["fuzzy"]


def test_two_arms_on_a_genuine_tradeoff_both_stay_on_the_frontier() -> None:
    """Neither the cheaper, less accurate arm nor the pricier, more accurate
    one dominates the other — both belong on the frontier, and neither is
    picked as a single winner (RFC-0002 "Decision output")."""
    record = _record(
        {
            "overlap": _arm_result(
                f1=0.7, peak_rss_bytes=100.0, warm_p50_seconds=0.001
            ),
            "embed-fp32": _arm_result(
                f1=0.95, peak_rss_bytes=9_000.0, warm_p50_seconds=0.01
            ),
        }
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert isinstance(family.frontier, tuple)
    assert {p.arm for p in family.frontier} == {"overlap", "embed-fp32"}
    assert family.excluded == {}


def test_arms_tied_on_every_dimension_do_not_dominate_each_other() -> None:
    record = _record(
        {
            "overlap": _arm_result(
                f1=0.8, peak_rss_bytes=500.0, warm_p50_seconds=0.005
            ),
            "fuzzy": _arm_result(f1=0.8, peak_rss_bytes=500.0, warm_p50_seconds=0.005),
        }
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert isinstance(family.frontier, tuple)
    assert {p.arm for p in family.frontier} == {"overlap", "fuzzy"}


def test_false_positives_is_zero_for_every_arm_on_an_all_positive_family_and_stays_inert() -> (
    None
):
    """Issue #106's second bullet: the false-positives axis is not switched
    on only where F1 is undefined. Here every arm reports the same formula
    (``predicted_positives - true_positives``), which happens to be ``0`` for
    every arm on an all-positive family — the axis is fully active, it
    simply contributes nothing, and domination is still decided by F1 and
    cost exactly as it was before this axis existed."""
    record = _record(
        {
            "overlap": _arm_result(
                f1=0.9, false_positives=0, peak_rss_bytes=100.0, warm_p50_seconds=0.001
            ),
            "fuzzy": _arm_result(
                f1=0.5, false_positives=0, peak_rss_bytes=200.0, warm_p50_seconds=0.002
            ),
        }
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    family = result.per_family[0]
    assert isinstance(family.frontier, tuple)
    assert [p.arm for p in family.frontier] == ["overlap"]
    assert "dominated by 'overlap'" in family.excluded["fuzzy"]


def test_an_arm_that_did_not_report_the_family_at_all_is_excluded_with_a_reason() -> (
    None
):
    record = _record(
        {
            "overlap": _arm_result(families=["exact", "character noise"]),
            "fuzzy": _arm_result(families=["exact"]),
        }
    )

    result = compute_frontier(record, _NO_CONSTRAINTS)

    character_noise = next(
        f for f in result.per_family if f.family == "character noise"
    )
    assert "was not reported" in character_noise.excluded["fuzzy"]


def test_no_arm_in_the_record_yields_an_empty_per_family_tuple() -> None:
    record = _record({})

    result = compute_frontier(record, _NO_CONSTRAINTS)

    assert result.per_family == ()


# --- _dominates: every branch (module-private, exercised directly) ---------


def test_dominates_strictly_better_on_f1_alone() -> None:
    better = FrontierPoint(
        arm="a", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    worse = FrontierPoint(
        arm="b", f1=0.5, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    assert _dominates(better, worse) is True


def test_dominates_strictly_better_on_memory_alone() -> None:
    better = FrontierPoint(
        arm="a", f1=0.9, peak_rss_bytes=50.0, warm_p50_seconds=0.1, false_positives=0
    )
    worse = FrontierPoint(
        arm="b", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    assert _dominates(better, worse) is True


def test_dominates_strictly_better_on_latency_alone() -> None:
    better = FrontierPoint(
        arm="a", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.05, false_positives=0
    )
    worse = FrontierPoint(
        arm="b", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    assert _dominates(better, worse) is True


def test_dominates_strictly_better_on_false_positives_alone() -> None:
    better = FrontierPoint(
        arm="a", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=1
    )
    worse = FrontierPoint(
        arm="b", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=5
    )
    assert _dominates(better, worse) is True


def test_does_not_dominate_when_worse_on_one_axis_despite_better_on_another() -> None:
    mixed = FrontierPoint(
        arm="a",
        f1=0.95,
        peak_rss_bytes=9_000.0,
        warm_p50_seconds=0.1,
        false_positives=0,
    )
    other = FrontierPoint(
        arm="b", f1=0.7, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    assert _dominates(mixed, other) is False


def test_does_not_dominate_an_identical_point() -> None:
    a = FrontierPoint(
        arm="a", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    b = FrontierPoint(
        arm="b", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    assert _dominates(a, b) is False


def test_dominates_ignores_f1_when_compare_f1_is_false() -> None:
    """Issue #106: within a family where F1 is undefined for every candidate,
    F1 plays no part in domination — it is not fabricated into "equal" or
    "better" from a value that does not exist, it is simply not one of the
    axes compared. Here both points carry ``f1=None``; the caller (never a
    per-pair guess) states plainly that F1 is not an axis for this
    comparison, and the arm with fewer false positives still dominates."""
    fewer_false_positives = FrontierPoint(
        arm="a", f1=None, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=1
    )
    more_false_positives = FrontierPoint(
        arm="b", f1=None, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=5
    )
    assert (
        _dominates(fewer_false_positives, more_false_positives, compare_f1=False)
        is True
    )


def test_dominates_compares_f1_by_default() -> None:
    better_f1 = FrontierPoint(
        arm="a", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    worse_f1 = FrontierPoint(
        arm="b", f1=0.5, peak_rss_bytes=100.0, warm_p50_seconds=0.1, false_positives=0
    )
    assert _dominates(better_f1, worse_f1) is True
    assert _dominates(better_f1, worse_f1, compare_f1=True) is True


# --- basic shape sanity --------------------------------------------------------


def test_frontier_result_and_family_frontier_are_frozen_dataclasses() -> None:
    result = compute_frontier(_record({"overlap": _arm_result()}), _NO_CONSTRAINTS)
    assert isinstance(result, FrontierResult)
    assert isinstance(result.per_family[0], FamilyFrontier)


def test_a_domination_reason_reads_in_the_units_the_tables_use() -> None:
    """The frontier's own reasons are published verbatim in the README's
    generated results section, beside tables that print ``22.7 MB`` and
    ``0.21µs``. Printing the same two quantities as ``22675456 bytes`` and
    ``0.000000208s`` in the sentence directly under those tables asks a reader
    to convert between two renderings of one figure to check they agree.

    Pinned by content: comparing against the formatter that produces it cannot
    see a change to what it produces.
    """
    from joinless.frontier import FrontierPoint, _describe

    point = FrontierPoint(
        arm="overlap",
        f1=1.0,
        false_positives=0,
        peak_rss_bytes=22675456,
        warm_p50_seconds=0.000000208,
    )

    assert _describe(point, compare_f1=True) == (
        "f1=1.000, false_positives=0, peak RSS=22.7 MB, warm p50=0.21µs"
    )
