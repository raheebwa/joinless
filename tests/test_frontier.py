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


def _family_row(family: str, f1: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": family,
        "precision": _metric(1.0),
        "recall": _metric(1.0),
        "f1": f1,
        "true_positives": 1,
        "predicted_positives": 1,
        "actual_positives": 1,
        "false_positives": 0,
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
    accuracy: dict[str, Any] | None = None,
    peak_rss_bytes: float = 1_000.0,
    peak_memory: dict[str, Any] | None = None,
    warm_p50_seconds: float = 0.001,
    warm_latency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if accuracy is None:
        families = families if families is not None else ["exact"]
        per_family = [
            _family_row(family, _metric(f1, f1_reason)) for family in families
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
            arm="overlap", f1=0.9, peak_rss_bytes=1_000.0, warm_p50_seconds=0.001
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


def test_undefined_f1_excludes_the_arm_even_with_no_stated_floor() -> None:
    record = _record({"overlap": _arm_result(f1=None, f1_reason="no actual positives")})

    result = compute_frontier(record, Constraints())

    family = result.per_family[0]
    assert isinstance(family.frontier, NoArmQualifies)


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
    better = FrontierPoint(arm="a", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1)
    worse = FrontierPoint(arm="b", f1=0.5, peak_rss_bytes=100.0, warm_p50_seconds=0.1)
    assert _dominates(better, worse) is True


def test_dominates_strictly_better_on_memory_alone() -> None:
    better = FrontierPoint(arm="a", f1=0.9, peak_rss_bytes=50.0, warm_p50_seconds=0.1)
    worse = FrontierPoint(arm="b", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1)
    assert _dominates(better, worse) is True


def test_dominates_strictly_better_on_latency_alone() -> None:
    better = FrontierPoint(arm="a", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.05)
    worse = FrontierPoint(arm="b", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1)
    assert _dominates(better, worse) is True


def test_does_not_dominate_when_worse_on_one_axis_despite_better_on_another() -> None:
    mixed = FrontierPoint(
        arm="a", f1=0.95, peak_rss_bytes=9_000.0, warm_p50_seconds=0.1
    )
    other = FrontierPoint(arm="b", f1=0.7, peak_rss_bytes=100.0, warm_p50_seconds=0.1)
    assert _dominates(mixed, other) is False


def test_does_not_dominate_an_identical_point() -> None:
    a = FrontierPoint(arm="a", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1)
    b = FrontierPoint(arm="b", f1=0.9, peak_rss_bytes=100.0, warm_p50_seconds=0.1)
    assert _dominates(a, b) is False


# --- basic shape sanity --------------------------------------------------------


def test_frontier_result_and_family_frontier_are_frozen_dataclasses() -> None:
    result = compute_frontier(_record({"overlap": _arm_result()}), _NO_CONSTRAINTS)
    assert isinstance(result, FrontierResult)
    assert isinstance(result.per_family[0], FamilyFrontier)
