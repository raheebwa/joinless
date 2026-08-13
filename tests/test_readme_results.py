# SPDX-License-Identifier: MIT
"""The README results section renderer (issue #72, epic #69, ADR-0010).

``render_results_section`` takes the plain, JSON-shaped record
:func:`joinless.runrecord.record_to_dict` produces and :func:`json.load` reads
back — the same convention ``tests/test_report.py`` and ``tests/test_frontier.py``
use, and for the same reason: this module renders the record a reader actually
has on disk, not a second representation of it. Every fixture below is built by
hand at that shape.
"""

from __future__ import annotations

from typing import Any

import pytest

from joinless.readme_results import (
    MARKER_BEGIN,
    MARKER_END,
    MissingMarkers,
    render_results_section,
    splice_into_readme,
)
from joinless.report import UnsupportedSchema


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


def _ok_accuracy(
    per_family: list[dict[str, Any]], aggregate_f1: dict[str, Any] | None = None
) -> dict[str, Any]:
    aggregate = {
        "precision": _metric(1.0),
        "recall": _metric(1.0),
        "f1": aggregate_f1 if aggregate_f1 is not None else _metric(1.0),
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


def _peak_memory(peak_rss_bytes: float = 1_000_000.0) -> dict[str, Any]:
    return {
        "status": "ok",
        "arm": "some-arm",
        "peak_rss_bytes": peak_rss_bytes,
        "thread_count": 1,
        "power_mode": "ac",
    }


def _unavailable_peak_memory(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "arm": "some-arm", "reason": reason}


def _warm_latency(p50_seconds: float = 0.000001) -> dict[str, Any]:
    return {
        "status": "ok",
        "arm": "some-arm",
        "p50_seconds": p50_seconds,
        "p99_seconds": p50_seconds * 2,
        "warmup_count": 5,
        "repetition_count": 20,
        "scope": "score only",
    }


def _artifact_size(value: float | None, reason: str | None = None) -> dict[str, Any]:
    return _metric(value, reason)


def _arm_result(
    *,
    families: list[str] | None = None,
    f1: float | None = 1.0,
    f1_reason: str | None = None,
    aggregate_f1: float | None = 1.0,
    false_positives: int = 0,
    accuracy: dict[str, Any] | None = None,
    peak_rss_bytes: float = 1_000_000.0,
    peak_memory: dict[str, Any] | None = None,
    warm_p50_seconds: float = 0.000001,
    warm_latency: dict[str, Any] | None = None,
    artifact_size: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if accuracy is None:
        families = families if families is not None else ["exact"]
        per_family = [
            _family_row(family, _metric(f1, f1_reason), false_positives)
            for family in families
        ]
        accuracy = _ok_accuracy(per_family, _metric(aggregate_f1))
    return {
        "accuracy": accuracy,
        "peak_memory": peak_memory
        if peak_memory is not None
        else _peak_memory(peak_rss_bytes),
        "warm_latency": warm_latency
        if warm_latency is not None
        else _warm_latency(warm_p50_seconds),
        "artifact_size": artifact_size
        if artifact_size is not None
        else _artifact_size(None, "classical arms carry no model artifact"),
    }


_HARDWARE = {
    "cpu_count": 12,
    "machine": "arm64",
    "python_version": "3.14.5",
    "release": "25.5.0",
    "system": "Darwin",
    "total_memory_bytes": 34359738368,
}


def _record(
    results: dict[str, dict[str, Any]],
    *,
    record_id: str = "20260813T153000Z-benchmark.json",
    schema: str = "benchmark-v7",
    seeds: list[int] | None = None,
    case_mixture: dict[str, int] | None = None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "record_id": record_id,
        "results": results,
        "evaluation_set": {
            "seeds": seeds if seeds is not None else [1, 2, 3, 4, 5],
            "case_mixture": case_mixture if case_mixture is not None else {"exact": 1},
        },
        "environment": {"hardware": hardware if hardware is not None else _HARDWARE},
    }


def test_cites_the_record_it_was_generated_from() -> None:
    record = _record(
        {"overlap": _arm_result()}, record_id="20260813T153000Z-benchmark.json"
    )

    section = render_results_section(record)

    assert "benchmarks/20260813T153000Z-benchmark.json" in section


def test_names_the_reference_machine() -> None:
    record = _record({"overlap": _arm_result()})

    section = render_results_section(record)

    assert "Darwin" in section
    assert "25.5.0" in section
    assert "arm64" in section
    assert "12" in section
    assert "32.0 GiB" in section
    assert "3.14.5" in section


def test_names_the_corpus_and_seeds() -> None:
    record = _record({"overlap": _arm_result()}, seeds=[1, 2, 3, 4, 5])

    section = render_results_section(record)

    assert "1, 2, 3, 4, 5" in section


def test_scope_statement_matches_adr_0010s_claim_boundary() -> None:
    record = _record({"overlap": _arm_result()})

    section = render_results_section(record)

    assert "synthetic" in section
    assert "not a universal ranking" in section
    assert "classical/neural crossover" in section
    assert "ADR-0010" in section


def test_aggregate_table_reports_each_arms_f1() -> None:
    record = _record({"overlap": _arm_result(aggregate_f1=0.721)})

    section = render_results_section(record)

    assert "0.721" in section


def test_aggregate_table_marks_an_unavailable_arm_with_its_reason() -> None:
    record = _record(
        {
            "overlap": _arm_result(
                accuracy=_invalid_accuracy("threshold selection invalid")
            )
        }
    )

    section = render_results_section(record)

    assert "unavailable" in section
    assert "threshold selection invalid" in section


def test_per_family_table_reports_f1_and_false_positives() -> None:
    record = _record(
        {
            "overlap": _arm_result(
                families=["near-miss negative"], f1=None, false_positives=115
            )
        }
    )

    section = render_results_section(record)

    assert "near-miss negative" in section
    assert "115" in section


def test_per_family_frontier_lists_non_dominated_arms() -> None:
    record = _record(
        {
            "overlap": _arm_result(
                families=["exact"],
                peak_rss_bytes=1_000.0,
                warm_p50_seconds=0.001,
            ),
            "fuzzy": _arm_result(
                families=["exact"],
                peak_rss_bytes=2_000.0,
                warm_p50_seconds=0.002,
            ),
        }
    )

    section = render_results_section(record)

    assert "`overlap`" in section
    assert "dominated by 'overlap'" in section


def test_per_family_no_arm_qualifies_is_rendered() -> None:
    # Accuracy is "ok" (so the family is still discoverable at all — module
    # docstring: family order comes from an "ok" arm's own report), but the
    # arm's peak memory is unavailable, so it has no candidate to place on
    # the frontier and the family's only arm is excluded outright.
    record = _record(
        {
            "overlap": _arm_result(
                families=["near-miss negative"],
                peak_memory=_unavailable_peak_memory("worker crashed"),
            )
        }
    )

    section = render_results_section(record)

    assert "No arm qualifies" in section


def test_artifact_size_undefined_reason_is_rendered_not_a_number() -> None:
    record = _record(
        {
            "overlap": _arm_result(
                artifact_size=_artifact_size(
                    None, "classical arms carry no model artifact"
                )
            )
        }
    )

    section = render_results_section(record)

    assert "classical arms carry no model artifact" in section


def test_artifact_size_is_reported_in_mb() -> None:
    record = _record(
        {"embed-fp32": _arm_result(artifact_size=_artifact_size(91_074_052.0))}
    )

    section = render_results_section(record)

    assert "91.1 MB" in section


def test_aggregate_table_marks_unavailable_warm_latency_with_its_reason() -> None:
    record = _record(
        {
            "embed-fp32": _arm_result(
                warm_latency={
                    "status": "unavailable",
                    "arm": "embed-fp32",
                    "reason": "no wheel for this platform",
                }
            )
        }
    )

    section = render_results_section(record)

    assert "unavailable — no wheel for this platform" in section


def test_per_family_table_marks_an_unavailable_arm_with_its_reason() -> None:
    record = _record(
        {
            "overlap": _arm_result(families=["exact"]),
            "embed-fp32": _arm_result(
                families=["exact"],
                accuracy=_invalid_accuracy(
                    "threshold selection touched the sealed test"
                ),
            ),
        }
    )

    section = render_results_section(record)

    assert "unavailable — threshold selection touched the sealed test" in section


def test_unsupported_schema_is_refused() -> None:
    record = _record({"overlap": _arm_result()}, schema="benchmark-v5")

    with pytest.raises(UnsupportedSchema):
        render_results_section(record)


# --- splicing the generated section into README.md's own text -----------------


def test_splice_replaces_content_between_the_markers() -> None:
    readme = f"# joinless\n\nintro\n\n{MARKER_BEGIN}\nstale content\n{MARKER_END}\n\nfooter\n"

    result = splice_into_readme(readme, "fresh content")

    assert "stale content" not in result
    assert "fresh content" in result
    assert result.startswith("# joinless\n\nintro\n\n")
    assert result.endswith("\n\nfooter\n")


def test_splice_is_idempotent() -> None:
    readme = f"# joinless\n\n{MARKER_BEGIN}\nstale\n{MARKER_END}\n"

    once = splice_into_readme(readme, "content")
    twice = splice_into_readme(once, "content")

    assert once == twice


def test_splice_raises_when_markers_are_missing() -> None:
    readme = "# joinless\n\nno markers here\n"

    with pytest.raises(MissingMarkers):
        splice_into_readme(readme, "content")
