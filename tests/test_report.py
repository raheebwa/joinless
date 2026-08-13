# SPDX-License-Identifier: MIT
"""The per-family renderer (issue #71, RFC-0002 "Output").

``render_per_family_table`` takes a plain, JSON-shaped record — exactly the shape
:func:`joinless.runrecord.record_to_dict` produces and :func:`json.load` returns
when reading a written record back — and renders it. Every fixture below is built
by hand at that same shape rather than through the ``RunRecord`` dataclasses:
this module renders the record a reader actually has on disk, so a test that
exercises the JSON shape directly is closer to what ``joinless report`` really
reads than one that goes through the dataclasses and never touches JSON at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from joinless.report import render_per_family_table


def _metric(value: float | None, reason: str | None = None) -> dict[str, Any]:
    return {"value": value, "undefined_reason": reason}


def _family_result(
    family: str,
    *,
    precision: dict[str, Any],
    recall: dict[str, Any],
    f1: dict[str, Any],
    true_positives: int = 0,
    predicted_positives: int = 0,
    actual_positives: int = 0,
    false_positives: int = 0,
) -> dict[str, Any]:
    return {
        "family": family,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "predicted_positives": predicted_positives,
        "actual_positives": actual_positives,
        "false_positives": false_positives,
    }


_DERIVATION = "sum true positives... pooled counts"


def _ok_accuracy(per_family: list[dict[str, Any]]) -> dict[str, Any]:
    metric = _metric(1.0)
    aggregate = {
        "precision": metric,
        "recall": metric,
        "f1": metric,
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


def _unavailable_accuracy(reason: str) -> dict[str, Any]:
    return {"status": "invalid", "reason": reason}


def _one_family_record(
    families: list[str] | None = None,
    *,
    arms: dict[str, dict[str, Any]] | None = None,
    contradictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    families = families if families is not None else ["exact"]
    if arms is None:
        exact_metric = _metric(1.0)
        per_family = [
            _family_result(
                family,
                precision=exact_metric,
                recall=exact_metric,
                f1=exact_metric,
                true_positives=1,
                predicted_positives=1,
                actual_positives=1,
                false_positives=0,
            )
            for family in families
        ]
        arms = {"overlap": {"accuracy": _ok_accuracy(per_family)}}
    return {
        "schema": "benchmark-v7",
        "record_id": "20260813T000000Z-benchmark.json",
        "results": arms,
        "contradictions": contradictions or [],
    }


def test_renders_a_block_per_family_with_arms_down_and_metrics_across() -> None:
    record = _one_family_record(["exact", "formatting"])

    table = render_per_family_table(record)

    assert "family: exact" in table
    assert "family: formatting" in table
    assert "overlap" in table
    assert "precision" in table
    assert "recall" in table
    assert "f1" in table


def test_a_defined_metric_renders_its_value() -> None:
    metric = _metric(0.721)
    per_family = [
        _family_result(
            "exact",
            precision=metric,
            recall=metric,
            f1=metric,
            true_positives=1,
            predicted_positives=1,
            actual_positives=1,
        )
    ]
    record = _one_family_record(
        ["exact"], arms={"overlap": {"accuracy": _ok_accuracy(per_family)}}
    )

    table = render_per_family_table(record)

    assert "0.721" in table


def test_an_undefined_metric_renders_null_with_its_reason_never_zero_never_blank() -> (
    None
):
    undefined = _metric(None, "no predicted positives")
    per_family = [
        _family_result(
            "near-miss negative",
            precision=undefined,
            recall=undefined,
            f1=undefined,
            false_positives=115,
        )
    ]
    record = _one_family_record(
        ["near-miss negative"],
        arms={"overlap": {"accuracy": _ok_accuracy(per_family)}},
    )

    table = render_per_family_table(record)

    assert "null (no predicted positives)" in table
    # the false positive count itself is a real figure, never undefined
    assert "115" in table
    # never rendered as a bare 0 standing in for the undefined ratio
    lines = [line for line in table.splitlines() if "overlap" in line]
    assert not any(line.strip().endswith("0.000") for line in lines)


def test_a_long_undefined_reason_never_glues_onto_the_next_column() -> None:
    """A ``null (<reason>)`` cell can run past the fixed column width on its
    own (issue #71's own regression: an f1 reason wrapping precision's own
    reason text) — the next column must still be visibly separated, never
    concatenated straight onto the reason string with no gap at all."""
    long_reason_f1 = _metric(None, "precision is undefined: no predicted positives")
    per_family = [
        _family_result(
            "near-miss negative",
            precision=_metric(None, "no predicted positives"),
            recall=_metric(0.0),
            f1=long_reason_f1,
            false_positives=7,
        )
    ]
    record = _one_family_record(
        ["near-miss negative"],
        arms={"overlap": {"accuracy": _ok_accuracy(per_family)}},
    )

    table = render_per_family_table(record)

    row = next(
        line for line in table.splitlines() if line.strip().startswith("overlap")
    )
    assert ")7" not in row
    assert ") 7" in row or "  7" in row


def test_an_unavailable_arm_keeps_its_row_carrying_its_reason() -> None:
    metric = _metric(1.0)
    per_family = [
        _family_result(
            "exact", precision=metric, recall=metric, f1=metric, true_positives=1
        )
    ]
    record = _one_family_record(
        ["exact"],
        arms={
            "overlap": {"accuracy": _ok_accuracy(per_family)},
            "embed-fp32": {
                "accuracy": _unavailable_accuracy("JOINLESS_MODEL_CACHE_DIR is not set")
            },
        },
    )

    table = render_per_family_table(record)

    assert "embed-fp32" in table
    assert "JOINLESS_MODEL_CACHE_DIR is not set" in table


def test_a_contradicted_family_is_marked() -> None:
    record = _one_family_record(
        ["near-miss negative"],
        contradictions=[
            {
                "family": "near-miss negative",
                "expected_winner": "embed-fp32",
                "actual_winners": ["overlap"],
            }
        ],
    )

    table = render_per_family_table(record)

    family_block = table.split("family: near-miss negative", 1)[1]
    heading_line = family_block.splitlines()[0]
    assert "embed-fp32" in heading_line
    assert "overlap" in heading_line


def test_a_family_with_no_contradiction_carries_no_contradiction_marker() -> None:
    record = _one_family_record(["exact"], contradictions=[])

    table = render_per_family_table(record)

    heading_line = next(line for line in table.splitlines() if "family: exact" in line)
    assert "expected" not in heading_line


def test_the_aggregate_is_labelled_as_derived_from_the_per_family_figures() -> None:
    record = _one_family_record(["exact"])

    table = render_per_family_table(record)

    assert "derived" in table
    assert _DERIVATION in table


def test_the_aggregate_values_match_the_records_own_aggregate_figures() -> None:
    per_family_metric = _metric(1.0)
    per_family = [
        _family_result(
            "exact",
            precision=per_family_metric,
            recall=per_family_metric,
            f1=per_family_metric,
            true_positives=1,
            predicted_positives=1,
            actual_positives=1,
        )
    ]
    accuracy = _ok_accuracy(per_family)
    accuracy["pooled"]["aggregate"] = {
        "precision": _metric(0.635),
        "recall": _metric(0.833),
        "f1": _metric(0.721),
        "derivation": _DERIVATION,
    }
    record = _one_family_record(["exact"], arms={"overlap": {"accuracy": accuracy}})

    table = render_per_family_table(record)

    assert "0.635" in table
    assert "0.833" in table
    assert "0.721" in table


def test_an_unavailable_arm_also_keeps_its_row_in_the_aggregate_section() -> None:
    metric = _metric(1.0)
    per_family = [
        _family_result(
            "exact", precision=metric, recall=metric, f1=metric, true_positives=1
        )
    ]
    record = _one_family_record(
        ["exact"],
        arms={
            "overlap": {"accuracy": _ok_accuracy(per_family)},
            "embed-fp32": {"accuracy": _unavailable_accuracy("no artefact configured")},
        },
    )

    table = render_per_family_table(record)

    aggregate_block = table.rsplit("aggregate", 1)[1]
    assert "embed-fp32" in aggregate_block
    assert "no artefact configured" in aggregate_block


def test_a_record_where_every_arm_is_unavailable_still_names_every_arm() -> None:
    record = _one_family_record(
        [],
        arms={
            "overlap": {"accuracy": _unavailable_accuracy("reason one")},
            "fuzzy": {"accuracy": _unavailable_accuracy("reason two")},
        },
    )

    table = render_per_family_table(record)

    assert "overlap" in table
    assert "reason one" in table
    assert "fuzzy" in table
    assert "reason two" in table


def test_families_are_rendered_in_the_first_available_arms_own_family_order() -> None:
    metric = _metric(1.0)
    per_family = [
        _family_result(
            "abbreviation", precision=metric, recall=metric, f1=metric, true_positives=1
        ),
        _family_result(
            "exact", precision=metric, recall=metric, f1=metric, true_positives=1
        ),
    ]
    record = _one_family_record(
        arms={"overlap": {"accuracy": _ok_accuracy(per_family)}}
    )

    table = render_per_family_table(record)

    assert table.index("family: abbreviation") < table.index("family: exact")


def test_the_header_names_the_record_id_and_schema() -> None:
    record = _one_family_record(["exact"])

    table = render_per_family_table(record)

    assert record["record_id"] in table
    assert record["schema"] in table


# --- a record this build cannot render is refused, not crashed on -------------------


def test_rendering_a_record_from_an_older_schema_names_both_schemas() -> None:
    """RFC-0002: "`report` is a pure function of the run record." Three of the
    four records committed to ``benchmarks/`` were written by earlier schema
    versions and carry no ``pooled`` accuracy block, because per-seed reporting
    (issue #97) did not exist when they were made. Rendering one raised a bare
    ``KeyError: 'pooled'`` — a stack trace naming an internal dictionary key,
    for a file that is a perfectly valid record of the run that produced it.

    The refusal is correct; only its shape was wrong. A reader is told which
    schema the record carries and which one this build renders, so the answer to
    "why can I not read my own record" is in the message rather than in the
    source.
    """
    from joinless.report import UnsupportedSchema, render_per_family_table

    with pytest.raises(UnsupportedSchema) as excinfo:
        render_per_family_table({"schema": "benchmark-v4", "results": {}})

    message = str(excinfo.value)
    assert "benchmark-v4" in message
    assert "benchmark-v7" in message


def test_the_renderable_schema_is_the_one_the_benchmark_writes() -> None:
    """Two constants naming the same version would drift apart the first time
    one moved. Pinned against the writer, so a schema bump that forgets the
    renderer fails here rather than at a reader's terminal.
    """
    from joinless.cli import _SCHEMA
    from joinless.report import RENDERABLE_SCHEMA

    assert RENDERABLE_SCHEMA == _SCHEMA
