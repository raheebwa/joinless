# SPDX-License-Identifier: MIT
"""Resource measurement, isolated per arm per metric (RFC-0002 Method step 7,
ADR-0013, ADR-0014, issues #51, #53, #54, #55, #56)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from joinless.measurement import ArtifactRequirement, verify_artifact


def test_verify_artifact_returns_none_when_the_checksum_matches(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()

    reason = verify_artifact(ArtifactRequirement(path=artifact, sha256=digest))

    assert reason is None


def test_verify_artifact_reports_a_reason_when_the_file_is_missing(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "missing.onnx"

    reason = verify_artifact(ArtifactRequirement(path=artifact, sha256="deadbeef"))

    assert reason == f"artifact missing at {artifact}"


def test_verify_artifact_reports_a_reason_when_the_checksum_does_not_match(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"weights")
    wrong_digest = hashlib.sha256(b"different weights").hexdigest()

    reason = verify_artifact(ArtifactRequirement(path=artifact, sha256=wrong_digest))

    assert reason == (
        f"artifact checksum mismatch at {artifact}: "
        f"expected {wrong_digest}, got {hashlib.sha256(b'weights').hexdigest()}"
    )


# --- _run_in_child(): the one isolation mechanism every metric shares (issue #53) --

import pytest

from joinless.measurement import _ChildFailed, _run_in_child


def test_run_in_child_returns_the_parsed_json_when_the_worker_succeeds() -> None:
    payload = _run_in_child("import json; print(json.dumps({'a': 1}))", {})

    assert payload == {"a": 1}


def test_run_in_child_passes_params_to_the_worker_via_environment() -> None:
    script = (
        "import json, os; "
        'print(json.dumps(json.loads(os.environ["JOINLESS_MEASURE_PARAMS"])))'
    )

    payload = _run_in_child(script, {"arm": "overlap", "n": 3})

    assert payload == {"arm": "overlap", "n": 3}


def test_run_in_child_raises_child_failed_when_the_worker_exits_non_zero() -> None:
    with pytest.raises(_ChildFailed, match="boom"):
        _run_in_child("import sys; sys.stderr.write('boom'); sys.exit(1)", {})


def test_run_in_child_raises_child_failed_when_the_worker_prints_no_json() -> None:
    with pytest.raises(_ChildFailed, match="no parseable output"):
        _run_in_child("print('not json')", {})


# --- _percentile(): the one percentile convention every repeated-sample metric in
# this module shares (RFC-0002 Method step 3), factored out so
# ``_PREPARATION_COST_SCRIPT`` computes p50/p99 the same way ``_WARM_LATENCY_SCRIPT``
# already does rather than defining the formula a second time (issue #103) --------

from joinless.measurement import _percentile


def test_percentile_p50_of_an_odd_length_series_is_its_middle_value() -> None:
    assert _percentile([3.0, 1.0, 2.0], 0.50) == 2.0


def test_percentile_p99_of_a_single_value_is_that_value() -> None:
    assert _percentile([0.5], 0.99) == 0.5


def test_percentile_clamps_to_the_last_value_rather_than_indexing_past_it() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.99) == 4.0


# --- measure_warm_latency(): p50/p99 per comparison, per arm (issue #54) -----------

from joinless.measurement import Unavailable, measure_warm_latency


def test_measure_warm_latency_reports_p50_p99_and_counts_for_a_real_arm() -> None:
    result = measure_warm_latency(
        "overlap", "Acme Traders", "Acme Trading Co", warmup_count=2, repetition_count=5
    )

    assert not isinstance(result, Unavailable)
    assert result.arm == "overlap"
    assert result.warmup_count == 2
    assert result.repetition_count == 5
    assert result.p50_seconds >= 0.0
    assert result.p99_seconds >= result.p50_seconds


def test_measure_warm_latency_is_unavailable_for_an_arm_that_cannot_initialise() -> (
    None
):
    result = measure_warm_latency(
        "embed-fp32",
        "Acme Traders",
        "Acme Trading Co",
        warmup_count=0,
        repetition_count=1,
    )

    assert isinstance(result, Unavailable)
    assert result.arm == "embed-fp32"
    assert "embed-fp32" in result.reason


def test_measure_warm_latency_rejects_a_repetition_count_below_one() -> None:
    with pytest.raises(ValueError, match="repetition_count"):
        measure_warm_latency(
            "overlap",
            "Acme Traders",
            "Acme Trading Co",
            warmup_count=0,
            repetition_count=0,
        )


def test_measure_warm_latency_rejects_a_negative_warmup_count() -> None:
    with pytest.raises(ValueError, match="warmup_count"):
        measure_warm_latency(
            "overlap",
            "Acme Traders",
            "Acme Trading Co",
            warmup_count=-1,
            repetition_count=1,
        )


def test_measure_warm_latency_refuses_to_run_when_the_artifact_is_missing_or_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0013's fourth fail-closed rule (issue #51): a missing or
    checksum-mismatched artefact refuses to run rather than fetching one. The
    worker is never spawned — ``subprocess.run`` would raise if it were."""
    monkeypatch.setattr(
        "joinless.measurement.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not spawn a worker")
        ),
    )
    from joinless.measurement import ArtifactRequirement

    missing = ArtifactRequirement(path=tmp_path / "missing.onnx", sha256="deadbeef")

    result = measure_warm_latency(
        "embed-fp32",
        "Acme Traders",
        "Acme Trading Co",
        warmup_count=0,
        repetition_count=1,
        artifact=missing,
    )

    assert isinstance(result, Unavailable)
    assert result.reason == f"artifact missing at {missing.path}"


def test_measure_warm_latency_runs_when_the_artifact_matches(tmp_path: Path) -> None:
    from joinless.measurement import ArtifactRequirement

    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"weights")
    matching = ArtifactRequirement(
        path=artifact, sha256=hashlib.sha256(b"weights").hexdigest()
    )

    result = measure_warm_latency(
        "overlap",
        "Acme Traders",
        "Acme Trading Co",
        warmup_count=0,
        repetition_count=1,
        artifact=matching,
    )

    assert not isinstance(result, Unavailable)


# --- measure_preparation_cost(): hoisted vs naive, per arm, isolated (issue #66) ---

from joinless.measurement import measure_preparation_cost


def test_measure_preparation_cost_reports_percentiles_and_counts_for_a_real_arm() -> (
    None
):
    """Issue #103: a single draw is replaced by repeated, warmed-up sampling
    over both paths, reported the same way ``measure_warm_latency`` reports
    warm scoring — p50 and p99, with ``warmup_count``/``repetition_count``
    travelling on the value so a reader holding it alone knows how it was
    made."""
    result = measure_preparation_cost(
        "overlap",
        ["Acme Traders", "Rocket Fuel Traders"],
        ["Acme Traders Ltd", "Zephyr Logistics"],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        warmup_count=2,
        repetition_count=5,
    )

    assert not isinstance(result, Unavailable)
    assert result.arm == "overlap"
    assert result.warmup_count == 2
    assert result.repetition_count == 5
    assert result.hoisted_p50_seconds >= 0.0
    assert result.hoisted_p99_seconds >= result.hoisted_p50_seconds
    assert result.naive_p50_seconds >= 0.0
    assert result.naive_p99_seconds >= result.naive_p50_seconds
    assert result.record_count == 4
    assert result.comparison_count == 4


def test_measure_preparation_cost_is_unavailable_for_an_arm_that_cannot_initialise() -> (
    None
):
    result = measure_preparation_cost(
        "embed-fp32",
        ["Acme Traders"],
        ["Acme Traders Ltd"],
        [(0, 0)],
        warmup_count=0,
        repetition_count=1,
    )

    assert isinstance(result, Unavailable)
    assert result.arm == "embed-fp32"
    assert "embed-fp32" in result.reason


def test_measure_preparation_cost_rejects_empty_left_names() -> None:
    with pytest.raises(ValueError, match="left_names"):
        measure_preparation_cost(
            "overlap",
            [],
            ["Acme Traders Ltd"],
            [(0, 0)],
            warmup_count=0,
            repetition_count=1,
        )


def test_measure_preparation_cost_rejects_empty_right_names() -> None:
    with pytest.raises(ValueError, match="right_names"):
        measure_preparation_cost(
            "overlap",
            ["Acme Traders"],
            [],
            [(0, 0)],
            warmup_count=0,
            repetition_count=1,
        )


def test_measure_preparation_cost_rejects_empty_comparison_pairs() -> None:
    with pytest.raises(ValueError, match="comparison_pairs"):
        measure_preparation_cost(
            "overlap",
            ["Acme Traders"],
            ["Acme Traders Ltd"],
            [],
            warmup_count=0,
            repetition_count=1,
        )


def test_measure_preparation_cost_rejects_a_repetition_count_below_one() -> None:
    with pytest.raises(ValueError, match="repetition_count"):
        measure_preparation_cost(
            "overlap",
            ["Acme Traders"],
            ["Acme Traders Ltd"],
            [(0, 0)],
            warmup_count=0,
            repetition_count=0,
        )


def test_measure_preparation_cost_rejects_a_negative_warmup_count() -> None:
    with pytest.raises(ValueError, match="warmup_count"):
        measure_preparation_cost(
            "overlap",
            ["Acme Traders"],
            ["Acme Traders Ltd"],
            [(0, 0)],
            warmup_count=-1,
            repetition_count=1,
        )


def test_measure_preparation_cost_refuses_to_run_when_the_artifact_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from joinless.measurement import ArtifactRequirement

    monkeypatch.setattr(
        "joinless.measurement.verify_artifact",
        lambda requirement: f"artifact missing at {requirement.path}",
    )
    missing = ArtifactRequirement(path=tmp_path / "missing.onnx", sha256="0" * 64)

    result = measure_preparation_cost(
        "overlap",
        ["Acme Traders"],
        ["Acme Traders Ltd"],
        [(0, 0)],
        warmup_count=0,
        repetition_count=1,
        artifact=missing,
    )

    assert isinstance(result, Unavailable)
    assert "missing" in result.reason


def test_measure_preparation_cost_reports_an_unexpected_crash_as_unavailable_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr("joinless.measurement.subprocess.run", _fake_run)

    result = measure_preparation_cost(
        "overlap",
        ["Acme Traders"],
        ["Acme Traders Ltd"],
        [(0, 0)],
        warmup_count=0,
        repetition_count=1,
    )

    assert isinstance(result, Unavailable)
    assert result.reason == "boom"


# --- _preparation_costs(): the isolated worker's own timing routine, proven to be
# the shipped resolver.prepare_hoisted/prepare_naive rather than a second copy of
# them (issue #100) --------------------------------------------------------------

from joinless import resolver
from joinless.measurement import _preparation_costs
from joinless.scoring import OverlapScorer


def test_preparation_costs_hoisted_phase_measures_through_resolver_prepare_hoisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that reimplemented the hoist itself would never reach
    ``resolver.prepare_hoisted`` at all — this monkeypatch would sit unused
    and ``calls`` would stay empty. That is the failure this test exists to
    catch (issue #100's second bullet: a test must fail on divergence, not a
    comment asserting agreement)."""
    calls: list[tuple[list[str | None], list[str | None]]] = []

    def _fake_prepare_hoisted(
        scorer: object, left: object, right: object
    ) -> tuple[dict[int, object], dict[int, object]]:
        del scorer
        calls.append(
            (
                [record.name for record in left],  # type: ignore[attr-defined]
                [record.name for record in right],  # type: ignore[attr-defined]
            )
        )
        return {}, {}

    monkeypatch.setattr(resolver, "prepare_hoisted", _fake_prepare_hoisted)

    hoisted_durations, _naive_durations = _preparation_costs(
        OverlapScorer(),
        ["Acme Traders", "Rocket Fuel Traders"],
        ["Acme Traders Ltd", "Zephyr Logistics"],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        warmup_count=0,
        repetition_count=1,
    )

    assert calls == [
        (
            ["Acme Traders", "Rocket Fuel Traders"],
            ["Acme Traders Ltd", "Zephyr Logistics"],
        )
    ]
    assert len(hoisted_durations) == 1
    assert hoisted_durations[0] >= 0.0


def test_preparation_costs_naive_phase_measures_through_resolver_prepare_naive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same proof as the hoisted test above, for the naive phase: a worker
    that reimplemented the naive loop itself would never reach
    ``resolver.prepare_naive``, and ``calls`` would stay empty."""
    calls: list[list[tuple[str | None, str | None]]] = []

    def _fake_prepare_naive(
        scorer: object, pairs: object
    ) -> list[tuple[object, object]]:
        del scorer
        calls.append([(left.name, right.name) for left, right in pairs])  # type: ignore[attr-defined]
        return []

    monkeypatch.setattr(resolver, "prepare_naive", _fake_prepare_naive)

    _hoisted_durations, naive_durations = _preparation_costs(
        OverlapScorer(),
        ["Acme Traders", "Rocket Fuel Traders"],
        ["Acme Traders Ltd", "Zephyr Logistics"],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        warmup_count=0,
        repetition_count=1,
    )

    assert calls == [
        [
            ("Acme Traders", "Acme Traders Ltd"),
            ("Acme Traders", "Zephyr Logistics"),
            ("Rocket Fuel Traders", "Acme Traders Ltd"),
            ("Rocket Fuel Traders", "Zephyr Logistics"),
        ]
    ]
    assert len(naive_durations) == 1
    assert naive_durations[0] >= 0.0


class _CountingScorer:
    """Counts calls made directly on this object, mirroring
    ``tests/test_resolver.py``'s own ``_CallCountingScorer`` (issue #65) —
    the same technique, applied here to prove ``_preparation_costs``
    reproduces ``score_candidates``'s own documented call pattern
    end-to-end, with no monkeypatch involved."""

    def __init__(self, inner: OverlapScorer) -> None:
        self._inner = inner
        self.prepare_calls = 0
        self.prepare_all_calls = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def prepare_all(self, names: Sequence[str | None]) -> list[frozenset[str]]:
        self.prepare_all_calls += 1
        return self._inner.prepare_all(names)

    def prepare(self, name: str | None) -> frozenset[str]:
        self.prepare_calls += 1
        return self._inner.prepare(name)

    def score(self, a: frozenset[str], b: frozenset[str]) -> float:
        return self._inner.score(a, b)


def test_preparation_costs_reproduces_the_documented_call_pattern() -> None:
    """``PreparationCost``'s own docstring: hoisted times ``prepare_all``
    called once per side; naive times ``prepare`` called fresh for both
    records of every comparison. Both are repeated ``warmup_count +
    repetition_count`` times each (issue #103) — one full pass per call,
    discarded for the first ``warmup_count`` and timed for the rest — so a
    scorer sees exactly that many calls of each shape, with no separate
    pre-warm call of its own."""
    counting = _CountingScorer(OverlapScorer())

    _preparation_costs(
        counting,
        ["Acme Traders", "Rocket Fuel Traders"],
        ["Acme Traders Ltd", "Zephyr Logistics"],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        warmup_count=1,
        repetition_count=2,
    )

    assert counting.prepare_all_calls == 2 * (1 + 2)
    assert counting.prepare_calls == 2 * 4 * (1 + 2)


def test_preparation_costs_discards_warmup_and_reports_one_duration_per_repetition() -> (
    None
):
    """``warmup_count`` runs of each path are timed and thrown away;
    ``repetition_count`` runs of each path are timed and kept — the same
    discard-then-collect discipline ``measure_warm_latency`` already
    applies to ``score`` (issue #103)."""
    hoisted_durations, naive_durations = _preparation_costs(
        OverlapScorer(),
        ["Acme Traders", "Rocket Fuel Traders"],
        ["Acme Traders Ltd", "Zephyr Logistics"],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        warmup_count=3,
        repetition_count=7,
    )

    assert len(hoisted_durations) == 7
    assert len(naive_durations) == 7
    assert all(duration >= 0.0 for duration in hoisted_durations)
    assert all(duration >= 0.0 for duration in naive_durations)


# --- measure_peak_memory(): peak RSS per arm, in its own process (issue #55) -------

from joinless.measurement import _PEAK_MEMORY_SCRIPT, measure_peak_memory


def test_measure_peak_memory_reports_a_positive_rss() -> None:
    result = measure_peak_memory(
        "overlap", "Acme Traders", "Acme Trading Co", power_mode="plugged"
    )

    assert not isinstance(result, Unavailable)
    assert result.arm == "overlap"
    assert result.peak_rss_bytes > 0


def test_measure_peak_memory_reports_the_thread_count_the_worker_actually_had() -> None:
    """``thread_count >= 1`` can never fail — a thread count is never below
    one — so it holds even for a value the worker never actually observed,
    including a hard-coded constant that ignores the process entirely
    (issue #55). This test ties the reported figure to something only the
    real worker can produce: three extra live threads are started in the
    child *before* ``_PEAK_MEMORY_SCRIPT``'s own ``threading.active_count()``
    call runs, so a correct worker must report one (the main thread) plus
    those three — a number a hard-coded return value cannot reliably match.
    """
    extra_thread_count = 3
    params_literal = json.dumps(
        {
            "arm": "overlap",
            "left_name": "Acme Traders",
            "right_name": "Acme Trading Co",
        }
    )
    probe = (
        "import json, os, threading\n"
        f'os.environ["JOINLESS_MEASURE_PARAMS"] = {params_literal!r}\n'
        "_hold = threading.Event()\n"
        f"_extra = [threading.Thread(target=_hold.wait, daemon=True) "
        f"for _ in range({extra_thread_count})]\n"
        "for _t in _extra:\n"
        "    _t.start()\n" + _PEAK_MEMORY_SCRIPT
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["thread_count"] == 1 + extra_thread_count


def test_measure_peak_memory_records_the_supplied_power_mode() -> None:
    result = measure_peak_memory(
        "overlap", "Acme Traders", "Acme Trading Co", power_mode="battery"
    )

    assert not isinstance(result, Unavailable)
    assert result.power_mode == "battery"


def test_measure_peak_memory_is_unavailable_for_an_arm_that_cannot_initialise() -> None:
    result = measure_peak_memory(
        "embed-int8", "Acme Traders", "Acme Trading Co", power_mode="plugged"
    )

    assert isinstance(result, Unavailable)
    assert result.arm == "embed-int8"
    assert "embed-int8" in result.reason


# --- measure_cold_start(): five phases, reported separately (issue #56) -----------

from joinless.measurement import (
    _NO_SESSION_REASON,
    _NO_TOKENIZER_REASON,
    measure_cold_start,
)


def test_measure_cold_start_reports_interpreter_start_and_import_as_defined() -> None:
    result = measure_cold_start("overlap", "Acme Traders", "Acme Trading Co")

    assert not isinstance(result, Unavailable)
    assert result.arm == "overlap"
    assert result.interpreter_start.value is not None
    assert result.interpreter_start.value >= 0.0
    assert result.import_phase.value is not None
    assert result.import_phase.value >= 0.0
    assert result.first_inference.value is not None
    assert result.first_inference.value >= 0.0


def test_measure_cold_start_reports_session_creation_as_undefined_for_a_classical_arm() -> (
    None
):
    result = measure_cold_start("overlap", "Acme Traders", "Acme Trading Co")

    assert not isinstance(result, Unavailable)
    assert result.session_creation.value is None
    assert (
        result.session_creation.undefined_reason
        == "classical arms construct no session"
    )


def test_measure_cold_start_reports_tokenizer_load_as_undefined_for_a_classical_arm() -> (
    None
):
    result = measure_cold_start("overlap", "Acme Traders", "Acme Trading Co")

    assert not isinstance(result, Unavailable)
    assert result.tokenizer_load.value is None
    assert result.tokenizer_load.undefined_reason == "classical arms load no tokenizer"


def test_measure_cold_start_marks_interpreter_start_as_not_attributable() -> None:
    result = measure_cold_start("overlap", "Acme Traders", "Acme Trading Co")

    assert not isinstance(result, Unavailable)
    assert result.not_attributable == frozenset({"interpreter start"})
    assert "import" not in result.not_attributable


def test_measure_cold_start_total_sums_only_the_defined_phases() -> None:
    result = measure_cold_start("overlap", "Acme Traders", "Acme Trading Co")

    assert not isinstance(result, Unavailable)
    assert result.interpreter_start.value is not None
    assert result.import_phase.value is not None
    assert result.first_inference.value is not None
    expected = (
        result.interpreter_start.value
        + result.import_phase.value
        + result.first_inference.value
    )
    assert result.total.value == expected


def _fake_cold_start_run(
    *, session_creation_seconds: float, tokenizer_load_seconds: float
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A worker payload shaped like a neural arm's real cold-start report — real
    numbers for the two phases a classical arm never carries — so
    ``measure_cold_start``'s own branching can be proven without a model artefact
    on disk (issue #108: the fake worker stands in for onnxruntime/tokenizers
    being installed and a real model being cached, neither of which this test
    needs to make its point about the parent function's branching).
    """

    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        payload = json.dumps(
            {
                "status": "ok",
                "child_start_epoch": time.time(),
                "import_seconds": 0.05,
                "session_creation_seconds": session_creation_seconds,
                "tokenizer_load_seconds": tokenizer_load_seconds,
                "first_inference_seconds": 0.01,
            }
        )
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=payload, stderr=""
        )

    return _fake_run


def test_measure_cold_start_reports_session_creation_and_tokenizer_load_as_defined_for_a_neural_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "joinless.measurement.subprocess.run",
        _fake_cold_start_run(session_creation_seconds=0.2, tokenizer_load_seconds=0.03),
    )

    result = measure_cold_start("embed-fp32", "Acme Traders", "Acme Trading Co")

    assert not isinstance(result, Unavailable)
    assert result.session_creation.value == 0.2
    assert result.session_creation.undefined_reason is None
    assert result.tokenizer_load.value == 0.03
    assert result.tokenizer_load.undefined_reason is None


def test_measure_cold_start_a_neural_arm_never_carries_the_classical_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Done-when bullet 3 (issue #108): a neural arm's phases must not read
    "classical arms construct no session" / "classical arms load no tokenizer"
    once construction actually reports a duration."""
    monkeypatch.setattr(
        "joinless.measurement.subprocess.run",
        _fake_cold_start_run(session_creation_seconds=0.2, tokenizer_load_seconds=0.03),
    )

    result = measure_cold_start("embed-int8", "Acme Traders", "Acme Trading Co")

    assert not isinstance(result, Unavailable)
    assert result.session_creation.undefined_reason != _NO_SESSION_REASON
    assert result.tokenizer_load.undefined_reason != _NO_TOKENIZER_REASON


def test_measure_cold_start_total_includes_session_creation_and_tokenizer_load_for_a_neural_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Done-when bullet 4 (issue #108): total no longer drops the model-load cost."""
    monkeypatch.setattr(
        "joinless.measurement.subprocess.run",
        _fake_cold_start_run(session_creation_seconds=0.2, tokenizer_load_seconds=0.03),
    )

    result = measure_cold_start("embed-fp32", "Acme Traders", "Acme Trading Co")

    assert not isinstance(result, Unavailable)
    assert result.interpreter_start.value is not None
    assert result.import_phase.value is not None
    assert result.session_creation.value is not None
    assert result.tokenizer_load.value is not None
    assert result.first_inference.value is not None
    expected = (
        result.interpreter_start.value
        + result.import_phase.value
        + result.session_creation.value
        + result.tokenizer_load.value
        + result.first_inference.value
    )
    assert result.total.value == expected


def test_measure_cold_start_is_unavailable_for_an_arm_that_cannot_initialise() -> None:
    result = measure_cold_start("embed-fp32", "Acme Traders", "Acme Trading Co")

    assert isinstance(result, Unavailable)
    assert result.arm == "embed-fp32"
    assert "embed-fp32" in result.reason


def test_measure_cold_start_refuses_to_run_when_the_artifact_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "joinless.measurement.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not spawn a worker")
        ),
    )
    from joinless.measurement import ArtifactRequirement

    missing = ArtifactRequirement(path=tmp_path / "missing.onnx", sha256="deadbeef")

    result = measure_cold_start(
        "embed-fp32", "Acme Traders", "Acme Trading Co", artifact=missing
    )

    assert isinstance(result, Unavailable)
    assert result.reason == f"artifact missing at {missing.path}"


# --- measure_artifact_size(): bytes on disk per arm (RFC-0002 Metrics, issue #63) --

from joinless.evaluation import Metric
from joinless.measurement import measure_artifact_size


def test_measure_artifact_size_sums_bytes_across_every_path(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    tokenizer = tmp_path / "tokenizer.json"
    model.write_bytes(b"0" * 100)
    tokenizer.write_bytes(b"0" * 25)

    result = measure_artifact_size([model, tokenizer])

    assert result == Metric(value=125.0, undefined_reason=None)


def test_measure_artifact_size_is_read_from_the_file_not_a_constant(
    tmp_path: Path,
) -> None:
    """The figure has to come from the filesystem at call time (issue #63's
    last bullet), not from anything compiled into the source — proven by
    changing the file on disk between two calls and getting two different
    answers, which a hardcoded constant could not do."""
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"0" * 10)

    first = measure_artifact_size([artifact])
    artifact.write_bytes(b"0" * 40)
    second = measure_artifact_size([artifact])

    assert first.value == 10.0
    assert second.value == 40.0


def test_measure_artifact_size_is_undefined_when_no_paths_are_given() -> None:
    result = measure_artifact_size([])

    assert result.value is None
    assert result.undefined_reason == "classical arms carry no model artifact"


# --- _sum_defined(): the "total" derivation, skipping (not zeroing) undefined -----

from joinless.measurement import _sum_defined


def test_sum_defined_sums_every_phase_when_all_are_defined() -> None:
    total = _sum_defined(
        [
            Metric(value=1.0, undefined_reason=None),
            Metric(value=2.0, undefined_reason=None),
        ]
    )

    assert total.value == 3.0


def test_sum_defined_carries_a_defined_phase_through_unchanged_when_another_is_undefined() -> (
    None
):
    """For a plain sum, no input with at least one defined phase can tell
    "skip the undefined phase" apart from "treat it as zero": zero is the
    addition identity, so both strategies total the defined phases to the
    same number regardless of how many undefined phases sit alongside them.
    ``test_sum_defined_is_undefined_when_every_phase_is_undefined`` below is
    the input that does distinguish them — skipping leaves the total
    undefined there, where zero-filling would produce a defined ``0.0``.
    This test instead proves the narrower claim its inputs can actually
    support: an undefined phase does not raise, and does not disturb the
    total of the phases that are defined.
    """
    total = _sum_defined(
        [
            Metric(value=1.0, undefined_reason=None),
            Metric(value=None, undefined_reason="classical arms construct no session"),
        ]
    )

    assert total.value == 1.0


def test_sum_defined_is_undefined_when_every_phase_is_undefined() -> None:
    total = _sum_defined(
        [
            Metric(value=None, undefined_reason="classical arms construct no session"),
            Metric(value=None, undefined_reason="classical arms load no tokenizer"),
        ]
    )

    assert total.value is None
    assert total.undefined_reason == "no phase produced a duration"


# --- Worker failure is unavailable, never a zero (issue #53's third bullet) -------


def test_measure_warm_latency_reports_an_unexpected_crash_as_unavailable_not_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr("joinless.measurement.subprocess.run", _fake_run)

    result = measure_warm_latency(
        "overlap", "Acme Traders", "Acme Trading Co", warmup_count=0, repetition_count=1
    )

    assert isinstance(result, Unavailable)
    assert result.reason == "boom"


# --- ADR-0014's invariant, scoped to every worker script this module spawns (issue #53) --

from joinless.measurement import _COLD_START_SCRIPT, _WARM_LATENCY_SCRIPT

# One list, one entry per worker script this module spawns. A worker added
# later (issue #53) is covered by adding one entry here, not by copying the
# test below a fourth time — that is what let the cold-start and peak-memory
# workers go uncovered while only the warm-latency worker had a test.
_WORKER_SCRIPTS_AND_PARAMS = [
    (
        "warm_latency",
        _WARM_LATENCY_SCRIPT,
        {
            "arm": "overlap",
            "left_name": "Acme Traders",
            "right_name": "Acme Trading Co",
            "warmup_count": 1,
            "repetition_count": 1,
        },
    ),
    (
        "peak_memory",
        _PEAK_MEMORY_SCRIPT,
        {
            "arm": "overlap",
            "left_name": "Acme Traders",
            "right_name": "Acme Trading Co",
        },
    ),
    (
        "cold_start",
        _COLD_START_SCRIPT,
        {
            "arm": "overlap",
            "left_name": "Acme Traders",
            "right_name": "Acme Trading Co",
        },
    ),
]


@pytest.mark.parametrize(
    ("script", "params"),
    [(script, params) for _, script, params in _WORKER_SCRIPTS_AND_PARAMS],
    ids=[name for name, _, _ in _WORKER_SCRIPTS_AND_PARAMS],
)
def test_a_classical_arms_worker_has_not_imported_the_inference_runtime(
    script: str, params: dict[str, object]
) -> None:
    """The invariant test has to run in a child interpreter, not this test's
    own process — see the module docstring of tests/test_import_boundary.py
    for why a same-process version would pass whether or not it holds.

    Parametrized over every worker script this module spawns (issue #53):
    covering only the warm-latency worker left the cold-start and
    peak-memory workers free to import the inference runtime with nothing
    to notice.
    """
    params_literal = json.dumps(params)
    probe = (
        "import json, os\n"
        f'os.environ["JOINLESS_MEASURE_PARAMS"] = {params_literal!r}\n' + script + "\n"
        "import sys\n"
        "offenders = sorted(\n"
        "    m for m in sys.modules\n"
        '    if m == "onnxruntime" or m.startswith("onnxruntime.")\n'
        ")\n"
        "sys.exit(1 if offenders else 0)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr


# --- warm latency names what it times (issue #65's third bullet) --------------------


def test_warm_latency_states_that_preparation_sits_outside_the_timed_section() -> None:
    """Issue #65's third bullet: "the run record states which one produced each
    figure." Warm latency's answer is that neither path produced it — the timed
    section is ``score`` alone, with both operands prepared before it. Without
    that on the figure, a reader comparing 24.29 µs for ``embed-fp32`` against
    25.00 µs for ``embed-int8`` would conclude quantization bought no speed at
    all, when what those numbers actually show is that the graph never runs
    inside the timed section: the inference cost is in preparation, where the
    same run records 12.06 ms against 8.25 ms.

    Pinned by content, not by comparison against the constant that produces it —
    a symbol comparison passes whatever the text says.
    """
    from joinless.measurement import WARM_LATENCY_SCOPE

    assert WARM_LATENCY_SCOPE == (
        "score only; both operands are prepared before the timed section, so no "
        "preparation cost is included in this figure"
    )


def test_a_warm_latency_value_carries_that_scope_with_it() -> None:
    from joinless.measurement import WARM_LATENCY_SCOPE, WarmLatency

    latency = WarmLatency(
        arm="overlap",
        p50_seconds=1.0,
        p99_seconds=2.0,
        warmup_count=5,
        repetition_count=20,
    )

    assert latency.scope == WARM_LATENCY_SCOPE
