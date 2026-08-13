# SPDX-License-Identifier: MIT
"""Resource measurement, isolated per arm per metric (RFC-0002 Method step 7,
ADR-0013, ADR-0014, issues #51, #53, #54, #55, #56)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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

from joinless.measurement import measure_cold_start


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
