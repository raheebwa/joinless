# SPDX-License-Identifier: MIT
"""RFC-0004 step 7: measure both arms in fresh child processes (issue #12).

Pure logic only: phase-duration arithmetic, RSS unit normalization, latency
summarization, and record assembly. The subprocess that actually spawns a fresh
interpreter per arm is a thin, untested wrapper — see the module docstring.
"""

from __future__ import annotations

import pytest

from spikes.quantization.measure import (
    CHECKPOINT_ORDER,
    assemble_arm_measurement,
    compute_phase_durations,
    normalize_peak_rss,
    summarize_latencies,
)


def test_compute_phase_durations_from_monotonically_increasing_checkpoints() -> None:
    checkpoints = {
        "spawn_start": 0.0,
        "ready": 0.05,
        "after_import": 0.30,
        "after_session": 0.45,
        "after_tokenizer": 0.47,
        "after_first_inference": 0.52,
    }

    durations = compute_phase_durations(checkpoints)

    assert durations["interpreter_start"] == pytest.approx(0.05)
    assert durations["import"] == pytest.approx(0.25)
    assert durations["session_creation"] == pytest.approx(0.15)
    assert durations["tokenizer_load"] == pytest.approx(0.02)
    assert durations["first_inference"] == pytest.approx(0.05)
    assert durations["cold_start_total"] == pytest.approx(0.52)


def test_compute_phase_durations_rejects_a_missing_checkpoint() -> None:
    incomplete = {k: 0.0 for k in CHECKPOINT_ORDER if k != "after_session"}

    with pytest.raises(ValueError, match="after_session"):
        compute_phase_durations(incomplete)


def test_normalize_peak_rss_darwin_is_already_bytes() -> None:
    assert normalize_peak_rss(104_857_600, "Darwin") == 104_857_600


def test_normalize_peak_rss_linux_is_kilobytes() -> None:
    assert normalize_peak_rss(102_400, "Linux") == 102_400 * 1024


def test_normalize_peak_rss_rejects_an_unknown_platform() -> None:
    with pytest.raises(ValueError, match="Windows"):
        normalize_peak_rss(1, "Windows")


def test_summarize_latencies_reports_count_median_and_p99() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]

    summary = summarize_latencies(samples)

    assert summary["count"] == 5
    assert summary["p50"] == pytest.approx(3.0)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(5.0)


def test_summarize_latencies_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize_latencies([])


def test_assemble_arm_measurement_combines_every_field() -> None:
    checkpoints = {
        "spawn_start": 0.0,
        "ready": 0.01,
        "after_import": 0.2,
        "after_session": 0.3,
        "after_tokenizer": 0.32,
        "after_first_inference": 0.35,
    }
    record = assemble_arm_measurement(
        arm="fp32",
        checkpoints=checkpoints,
        single_pair_latencies=[0.001, 0.0012, 0.0009],
        batched_latencies={1: [0.001], 8: [0.006]},
        peak_rss_raw=102_400,
        platform_name="Linux",
        artifact_bytes=90_123_456,
    )

    cold_start = record["cold_start"]
    warm_single_pair = record["warm_single_pair"]
    warm_batched = record["warm_batched"]
    assert isinstance(cold_start, dict)
    assert isinstance(warm_single_pair, dict)
    assert isinstance(warm_batched, dict)

    assert record["arm"] == "fp32"
    assert cold_start["cold_start_total"] == pytest.approx(0.35)
    assert warm_single_pair["count"] == 3
    assert warm_batched["1"]["count"] == 1
    assert warm_batched["8"]["count"] == 1
    assert record["peak_rss_bytes"] == 102_400 * 1024
    assert record["peak_rss_method"]
    assert record["artifact_bytes"] == 90_123_456
