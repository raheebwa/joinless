# SPDX-License-Identifier: MIT
"""RFC-0004 step 7: measure both arms, each in a fresh child process (issue #12).

Cold start measured in-process, after another arm already imported the runtime, measures
nothing — a warm allocator and a warm page cache flatter whichever arm runs second. Every
figure here is taken inside :mod:`spikes.quantization.workers.measure_arm`, spawned once
per arm via ``subprocess``, never reused across arms.

The functions below are pure: phase-duration arithmetic, RSS unit normalization, latency
summarization and record assembly, all operating on values a worker process already
produced. Spawning that worker is the untested edge, in :func:`main`.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

CHECKPOINT_ORDER = (
    "spawn_start",
    "ready",
    "after_import",
    "after_session",
    "after_tokenizer",
    "after_first_inference",
)
"""Six checkpoints bracketing cold start. ``spawn_start`` is taken by the parent,
immediately before the child is spawned; the rest are taken inside the child, in the
order PRD MR-11 names: interpreter start, import, session creation, tokenizer load,
first inference."""

_PHASE_LABELS = (
    "interpreter_start",
    "import",
    "session_creation",
    "tokenizer_load",
    "first_inference",
)

BATCH_SIZES = (1, 8, 32)
"""Documented batch sizes for warm batched preparation (RFC-0004 step 7)."""

WARM_REPEATS = 20
"""Repetitions for warm single-pair scoring, after warm-up (RFC-0002 method step 2)."""

THREAD_COUNT = 1
"""Intra-op thread count each session is configured with, and the value the record
names for both arms (PRD MR-6). Fixed rather than left to the runtime's own default so
a difference between arms cannot be a difference in how many cores each one used."""

PEAK_RSS_METHOD = (
    "resource.getrusage(RUSAGE_SELF).ru_maxrss, taken once at process exit and "
    "normalized to bytes per platform (Linux reports kibibytes, Darwin bytes)"
)


def compute_phase_durations(checkpoints: Mapping[str, float]) -> dict[str, float]:
    """Turn six checkpoints into five phase durations plus the cold-start total."""
    missing = [key for key in CHECKPOINT_ORDER if key not in checkpoints]
    if missing:
        raise ValueError(f"missing checkpoint(s): {missing}")

    values = [checkpoints[key] for key in CHECKPOINT_ORDER]
    durations = {
        label: values[i + 1] - values[i] for i, label in enumerate(_PHASE_LABELS)
    }
    durations["cold_start_total"] = values[-1] - values[0]
    return durations


def normalize_peak_rss(raw: int, platform_name: str) -> int:
    """Normalize ``ru_maxrss`` to bytes. The unit is platform-dependent, not a libc
    constant: Linux reports kibibytes, Darwin (and the BSDs) report bytes."""
    normalized = platform_name.lower()
    if normalized == "darwin":
        return raw
    if normalized == "linux":
        return raw * 1024
    raise ValueError(
        f"no known ru_maxrss unit for platform {platform_name!r}; add one rather than guess"
    )


def summarize_latencies(samples: Sequence[float]) -> dict[str, float]:
    """Count, median (p50), min and max over a set of repeated timings.

    Raises on an empty sample: a zero-sample measurement is a harness defect (the
    repeat count is a constant this module controls), not an undefined metric in the
    ADR-0013 sense — that rule covers a denominator the *evaluation set* determines.
    """
    if not samples:
        raise ValueError("cannot summarize an empty latency sample")
    ordered = sorted(samples)
    return {
        "count": float(len(ordered)),
        "p50": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def artifact_size_bytes(path: str | Path) -> int:
    """Bytes on disk for one artefact (RFC-0004 step 7 / MR-5)."""
    return Path(path).stat().st_size


def assemble_arm_measurement(
    *,
    arm: str,
    checkpoints: Mapping[str, float],
    single_pair_latencies: Sequence[float],
    batched_latencies: Mapping[int, Sequence[float]],
    peak_rss_raw: int,
    platform_name: str,
    artifact_bytes: int,
) -> dict[str, object]:
    """Combine one arm's measurements into the shape the record carries."""
    return {
        "arm": arm,
        "cold_start": compute_phase_durations(checkpoints),
        "warm_single_pair": summarize_latencies(single_pair_latencies),
        "warm_batched": {
            str(batch_size): summarize_latencies(samples)
            for batch_size, samples in sorted(batched_latencies.items())
        },
        "peak_rss_bytes": normalize_peak_rss(peak_rss_raw, platform_name),
        "peak_rss_method": PEAK_RSS_METHOD,
        "artifact_bytes": artifact_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    """Spawn a fresh child process per arm and record both arms' measurements."""
    import argparse
    import json
    import os
    import platform
    import subprocess
    import sys
    import time

    from spikes.quantization.cli_common import (
        hf_cache_dir,
        read_fragment,
        resolve_cache_dir,
        write_fragment,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=WARM_REPEATS)
    args = parser.parse_args(argv)

    cache_dir = resolve_cache_dir(os.environ)
    quantize_record = read_fragment(cache_dir, "step3_quantize")
    call = quantize_record["call"]
    assert isinstance(call, dict)
    model_paths = {"fp32": str(call["model_input"]), "int8": str(call["model_output"])}

    arm_records: dict[str, object] = {}
    for arm, model_path in model_paths.items():
        out_path = cache_dir / f"step7_measure_{arm}_worker.json"
        command = [
            sys.executable,
            "-m",
            "spikes.quantization.workers.measure_arm",
            "--arm",
            arm,
            "--model-path",
            model_path,
            "--hf-cache-dir",
            str(hf_cache_dir(cache_dir)),
            "--batch-sizes",
            ",".join(str(b) for b in BATCH_SIZES),
            "--repeats",
            str(args.repeats),
            "--intra-op-threads",
            str(THREAD_COUNT),
            "--out",
            str(out_path),
        ]
        spawn_start = time.perf_counter()
        subprocess.run(command, check=True)
        worker_output = json.loads(out_path.read_text(encoding="utf-8"))

        checkpoints = {"spawn_start": spawn_start, **worker_output["checkpoints"]}
        arm_records[arm] = assemble_arm_measurement(
            arm=arm,
            checkpoints=checkpoints,
            single_pair_latencies=worker_output["single_pair_latencies"],
            batched_latencies={
                int(k): v for k, v in worker_output["batched_latencies"].items()
            },
            peak_rss_raw=worker_output["peak_rss_raw"],
            platform_name=platform.system(),
            artifact_bytes=artifact_size_bytes(model_path),
        )

    write_fragment(
        cache_dir, "step7_measure", {"thread_count": THREAD_COUNT, "arms": arm_records}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
