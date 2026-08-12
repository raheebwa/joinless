# SPDX-License-Identifier: MIT
"""RFC-0004 step 8: assemble the spike record in benchmarks/README.md's schema.

"Full environment capture" means the fields benchmarks/README.md names — hardware, OS,
Python version, runtime versions, thread count, model identity and revision — assembled
from an explicit allow-list, never a dump of the process environment. The reference
machine's own environment carries an API token for a model host; a benchmark record is a
public artefact, and :data:`ALLOWED_ENV_KEYS` is the only gate between the two.

Step 13 (the go/no-go against RFC-0004's abort clauses) is deliberately left as ``None``
here — it is written from this record's real output, after a real run, which this module
does not perform.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from datetime import datetime

from spikes.quantization.cli_common import CACHE_DIR_ENV_VAR

RECORD_SCHEMA = "quantization-spike-v1"

ALLOWED_ENV_KEYS = frozenset({CACHE_DIR_ENV_VAR})
"""The only environment variable names this module will ever read out of a
caller-supplied mapping. Everything else in the process environment — including a
model-host API token — is invisible to :func:`capture_allowed_env`, by construction:
the function has no way to return a key it was not told to look for."""


def capture_allowed_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Read only :data:`ALLOWED_ENV_KEYS` out of ``environ``.

    Takes the environment mapping as a parameter, the same discipline
    :func:`spikes.quantization.cli_common.resolve_cache_dir` uses — nothing here reads
    ``os.environ`` itself.
    """
    return {key: environ[key] for key in ALLOWED_ENV_KEYS if key in environ}


def total_memory_bytes() -> int | None:
    """Total physical memory, or None where the platform does not report it.

    None rather than 0: benchmarks/README.md requires memory, and a zero would be a
    claim about the machine rather than an admission that it was not measured
    (ADR-0013).
    """
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def capture_platform_facts() -> dict[str, object]:
    """Hardware and OS facts from the stdlib ``platform`` module — not from any
    environment variable, so nothing here needs an allow-list of its own."""
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes(),
    }


def parse_pmset_battery_output(text: str) -> str:
    """Normalize macOS ``pmset -g batt`` output to ``"ac"``, ``"battery"`` or
    ``"unknown"``."""
    if "AC Power" in text:
        return "ac"
    if "Battery Power" in text:
        return "battery"
    return "unknown"


def parse_linux_power_supply_status(text: str) -> str:
    """Normalize the contents of ``/sys/class/power_supply/*/status`` the same way."""
    normalized = text.strip().lower()
    if normalized in {"charging", "full"}:
        return "ac"
    if normalized == "discharging":
        return "battery"
    return "unknown"


def detect_power_mode() -> str:
    """Dispatch to the platform-appropriate power-mode probe.

    Untested: it shells out on Darwin and reads a virtual filesystem entry on Linux.
    The parsing each branch delegates to is pure and tested above.
    """
    import subprocess

    system = platform.system()
    if system == "Darwin":
        result = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
        )
        return parse_pmset_battery_output(result.stdout)
    if system == "Linux":
        supply_dir = "/sys/class/power_supply"
        entries = sorted(os.listdir(supply_dir)) if os.path.isdir(supply_dir) else []
        for entry in entries:
            status_path = f"{supply_dir}/{entry}/status"
            if os.path.exists(status_path):
                with open(status_path, encoding="utf-8") as handle:
                    return parse_linux_power_supply_status(handle.read())
    return "unknown"


def build_record_filename(started_at: datetime) -> str:
    """A timestamped filename — records are never overwritten (benchmarks/README.md)."""
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-quantization-spike.json"


def assemble_spike_record(
    *,
    started_at: datetime,
    allowed_env: Mapping[str, str],
    platform_facts: Mapping[str, object],
    power_mode: str,
    runtime_versions: Mapping[str, str],
    thread_count: int,
    model: Mapping[str, object],
    export: Mapping[str, object],
    quantize: Mapping[str, object],
    signatures: Mapping[str, object],
    operators: Mapping[str, object],
    smoke: Mapping[str, object],
    measurements: Mapping[str, object],
) -> dict[str, object]:
    """Combine every RFC-0004 step's fragment into one record.

    Every field is named explicitly rather than merged from an unknown-shaped source —
    the same discipline :data:`ALLOWED_ENV_KEYS` applies to the environment applies here
    to the record as a whole: what appears is what a parameter named, nothing more.
    """
    return {
        "schema": RECORD_SCHEMA,
        "spike": "int8-quantization-feasibility",
        "started_at": started_at.isoformat(),
        "environment": {
            "hardware": dict(platform_facts),
            "power_mode": power_mode,
            "runtime_versions": dict(runtime_versions),
            "thread_count": thread_count,
            "env": dict(allowed_env),
        },
        "model": dict(model),
        "export": dict(export),
        "quantize": dict(quantize),
        "signatures": dict(signatures),
        "operators": dict(operators),
        "smoke": dict(smoke),
        "measurements": dict(measurements),
        "go_no_go": None,
    }


def main(argv: list[str] | None = None) -> int:
    """Read every step's fragment and write the consolidated record to benchmarks/.

    This is RFC-0004 step 8, run last, after steps 1 through 7 have each written their
    fragment to the model cache directory.
    """
    import argparse
    import importlib.metadata
    import json
    import os
    from datetime import UTC
    from pathlib import Path

    from spikes.quantization.cli_common import read_fragment, resolve_cache_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks-dir", default="benchmarks")
    args = parser.parse_args(argv)

    cache_dir = resolve_cache_dir(os.environ)
    step7 = read_fragment(cache_dir, "step7_measure")
    measurements = step7["arms"]
    thread_count = step7["thread_count"]
    assert isinstance(measurements, dict)
    assert isinstance(thread_count, int)

    started_at = datetime.now(UTC)
    record = assemble_spike_record(
        started_at=started_at,
        allowed_env=capture_allowed_env(os.environ),
        platform_facts=capture_platform_facts(),
        power_mode=detect_power_mode(),
        runtime_versions={"onnxruntime": importlib.metadata.version("onnxruntime")},
        thread_count=thread_count,
        model=read_fragment(cache_dir, "step1_model"),
        export=read_fragment(cache_dir, "step2_export"),
        quantize=read_fragment(cache_dir, "step3_quantize"),
        signatures=read_fragment(cache_dir, "step4_signatures"),
        operators=read_fragment(cache_dir, "step5_operators"),
        smoke=read_fragment(cache_dir, "step6_smoke"),
        measurements=measurements,
    )

    benchmarks_dir = Path(args.benchmarks_dir)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    out_path = benchmarks_dir / build_record_filename(started_at)
    out_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
