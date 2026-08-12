# SPDX-License-Identifier: MIT
"""RFC-0004 step 8: assemble the spike record, environment capture included (issue #6).

The reference machine's environment carries an API token for a model host, and a
benchmark record is a public artefact. capture_allowed_env is the one function allowed
to read from a caller-supplied environment mapping, and it is tested here specifically
for what it must never let through.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from spikes.quantization.cli_common import CACHE_DIR_ENV_VAR
from spikes.quantization.record import (
    ALLOWED_ENV_KEYS,
    assemble_spike_record,
    build_record_filename,
    capture_allowed_env,
    capture_platform_facts,
    parse_linux_power_supply_status,
    parse_pmset_battery_output,
)


def test_allowed_env_keys_names_only_the_cache_dir() -> None:
    assert ALLOWED_ENV_KEYS == frozenset({CACHE_DIR_ENV_VAR})


def test_capture_allowed_env_keeps_the_named_key() -> None:
    environ = {CACHE_DIR_ENV_VAR: "/cache"}

    assert capture_allowed_env(environ) == {CACHE_DIR_ENV_VAR: "/cache"}


def test_capture_allowed_env_excludes_a_key_outside_the_list() -> None:
    poisoned = {
        CACHE_DIR_ENV_VAR: "/cache",
        "HF_TOKEN": "do-not-leak-this-token",
        "SOME_OTHER_SERVICE_TOKEN": "or-this-one",
    }

    captured = capture_allowed_env(poisoned)

    assert captured == {CACHE_DIR_ENV_VAR: "/cache"}
    assert "HF_TOKEN" not in captured
    assert "do-not-leak-this-token" not in captured.values()


def test_capture_platform_facts_names_the_required_fields() -> None:
    facts = capture_platform_facts()

    for key in ("machine", "system", "release", "python_version", "cpu_count"):
        assert key in facts


def test_parse_pmset_battery_output_ac_power() -> None:
    assert (
        parse_pmset_battery_output(
            "Now drawing from 'AC Power'\n -InternalBattery-0 (id=123)\t100%; charged;"
        )
        == "ac"
    )


def test_parse_pmset_battery_output_battery_power() -> None:
    assert parse_pmset_battery_output("Now drawing from 'Battery Power'\n") == "battery"


def test_parse_pmset_battery_output_unknown() -> None:
    assert parse_pmset_battery_output("garbage") == "unknown"


def test_parse_linux_power_supply_status_charging_is_ac() -> None:
    assert parse_linux_power_supply_status("Charging\n") == "ac"
    assert parse_linux_power_supply_status("Full\n") == "ac"


def test_parse_linux_power_supply_status_discharging_is_battery() -> None:
    assert parse_linux_power_supply_status("Discharging\n") == "battery"


def test_parse_linux_power_supply_status_unknown() -> None:
    assert parse_linux_power_supply_status("Not charging\n") == "unknown"


def test_build_record_filename_is_iso_timestamped() -> None:
    started_at = datetime(2026, 8, 12, 13, 5, 7, tzinfo=UTC)

    assert (
        build_record_filename(started_at) == "20260812T130507Z-quantization-spike.json"
    )


def _minimal_record(allowed_env: dict[str, str]) -> dict[str, object]:
    return assemble_spike_record(
        started_at=datetime(2026, 8, 12, 13, 5, 7, tzinfo=UTC),
        allowed_env=allowed_env,
        platform_facts={"machine": "arm64", "system": "Darwin"},
        power_mode="ac",
        runtime_versions={"onnxruntime": "1.28.0"},
        thread_count=1,
        model={"model_id": "m"},
        export={"command": []},
        quantize={"call": {}},
        signatures={"equivalent": True},
        operators={"added": []},
        smoke={"max_divergence": None},
        measurements={"fp32": {}},
    )


def test_assemble_spike_record_carries_every_named_section() -> None:
    record = _minimal_record({CACHE_DIR_ENV_VAR: "/cache"})

    for key in (
        "schema",
        "started_at",
        "environment",
        "model",
        "export",
        "quantize",
        "signatures",
        "operators",
        "smoke",
        "measurements",
        "go_no_go",
    ):
        assert key in record

    assert record["go_no_go"] is None  # step 13 — deliberately left open


def test_assemble_spike_record_never_lets_an_unlisted_key_through() -> None:
    poisoned_environ = {
        CACHE_DIR_ENV_VAR: "/cache",
        "HF_TOKEN": "do-not-leak-this-token",
    }
    allowed = capture_allowed_env(poisoned_environ)

    record = _minimal_record(allowed)
    serialized = json.dumps(record)

    assert "HF_TOKEN" not in serialized
    assert "do-not-leak-this-token" not in serialized


def test_platform_facts_carry_total_memory() -> None:
    """benchmarks/README.md requires 'Hardware — CPU, core count, memory'. Without a
    memory figure a reader cannot evaluate ADR-0009's argument that a model this small
    may already be resident in cache."""
    facts = capture_platform_facts()
    assert "total_memory_bytes" in facts
    value = facts["total_memory_bytes"]
    assert value is None or (isinstance(value, int) and value > 0)
