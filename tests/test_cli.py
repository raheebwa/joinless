# SPDX-License-Identifier: MIT
"""The console entry point: ``resolve``, ``compare``, ``doctor``, ``benchmark``.

``report`` (issue #46) does not exist yet, and this suite guards that as a fact
about the CLI, not an accident of what nobody got round to adding — a word that
would be a subcommand name in a later version must still be rejected as an
unrecognized argument today.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from joinless.resolver import _REASON_NO_COORDINATES, _REASON_NOT_SELECTED


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- no commands beyond resolve, compare, doctor exist ----------------------


def test_help_exits_zero() -> None:
    from joinless.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0


def test_help_names_only_the_commands_that_exist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from joinless.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])

    out = capsys.readouterr().out
    assert "resolve" in out
    assert "compare" in out
    assert "doctor" in out
    assert "benchmark" in out
    # report is #46, not built yet - --help must not imply it exists.
    assert "report" not in out


@pytest.mark.parametrize("word", ["report"])
def test_a_not_yet_built_command_is_rejected_as_unrecognized(word: str) -> None:
    from joinless.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main([word])

    assert excinfo.value.code != 0


def test_characterizes_successful_invocation_returns_zero() -> None:
    """Characterization test — written against ``main`` as it already stands.

    ``[project.scripts]`` wires ``joinless`` to this function, and the shell sees
    whatever it returns. With no subcommand given, ``main`` parses cleanly and
    returns ``0`` rather than raising.
    """
    from joinless.cli import main

    assert main([]) == 0


# --- resolve ------------------------------------------------------------


def test_resolve_writes_a_matched_pair_merged_under_the_fr5_policy(
    tmp_path: Path,
) -> None:
    from joinless.cli import main

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(
        left,
        [{"name": "Acme Trading Co", "latitude": 0.31, "longitude": 32.58}],
    )
    _write_jsonl(
        right,
        [{"name": "Acme Trading Co", "latitude": 0.31, "longitude": 32.58}],
    )

    exit_code = main(
        ["resolve", "--left", str(left), "--right", str(right), "--output", str(output)]
    )

    assert exit_code == 0
    rows = _read_jsonl(output)
    assert len(rows) == 1
    assert rows[0]["status"] == "matched"
    assert rows[0]["name"] == "Acme Trading Co"
    assert rows[0]["sources"] == ["left", "right"]


def test_resolve_default_scorer_is_named_overlap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #42: 'defaulting to a named arm rather than an implicit one' - the
    default is observable in the summary the command prints, not merely
    documented, so a change to the default is a test failure rather than a
    silent drift."""
    from joinless.cli import main

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(left, [{"name": "Acme"}])
    _write_jsonl(right, [{"name": "Acme"}])

    exit_code = main(
        ["resolve", "--left", str(left), "--right", str(right), "--output", str(output)]
    )

    assert exit_code == 0
    assert "under 'overlap'" in capsys.readouterr().out


def test_resolve_scorer_flag_selects_a_named_alternative_arm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from joinless.cli import main

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(left, [{"name": "Acme"}])
    _write_jsonl(right, [{"name": "Acme"}])

    exit_code = main(
        [
            "resolve",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output",
            str(output),
            "--scorer",
            "fuzzy",
        ]
    )

    assert exit_code == 0
    assert "under 'fuzzy'" in capsys.readouterr().out


def test_resolve_rejects_an_unknown_scorer_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from joinless.cli import main

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(left, [{"name": "Acme"}])
    _write_jsonl(right, [{"name": "Acme"}])

    exit_code = main(
        [
            "resolve",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output",
            str(output),
            "--scorer",
            "nonesuch",
        ]
    )

    assert exit_code == 1
    assert "Unknown scorer" in capsys.readouterr().err
    assert not output.exists()


def test_resolve_reports_a_missing_scorer_dependency_as_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from joinless.cli import main

    monkeypatch.setitem(sys.modules, "rapidfuzz", None)

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(left, [{"name": "Acme"}])
    _write_jsonl(right, [{"name": "Acme"}])

    exit_code = main(
        [
            "resolve",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output",
            str(output),
            "--scorer",
            "fuzzy",
        ]
    )

    assert exit_code == 1
    assert "rapidfuzz" in capsys.readouterr().err
    assert not output.exists()


def test_resolve_writes_unmatched_records_with_their_reason(tmp_path: Path) -> None:
    """Issue #42: 'the merge dropped rows' is the failure mode this bullet
    guards against - an unmatched record must carry a machine-readable reason
    in the output file, not vanish from it."""
    from joinless.cli import main

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(left, [{"name": "Acme Trading Co"}])  # no coordinates
    _write_jsonl(
        right,
        [{"name": "Unrelated Bakery", "latitude": 0.31, "longitude": 32.58}],
    )

    exit_code = main(
        ["resolve", "--left", str(left), "--right", str(right), "--output", str(output)]
    )

    assert exit_code == 0
    rows = _read_jsonl(output)
    assert len(rows) == 2
    by_source = {(r["source"], r["ordinal"]): r for r in rows}

    left_row = by_source[("left", 0)]
    assert left_row["status"] == "unmatched"
    assert left_row["reason"] == _REASON_NO_COORDINATES
    assert "no coordinates" in left_row["reason"]

    right_row = by_source[("right", 0)]
    assert right_row["status"] == "unmatched"
    assert right_row["reason"] == _REASON_NOT_SELECTED


def test_resolve_skips_blank_lines_when_numbering_ordinals(tmp_path: Path) -> None:
    """A blank line must neither crash the parser (a bare ``json.loads("")``
    raises) nor shift the real row's ``ordinal`` by counting the blank line as
    a position of its own - the surrounding blank lines here would push the
    one real row from ordinal 0 to ordinal 1 if they were counted."""
    from joinless.cli import main

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    left.write_text('\n{"name": "Acme"}\n\n', encoding="utf-8")
    right.write_text("", encoding="utf-8")

    exit_code = main(
        ["resolve", "--left", str(left), "--right", str(right), "--output", str(output)]
    )

    assert exit_code == 0
    rows = _read_jsonl(output)
    assert len(rows) == 1
    assert rows[0]["status"] == "unmatched"
    assert rows[0]["source"] == "left"
    assert rows[0]["ordinal"] == 0


def test_resolve_writes_an_empty_file_for_two_empty_record_sets(tmp_path: Path) -> None:
    from joinless.cli import main

    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    left.write_text("", encoding="utf-8")
    right.write_text("", encoding="utf-8")

    exit_code = main(
        ["resolve", "--left", str(left), "--right", str(right), "--output", str(output)]
    )

    assert exit_code == 0
    assert output.read_text(encoding="utf-8") == ""


def test_resolve_completes_with_no_network_interface_available(tmp_path: Path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(
        left, [{"name": "Acme Trading Co", "latitude": 0.31, "longitude": 32.58}]
    )
    _write_jsonl(
        right, [{"name": "Acme Trading Co", "latitude": 0.31, "longitude": 32.58}]
    )

    argv = [
        "resolve",
        "--left",
        str(left),
        "--right",
        str(right),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        [sys.executable, "-c", _no_network_probe(argv)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists()


# --- compare --------------------------------------------------------------


def test_compare_prints_score_and_a_match_decision(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from joinless.cli import main

    exit_code = main(["compare", "Acme Trading Co", "Acme Trading Co"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "scorer: overlap" in out
    assert "score: 1.0000" in out
    assert "decision: match" in out


def test_compare_prints_a_no_match_decision_for_dissimilar_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from joinless.cli import main

    exit_code = main(["compare", "Acme Trading Co", "Zebra Fisheries"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "score: 0.0000" in out
    assert "decision: no match" in out


def test_compare_threshold_flag_moves_the_decision(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RFC-0003 open question 2: `compare` is where a reader can move the
    threshold and watch the decision change. This pair's overlap score is a
    known, exact 0.5 - {"acme", "trading", "company"} vs {"acme",
    "retailers"} share one token out of the smaller side's two - so the two
    thresholds below are deliberately chosen either side of it and the
    printed score is pinned as well, proving the flag is wired to the actual
    decision rather than merely accepted and ignored."""
    from joinless.cli import main

    exit_code_high = main(
        ["compare", "Acme Trading Company", "Acme Retailers", "--threshold", "0.6"]
    )
    assert exit_code_high == 0
    out_high = capsys.readouterr().out
    assert "score: 0.5000" in out_high
    assert "decision: no match" in out_high

    exit_code_low = main(
        ["compare", "Acme Trading Company", "Acme Retailers", "--threshold", "0.4"]
    )
    assert exit_code_low == 0
    out_low = capsys.readouterr().out
    assert "score: 0.5000" in out_low
    assert "decision: match" in out_low


def test_compare_labels_its_timing_as_illustrative_in_the_output_itself(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #43: the illustrative label has to be in the printed output, not
    only in the documentation around it."""
    from joinless.cli import main

    main(["compare", "Acme", "Acme"])

    out = capsys.readouterr().out
    assert "illustrative" in out
    assert "never benchmark evidence" in out


def test_compare_scorer_flag_selects_a_named_alternative_arm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from joinless.cli import main

    main(["compare", "Acme", "Acme", "--scorer", "fuzzy"])

    assert "scorer: fuzzy" in capsys.readouterr().out


def test_compare_rejects_an_unknown_scorer_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from joinless.cli import main

    exit_code = main(["compare", "Acme", "Acme", "--scorer", "nonesuch"])

    assert exit_code == 1
    assert "Unknown scorer" in capsys.readouterr().err


def test_compare_reports_a_missing_scorer_dependency_as_unavailable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from joinless.cli import main

    monkeypatch.setitem(sys.modules, "rapidfuzz", None)

    exit_code = main(["compare", "Acme", "Acme", "--scorer", "fuzzy"])

    assert exit_code == 1
    assert "rapidfuzz" in capsys.readouterr().err


def test_compare_completes_with_no_network_interface_available() -> None:
    argv = ["compare", "Acme Trading Co", "Acme Trading Co"]
    result = subprocess.run(
        [sys.executable, "-c", _no_network_probe(argv)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


# --- doctor -----------------------------------------------------------------


def test_doctor_reports_the_four_fields_the_command_is_named_for(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from joinless.cli import main

    exit_code = main(["doctor"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "architecture:" in out
    assert "execution provider:" in out
    assert "installed profile:" in out
    assert "offline status:" in out


def test_doctor_output_is_copy_pasteable_into_the_bug_form(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The bug report template requires architecture, OS, Python version and
    the benchmark run record (CONTRIBUTING's standing rule on the template) -
    `doctor` must print all four labels so its output can be pasted straight
    into the form."""
    from joinless.cli import main

    main(["doctor"])

    out = capsys.readouterr().out
    assert "architecture:" in out
    assert "operating system:" in out
    assert "python version:" in out
    assert "benchmark run record:" in out


def test_doctor_reports_the_base_profile_without_importing_the_runtime(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    from joinless.cli import main

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    exit_code = main(["doctor"])

    assert exit_code == 0
    assert "installed profile: base" in capsys.readouterr().out


def test_doctor_reports_the_neural_profile_without_asking_the_runtime(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #44: report the profile "without importing the inference runtime to
    find out". Faking a positive ``find_spec`` result, rather than installing the
    extra, is what shows the detection path answers from metadata alone — the
    fake spec carries ``loader=None``, so anything that tried to import from it
    would fail rather than quietly succeed."""
    import importlib.machinery
    import importlib.util

    from joinless.cli import main

    fake_spec = importlib.machinery.ModuleSpec("onnxruntime", loader=None)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: fake_spec if name == "onnxruntime" else None,
    )

    exit_code = main(["doctor"])

    assert exit_code == 0
    assert "installed profile: neural" in capsys.readouterr().out


_DOCTOR_IMPORT_PROBE = """
import sys
from joinless.cli import main

main(["doctor"])
offenders = sorted(m for m in sys.modules if m == "onnxruntime" or m.startswith("onnxruntime."))
print("\\n".join(offenders))
sys.exit(1 if offenders else 0)
"""


def test_running_doctor_never_initialises_the_runtime() -> None:
    """The profile it reports is read from metadata, so running it must not load
    the runtime even where the neural extra is installed.

    A child interpreter, and that is load-bearing. ``sys.modules`` is global to a
    process, so an in-process assertion that ``onnxruntime`` is absent says only
    that nothing in the whole session imported it — it passes or fails on test
    ordering and on which install profile CI happened to build, neither of which
    is a property of this command."""
    result = subprocess.run(
        [sys.executable, "-c", _DOCTOR_IMPORT_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "running doctor pulled in the inference runtime: "
        f"{result.stdout.strip() or result.stderr.strip()}"
    )


def test_doctor_completes_with_no_network_interface_available() -> None:
    argv = ["doctor"]
    result = subprocess.run(
        [sys.executable, "-c", _no_network_probe(argv)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


# --- benchmark ----------------------------------------------------------------


def _run_benchmark_with_small_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Run the real ``benchmark`` command with the seed set shrunk to one seed
    (issue #45's TDD bullet: "a test that runs the real command over a small
    input proves more than one that mocks the thing under test") and the working
    directory pointed at ``tmp_path``, so the record lands there rather than in
    the repository's own ``benchmarks/``. There is no ``--pairs`` flag to shrink
    the corpus with (issue #45 forbids one), so the seed set is shrunk the same
    way a test would shrink any other module-level default: by monkeypatching
    the attribute the command reads at call time.

    Returns the one record written, parsed from JSON.
    """
    from joinless import corpus as corpus_module
    from joinless.cli import main

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])
    assert exit_code == 0

    written = list((tmp_path / "benchmarks").glob("*.json"))
    assert len(written) == 1
    result: dict[str, Any] = json.loads(written[0].read_text(encoding="utf-8"))
    return result


def test_benchmark_writes_one_record_carrying_its_schema_and_exact_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    assert record["schema"] == "benchmark-v2"
    assert record["command"] == ["joinless", "benchmark"]


def test_benchmark_prints_the_path_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from joinless import corpus as corpus_module
    from joinless.cli import main

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])

    assert exit_code == 0
    written = list((tmp_path / "benchmarks").glob("*.json"))
    assert len(written) == 1
    assert written[0].name in capsys.readouterr().out


def test_benchmark_records_every_configured_arm_including_the_unregistered_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #45's third bullet: "an arm that cannot initialise is recorded as
    unavailable with a reason, not omitted" - all four configured arms must have
    a row, not just the two joinless.scoring currently registers."""
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    assert set(record["results"]) == {"overlap", "fuzzy", "embed-fp32", "embed-int8"}


def test_benchmark_records_an_unregistered_arm_as_unavailable_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    for arm in ("embed-fp32", "embed-int8"):
        for field in ("warm_latency", "peak_memory", "cold_start", "artifact_size"):
            entry = record["results"][arm][field]
            assert entry["status"] == "unavailable"
            assert arm in entry["reason"]
        accuracy = record["results"][arm]["accuracy"]
        assert accuracy["status"] == "invalid"
        assert arm in accuracy["reason"]


def test_benchmark_records_a_real_measured_result_for_a_registered_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    accuracy = record["results"]["overlap"]["accuracy"]
    assert accuracy["status"] == "ok"
    assert accuracy["per_family"]

    warm_latency = record["results"]["overlap"]["warm_latency"]
    assert warm_latency["status"] == "ok"
    assert warm_latency["p50_seconds"] >= 0.0
    assert warm_latency["warmup_count"] == 5
    assert warm_latency["repetition_count"] == 20

    peak_memory = record["results"]["fuzzy"]["peak_memory"]
    assert peak_memory["status"] == "ok"
    assert peak_memory["peak_rss_bytes"] > 0

    cold_start = record["results"]["fuzzy"]["cold_start"]
    assert cold_start["status"] == "ok"
    assert cold_start["session_creation"]["value"] is None


def test_benchmark_records_the_classical_arms_artifact_size_as_an_explicit_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #63: "the classical arms are not exempt... a zero or an empty
    cell for the classical arms is a fact worth stating explicitly" - both
    classical arms get a defined ``artifact_size`` entry naming why there is
    no figure, not an omitted field and not ``0.0`` (which would claim a
    zero-byte artefact exists rather than none at all)."""
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    for arm in ("overlap", "fuzzy"):
        artifact_size = record["results"][arm]["artifact_size"]
        assert "status" not in artifact_size
        assert artifact_size == {
            "value": None,
            "undefined_reason": "classical arms carry no model artifact",
        }


def test_benchmark_records_the_environment_the_readme_requires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    environment = record["environment"]
    hardware = environment["hardware"]
    assert hardware["cpu_count"] >= 1
    assert hardware["system"]
    assert hardware["python_version"]
    assert hardware["total_memory_bytes"] > 0
    assert environment["runtime_versions"]["rapidfuzz"]
    assert environment["runtime_versions"]["onnxruntime"] == {
        "value": None,
        "reason": "no neural arm in this run",
    }
    assert environment["thread_count"] == 1
    assert environment["warmup_count"] == 5
    assert environment["repetition_count"] == 20
    assert environment["power_mode"] in {"ac", "battery", "unknown"}


def test_benchmark_records_model_identity_and_runtime_version_when_the_neural_arm_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run in which ``embed-fp32`` initialises must record the runtime version
    and the model identity it actually loaded (ADR-0002, issue #59). The
    inapplicable-with-a-reason form is correct only for a run where no neural arm
    ran, and stating it for a run where one did would make the record assert
    something untrue of itself.

    This test is the only one that reaches that path. The fixture the other
    benchmark tests rely on never sets ``JOINLESS_MODEL_CACHE_DIR``, so
    ``embed-fp32`` genuinely fails there and its absence is honestly reported;
    only a run where the arm succeeds can distinguish the two. It uses a
    fake ``onnxruntime``/``tokenizers`` pair standing in for the 90 MB production
    artefact (ADR-0016 rule 2: awkward enough to prove the pooling arithmetic
    runs — see ``tests/test_embedding.py``'s own such fakes — not a claim about
    matching the real graph's output).
    """
    import hashlib
    import types
    from collections.abc import Mapping, Sequence
    from typing import cast

    from joinless import corpus as corpus_module
    from joinless import embedding
    from joinless.cli import main

    class _FakeEncoding:
        def __init__(
            self, ids: list[int], attention_mask: list[int], type_ids: list[int]
        ) -> None:
            self.ids = ids
            self.attention_mask = attention_mask
            self.type_ids = type_ids

    class _FakeTokenizer:
        def enable_padding(self, *, pad_token: str, pad_id: int) -> None:
            del pad_token, pad_id

        def enable_truncation(self, *, max_length: int) -> None:
            del max_length

        def encode_batch(self, texts: Sequence[str]) -> list[_FakeEncoding]:
            return [
                _FakeEncoding(ids=[1], attention_mask=[1], type_ids=[0]) for _ in texts
            ]

    class _TokenizerNamespace:
        @staticmethod
        def from_file(path: str) -> _FakeTokenizer:
            del path
            return _FakeTokenizer()

    fake_tokenizers = types.ModuleType("tokenizers")
    fake_tokenizers.Tokenizer = _TokenizerNamespace  # type: ignore[attr-defined]

    class _FakeSession:
        def __init__(self, path: str, providers: list[str] | None = None) -> None:
            del path, providers

        def run(
            self, output_names: list[str] | None, input_feed: Mapping[str, object]
        ) -> list[object]:
            del output_names
            rows = cast(list[list[int]], input_feed["input_ids"])
            hidden = [[[1.0, 0.0] for _ in row] for row in rows]
            return [hidden]

    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.InferenceSession = _FakeSession  # type: ignore[attr-defined]
    fake_onnxruntime.__version__ = "9.9.9-fake"  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tokenizers)

    cache_dir = tmp_path / "cache"
    fp32_dir = cache_dir / "fp32"
    fp32_dir.mkdir(parents=True)
    model_bytes = b"a fake fp32 graph"
    tokenizer_bytes = b"a fake tokenizer config"
    (fp32_dir / "model.onnx").write_bytes(model_bytes)
    (fp32_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    monkeypatch.setattr(
        embedding, "FP32_MODEL_SHA256", hashlib.sha256(model_bytes).hexdigest()
    )
    monkeypatch.setattr(
        embedding, "FP32_TOKENIZER_SHA256", hashlib.sha256(tokenizer_bytes).hexdigest()
    )
    monkeypatch.setenv("JOINLESS_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])
    assert exit_code == 0

    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))

    environment = record["environment"]
    assert environment["model"] == {
        "value": {
            "model_id": embedding.MODEL_ID,
            "revision": embedding.MODEL_REVISION,
            "checksum_sha256": embedding.FP32_MODEL_SHA256,
            "license": embedding.MODEL_LICENSE,
        },
        "reason": None,
    }
    assert environment["runtime_versions"]["onnxruntime"] == {
        "value": "9.9.9-fake",
        "reason": None,
    }

    accuracy = record["results"]["embed-fp32"]["accuracy"]
    assert accuracy["status"] == "ok"


# --- benchmark: environment fields are pinned to their real source, not a range
# (a field decoupled from what it reports must turn the suite red) ------------


def test_benchmark_records_the_cpu_count_os_actually_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from joinless import corpus as corpus_module
    from joinless.cli import main

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.setattr(os, "cpu_count", lambda: 17)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])

    assert exit_code == 0
    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["environment"]["hardware"]["cpu_count"] == 17


def test_benchmark_records_total_memory_bytes_from_os_sysconf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from joinless import corpus as corpus_module
    from joinless.cli import main

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.setattr(
        os, "sysconf", lambda name: {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1000}[name]
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])

    assert exit_code == 0
    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["environment"]["hardware"]["total_memory_bytes"] == 4096 * 1000


def test_benchmark_records_the_python_version_platform_actually_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from joinless import corpus as corpus_module
    from joinless.cli import main

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.setattr(platform, "python_version", lambda: "9.9.9")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])

    assert exit_code == 0
    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["environment"]["hardware"]["python_version"] == "9.9.9"


def test_benchmark_records_the_system_platform_actually_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from joinless import corpus as corpus_module
    from joinless.cli import main

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.setattr(platform, "system", lambda: "PinnedOS")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])

    assert exit_code == 0
    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["environment"]["hardware"]["system"] == "PinnedOS"


def test_benchmark_records_the_power_mode_detect_power_mode_actually_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from joinless import cli as cli_module
    from joinless import corpus as corpus_module

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.setattr(cli_module, "_detect_power_mode", lambda: "battery")
    monkeypatch.chdir(tmp_path)

    exit_code = cli_module.main(["benchmark"])

    assert exit_code == 0
    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["environment"]["power_mode"] == "battery"


def test_benchmark_records_the_evaluation_set_seeds_it_actually_drew_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    assert record["evaluation_set"]["seeds"] == [1]
    assert sum(record["evaluation_set"]["case_mixture"].values()) > 0


def test_benchmark_records_a_selected_threshold_per_registered_arm_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    scorer_names = {entry["scorer_name"] for entry in record["selected_thresholds"]}
    assert scorer_names == {"overlap", "fuzzy"}


def test_benchmark_records_the_pre_registered_expected_winners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    assert record["expected_winners"]["winners"]["character noise"] == "fuzzy"


# --- benchmark: contradictions (ADR-0011 rule 4, issue #50) -------------------

from joinless.evaluation import (
    AggregateResult,
    Contradiction,
    EvaluationReport,
    ExpectedWinners,
    FamilyResult,
    InvalidRun,
    Metric,
)
from joinless.measurement import Unavailable
from joinless.runrecord import ArmResult


def _report_with_f1(f1_value: float) -> EvaluationReport:
    winning_metric = Metric(value=f1_value, undefined_reason=None)
    other_metric = Metric(value=1.0, undefined_reason=None)
    family = FamilyResult(
        family="exact",
        precision=other_metric,
        recall=other_metric,
        f1=winning_metric,
        true_positives=1,
        predicted_positives=1,
        actual_positives=1,
    )
    aggregate = AggregateResult(
        precision=other_metric,
        recall=other_metric,
        f1=winning_metric,
        derivation="pooled",
    )
    return EvaluationReport(per_family=(family,), aggregate=aggregate, n_pairs=1)


def _result_with_accuracy(
    arm: str, accuracy: EvaluationReport | InvalidRun
) -> ArmResult:
    unavailable = Unavailable(arm=arm, reason="not measured in this test")
    return ArmResult(
        accuracy=accuracy,
        warm_latency=unavailable,
        peak_memory=unavailable,
        cold_start=unavailable,
        artifact_size=unavailable,
    )


def test_find_contradictions_reports_a_family_whose_actual_winner_differs() -> None:
    from joinless.cli import _find_contradictions

    expected = ExpectedWinners(winners={"exact": "fuzzy"})
    arm_results = {
        "overlap": _result_with_accuracy("overlap", _report_with_f1(1.0)),
        "fuzzy": _result_with_accuracy("fuzzy", _report_with_f1(0.5)),
    }

    contradictions = _find_contradictions(expected, arm_results)

    assert len(contradictions) == 1
    assert contradictions[0].family == "exact"
    assert contradictions[0].expected_winner == "fuzzy"
    assert contradictions[0].actual_winners == ("overlap",)


def test_find_contradictions_is_empty_when_the_expectation_holds() -> None:
    from joinless.cli import _find_contradictions

    expected = ExpectedWinners(winners={"exact": "overlap"})
    arm_results = {
        "overlap": _result_with_accuracy("overlap", _report_with_f1(1.0)),
        "fuzzy": _result_with_accuracy("fuzzy", _report_with_f1(0.5)),
    }

    assert _find_contradictions(expected, arm_results) == ()


def test_find_contradictions_skips_an_arm_whose_accuracy_is_not_a_report() -> None:
    """Only two arms are registered today (module docstring); an arm whose
    ``get_scorer`` call failed carries ``InvalidRun``, not an
    ``EvaluationReport`` - it must not be handed to
    ``joinless.evaluation.find_contradictions`` as if it had a real per-family
    table, and with only one comparable arm left the family is skipped
    entirely (that function's own docstring)."""
    from joinless.cli import _find_contradictions

    expected = ExpectedWinners(winners={"exact": "overlap"})
    arm_results = {
        "overlap": _result_with_accuracy("overlap", _report_with_f1(1.0)),
        "embed-fp32": _result_with_accuracy(
            "embed-fp32", InvalidRun(reason="'embed-fp32' is not a known scorer")
        ),
    }

    assert _find_contradictions(expected, arm_results) == ()


def test_format_contradictions_reports_none_broke_when_empty() -> None:
    from joinless.cli import _format_contradictions

    assert _format_contradictions(()) == [
        "contradictions: none — every pre-registered expectation held"
    ]


def test_format_contradictions_names_the_family_and_both_winners() -> None:
    from joinless.cli import _format_contradictions

    lines = _format_contradictions(
        (
            Contradiction(
                family="exact", expected_winner="fuzzy", actual_winners=("overlap",)
            ),
        )
    )

    assert lines[0] == "contradictions: 1 pre-registered expectation(s) did not hold"
    assert "exact" in lines[1]
    assert "fuzzy" in lines[1]
    assert "overlap" in lines[1]


def test_format_contradictions_names_every_tied_actual_winner() -> None:
    """A tie at the top is printed as the tie it is - every arm that reached
    the top score, not just one of them."""
    from joinless.cli import _format_contradictions

    lines = _format_contradictions(
        (
            Contradiction(
                family="transliteration",
                expected_winner="fuzzy",
                actual_winners=("embed-fp32", "overlap"),
            ),
        )
    )

    assert "embed-fp32" in lines[1]
    assert "overlap" in lines[1]


def test_benchmark_persists_the_same_contradictions_it_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report (#46) must render what the command found, not recompute it
    (issue #50: "two answers that can disagree") - this pins the printed
    output and the persisted ``contradictions`` field to the same computed
    value by forcing what that value is."""
    from joinless import cli as cli_module
    from joinless import corpus as corpus_module

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)
    forced = (
        Contradiction(
            family="exact", expected_winner="fuzzy", actual_winners=("overlap",)
        ),
    )
    monkeypatch.setattr(cli_module, "_find_contradictions", lambda *a, **k: forced)

    exit_code = cli_module.main(["benchmark"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "exact" in out
    assert "fuzzy" in out
    assert "overlap" in out

    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["contradictions"] == [
        {"family": "exact", "expected_winner": "fuzzy", "actual_winners": ["overlap"]}
    ]


def test_benchmark_prints_when_no_expectation_broke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from joinless import cli as cli_module
    from joinless import corpus as corpus_module

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_find_contradictions", lambda *a, **k: ())

    exit_code = cli_module.main(["benchmark"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "contradictions: none — every pre-registered expectation held" in out

    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["contradictions"] == []


# --- benchmark: a same-identifier collision is reported, not a traceback -----


def test_benchmark_reports_a_same_identifier_collision_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``write_record`` refuses to overwrite (issue #57) - that refusal must
    reach the operator as a readable message naming the path, not as an
    unhandled ``FileExistsError`` traceback."""
    import datetime as datetime_module

    from joinless import cli as cli_module
    from joinless import corpus as corpus_module

    class _FixedDatetime(datetime_module.datetime):
        @classmethod
        def now(cls, tz: datetime_module.tzinfo | None = None) -> _FixedDatetime:
            return cls(2026, 8, 13, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.setattr(cli_module, "datetime", _FixedDatetime)
    monkeypatch.chdir(tmp_path)

    first_exit = cli_module.main(["benchmark"])
    assert first_exit == 0
    capsys.readouterr()  # discard the first run's own output

    second_exit = cli_module.main(["benchmark"])

    assert second_exit == 1
    captured = capsys.readouterr()
    assert captured.err.strip() != ""
    assert "20260813T120000Z-benchmark.json" in captured.err
    assert "Traceback" not in captured.err
    written = list((tmp_path / "benchmarks").glob("*.json"))
    assert len(written) == 1


_BENCHMARK_NO_NETWORK_PROBE = """
import socket

def _blocked(*args, **kwargs):
    raise AssertionError("a network call was attempted")

socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

from joinless import corpus as corpus
corpus.SEEDS = (1,)

import sys
from joinless import cli as cli
sys.exit(cli.main(["benchmark"]))
"""


def test_benchmark_completes_with_no_network_interface_available(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _BENCHMARK_NO_NETWORK_PROBE],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert list((tmp_path / "benchmarks").glob("*.json"))


# --- benchmark: power-mode detection (RFC-0002 Method step 5) -----------------


def test_parse_pmset_output_ac_power() -> None:
    from joinless.cli import _parse_pmset_output

    assert _parse_pmset_output("Now drawing from 'AC Power'\n") == "ac"


def test_parse_pmset_output_battery_power() -> None:
    from joinless.cli import _parse_pmset_output

    assert _parse_pmset_output("Now drawing from 'Battery Power'\n") == "battery"


def test_parse_pmset_output_unknown() -> None:
    from joinless.cli import _parse_pmset_output

    assert _parse_pmset_output("garbage") == "unknown"


def test_parse_linux_power_supply_status_charging_or_full_is_ac() -> None:
    from joinless.cli import _parse_linux_power_supply_status

    assert _parse_linux_power_supply_status("Charging\n") == "ac"
    assert _parse_linux_power_supply_status("Full\n") == "ac"


def test_parse_linux_power_supply_status_discharging_is_battery() -> None:
    from joinless.cli import _parse_linux_power_supply_status

    assert _parse_linux_power_supply_status("Discharging\n") == "battery"


def test_parse_linux_power_supply_status_other_is_unknown() -> None:
    from joinless.cli import _parse_linux_power_supply_status

    assert _parse_linux_power_supply_status("Not charging\n") == "unknown"


def test_detect_linux_power_mode_reads_the_first_status_file(tmp_path: Path) -> None:
    from joinless.cli import _detect_linux_power_mode

    (tmp_path / "AC").mkdir()
    (tmp_path / "AC" / "status").write_text("Charging\n", encoding="utf-8")

    assert _detect_linux_power_mode(tmp_path) == "ac"


def test_detect_linux_power_mode_is_unknown_with_no_status_file(
    tmp_path: Path,
) -> None:
    from joinless.cli import _detect_linux_power_mode

    assert _detect_linux_power_mode(tmp_path) == "unknown"


def test_detect_power_mode_dispatches_to_pmset_on_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from joinless.cli import _detect_power_mode

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Now drawing from 'AC Power'\n"
        ),
    )

    assert _detect_power_mode() == "ac"


def test_detect_power_mode_dispatches_to_the_linux_supply_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from joinless import cli as cli_module

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli_module, "_LINUX_POWER_SUPPLY_DIR", tmp_path)
    (tmp_path / "BAT0").mkdir()
    (tmp_path / "BAT0" / "status").write_text("Discharging\n", encoding="utf-8")

    assert cli_module._detect_power_mode() == "battery"


def test_detect_power_mode_is_unknown_on_an_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from joinless.cli import _detect_power_mode

    monkeypatch.setattr(platform, "system", lambda: "Windows")

    assert _detect_power_mode() == "unknown"


# --- benchmark: pooling every seed's corpus into one -------------------------


def test_pool_corpora_combines_pairs_and_roles_from_every_seed() -> None:
    from joinless.cli import _pool_corpora
    from joinless.corpus import generate_corpus

    first = generate_corpus(1)
    second = generate_corpus(2)

    pooled = _pool_corpora([first, second])

    assert len(pooled.pairs) == len(first.pairs) + len(second.pairs)
    assert set(pooled.roles) == set(first.roles) | set(second.roles)


# --- shared: prove a command runs with no network interface available ------


def _no_network_probe(argv: list[str]) -> str:
    """A script for a *child* interpreter that makes every socket-construction
    entry point raise, then runs ``joinless.cli.main(argv)``.

    Blocking in this process would prove nothing: pytest's own machinery, and
    every fixture already used above, would need the network blocked too, and
    nothing here is trying to test them. A child interpreter isolates the
    claim to exactly the command under test - and if the command opened a
    connection, the raised exception is uncaught, so the child exits non-zero
    and the assertion in the calling test fails.
    """
    return f"""
import socket
import sys

def _blocked(*args, **kwargs):
    raise AssertionError("a network call was attempted")

socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

from joinless import cli as cli
sys.exit(cli.main({argv!r}))
"""
