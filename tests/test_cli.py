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
import types
from pathlib import Path
from typing import Any

import pytest

from joinless.resolver import _REASON_NO_COORDINATES, _REASON_NOT_SELECTED


def _install_fake_neural_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a fake ``onnxruntime``/``tokenizers`` pair in :data:`sys.modules`.

    One definition for every benchmark test that needs a neural arm to load
    without the 91 MB and 59 MB artefacts. ADR-0016 rule 2 asks a stand-in to be
    at least as awkward as the thing it replaces, and this pair is: the session
    returns one hidden state per token id it was given, so a caller that skipped
    the mask, mispaired a batch, or pooled the wrong axis gets a wrong answer
    rather than a convenient one.

    The embedding it produces is the same for every name, which makes each arm's
    predictions identical and every family's divergence exactly ``0.0``. That is
    a deterministic property of this double, never a claim about what the real
    graphs agree on.
    """
    import types
    from collections.abc import Mapping, Sequence
    from typing import cast

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


def _fake_onnx_module(op_types: list[str]) -> types.ModuleType:
    """A fake ``onnx`` module whose ``load`` returns an object exposing
    ``graph.node`` with exactly ``op_types`` (as each node's ``.op_type``) -
    standing in for the real ``onnx.load`` the same way the neighbouring
    fakes stand in for ``onnxruntime``/``tokenizers`` (ADR-0016 rule 2), so
    quantized-operator verification (issue #68) can be exercised end to end
    without a real graph on disk.
    """

    class _FakeNode:
        def __init__(self, op_type: str) -> None:
            self.op_type = op_type

    class _FakeGraph:
        def __init__(self, nodes: list[str]) -> None:
            self.node = [_FakeNode(op_type) for op_type in nodes]

    class _FakeModel:
        def __init__(self, nodes: list[str]) -> None:
            self.graph = _FakeGraph(nodes)

    module = types.ModuleType("onnx")

    def _load(path: str) -> _FakeModel:
        del path
        return _FakeModel(op_types)

    module.load = _load  # type: ignore[attr-defined]
    return module


def test_benchmark_writes_one_record_carrying_its_schema_and_exact_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    assert record["schema"] == "benchmark-v4"
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


def test_benchmark_records_every_configured_arm_including_the_two_neural_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #45's third bullet: "an arm that cannot initialise is recorded as
    unavailable with a reason, not omitted" - all four configured arms must have
    a row. The fixture this test shares with its neighbours never sets
    ``JOINLESS_MODEL_CACHE_DIR``, so both neural arms are registered but
    unavailable here (issue #67) - the "not omitted" half of ADR-0013 applies
    to that case exactly as it does to a name nothing registers at all."""
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    assert set(record["results"]) == {"overlap", "fuzzy", "embed-fp32", "embed-int8"}


def test_benchmark_records_an_arm_without_a_configured_cache_dir_as_unavailable(
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


def test_benchmark_records_both_preparation_paths_and_they_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #65's third bullet: "the run record states which one produced
    each figure" - each registered arm's ``preparation`` entry carries both
    the hoisted and the naive path's own score, each self-tagged
    (``ScoredComparisons.path``), and the two agree exactly - the same
    invariant ``tests/test_resolver.py`` already asserts at the resolver
    level, now exercised through the real command rather than mocked."""
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    for arm in ("overlap", "fuzzy"):
        preparation = record["results"][arm]["preparation"]
        assert preparation["status"] == "ok"
        assert preparation["hoisted"]["path"] == "hoisted"
        assert preparation["naive"]["path"] == "naive"
        assert preparation["hoisted"]["scores"] == preparation["naive"]["scores"]
        assert preparation["hoisted"]["scores"]


def test_benchmark_records_preparation_as_unavailable_for_an_unregistered_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither the accuracy pipeline nor either preparation path ever runs
    for an arm whose dependency is missing (ADR-0013) - ``preparation`` is
    unavailable for the same reason its siblings are, not silently absent."""
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    for arm in ("embed-fp32", "embed-int8"):
        preparation = record["results"][arm]["preparation"]
        assert preparation["status"] == "unavailable"
        assert arm in preparation["reason"]


# --- preparation cost and bucket occupancy (issue #66) ------------------------


def test_benchmark_records_the_bucket_occupancy_the_preparation_sample_actually_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #66's second bullet: "the bucket occupancy distribution for the
    run is recorded in the same record" - a real distribution (not a mean),
    read from the resolver's own grid blocking over the run's shared
    preparation sample, not a configured parameter echoed back.

    Finding 1: it lives nested under ``preparation_asymmetry``, not as a
    bare top-level field a reader could mistake for describing the accuracy
    evaluation's own 470-pair corpus rather than this one 20-record sample -
    and it carries its own ``max_occupancy`` rather than requiring a reader
    to reduce ``counts`` themselves."""
    from joinless.resolver import DEFAULT_CELL_SIZE_DEGREES

    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    assert "bucket_occupancy" not in record
    occupancy = record["preparation_asymmetry"]["occupancy"]
    assert sorted(occupancy["counts"]) == [1, 2, 3, 4]
    assert occupancy["cell_size_degrees"] == DEFAULT_CELL_SIZE_DEGREES
    assert occupancy["max_occupancy"] == 4


def test_benchmark_records_preparation_cost_for_a_registered_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #66's first bullet: "hoisted and naive preparation cost are
    recorded per arm" - measured in an isolated worker (RFC-0002 Method
    step 7), over the exact candidate set whose occupancy this run also
    reports (ADR-0009)."""
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    for arm in ("overlap", "fuzzy"):
        cost = record["results"][arm]["preparation_cost"]
        assert cost["status"] == "ok"
        assert cost["arm"] == arm
        assert cost["hoisted_seconds"] >= 0.0
        assert cost["naive_seconds"] >= 0.0
        assert cost["record_count"] == 20
        assert cost["comparison_count"] == 30


def test_benchmark_records_preparation_cost_as_unavailable_for_an_unregistered_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    for arm in ("embed-fp32", "embed-int8"):
        cost = record["results"][arm]["preparation_cost"]
        assert cost["status"] == "unavailable"
        assert arm in cost["reason"]


def test_benchmark_records_the_classical_arm_preparation_speedup_and_no_neural_speedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #66's third bullet: "the asymmetry between classical and
    neural arms is reported as a result" - derivable from the record alone.
    Neither neural arm has a configured artefact in this fixture, so
    ``neural_speedups`` is empty rather than populated with a fabricated
    figure (ADR-0013)."""
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    asymmetry = record["preparation_asymmetry"]
    assert set(asymmetry["classical_speedups"]) == {"overlap", "fuzzy"}
    assert asymmetry["neural_speedups"] == {}
    for metric in asymmetry["classical_speedups"].values():
        assert (metric["value"] is None) != (metric["undefined_reason"] is None)


def test_benchmark_prints_the_preparation_asymmetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from joinless import corpus as corpus_module
    from joinless.cli import main

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "preparation hoist speed-up" in out
    assert "overlap" in out
    assert "fuzzy" in out
    # Finding 1: a reader of the printed output, not only of the JSON
    # record, can see the occupancy the speed-ups were measured at.
    assert "candidate-bucket occupancy" in out
    assert "max 4" in out


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
    assert environment["models"] == {}
    assert environment["quantized_operators"] == {
        "value": None,
        "reason": "no int8 arm in this run",
    }
    assert environment["thread_count"] == 1
    assert environment["warmup_count"] == 5
    assert environment["repetition_count"] == 20
    assert environment["power_mode"] in {"ac", "battery", "unknown"}
    # Finding 2: states which path produced every arm's `accuracy` and
    # `warm_latency` - the two figures a reader actually compares arms on
    # that previously carried no such attribution.
    assert environment["measurement_preparation_path"] == "hoisted"


def test_benchmark_records_model_identity_and_runtime_version_when_the_neural_arm_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run in which ``embed-fp32`` initialises must record the runtime version
    and the model identity it actually loaded (ADR-0002, issue #59). The
    inapplicable-with-a-reason form is correct only for a run where no neural arm
    ran, and stating it for a run where one did would make the record assert
    something untrue of itself.

    The fixture the other benchmark tests rely on never sets
    ``JOINLESS_MODEL_CACHE_DIR``, so ``embed-fp32`` genuinely fails there and
    its absence is honestly reported; only a run where the arm succeeds can
    distinguish the two. Only ``embed-fp32``'s artefact is configured here —
    ``embed-int8`` still fails, so ``environment["models"]`` carries one entry;
    ``test_benchmark_records_both_neural_arms_models_and_their_accuracy_divergence_when_both_run``
    below covers the two-neural-arms case. It uses a fake
    ``onnxruntime``/``tokenizers`` pair standing in for the 90 MB production
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
    assert environment["models"] == {
        "embed-fp32": {
            "model_id": embedding.MODEL_ID,
            "revision": embedding.MODEL_REVISION,
            "checksum_sha256": embedding.FP32_MODEL_SHA256,
            "license": embedding.MODEL_LICENSE,
        }
    }
    assert environment["runtime_versions"]["onnxruntime"] == {
        "value": "9.9.9-fake",
        "reason": None,
    }

    accuracy = record["results"]["embed-fp32"]["accuracy"]
    assert accuracy["status"] == "ok"


def test_benchmark_records_both_neural_arms_models_and_their_accuracy_divergence_when_both_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run in which both ``embed-fp32`` and ``embed-int8`` initialise must
    record both model identities (issue #67) - not just whichever one the
    loop happened to measure last, which is exactly the bug a single
    ``Environment.model`` slot could not avoid once a second neural arm
    existed - and the int8 arm's per-family F1 divergence from the fp32 arm,
    computed from this run's own two accuracy reports rather than asserted in
    prose (issue #67's third bullet).

    The two arms share one fake ``onnxruntime``/``tokenizers`` pair, so their
    predictions are identical and every family's divergence is exactly
    ``0.0`` — a small, deterministic outcome, not a claim about matching the
    real graphs' output (ADR-0016 rule 2).
    """
    import hashlib

    from joinless import corpus as corpus_module
    from joinless import embedding
    from joinless.cli import main

    _install_fake_neural_runtime(monkeypatch)
    # issue #68: the real graph's own quantized-operator census, read at run
    # time - here made to match both embedding.INT8_QUANTIZED_OPERATORS and
    # embedding.INT8_MATMUL_CONVERSION exactly (36 converted, 12 remaining), so
    # this "both arms succeed" run also succeeds past that check.
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        _fake_onnx_module(
            ["DynamicQuantizeLinear"] + ["MatMulInteger"] * 36 + ["MatMul"] * 12
        ),
    )

    cache_dir = tmp_path / "cache"
    fp32_dir = cache_dir / "fp32"
    int8_dir = cache_dir / "int8"
    fp32_dir.mkdir(parents=True)
    int8_dir.mkdir(parents=True)
    tokenizer_bytes = b"a fake tokenizer config"
    fp32_model_bytes = b"a fake fp32 graph"
    int8_model_bytes = b"a fake int8 graph"
    (fp32_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    (fp32_dir / "model.onnx").write_bytes(fp32_model_bytes)
    (int8_dir / "model.onnx").write_bytes(int8_model_bytes)
    monkeypatch.setattr(
        embedding, "FP32_TOKENIZER_SHA256", hashlib.sha256(tokenizer_bytes).hexdigest()
    )
    monkeypatch.setattr(
        embedding, "FP32_MODEL_SHA256", hashlib.sha256(fp32_model_bytes).hexdigest()
    )
    monkeypatch.setattr(
        embedding, "INT8_MODEL_SHA256", hashlib.sha256(int8_model_bytes).hexdigest()
    )
    monkeypatch.setenv("JOINLESS_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])
    assert exit_code == 0

    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))

    environment = record["environment"]
    assert set(environment["models"]) == {"embed-fp32", "embed-int8"}
    assert (
        environment["models"]["embed-fp32"]["checksum_sha256"]
        == embedding.FP32_MODEL_SHA256
    )
    assert (
        environment["models"]["embed-int8"]["checksum_sha256"]
        == embedding.INT8_MODEL_SHA256
    )
    assert (
        environment["models"]["embed-fp32"]["checksum_sha256"]
        != environment["models"]["embed-int8"]["checksum_sha256"]
    )
    # issue #68's first and second bullets: every record for an int8 arm
    # carries the matmul-conversion census (finding 1: converted/fp32/remaining
    # counts, not a bare operator-type list), read from the graph above - not
    # asserted from embedding.INT8_MATMUL_CONVERSION directly, but produced by
    # the same command path a reader would run themselves.
    assert environment["quantized_operators"] == {
        "value": {
            "Gemm": {"converted_count": 0, "fp32_count": 0, "int8_count_remaining": 0},
            "MatMul": {
                "converted_count": 36,
                "fp32_count": 48,
                "int8_count_remaining": 12,
            },
        },
        "reason": None,
    }

    assert record["results"]["embed-fp32"]["accuracy"]["status"] == "ok"
    assert record["results"]["embed-int8"]["accuracy"]["status"] == "ok"

    divergence = record["int8_accuracy_divergence"]
    assert divergence["reason"] is None
    assert divergence["value"]
    # Both arms share one fake session/tokenizer pair, so every family's fp32
    # and int8 F1 is identical - either both defined and equal (delta 0.0) or
    # both undefined for the same reason (an empty split for that family under
    # this corpus's small fixture is a real, if uninteresting, possibility).
    for row in divergence["value"]:
        if row["baseline_f1"]["value"] is None:
            assert row["delta_f1"]["value"] is None
        else:
            assert row["delta_f1"]["value"] == pytest.approx(0.0)


def test_benchmark_writes_no_record_when_the_int8_graph_does_not_match_its_recorded_operator_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #68's third bullet, constructed directly rather than reasoned
    about: the fake int8 graph below reports a quantized-operator census
    that is not ``embedding.INT8_QUANTIZED_OPERATORS`` (no operators
    converted at all), so the whole run must refuse to write a record -
    not mark just the int8 arm unavailable while ``overlap``, ``fuzzy`` and
    ``embed-fp32``'s already-measured results are written as if nothing
    were wrong. That is a stronger response than an ordinary checksum
    mismatch (ADR-0013 rule 3, one arm marked unavailable, the record
    still written): a graph whose operator census contradicts what the
    checksum-verified artefact is recorded to contain means this run's own
    understanding of what it measured cannot be trusted, not just one row
    of it.
    """
    import hashlib

    from joinless import corpus as corpus_module
    from joinless import embedding
    from joinless.cli import main

    _install_fake_neural_runtime(monkeypatch)
    # The mismatch: a graph that converted nothing at all, not
    # embedding.INT8_QUANTIZED_OPERATORS.
    monkeypatch.setitem(sys.modules, "onnx", _fake_onnx_module(["MatMul", "Add"]))

    cache_dir = tmp_path / "cache"
    fp32_dir = cache_dir / "fp32"
    int8_dir = cache_dir / "int8"
    fp32_dir.mkdir(parents=True)
    int8_dir.mkdir(parents=True)
    tokenizer_bytes = b"a fake tokenizer config"
    fp32_model_bytes = b"a fake fp32 graph"
    int8_model_bytes = b"a fake int8 graph"
    (fp32_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    (fp32_dir / "model.onnx").write_bytes(fp32_model_bytes)
    (int8_dir / "model.onnx").write_bytes(int8_model_bytes)
    monkeypatch.setattr(
        embedding, "FP32_TOKENIZER_SHA256", hashlib.sha256(tokenizer_bytes).hexdigest()
    )
    monkeypatch.setattr(
        embedding, "FP32_MODEL_SHA256", hashlib.sha256(fp32_model_bytes).hexdigest()
    )
    monkeypatch.setattr(
        embedding, "INT8_MODEL_SHA256", hashlib.sha256(int8_model_bytes).hexdigest()
    )
    monkeypatch.setenv("JOINLESS_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])

    assert exit_code != 0
    benchmarks_dir = tmp_path / "benchmarks"
    assert not benchmarks_dir.exists() or list(benchmarks_dir.glob("*.json")) == []
    err = capsys.readouterr().err
    assert str(int8_dir / "model.onnx") in err
    assert "MatMulInteger" in err


def test_benchmark_writes_no_record_when_the_matmul_conversion_counts_do_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Issue #68 finding 1's extension of the same guard, constructed
    directly: a graph whose replacement operator *types* match
    ``embedding.INT8_QUANTIZED_OPERATORS`` exactly (``DynamicQuantizeLinear``
    and ``MatMulInteger`` both present, nothing extra) but whose *counts*
    differ from ``embedding.INT8_MATMUL_CONVERSION`` (30 conversions where 36
    are recorded) must refuse to write a record too - the old, type-only
    check would have let this graph through.
    """
    import hashlib

    from joinless import corpus as corpus_module
    from joinless import embedding
    from joinless.cli import main

    _install_fake_neural_runtime(monkeypatch)
    # The mismatch: types present are exactly right, but only 30 MatMulInteger
    # nodes exist where the pinned census records 36.
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        _fake_onnx_module(
            ["DynamicQuantizeLinear"] + ["MatMulInteger"] * 30 + ["MatMul"] * 18
        ),
    )

    cache_dir = tmp_path / "cache"
    fp32_dir = cache_dir / "fp32"
    int8_dir = cache_dir / "int8"
    fp32_dir.mkdir(parents=True)
    int8_dir.mkdir(parents=True)
    tokenizer_bytes = b"a fake tokenizer config"
    fp32_model_bytes = b"a fake fp32 graph"
    int8_model_bytes = b"a fake int8 graph"
    (fp32_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    (fp32_dir / "model.onnx").write_bytes(fp32_model_bytes)
    (int8_dir / "model.onnx").write_bytes(int8_model_bytes)
    monkeypatch.setattr(
        embedding, "FP32_TOKENIZER_SHA256", hashlib.sha256(tokenizer_bytes).hexdigest()
    )
    monkeypatch.setattr(
        embedding, "FP32_MODEL_SHA256", hashlib.sha256(fp32_model_bytes).hexdigest()
    )
    monkeypatch.setattr(
        embedding, "INT8_MODEL_SHA256", hashlib.sha256(int8_model_bytes).hexdigest()
    )
    monkeypatch.setenv("JOINLESS_MODEL_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark"])

    assert exit_code != 0
    benchmarks_dir = tmp_path / "benchmarks"
    assert not benchmarks_dir.exists() or list(benchmarks_dir.glob("*.json")) == []
    err = capsys.readouterr().err
    assert str(int8_dir / "model.onnx") in err
    assert "matmul-conversion census" in err
    assert "30" in err
    assert "36" in err


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


def test_benchmark_records_an_absent_int8_accuracy_divergence_when_neither_neural_arm_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture every other benchmark test shares never sets
    ``JOINLESS_MODEL_CACHE_DIR``, so neither neural arm produces a comparable
    accuracy report here - ``int8_accuracy_divergence`` is therefore an
    explicit absence with a reason (ADR-0013), never an empty list or a
    silently-omitted field."""
    record = _run_benchmark_with_small_corpus(tmp_path, monkeypatch)

    divergence = record["int8_accuracy_divergence"]
    assert divergence["value"] is None
    assert divergence["reason"] is not None


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
from joinless.measurement import PreparationCost, Unavailable
from joinless.runrecord import ArmResult, BucketOccupancy


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
        preparation=unavailable,
        preparation_cost=unavailable,
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
    """An arm whose ``get_scorer`` call failed - unregistered, or registered
    but unavailable (module docstring) - carries ``InvalidRun``, not an
    ``EvaluationReport``. It must not be handed to
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


# --- int8 accuracy divergence: computed once from the two arms' own reports
# (issue #67's third bullet) -----------------------------------------------


def test_int8_accuracy_divergence_computes_from_both_arms_own_reports() -> None:
    from joinless.cli import _int8_accuracy_divergence

    arm_results = {
        "overlap": _result_with_accuracy("overlap", _report_with_f1(1.0)),
        "embed-fp32": _result_with_accuracy("embed-fp32", _report_with_f1(1.0)),
        "embed-int8": _result_with_accuracy("embed-int8", _report_with_f1(0.9)),
    }

    divergence = _int8_accuracy_divergence(arm_results)

    assert divergence.reason is None
    assert divergence.value is not None
    [row] = divergence.value
    assert row.family == "exact"
    assert row.delta_f1.value == pytest.approx(-0.1)


def test_int8_accuracy_divergence_is_absent_with_a_reason_when_fp32_did_not_run() -> (
    None
):
    from joinless.cli import _int8_accuracy_divergence

    arm_results = {
        "embed-fp32": _result_with_accuracy(
            "embed-fp32", InvalidRun(reason="'embed-fp32' is unavailable")
        ),
        "embed-int8": _result_with_accuracy("embed-int8", _report_with_f1(0.9)),
    }

    divergence = _int8_accuracy_divergence(arm_results)

    assert divergence.value is None
    assert divergence.reason is not None
    assert "embed-fp32" in divergence.reason


def test_int8_accuracy_divergence_is_absent_with_a_reason_when_int8_did_not_run() -> (
    None
):
    from joinless.cli import _int8_accuracy_divergence

    arm_results = {
        "embed-fp32": _result_with_accuracy("embed-fp32", _report_with_f1(1.0)),
        "embed-int8": _result_with_accuracy(
            "embed-int8", InvalidRun(reason="'embed-int8' is unavailable")
        ),
    }

    divergence = _int8_accuracy_divergence(arm_results)

    assert divergence.value is None
    assert divergence.reason is not None
    assert "embed-int8" in divergence.reason


def test_int8_accuracy_divergence_is_absent_with_a_reason_when_neither_arm_ran() -> (
    None
):
    """Neither key is present at all (a caller building ``arm_results`` from
    only the classical arms) - the same absence, reached the other way."""
    from joinless.cli import _int8_accuracy_divergence

    arm_results = {
        "overlap": _result_with_accuracy("overlap", _report_with_f1(1.0)),
    }

    divergence = _int8_accuracy_divergence(arm_results)

    assert divergence.value is None
    assert divergence.reason is not None


# --- preparation cost & bucket occupancy helpers (issue #66) -----------------------


def _cost(arm: str, *, hoisted_seconds: float, naive_seconds: float) -> PreparationCost:
    return PreparationCost(
        arm=arm,
        hoisted_seconds=hoisted_seconds,
        naive_seconds=naive_seconds,
        record_count=20,
        comparison_count=30,
    )


def _occupancy() -> BucketOccupancy:
    return BucketOccupancy(counts=(1, 2, 3, 4), cell_size_degrees=0.01, max_occupancy=4)


def test_build_preparation_sample_produces_the_configured_occupancy_distribution() -> (
    None
):
    """The sample's occupancy is a real, uneven distribution - not one
    repeated value - so the run's own ``bucket_occupancy`` figure is a
    genuine distribution to report (issue #66's second bullet)."""
    from joinless import corpus as corpus_module
    from joinless.cli import _build_preparation_sample, _pool_corpora

    pooled = _pool_corpora(corpus_module.generate_corpora((1,)))

    sample = _build_preparation_sample(pooled)

    assert len(sample.left_names) == 20 // 2
    assert len(sample.right_names) == 20 // 2
    assert sorted(sample.occupancy.counts) == [1, 2, 3, 4]
    assert sample.occupancy.max_occupancy == 4
    assert len(sample.comparison_pairs) == 30


def test_build_preparation_sample_rejects_a_corpus_too_small_to_draw_from() -> None:
    from joinless.cli import _build_preparation_sample
    from joinless.corpus import Corpus, LabelledPair

    tiny = Corpus(
        seed=1,
        pairs=(
            LabelledPair(
                left_name="A", right_name="B", label=1, pair_id="0001-exact-000"
            ),
        ),
        roles={"0001-exact-000": "development"},
    )

    with pytest.raises(ValueError, match="fewer pairs"):
        _build_preparation_sample(tiny)


def test_hoist_speedup_is_the_naive_over_hoisted_ratio() -> None:
    from joinless.cli import _hoist_speedup

    metric = _hoist_speedup(
        _cost("embed-fp32", hoisted_seconds=0.01, naive_seconds=0.5)
    )

    assert metric.value == 50.0
    assert metric.undefined_reason is None


def test_hoist_speedup_is_undefined_when_hoisted_measured_at_zero_seconds() -> None:
    from joinless.cli import _hoist_speedup

    metric = _hoist_speedup(_cost("overlap", hoisted_seconds=0.0, naive_seconds=0.0))

    assert metric.value is None
    assert metric.undefined_reason is not None


def test_preparation_asymmetry_partitions_classical_and_neural_arms() -> None:
    import dataclasses

    from joinless.cli import _preparation_asymmetry

    arm_results = {
        "overlap": dataclasses.replace(
            _result_with_accuracy("overlap", _report_with_f1(1.0)),
            preparation_cost=_cost("overlap", hoisted_seconds=0.01, naive_seconds=0.02),
        ),
        "embed-fp32": dataclasses.replace(
            _result_with_accuracy("embed-fp32", _report_with_f1(1.0)),
            preparation_cost=_cost(
                "embed-fp32", hoisted_seconds=0.01, naive_seconds=1.0
            ),
        ),
    }

    asymmetry = _preparation_asymmetry(arm_results, _occupancy())

    assert set(asymmetry.classical_speedups) == {"overlap"}
    assert set(asymmetry.neural_speedups) == {"embed-fp32"}
    assert asymmetry.neural_speedups["embed-fp32"].value == 100.0
    assert asymmetry.occupancy == _occupancy()


def test_preparation_asymmetry_skips_an_arm_whose_cost_is_unavailable() -> None:
    from joinless.cli import _preparation_asymmetry

    arm_results = {
        "overlap": _result_with_accuracy("overlap", _report_with_f1(1.0)),
    }

    asymmetry = _preparation_asymmetry(arm_results, _occupancy())

    assert asymmetry.classical_speedups == {}
    assert asymmetry.neural_speedups == {}


def test_preparation_asymmetry_ignores_an_arm_outside_both_named_families() -> None:
    """A defensive branch this project's own four registered arms never
    reach in practice (every name in ``_ARMS`` is one or the other) - kept
    testable directly rather than asserted only by omission."""
    import dataclasses

    from joinless.cli import _preparation_asymmetry

    result = dataclasses.replace(
        _result_with_accuracy("mystery-arm", _report_with_f1(1.0)),
        preparation_cost=_cost("mystery-arm", hoisted_seconds=0.01, naive_seconds=0.02),
    )

    asymmetry = _preparation_asymmetry({"mystery-arm": result}, _occupancy())

    assert asymmetry.classical_speedups == {}
    assert asymmetry.neural_speedups == {}


def test_format_preparation_asymmetry_reports_no_arm_when_both_groups_are_empty() -> (
    None
):
    from joinless.cli import _format_preparation_asymmetry
    from joinless.runrecord import PreparationAsymmetry

    lines = _format_preparation_asymmetry(
        PreparationAsymmetry(
            occupancy=_occupancy(), classical_speedups={}, neural_speedups={}
        )
    )

    assert lines[0] == "preparation hoist speed-up (naive seconds / hoisted seconds):"
    assert any("candidate-bucket occupancy" in line for line in lines)
    assert any("classical: no arm" in line for line in lines)
    assert any("neural: no arm" in line for line in lines)


def test_format_preparation_asymmetry_names_each_arms_speed_up_factor() -> None:
    from joinless.cli import _format_preparation_asymmetry
    from joinless.runrecord import PreparationAsymmetry

    lines = _format_preparation_asymmetry(
        PreparationAsymmetry(
            occupancy=_occupancy(),
            classical_speedups={"overlap": Metric(value=1.02, undefined_reason=None)},
            neural_speedups={"embed-fp32": Metric(value=48.6, undefined_reason=None)},
        )
    )

    assert any("overlap" in line and "1.02x" in line for line in lines)
    assert any("embed-fp32" in line and "48.60x" in line for line in lines)


def test_format_preparation_asymmetry_reports_undefined_when_the_ratio_has_none() -> (
    None
):
    from joinless.cli import _format_preparation_asymmetry
    from joinless.runrecord import PreparationAsymmetry

    lines = _format_preparation_asymmetry(
        PreparationAsymmetry(
            occupancy=_occupancy(),
            classical_speedups={
                "overlap": Metric(
                    value=None, undefined_reason="hoisted preparation measured at 0"
                )
            },
            neural_speedups={},
        )
    )

    assert any("overlap" in line and "undefined" in line for line in lines)


def test_benchmark_persists_the_same_preparation_asymmetry_it_prints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from joinless import cli as cli_module
    from joinless import corpus as corpus_module
    from joinless.runrecord import PreparationAsymmetry

    monkeypatch.setattr(corpus_module, "SEEDS", (1,))
    monkeypatch.chdir(tmp_path)
    forced = PreparationAsymmetry(
        occupancy=_occupancy(),
        classical_speedups={"overlap": Metric(value=2.0, undefined_reason=None)},
        neural_speedups={},
    )
    monkeypatch.setattr(cli_module, "_preparation_asymmetry", lambda *a, **k: forced)

    exit_code = cli_module.main(["benchmark"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "overlap" in out
    assert "2.00x" in out

    written = list((tmp_path / "benchmarks").glob("*.json"))
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["preparation_asymmetry"] == {
        "occupancy": {
            "counts": [1, 2, 3, 4],
            "cell_size_degrees": 0.01,
            "max_occupancy": 4,
        },
        "classical_speedups": {"overlap": {"value": 2.0, "undefined_reason": None}},
        "neural_speedups": {},
    }


# --- quantized-operator census: read live from the int8 graph, keyed by candidate
# operator type with converted/fp32/remaining counts (issue #68 finding 1) ----------


def test_quantized_operators_is_absent_with_a_reason_when_no_int8_arm_ran() -> None:
    from joinless.cli import _quantized_operators

    result = _quantized_operators({})

    assert result.value is None
    assert result.reason == "no int8 arm in this run"


def test_quantized_operators_reads_the_matmul_conversion_census_live_from_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from joinless import embedding
    from joinless.cli import _quantized_operators
    from joinless.runrecord import ModelIdentity

    monkeypatch.setitem(
        sys.modules,
        "onnx",
        _fake_onnx_module(
            ["DynamicQuantizeLinear"] + ["MatMulInteger"] * 36 + ["MatMul"] * 12
        ),
    )
    monkeypatch.setenv("JOINLESS_MODEL_CACHE_DIR", "/some/cache")
    models = {
        "embed-int8": ModelIdentity(
            model_id=embedding.MODEL_ID,
            revision=embedding.MODEL_REVISION,
            checksum_sha256=embedding.INT8_MODEL_SHA256,
            license=embedding.MODEL_LICENSE,
        )
    }

    result = _quantized_operators(models)

    assert result.reason is None
    assert result.value == embedding.INT8_MATMUL_CONVERSION


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


def test_the_schema_tag_names_one_transition_from_the_last_published_version() -> None:
    """The persisted shape changed several times across this branch's work, but
    only ``benchmark-v3`` was ever written into a record anyone can hold. A tag
    that counted intermediate, uncommitted steps would imply versions that never
    existed to read.
    """
    from joinless.cli import _SCHEMA

    assert _SCHEMA == "benchmark-v4"
