# SPDX-License-Identifier: MIT
"""The console entry point: ``resolve``, ``compare``, ``doctor``.

``report`` and ``benchmark`` (issues #45, #46) do not exist yet, and this suite
guards that as a fact about the CLI, not an accident of what nobody got round to
adding — a word that would be a subcommand name in a later version must still be
rejected as an unrecognized argument today.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
    # report and benchmark are #45/#46, not built yet - --help must not imply
    # they exist.
    assert "report" not in out
    assert "benchmark" not in out


@pytest.mark.parametrize("word", ["report", "benchmark"])
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
    assert "onnxruntime" not in sys.modules


def test_doctor_reports_the_neural_profile_without_importing_the_runtime(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #44: 'without importing the inference runtime to find out' -
    faking a positive find_spec result (rather than actually installing the
    extra) is what proves the detection path itself never imports
    onnxruntime, regardless of which way the answer comes out."""
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
    assert "onnxruntime" not in sys.modules


def test_doctor_completes_with_no_network_interface_available() -> None:
    argv = ["doctor"]
    result = subprocess.run(
        [sys.executable, "-c", _no_network_probe(argv)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


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

import joinless.cli as cli
sys.exit(cli.main({argv!r}))
"""
