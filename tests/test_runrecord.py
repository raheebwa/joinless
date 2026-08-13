# SPDX-License-Identifier: MIT
"""The run record: the durable artefact `joinless benchmark` writes to
``benchmarks/`` (RFC-0002 "Output", benchmarks/README.md, issue #57)."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

import pytest

from joinless.evaluation import (
    AggregateResult,
    EvaluationReport,
    ExpectedWinners,
    FamilyResult,
    Metric,
    SealedTestAccuracy,
    compute_family_variation,
)
from joinless.runrecord import (
    ArmResult,
    BucketOccupancy,
    Environment,
    EvaluationSetIdentity,
    Hardware,
    Maybe,
    PreparationAsymmetry,
    RunAssembly,
    RunRecord,
    RuntimeVersions,
)

_STARTED_AT = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)


def _environment() -> Environment:
    return Environment(
        hardware=Hardware(
            cpu_count=8,
            machine="arm64",
            python_version="3.14.5",
            release="25.5.0",
            system="Darwin",
            total_memory_bytes=34359738368,
        ),
        runtime_versions=RuntimeVersions(
            onnxruntime=Maybe(value=None, reason="no neural arm in this run"),
            rapidfuzz="3.10.0",
        ),
        power_mode="ac",
        thread_count=1,
        warmup_count=5,
        repetition_count=20,
        models={},
        quantized_operators=Maybe(value=None, reason="no int8 arm in this run"),
        measurement_preparation_path="hoisted",
    )


def _evaluation_set() -> EvaluationSetIdentity:
    return EvaluationSetIdentity(seeds=(1,), case_mixture={"exact": 30})


def _bucket_occupancy() -> BucketOccupancy:
    return BucketOccupancy(counts=(1, 2, 3), cell_size_degrees=0.01, max_occupancy=3)


def _preparation_asymmetry() -> PreparationAsymmetry:
    return PreparationAsymmetry(
        occupancy=_bucket_occupancy(), classical_speedups={}, neural_speedups={}
    )


def _ok_report() -> EvaluationReport:
    metric = Metric(value=1.0, undefined_reason=None)
    family = FamilyResult(
        family="exact",
        precision=metric,
        recall=metric,
        f1=metric,
        true_positives=1,
        predicted_positives=1,
        actual_positives=1,
        false_positives=0,
    )
    aggregate = AggregateResult(
        precision=metric, recall=metric, f1=metric, derivation="pooled"
    )
    return EvaluationReport(per_family=(family,), aggregate=aggregate, n_pairs=1)


def _ok_accuracy() -> SealedTestAccuracy:
    """A minimal, valid :class:`SealedTestAccuracy` (issue #97) — one seed's
    report standing in for both the pooled figure and its own by-seed entry,
    with the real :func:`compute_family_variation` behind it, so every test
    below that only needs *an* ok accuracy value gets one that could not
    have been constructed if the pooled-without-variation invariant were
    violated."""
    report = _ok_report()
    by_seed = {1: report}
    return SealedTestAccuracy(
        pooled=report,
        pooled_answers="what the pooled figure answers in this test",
        by_seed=by_seed,
        variation=compute_family_variation(by_seed),
        by_seed_answers="what the per-seed figures answer in this test",
    )


def test_a_present_maybe_carries_no_reason() -> None:
    present = Maybe(value="1.28.0", reason=None)
    assert present.value == "1.28.0"
    assert present.reason is None


def test_an_absent_maybe_carries_a_reason_instead_of_a_value() -> None:
    absent: Maybe[str] = Maybe(value=None, reason="no neural arm in this run")
    assert absent.value is None
    assert absent.reason == "no neural arm in this run"


def test_a_maybe_cannot_carry_both_a_value_and_a_reason() -> None:
    with pytest.raises(ValueError, match="must be None exactly when"):
        Maybe(value="1.28.0", reason="no neural arm in this run")


def test_a_maybe_cannot_omit_both_a_value_and_a_reason() -> None:
    with pytest.raises(ValueError, match="must be None exactly when"):
        Maybe(value=None, reason=None)


def test_model_identity_carries_the_model_cards_licence_alongside_its_identity() -> (
    None
):
    """Issue #59: "the model card's licence is recorded alongside the artefact
    identity" - a fourth field on the same type that already carries
    ``model_id``/``revision``/``checksum_sha256``, not a second place a reader
    has to look to find it."""
    from joinless.runrecord import ModelIdentity

    identity = ModelIdentity(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        checksum_sha256="e3fe9a9a8c877bd5ca0deebb6303aba138acc6818440211377afaca1ba78b511",
        license="apache-2.0",
    )
    assert identity.license == "apache-2.0"


def test_matmul_conversion_carries_converted_fp32_and_remaining_counts() -> None:
    """Issue #68's stated purpose: "how many of the graph's matmuls were
    converted and how many remain in fp32" - the three counts this type
    exists to hold together, mirroring
    benchmarks/20260812T181752Z-quantization-spike.json's own
    "operators.matmul_conversion" shape (converted_count/fp32_count/
    int8_count_remaining per operator type)."""
    from joinless.runrecord import MatmulConversion

    conversion = MatmulConversion(
        converted_count=36, fp32_count=48, int8_count_remaining=12
    )
    assert conversion.converted_count == 36
    assert conversion.fp32_count == 48
    assert conversion.int8_count_remaining == 12


def test_preparation_comparison_carries_both_paths_own_scored_comparisons() -> None:
    """Issue #65's third bullet: "the run record states which one produced
    each figure" - satisfied structurally by holding one
    :class:`~joinless.resolver.ScoredComparisons` per path, each already
    self-tagged with the path that produced it (that type's own docstring),
    rather than a bare pair of numbers a reader would have to trust a
    variable name or field position to tell apart."""
    from joinless.resolver import ScoredComparisons
    from joinless.runrecord import PreparationComparison

    comparison = PreparationComparison(
        hoisted=ScoredComparisons(path="hoisted", scores=(0.5,)),
        naive=ScoredComparisons(path="naive", scores=(0.5,)),
    )

    assert comparison.hoisted.path == "hoisted"
    assert comparison.hoisted.scores == (0.5,)
    assert comparison.naive.path == "naive"
    assert comparison.naive.scores == (0.5,)


def test_bucket_occupancy_carries_the_full_distribution_and_the_cell_size_it_was_measured_under() -> (
    None
):
    """Issue #66: "occupancy is not context - it is the independent
    variable" - the raw per-cell counts, not a mean, so a reader can compute
    whatever summary they need rather than trust one this module chose for
    them (ADR-0009: bucket density has to travel with the hoist's result)."""
    occupancy = BucketOccupancy(
        counts=(1, 2, 4), cell_size_degrees=0.01, max_occupancy=4
    )

    assert occupancy.counts == (1, 2, 4)
    assert occupancy.cell_size_degrees == 0.01
    assert occupancy.max_occupancy == 4


def test_bucket_occupancy_carries_its_max_occupancy_directly() -> None:
    """Finding 1: a reader must be able to see how full the fullest bucket
    got without reducing ``counts`` themselves - the same "computed once,
    named directly" precedent :class:`MatmulConversion`'s own ``fp32_count``
    already sets for a value that is redundant with its siblings but kept
    anyway so a reader never has to recompute it."""
    occupancy = BucketOccupancy(
        counts=(1, 2, 3, 4), cell_size_degrees=0.01, max_occupancy=4
    )

    assert occupancy.max_occupancy == max(occupancy.counts)


def test_preparation_asymmetry_partitions_speed_up_by_arm_class() -> None:
    """Issue #66's third bullet: "the asymmetry between classical and
    neural arms is reported as a result" - a reader partitions the two
    groups from this type alone, without first knowing which of the four
    registered arms are classical and which are neural."""
    fast = Metric(value=1.02, undefined_reason=None)
    slow = Metric(value=48.6, undefined_reason=None)
    occupancy = BucketOccupancy(counts=(1, 2), cell_size_degrees=0.01, max_occupancy=2)
    asymmetry = PreparationAsymmetry(
        occupancy=occupancy,
        classical_speedups={"overlap": fast},
        neural_speedups={"embed-fp32": slow},
    )

    assert asymmetry.classical_speedups == {"overlap": fast}
    assert asymmetry.neural_speedups == {"embed-fp32": slow}


def test_preparation_asymmetry_carries_the_occupancy_it_explains() -> None:
    """Finding 1: occupancy is the independent variable the asymmetry is a
    function of (ADR-0009), so it travels on the finding it explains rather
    than sitting as an unrelated top-level field beside ``evaluation_set`` -
    where nothing marks it as describing only the preparation-cost sample,
    not the accuracy evaluation."""
    occupancy = BucketOccupancy(
        counts=(1, 2, 3, 4), cell_size_degrees=0.01, max_occupancy=4
    )

    asymmetry = PreparationAsymmetry(
        occupancy=occupancy, classical_speedups={}, neural_speedups={}
    )

    assert asymmetry.occupancy is occupancy


# --- RunAssembly: expectations before any report (ADR-0011 rule 4, issue #50) -------

from joinless.measurement import Unavailable


def _unavailable(arm: str) -> Unavailable:
    return Unavailable(arm=arm, reason="not measured in this test")


def _arm_result() -> ArmResult:
    return ArmResult(
        accuracy=_ok_accuracy(),
        warm_latency=_unavailable("overlap"),
        peak_memory=_unavailable("overlap"),
        cold_start=_unavailable("overlap"),
        artifact_size=Metric(
            value=None, undefined_reason="classical arms carry no model artifact"
        ),
        preparation=_unavailable("overlap"),
        preparation_cost=_unavailable("overlap"),
    )


def test_run_assembly_requires_expected_winners_to_construct() -> None:
    with pytest.raises(TypeError):
        RunAssembly()  # type: ignore[call-arg]


def test_build_rejects_a_run_with_no_arm_added() -> None:
    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)

    with pytest.raises(ValueError, match="at least one arm"):
        assembly.build(
            schema="benchmark-v1",
            started_at=_STARTED_AT,
            command=("joinless", "benchmark"),
            environment=_environment(),
            evaluation_set=_evaluation_set(),
            selected_thresholds=(),
            contradictions=(),
            preparation_asymmetry=_preparation_asymmetry(),
            int8_accuracy_divergence=Maybe(
                value=None, reason="not computed in this test"
            ),
        )


def test_a_built_record_carries_the_expected_winners_given_at_construction() -> None:
    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", _arm_result())

    record = assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=_environment(),
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=None, reason="not computed in this test"),
    )

    assert record.expected_winners.winners == {"exact": "overlap"}
    assert isinstance(record.results["overlap"].accuracy, SealedTestAccuracy)


# --- RunAssembly: contradictions are persisted, never recomputed (ADR-0011 rule 4,
# issue #50) --------------------------------------------------------------------


def test_a_built_record_carries_the_contradictions_it_was_given() -> None:
    from joinless.evaluation import Contradiction

    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", _arm_result())
    contradiction = Contradiction(
        family="character noise",
        expected_winner="fuzzy",
        actual_winners=("overlap",),
    )

    record = assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=_environment(),
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(contradiction,),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=None, reason="not computed in this test"),
    )

    assert record.contradictions == (contradiction,)


def test_a_run_with_no_broken_expectation_records_an_empty_contradictions_tuple() -> (
    None
):
    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", _arm_result())

    record = assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=_environment(),
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=None, reason="not computed in this test"),
    )

    assert record.contradictions == ()


# --- record_to_dict(): the one place a RunRecord becomes JSON-serialisable ----------

from joinless.runrecord import record_to_dict


def _record_with(arm_result: ArmResult) -> RunRecord:
    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", arm_result)
    return assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=_environment(),
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=None, reason="not computed in this test"),
    )


def test_record_to_dict_writes_the_record_id_and_started_at() -> None:
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    assert payload["record_id"] == "20260813T120000Z-benchmark.json"
    assert payload["started_at"] == "2026-08-13T12:00:00+00:00"


def test_record_to_dict_renders_a_present_maybe_as_its_value_with_no_reason() -> None:
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    rapidfuzz = payload["environment"]["runtime_versions"]["rapidfuzz"]
    assert rapidfuzz == "3.10.0"


def test_record_to_dict_renders_an_absent_maybe_as_null_with_its_reason() -> None:
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    onnxruntime = payload["environment"]["runtime_versions"]["onnxruntime"]
    assert onnxruntime == {"value": None, "reason": "no neural arm in this run"}


# --- Environment.models: one entry per neural arm that actually loaded, keyed by
# arm name (issue #67) - not a single Maybe[ModelIdentity] slot, because a run can
# now load both the fp32 and the int8 arm's models at once, and a singular slot
# would have to silently drop one of the two ------------------------------------


def test_environment_models_is_empty_when_no_neural_arm_ran() -> None:
    assert _environment().models == {}


# --- Environment.measurement_preparation_path: which path produced
# `results.<arm>.accuracy` and `results.<arm>.warm_latency` (Finding 2) -------------


def test_environment_states_the_path_that_produced_accuracy_and_warm_latency() -> None:
    """Finding 2: neither ``results.<arm>.accuracy`` nor
    ``results.<arm>.warm_latency`` states which preparation path produced
    it - only ``PreparationComparison``'s own ``path`` did, which describes
    itself and nothing else. This field is the run record stating it for
    the two figures that don't (issue #65's third bullet)."""
    assert _environment().measurement_preparation_path == "hoisted"


def test_record_to_dict_renders_the_measurement_preparation_path() -> None:
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    assert payload["environment"]["measurement_preparation_path"] == "hoisted"


def test_record_to_dict_renders_one_model_identity_by_arm_name_when_one_neural_arm_ran() -> (
    None
):
    from joinless.runrecord import ModelIdentity

    fp32_identity = ModelIdentity(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        checksum_sha256="e3fe9a9a8c877bd5ca0deebb6303aba138acc6818440211377afaca1ba78b511",
        license="apache-2.0",
    )
    environment = Environment(
        hardware=_environment().hardware,
        runtime_versions=_environment().runtime_versions,
        power_mode="ac",
        thread_count=1,
        warmup_count=5,
        repetition_count=20,
        models={"embed-fp32": fp32_identity},
        quantized_operators=Maybe(value=None, reason="no int8 arm in this run"),
        measurement_preparation_path="hoisted",
    )
    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", _arm_result())
    record = assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=environment,
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=None, reason="not computed in this test"),
    )

    payload = record_to_dict(record)

    assert payload["environment"]["models"] == {
        "embed-fp32": {
            "model_id": fp32_identity.model_id,
            "revision": fp32_identity.revision,
            "checksum_sha256": fp32_identity.checksum_sha256,
            "license": fp32_identity.license,
        }
    }


def test_record_to_dict_renders_both_neural_arms_model_identities_when_both_ran() -> (
    None
):
    """The fact this module exists to guard: a run where both ``embed-fp32``
    and ``embed-int8`` initialise records *both* checksums, not whichever the
    run happened to measure last (RFC-0001: the two arms differ only in their
    model artefact, and that artefact's checksum is exactly the fact a
    dropped entry would lose)."""
    from joinless.runrecord import ModelIdentity

    fp32_identity = ModelIdentity(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        checksum_sha256="e3fe9a9a8c877bd5ca0deebb6303aba138acc6818440211377afaca1ba78b511",
        license="apache-2.0",
    )
    int8_identity = ModelIdentity(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        checksum_sha256=(
            "eebed71d4f7671a4d8093decee1fb23018992e139813f30d502bf16ee408208e"
        ),
        license="apache-2.0",
    )
    environment = Environment(
        hardware=_environment().hardware,
        runtime_versions=_environment().runtime_versions,
        power_mode="ac",
        thread_count=1,
        warmup_count=5,
        repetition_count=20,
        models={"embed-fp32": fp32_identity, "embed-int8": int8_identity},
        quantized_operators=Maybe(value=None, reason="not read in this test"),
        measurement_preparation_path="hoisted",
    )
    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", _arm_result())
    record = assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=environment,
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=None, reason="not computed in this test"),
    )

    payload = record_to_dict(record)

    models = payload["environment"]["models"]
    assert models["embed-fp32"]["checksum_sha256"] == fp32_identity.checksum_sha256
    assert models["embed-int8"]["checksum_sha256"] == int8_identity.checksum_sha256


# --- Environment.quantized_operators: matmul-conversion counts, not a flat operator-
# type list (issue #68's stated purpose: "how many of the graph's matmuls were
# converted and how many remain in fp32" - a reader cannot answer that from a bare
# ``["DynamicQuantizeLinear", "MatMulInteger"]``) ------------------------------------


def test_record_to_dict_renders_quantized_operators_as_a_mapping_of_matmul_conversion_counts() -> (
    None
):
    from joinless.runrecord import MatmulConversion

    census = {
        "Gemm": MatmulConversion(
            converted_count=0, fp32_count=0, int8_count_remaining=0
        ),
        "MatMul": MatmulConversion(
            converted_count=36, fp32_count=48, int8_count_remaining=12
        ),
    }
    environment = Environment(
        hardware=_environment().hardware,
        runtime_versions=_environment().runtime_versions,
        power_mode="ac",
        thread_count=1,
        warmup_count=5,
        repetition_count=20,
        models={},
        quantized_operators=Maybe(value=census, reason=None),
        measurement_preparation_path="hoisted",
    )
    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", _arm_result())
    record = assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=environment,
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=None, reason="not computed in this test"),
    )

    payload = record_to_dict(record)

    assert payload["environment"]["quantized_operators"] == {
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


# --- RunRecord.int8_accuracy_divergence: per-family, computed once (issue #67) -----


def test_record_to_dict_renders_an_absent_int8_accuracy_divergence_with_its_reason() -> (
    None
):
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    assert payload["int8_accuracy_divergence"] == {
        "value": None,
        "reason": "not computed in this test",
    }


def test_record_to_dict_renders_a_present_int8_accuracy_divergence_per_family() -> None:
    from joinless.evaluation import AccuracyDivergence

    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", _arm_result())
    divergence = (
        AccuracyDivergence(
            family="exact",
            baseline_f1=Metric(value=1.0, undefined_reason=None),
            candidate_f1=Metric(value=0.9, undefined_reason=None),
            delta_f1=Metric(value=-0.1, undefined_reason=None),
        ),
    )
    record = assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=_environment(),
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=divergence, reason=None),
    )

    payload = record_to_dict(record)

    assert payload["int8_accuracy_divergence"] == {
        "value": [
            {
                "family": "exact",
                "baseline_f1": {"value": 1.0, "undefined_reason": None},
                "candidate_f1": {"value": 0.9, "undefined_reason": None},
                "delta_f1": {"value": -0.1, "undefined_reason": None},
            }
        ],
        "reason": None,
    }


def test_record_to_dict_tags_an_ok_accuracy_report_with_its_status() -> None:
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    accuracy = payload["results"]["overlap"]["accuracy"]
    assert accuracy["status"] == "ok"
    assert accuracy["pooled"]["per_family"][0]["family"] == "exact"


def test_record_to_dict_renders_the_pooled_figure_alongside_its_seed_variation() -> (
    None
):
    """Issue #97: per-family results reported per seed, not only pooled, and
    the seed-to-seed variation stated as a figure — both present on the same
    ``accuracy`` value a pooled figure is read from, each stating which
    question it answers (issue #97's third bullet)."""
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    accuracy = payload["results"]["overlap"]["accuracy"]
    assert accuracy["by_seed"]["1"]["per_family"][0]["family"] == "exact"
    assert accuracy["variation"][0]["family"] == "exact"
    assert accuracy["pooled_answers"]
    assert accuracy["by_seed_answers"]
    # The nested per-seed/pooled reports are not themselves re-tagged with a
    # second "status" — only the ADR-0013 either/or slot itself (this
    # ``accuracy`` value against ``InvalidRun``) carries that tag.
    assert "status" not in accuracy["pooled"]


def test_record_to_dict_tags_an_invalid_accuracy_report_with_its_status() -> None:
    from joinless.evaluation import InvalidRun

    invalid_result = ArmResult(
        accuracy=InvalidRun(reason="threshold selection touched the sealed test"),
        warm_latency=_unavailable("overlap"),
        peak_memory=_unavailable("overlap"),
        cold_start=_unavailable("overlap"),
        artifact_size=_unavailable("overlap"),
        preparation=_unavailable("overlap"),
        preparation_cost=_unavailable("overlap"),
    )
    record = _record_with(invalid_result)

    payload = record_to_dict(record)

    accuracy = payload["results"]["overlap"]["accuracy"]
    assert accuracy == {
        "status": "invalid",
        "reason": "threshold selection touched the sealed test",
    }


def test_record_to_dict_tags_an_unavailable_measurement_with_its_status() -> None:
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    warm_latency = payload["results"]["overlap"]["warm_latency"]
    assert warm_latency == {
        "status": "unavailable",
        "arm": "overlap",
        "reason": "not measured in this test",
    }


def test_record_to_dict_tags_an_ok_preparation_comparison_with_its_status() -> None:
    from joinless.resolver import ScoredComparisons
    from joinless.runrecord import PreparationComparison

    result = ArmResult(
        accuracy=_ok_accuracy(),
        warm_latency=_unavailable("overlap"),
        peak_memory=_unavailable("overlap"),
        cold_start=_unavailable("overlap"),
        artifact_size=_unavailable("overlap"),
        preparation=PreparationComparison(
            hoisted=ScoredComparisons(path="hoisted", scores=(0.5,)),
            naive=ScoredComparisons(path="naive", scores=(0.5,)),
        ),
        preparation_cost=_unavailable("overlap"),
    )
    record = _record_with(result)

    payload = record_to_dict(record)

    preparation = payload["results"]["overlap"]["preparation"]
    assert preparation["status"] == "ok"
    assert preparation["hoisted"] == {"path": "hoisted", "scores": [0.5]}
    assert preparation["naive"] == {"path": "naive", "scores": [0.5]}


def test_record_to_dict_tags_an_unavailable_preparation_comparison_with_its_status() -> (
    None
):
    result = ArmResult(
        accuracy=_ok_accuracy(),
        warm_latency=_unavailable("embed-int8"),
        peak_memory=_unavailable("embed-int8"),
        cold_start=_unavailable("embed-int8"),
        artifact_size=_unavailable("embed-int8"),
        preparation=_unavailable("embed-int8"),
        preparation_cost=_unavailable("embed-int8"),
    )
    record = _record_with(result)

    payload = record_to_dict(record)

    preparation = payload["results"]["overlap"]["preparation"]
    assert preparation == {
        "status": "unavailable",
        "arm": "embed-int8",
        "reason": "not measured in this test",
    }


def test_record_to_dict_tags_an_ok_preparation_cost_with_its_status() -> None:
    from joinless.measurement import PreparationCost

    result = ArmResult(
        accuracy=_ok_accuracy(),
        warm_latency=_unavailable("overlap"),
        peak_memory=_unavailable("overlap"),
        cold_start=_unavailable("overlap"),
        artifact_size=_unavailable("overlap"),
        preparation=_unavailable("overlap"),
        preparation_cost=PreparationCost(
            arm="overlap",
            hoisted_p50_seconds=0.001,
            hoisted_p99_seconds=0.002,
            naive_p50_seconds=0.004,
            naive_p99_seconds=0.006,
            warmup_count=5,
            repetition_count=20,
            record_count=4,
            comparison_count=4,
        ),
    )
    record = _record_with(result)

    payload = record_to_dict(record)

    preparation_cost = payload["results"]["overlap"]["preparation_cost"]
    assert preparation_cost == {
        "status": "ok",
        "arm": "overlap",
        "hoisted_p50_seconds": 0.001,
        "hoisted_p99_seconds": 0.002,
        "naive_p50_seconds": 0.004,
        "naive_p99_seconds": 0.006,
        "warmup_count": 5,
        "repetition_count": 20,
        "record_count": 4,
        "comparison_count": 4,
    }


def test_record_to_dict_tags_an_unavailable_preparation_cost_with_its_status() -> None:
    result = ArmResult(
        accuracy=_ok_accuracy(),
        warm_latency=_unavailable("embed-int8"),
        peak_memory=_unavailable("embed-int8"),
        cold_start=_unavailable("embed-int8"),
        artifact_size=_unavailable("embed-int8"),
        preparation=_unavailable("embed-int8"),
        preparation_cost=_unavailable("embed-int8"),
    )
    record = _record_with(result)

    payload = record_to_dict(record)

    preparation_cost = payload["results"]["overlap"]["preparation_cost"]
    assert preparation_cost == {
        "status": "unavailable",
        "arm": "embed-int8",
        "reason": "not measured in this test",
    }


def test_record_to_dict_renders_the_preparation_asymmetrys_occupancy_untagged() -> None:
    """Finding 1: the occupancy distribution lives nested under
    ``preparation_asymmetry``, not as a bare top-level field a reader could
    mistake for describing ``evaluation_set``'s 470-pair accuracy corpus
    rather than the one small preparation-cost sample it actually describes.
    ``BucketOccupancy`` is always computed for a run - unlike a per-arm
    measurement it carries no ADR-0013 either/or slot, so it renders as a
    plain nested object with no ``status`` wrapper (issue #66)."""
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    assert "bucket_occupancy" not in payload
    assert payload["preparation_asymmetry"]["occupancy"] == {
        "counts": [1, 2, 3],
        "cell_size_degrees": 0.01,
        "max_occupancy": 3,
    }


def test_record_to_dict_renders_the_preparation_asymmetry_untagged() -> None:
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    assert payload["preparation_asymmetry"] == {
        "occupancy": {
            "counts": [1, 2, 3],
            "cell_size_degrees": 0.01,
            "max_occupancy": 3,
        },
        "classical_speedups": {},
        "neural_speedups": {},
    }


def test_record_to_dict_renders_a_classical_arms_artifact_size_as_an_explicit_undefined_metric() -> (
    None
):
    """Issue #63: "arms with no model artifact record that explicitly rather
    than as an absence" - the field is present with a reason, not missing."""
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    artifact_size = payload["results"]["overlap"]["artifact_size"]
    assert artifact_size == {
        "value": None,
        "undefined_reason": "classical arms carry no model artifact",
    }


def test_record_to_dict_renders_a_defined_artifact_size_metric_with_no_status_wrapper() -> (
    None
):
    """Unlike ``warm_latency``/``peak_memory``/``cold_start``, a defined
    ``artifact_size`` is a bare :class:`~joinless.evaluation.Metric`, not one
    of :func:`~joinless.runrecord._OK_TAGGED_TYPES` - it renders exactly like
    ``Metric`` renders everywhere else it is nested (e.g. inside an accuracy
    report's ``precision``), with no ``status`` key added."""
    result = ArmResult(
        accuracy=_ok_accuracy(),
        warm_latency=_unavailable("embed-fp32"),
        peak_memory=_unavailable("embed-fp32"),
        cold_start=_unavailable("embed-fp32"),
        artifact_size=Metric(value=94000000.0, undefined_reason=None),
        preparation=_unavailable("embed-fp32"),
        preparation_cost=_unavailable("embed-fp32"),
    )
    record = _record_with(result)

    payload = record_to_dict(record)

    assert payload["results"]["overlap"]["artifact_size"] == {
        "value": 94000000.0,
        "undefined_reason": None,
    }


def test_record_to_dict_tags_an_unavailable_artifact_size_with_its_status() -> None:
    from joinless.evaluation import InvalidRun

    result = ArmResult(
        accuracy=InvalidRun(reason="'embed-int8' is not a known scorer"),
        warm_latency=_unavailable("embed-int8"),
        peak_memory=_unavailable("embed-int8"),
        cold_start=_unavailable("embed-int8"),
        artifact_size=_unavailable("embed-int8"),
        preparation=_unavailable("embed-int8"),
        preparation_cost=_unavailable("embed-int8"),
    )
    record = _record_with(result)

    payload = record_to_dict(record)

    assert payload["results"]["overlap"]["artifact_size"] == {
        "status": "unavailable",
        "arm": "embed-int8",
        "reason": "not measured in this test",
    }


def test_record_to_dict_renders_a_frozenset_as_a_sorted_list() -> None:
    from joinless.measurement import ColdStartPhases

    cold_start = ColdStartPhases(
        arm="overlap",
        interpreter_start=Metric(value=0.01, undefined_reason=None),
        import_phase=Metric(value=0.02, undefined_reason=None),
        session_creation=Metric(
            value=None, undefined_reason="classical arms construct no session"
        ),
        tokenizer_load=Metric(
            value=None, undefined_reason="classical arms load no tokenizer"
        ),
        first_inference=Metric(value=0.03, undefined_reason=None),
        not_attributable=frozenset({"interpreter start"}),
    )
    result = ArmResult(
        accuracy=_ok_accuracy(),
        warm_latency=_unavailable("overlap"),
        peak_memory=_unavailable("overlap"),
        cold_start=cold_start,
        artifact_size=_unavailable("overlap"),
        preparation=_unavailable("overlap"),
        preparation_cost=_unavailable("overlap"),
    )
    record = _record_with(result)

    payload = record_to_dict(record)

    cold_start_payload = payload["results"]["overlap"]["cold_start"]
    assert cold_start_payload["status"] == "ok"
    assert cold_start_payload["not_attributable"] == ["interpreter start"]


def test_record_to_dict_is_json_serialisable() -> None:
    import json

    record = _record_with(_arm_result())

    json.dumps(record_to_dict(record))  # raises TypeError if anything is not plain data


def test_record_to_dict_renders_no_contradictions_as_an_empty_list() -> None:
    record = _record_with(_arm_result())

    payload = record_to_dict(record)

    assert payload["contradictions"] == []


def test_record_to_dict_renders_a_contradiction_with_its_family_and_both_winners() -> (
    None
):
    from joinless.evaluation import Contradiction

    expected = ExpectedWinners(winners={"exact": "overlap"})
    assembly = RunAssembly(expected_winners=expected)
    assembly.add_arm("overlap", _arm_result())
    contradiction = Contradiction(
        family="character noise",
        expected_winner="fuzzy",
        actual_winners=("overlap",),
    )
    record = assembly.build(
        schema="benchmark-v1",
        started_at=_STARTED_AT,
        command=("joinless", "benchmark"),
        environment=_environment(),
        evaluation_set=_evaluation_set(),
        selected_thresholds=(),
        contradictions=(contradiction,),
        preparation_asymmetry=_preparation_asymmetry(),
        int8_accuracy_divergence=Maybe(value=None, reason="not computed in this test"),
    )

    payload = record_to_dict(record)

    assert payload["contradictions"] == [
        {
            "family": "character noise",
            "expected_winner": "fuzzy",
            "actual_winners": ["overlap"],
        }
    ]


# --- record ids: the naming convention issue #57 settles ---------------------------

from joinless.runrecord import build_record_id


def test_the_record_id_is_the_utc_timestamp_and_a_fixed_slug() -> None:
    started_at = datetime(2026, 8, 12, 18, 17, 52, tzinfo=UTC)

    assert build_record_id(started_at) == "20260812T181752Z-benchmark.json"


def test_the_record_id_normalises_a_non_utc_timestamp_to_utc() -> None:
    from datetime import timedelta

    plus_two = timezone(timedelta(hours=2))
    started_at = datetime(2026, 8, 12, 20, 17, 52, tzinfo=plus_two)

    assert build_record_id(started_at) == "20260812T181752Z-benchmark.json"


def test_a_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_record_id(datetime(2026, 8, 12, 18, 17, 52))  # noqa: DTZ001


# --- write_record(): never overwrites an existing record (issue #57) ---------------

from pathlib import Path

from joinless.runrecord import write_record


def test_write_record_writes_the_record_under_its_own_record_id(tmp_path: Path) -> None:
    record = _record_with(_arm_result())

    path = write_record(record, tmp_path)

    assert path == tmp_path / "20260813T120000Z-benchmark.json"
    assert path.is_file()


def test_write_record_content_round_trips_through_json(tmp_path: Path) -> None:
    import json

    record = _record_with(_arm_result())

    path = write_record(record, tmp_path)

    assert json.loads(path.read_text(encoding="utf-8")) == record_to_dict(record)


def test_write_record_creates_the_directory_if_it_does_not_exist(
    tmp_path: Path,
) -> None:
    record = _record_with(_arm_result())
    directory = tmp_path / "records" / "nested"

    path = write_record(record, directory)

    assert path.is_file()


def test_write_record_never_overwrites_an_existing_record(tmp_path: Path) -> None:
    first = _record_with(_arm_result())
    write_record(first, tmp_path)
    path = tmp_path / "20260813T120000Z-benchmark.json"
    original_content = path.read_text(encoding="utf-8")

    second = _record_with(
        ArmResult(
            accuracy=_ok_accuracy(),
            warm_latency=_unavailable("a different arm entirely"),
            peak_memory=_unavailable("overlap"),
            cold_start=_unavailable("overlap"),
            artifact_size=_unavailable("overlap"),
            preparation=_unavailable("overlap"),
            preparation_cost=_unavailable("overlap"),
        )
    )

    with pytest.raises(FileExistsError):
        write_record(second, tmp_path)

    assert path.read_text(encoding="utf-8") == original_content


# --- build_evaluation_set_identity(): grounded in real generated corpora -----------

from joinless.corpus import FAMILIES, Corpus, generate_corpus
from joinless.runrecord import build_evaluation_set_identity


def test_build_evaluation_set_identity_names_every_corpus_seed_used() -> None:
    corpora = [generate_corpus(1), generate_corpus(2)]

    identity = build_evaluation_set_identity(corpora)

    assert identity.seeds == (1, 2)


def test_build_evaluation_set_identity_pools_family_counts_across_corpora() -> None:
    corpus = generate_corpus(1)

    identity = build_evaluation_set_identity([corpus])

    expected = {family: 0 for family in FAMILIES}
    for pair in corpus.pairs:
        assert pair.category is not None
        expected[pair.category] += 1
    assert dict(identity.case_mixture) == expected
    assert sum(identity.case_mixture.values()) == len(corpus.pairs)


def test_build_evaluation_set_identity_requires_at_least_one_corpus() -> None:
    with pytest.raises(ValueError, match="at least one corpus"):
        build_evaluation_set_identity([])


def test_build_evaluation_set_identity_rejects_an_uncategorised_pair() -> None:
    from types import MappingProxyType as _MappingProxyType

    from joinless.corpus import LabelledPair

    pair = LabelledPair(
        pair_id="p1", left_name="A", right_name="B", label=1, category=None
    )
    corpus = Corpus(
        seed=1, pairs=(pair,), roles=_MappingProxyType({"p1": "development"})
    )

    with pytest.raises(ValueError, match="has no category"):
        build_evaluation_set_identity([corpus])
