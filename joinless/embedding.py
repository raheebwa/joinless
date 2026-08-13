# SPDX-License-Identifier: MIT
"""The embedding arms — fp32 and int8: tokenize, embed, mean-pool, compare (RFC-0001,
ADR-0002, ADR-0007).

**Model, pooling, similarity — the two arms in one paragraph.** Both load
``sentence-transformers/all-MiniLM-L6-v2`` at revision :data:`MODEL_REVISION`, executed on
ONNX Runtime's CPU provider (ADR-0002): the fp32 arm the plain ONNX export, the int8 arm
the same graph after ``onnxruntime.quantization.quantize_dynamic`` (ADR-0007) — "the fp32
and int8 arms are the same class with different model artefacts" (RFC-0001), and that
sentence is this module's organising fact: everywhere below that reads "this arm" or
names one arm's function, its sibling has the identically-shaped counterpart named
alongside it. Tokenization is ``tokenizers.Tokenizer.from_file`` over a locally-fetched
``tokenizer.json`` — never ``Tokenizer.from_pretrained``, which is the one member of that
package that would put a network client on the path a comparison takes (ADR-0017). Both
arms share one tokenizer file (the int8 graph carries no tokenizer of its own to export —
quantization touches weights, not vocabulary), read from the fp32 arm's own directory by
:func:`resolve_int8_model_paths` rather than a second copy under ``int8/``.
:func:`_mean_pool` averages the graph's per-token ``last_hidden_state`` over the positions
``attention_mask`` marks real, excluding padding; the pooled vector is then
L2-normalised. :func:`_cosine_similarity` compares two such vectors and
:meth:`EmbeddingScorer.score` rescales the result from ``[-1, 1]`` to ``[0, 1]``
(RFC-0001's "Comparability"), so either arm's score sits on the same scale as ``overlap``
and ``fuzzy`` without being the same quantity.

**Every arithmetic step is standard-library ``math``, not NumPy** (:func:`_mean_pool`,
:func:`_l2_normalize`, :func:`_cosine_similarity`). ``onnxruntime.InferenceSession.run``
accepts plain nested Python lists for its integer inputs and hands back a NumPy array for
its output — but iterating and indexing a NumPy array works exactly like iterating a
nested list, so a function written against ``Sequence[Sequence[float]]`` runs unchanged
against either one, which is exactly how ``spikes/quantization/smoke.py`` already proved
this same arithmetic out (read, not imported — that tooling lives behind the ``export``
extra, a different install surface, and reimplementing here keeps this module's own
history genuinely its own). The consequence is a real one: this module needs no NumPy
import of its own, at module level or otherwise, so nothing about testing its pooling and
similarity arithmetic requires NumPy to be installed at all.

**The lazy-import boundary (ADR-0014, ADR-0017).** ``onnxruntime`` and ``tokenizers`` are
each imported exactly once per arm, inside that arm's own ``probe_*`` and ``load_*``
functions (:func:`probe_fp32`/:func:`load_fp32_scorer`, :func:`probe_int8`/
:func:`load_int8_scorer`) — never at this module's top level. :class:`EmbeddingScorer`
itself imports neither: it is constructed with an already-built tokenizer and inference
session (RFC-0001's "``EmbeddingScorer`` is constructed with a model path and a runtime
session"), so its own ``prepare_all``/``prepare``/``score`` logic has no import of its own
to hide behind a boundary — there is nothing there for a test to fake but two small
objects exposing ``encode_batch`` and ``run``, and no reason for those tests to need the
real packages installed at all. :mod:`joinless.scoring` reaches this module only from
inside its own lazy ``embed-fp32``/``embed-int8`` registration functions, mirroring
exactly how it reaches ``rapidfuzz`` from inside :class:`~joinless.scoring.FuzzyScorer` —
so a classical-only run never imports this module, and this module never imports the
runtime, until an embedding arm is actually requested and its own probe has already said
yes.

**Fail closed (ADR-0013, issue #59, issue #67).** Three independent things must hold
before either arm will run at all, checked cheapest-first by that arm's own ``probe_*``
so a missing dependency is never misreported as a missing artefact:
:func:`_probe_dependencies` (shared by both arms — the same two packages), then
configuration (``JOINLESS_MODEL_CACHE_DIR`` names a directory), then
:func:`_probe_artifacts` — both artefact files an arm depends on exist and hash to the
value recorded for it. Verification is :func:`joinless.measurement.verify_artifact`,
called once per file and never reimplemented here (RFC-0017's consequences: "one
checksum-verification mechanism for every artefact file an arm depends on, not a second
one specific to the tokenizer"). Nothing on this path fetches a replacement for a missing
or mismatched file; refusing, with a reason that names the setup command, is the correct
response (ADR-0013), and :func:`resolve_model_paths`/:func:`resolve_int8_model_paths` are
the only two places this module reads the cache-directory environment variable, so a
caller supplying its own mapping (every test in this module) never touches the process
environment to do it.

**Batched preparation (RFC-0001, ADR-0009, issue #61).** ``prepare_all`` is the production
call pattern: it collects the distinct non-blank names in a batch — a name that recurs
within one batch is embedded once, not once per occurrence — and embeds all of them in a
single tokenizer/session call, then hands each position in the input its own prepared
value back, duplicates included. ``prepare`` is the naive per-record control that hoist is
measured against (same RFC): it tokenizes and embeds one name per call, independently of
whatever else is being prepared, so a caller measuring it in a loop reproduces exactly the
per-comparison cost the hoist removes. Both routes call the same pooling and normalisation
arithmetic, so they can only ever disagree if the batching or deduplication bookkeeping
itself is wrong — which is what ``tests/test_embedding.py``'s batched/unbatched parity
test exists to catch.

A record without a name — ``None``, or a string that is empty or whitespace-only —
carries no identity to embed. Rather than tokenize an empty string (which BERT-style
tokenizers happily do, producing a vector that is not obviously distinguishable from a
real short name), this arm gives such a record a dedicated ``None`` prepared value, and
:meth:`EmbeddingScorer.score` returns ``0.0`` whenever either side is ``None`` — the same
rule :mod:`joinless.scoring`'s module docstring states for the classical arms: an unnamed
record is not evidence that two records are the same entity, and every arm applies that
rule rather than letting it fall out of whatever its own arithmetic happens to do with
empty input.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from joinless.measurement import ArtifactRequirement, verify_artifact
from joinless.runrecord import MatmulConversion, ModelIdentity

# Recorded identity of the source model (benchmarks/20260812T181752Z-quantization-spike.json,
# the record produced by RFC-0004's spike). ADR-0002 constraint 3: the source model and
# revision are among the things every arm built from this graph must hold constant, so
# they are pinned here as literals rather than read from anywhere that could drift
# silently between runs.
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# cardData.license, recorded alongside the model identity in the same run record
# (issue #59: "the model card's licence is recorded alongside the artefact identity").
MODEL_LICENSE = "apache-2.0"

# Where the fetched artefact lives is configuration, not a constant compiled into the
# package (issue #59) — the directory itself is named by this environment variable, and
# the layout beneath it (an "fp32" subdirectory holding the graph and its tokenizer) is
# the convention spikes/quantization/cli_common.py already established for the same
# artefact, reused here rather than invented a second time.
CACHE_DIR_ENV_VAR = "JOINLESS_MODEL_CACHE_DIR"
_FP32_SUBDIRECTORY = "fp32"
_INT8_SUBDIRECTORY = "int8"
_MODEL_FILENAME = "model.onnx"
_TOKENIZER_FILENAME = "tokenizer.json"

# The checksums of the exact fp32 artefact files this arm was built and verified against
# under $JOINLESS_MODEL_CACHE_DIR/fp32 (sha256, computed directly from those files — see
# ADR-0013's fourth rule and RFC-0017's consequences: one checksum per artefact file, the
# same verify_artifact path for both). A module-level name rather than a value folded
# into a function body so a test can patch it to a small fixture's own checksum without
# needing the 90 MB production graph.
FP32_MODEL_SHA256 = "e3fe9a9a8c877bd5ca0deebb6303aba138acc6818440211377afaca1ba78b511"
FP32_TOKENIZER_SHA256 = (
    "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"
)

# The int8 graph's own checksum under $JOINLESS_MODEL_CACHE_DIR/int8 (ADR-0007,
# issue #67). The int8 arm's tokenizer is not a second, independently-verified
# file: it is the fp32 arm's own tokenizer.json (RFC-0004 records the two arms
# as sharing one tokenizer), so there is no INT8_TOKENIZER_SHA256 alongside this
# — FP32_TOKENIZER_SHA256 is reused, verified against the copy under fp32/.
INT8_MODEL_SHA256 = "eebed71d4f7671a4d8093decee1fb23018992e139813f30d502bf16ee408208e"

# The operator types RFC-0004's spike recorded present in the int8 graph and absent
# from the fp32 graph — benchmarks/20260812T181752Z-quantization-spike.json's own
# "operators.added" — for the exact artefact INT8_MODEL_SHA256 checksums (issue #68).
# Pinned as a literal the same way INT8_MODEL_SHA256 itself is: a fact about this one
# checksum-verified artefact, not re-derived from the spike record at run time. This
# is what :func:`verify_int8_operators` checks a fresh, live read against — the run
# record's own ``quantized_operators`` value is always that live read
# (:func:`joinless.cli._quantized_operators`), never this constant transplanted
# directly, which is what keeps "read from the graph at run time" (issue #68's second
# bullet) true even though this expectation was itself sourced from the spike.
INT8_QUANTIZED_OPERATORS: tuple[str, ...] = ("DynamicQuantizeLinear", "MatMulInteger")

# The operator types onnxruntime.quantization.quantize_dynamic introduces in place of
# a converted MatMul/Gemm (RFC-0004 step 5) — reimplemented from
# spikes/quantization/operators.py's QUANTIZED_MATMUL_REPLACEMENTS rather than
# imported: nothing under joinless/ imports spikes/ (spikes/quantization/README.md,
# "Why this lives outside joinless/" — importing spike tooling into the shipped
# package would make joinless/ depend on code that ships in no wheel and will not run
# again, the same reasoning this module's own docstring already gives for
# reimplementing spikes/quantization/smoke.py's pooling arithmetic rather than
# importing it).
_QUANTIZATION_REPLACEMENT_OPERATORS = (
    "DynamicQuantizeLinear",
    "MatMulInteger",
    "QGemm",
    "QLinearMatMul",
)

# Candidate operator type -> the single replacement type
# onnxruntime.quantization.quantize_dynamic introduces when it converts a node of
# that type (RFC-0004 step 5) — the two keys
# benchmarks/20260812T181752Z-quantization-spike.json's own
# "operators.matmul_conversion" reports on ("Gemm", "MatMul"). A converted node's
# *type* changes to its replacement, so a candidate type's pre-conversion count is
# always exactly its own remaining count plus its replacement's count in the same
# graph — :func:`matmul_conversion_census` needs no second, earlier graph to
# establish "how many were there before" (issue #68's second bullet: read from
# *the* graph, singular).
_MATMUL_CANDIDATE_REPLACEMENTS: Mapping[str, str] = {
    "Gemm": "QGemm",
    "MatMul": "MatMulInteger",
}

# The matmul-conversion census RFC-0004's spike recorded for the exact artefact
# INT8_MODEL_SHA256 checksums — benchmarks/20260812T181752Z-quantization-spike.json's
# own "operators.matmul_conversion" (issue #68's stated purpose: "how many of the
# graph's matmuls were converted and how many remain in fp32"). Pinned as a literal
# the same way INT8_QUANTIZED_OPERATORS is, and checked the same way: this is what
# :func:`verify_int8_operators` compares a fresh, live-read census against — the run
# record's own value is always that live read, never this constant transplanted
# directly (issue #68's second bullet).
INT8_MATMUL_CONVERSION: Mapping[str, MatmulConversion] = {
    "Gemm": MatmulConversion(converted_count=0, fp32_count=0, int8_count_remaining=0),
    "MatMul": MatmulConversion(
        converted_count=36, fp32_count=48, int8_count_remaining=12
    ),
}

# tokenizer_config.json's own recorded "max_length" for this artefact — truncation is
# configured to match what the artefact already declares for itself, not a value chosen
# independently of it.
_MAX_SEQUENCE_LENGTH = 128
_PAD_TOKEN = "[PAD]"
_PAD_ID = 0

_ARM_NAME = "embed-fp32"
_INT8_ARM_NAME = "embed-int8"

_SETUP_HINT = (
    f"set {CACHE_DIR_ENV_VAR} to a writable directory and fetch the model artefact by "
    "following spikes/quantization/README.md's Setup section, then running "
    "`python -m spikes.quantization.model` and `python -m spikes.quantization.export_fp32`"
)

# One extra step beyond _SETUP_HINT: the int8 graph is quantize_dynamic's output
# over the fp32 export (ADR-0007), so producing it needs that export to already
# exist plus the quantization step itself.
_INT8_SETUP_HINT = (
    f"set {CACHE_DIR_ENV_VAR} to a writable directory and fetch the model artefact by "
    "following spikes/quantization/README.md's Setup section, then running "
    "`python -m spikes.quantization.model`, `python -m spikes.quantization.export_fp32` "
    "and `python -m spikes.quantization.quantize_int8`"
)


class CacheDirNotSetError(RuntimeError):
    """``CACHE_DIR_ENV_VAR`` is unset or empty.

    Issue #59: "a user who has not run setup must get an error that names the command to
    run" — :data:`_SETUP_HINT` is folded into this exception's message rather than left
    for a caller to append, so the one place this error can be raised is also the one
    place its remedy is stated.
    """


class QuantizedOperatorMismatchError(RuntimeError):
    """The int8 graph's own quantized-operator census does not equal
    :data:`INT8_QUANTIZED_OPERATORS` (issue #68's third bullet: "a record for a run
    whose graph does not match its recorded operator list is not produced").

    Raised by :func:`verify_int8_operators`, never caught inside this module — a
    checksum mismatch (:func:`probe_int8`) marks *only* the int8 arm unavailable and
    the run record is still written for whichever arms did initialise (ADR-0013 rule
    3, "an arm that cannot initialise is recorded... never omitted"). This is a
    stronger signal: the checksum-verified bytes did not produce the operator census
    this exact artefact is recorded to have, which means what this run believes about
    quantization is not to be trusted, not just that one arm's row. Deliberately not
    caught in :mod:`joinless.embedding` itself, so :mod:`joinless.cli` — the one place
    that decides what "no record is produced" means for a whole run — is the only
    place this is handled.
    """


def _resolve_paths(
    environ: Mapping[str, str], *, model_subdirectory: str, tokenizer_subdirectory: str
) -> tuple[Path, Path]:
    """The model and tokenizer paths under the configured cache directory, each
    resolved from its own subdirectory — shared by :func:`resolve_model_paths`
    (fp32, both subdirectories the same) and :func:`resolve_int8_model_paths`
    (int8 model, fp32 tokenizer), so "how the cache directory is read" is one
    function both arms call rather than two copies that could drift apart.

    Takes the environment mapping as a parameter rather than reading ``os.environ``
    itself, mirroring ``spikes/quantization/cli_common.resolve_cache_dir`` — not
    imported, since that tooling sits behind the ``export`` extra, a different install
    surface than this arm's, but reimplemented against the same convention so a caller
    supplying its own mapping (every test in this module) never touches the process
    environment to exercise this function.
    """
    value = environ.get(CACHE_DIR_ENV_VAR)
    if not value:
        raise CacheDirNotSetError(f"{CACHE_DIR_ENV_VAR} is not set; {_SETUP_HINT}.")
    base = Path(value)
    return (
        base / model_subdirectory / _MODEL_FILENAME,
        base / tokenizer_subdirectory / _TOKENIZER_FILENAME,
    )


def resolve_model_paths(environ: Mapping[str, str]) -> tuple[Path, Path]:
    """The fp32 model and tokenizer paths under the configured cache directory."""
    return _resolve_paths(
        environ,
        model_subdirectory=_FP32_SUBDIRECTORY,
        tokenizer_subdirectory=_FP32_SUBDIRECTORY,
    )


def resolve_int8_model_paths(environ: Mapping[str, str]) -> tuple[Path, Path]:
    """The int8 model path under its own subdirectory, and the fp32 arm's own
    tokenizer path — the two arms share one tokenizer file (RFC-0004: "the
    tokenizer is shared with fp32"), so this never reads a second copy from an
    ``int8`` subdirectory that does not exist."""
    return _resolve_paths(
        environ,
        model_subdirectory=_INT8_SUBDIRECTORY,
        tokenizer_subdirectory=_FP32_SUBDIRECTORY,
    )


def artifact_requirements_fp32(
    environ: Mapping[str, str] | None = None,
) -> tuple[ArtifactRequirement, ArtifactRequirement]:
    """The model and tokenizer :class:`~joinless.measurement.ArtifactRequirement`
    pair for the fp32 arm — the model file, then the tokenizer file, in that
    order.

    Exposed on its own, separately from :func:`probe_fp32`, so a caller that
    wants these paths without also running the checksum check they gate
    (:func:`joinless.measurement.measure_artifact_size`, issue #63) does not
    have to re-derive the path/checksum pairing a second time. :func:`probe_fp32`
    itself calls this rather than constructing the pair inline, so there is
    exactly one place that pairing is made.
    """
    env = os.environ if environ is None else environ
    model_path, tokenizer_path = resolve_model_paths(env)
    return (
        ArtifactRequirement(path=model_path, sha256=FP32_MODEL_SHA256),
        ArtifactRequirement(path=tokenizer_path, sha256=FP32_TOKENIZER_SHA256),
    )


def artifact_requirements_int8(
    environ: Mapping[str, str] | None = None,
) -> tuple[ArtifactRequirement, ArtifactRequirement]:
    """The int8 model and shared fp32 tokenizer :class:`~joinless.measurement.
    ArtifactRequirement` pair — mirrors :func:`artifact_requirements_fp32`
    exactly, over :func:`resolve_int8_model_paths` and :data:`INT8_MODEL_SHA256`
    instead."""
    env = os.environ if environ is None else environ
    model_path, tokenizer_path = resolve_int8_model_paths(env)
    return (
        ArtifactRequirement(path=model_path, sha256=INT8_MODEL_SHA256),
        ArtifactRequirement(path=tokenizer_path, sha256=FP32_TOKENIZER_SHA256),
    )


def model_identity_fp32() -> ModelIdentity:
    """This arm's model identity, revision, checksum and licence — issue #59's
    "the model card's licence is recorded alongside the artefact identity,"
    read from the same pinned literals :data:`MODEL_ID`, :data:`MODEL_REVISION`,
    :data:`FP32_MODEL_SHA256` and :data:`MODEL_LICENSE` that
    :func:`artifact_requirements_fp32` already treats as this arm's single
    source of truth for what it was built and verified against, rather than
    re-derived from the artefact at call time — a second read of a 90 MB file
    to confirm a fact :func:`probe_fp32` already established for this same
    process. Callable meaningfully only once :func:`probe_fp32` has returned
    ``None``: what it reports is then a description of the artefact whose
    checksum was just verified to match :data:`FP32_MODEL_SHA256`, not a
    value that could have drifted from it.
    """
    return ModelIdentity(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        checksum_sha256=FP32_MODEL_SHA256,
        license=MODEL_LICENSE,
    )


def model_identity_int8() -> ModelIdentity:
    """This arm's model identity, revision, checksum and licence — mirrors
    :func:`model_identity_fp32` exactly: same :data:`MODEL_ID`,
    :data:`MODEL_REVISION` and :data:`MODEL_LICENSE` (RFC-0001: "the fp32 and
    int8 arms are the same class with different model artefacts" — the
    identity and licence describe the *model*, not the precision it was
    exported at), but :data:`INT8_MODEL_SHA256`, the one fact that actually
    differs between the two artefacts."""
    return ModelIdentity(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        checksum_sha256=INT8_MODEL_SHA256,
        license=MODEL_LICENSE,
    )


def _probe_dependencies() -> str | None:
    """``None`` when both ``onnxruntime`` and ``tokenizers`` import; otherwise the
    reason one of them does not. Shared by :func:`probe_fp32` and :func:`probe_int8`
    — both arms need the same two packages, so whether they are importable is a fact
    about the process, not about which arm asked (RFC-0001's "same class, different
    artefacts" applies to this check too)."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        return (
            "the 'onnxruntime' package is not installed "
            f"({exc}); install with `pip install 'joinless[neural]'`"
        )

    try:
        import tokenizers  # noqa: F401
    except ImportError as exc:
        return (
            "the 'tokenizers' package is not installed "
            f"({exc}); install with `pip install 'joinless[neural]'`"
        )

    return None


def _probe_artifacts(
    requirements: tuple[ArtifactRequirement, ArtifactRequirement], setup_hint: str
) -> str | None:
    """``None`` when every requirement in ``requirements`` verifies; otherwise the
    first failure's reason, with ``setup_hint`` appended — shared by
    :func:`probe_fp32` and :func:`probe_int8`, which differ only in which
    requirements and which hint they pass in."""
    for requirement in requirements:
        reason = verify_artifact(requirement)
        if reason is not None:
            return f"{reason}; {setup_hint}"
    return None


def probe_fp32(environ: Mapping[str, str] | None = None) -> str | None:
    """``None`` when the fp32 arm can be constructed; otherwise the reason it cannot.

    This is the check :func:`joinless.scoring.get_scorer` runs, through
    :mod:`joinless.scoring`'s own lazy ``embed-fp32`` registration, before it will call
    :func:`load_fp32_scorer` (ADR-0013). Checks run cheapest-first and stop at the first
    failure: dependency importability (:func:`_probe_dependencies`), then configuration,
    then the artefact files themselves (:func:`_probe_artifacts`) — hashing a 90 MB
    graph is the most expensive check here, so it never runs when a cheaper check
    already explains why the arm is unavailable.

    ``environ`` defaults to ``os.environ`` — this is one of two functions in this
    module that reads it (the other is :func:`probe_int8`), and only when a caller has
    not supplied its own mapping.
    """
    dependency_reason = _probe_dependencies()
    if dependency_reason is not None:
        return dependency_reason

    env = os.environ if environ is None else environ
    try:
        requirements = artifact_requirements_fp32(env)
    except CacheDirNotSetError as exc:
        return str(exc)

    return _probe_artifacts(requirements, _SETUP_HINT)


def probe_int8(environ: Mapping[str, str] | None = None) -> str | None:
    """``None`` when the int8 arm can be constructed; otherwise the reason it cannot.

    Mirrors :func:`probe_fp32` exactly, over :func:`artifact_requirements_int8` and
    :data:`_INT8_SETUP_HINT` instead — this is the check
    :func:`joinless.scoring.get_scorer` runs before it will call
    :func:`load_int8_scorer` (ADR-0013).
    """
    dependency_reason = _probe_dependencies()
    if dependency_reason is not None:
        return dependency_reason

    env = os.environ if environ is None else environ
    try:
        requirements = artifact_requirements_int8(env)
    except CacheDirNotSetError as exc:
        return str(exc)

    return _probe_artifacts(requirements, _INT8_SETUP_HINT)


def read_operator_counts(model_path: Path) -> Mapping[str, int]:
    """A node count per distinct ONNX operator type in ``model_path``'s graph.

    ``onnx`` is imported here, at call time (ADR-0014, ADR-0017) — not
    ``onnxruntime``, whose loaded ``InferenceSession`` optimises the graph and never
    exposes the raw node list this reads. ``onnx`` is already part of the ``neural``
    install profile in ``pyproject.toml``, the same profile ``onnxruntime`` and
    ``tokenizers`` belong to, so reaching for it here introduces no dependency this
    arm did not already carry.

    The one place this module reads ``model_path``'s graph — :func:`read_operator_types`
    and :func:`matmul_conversion_census` (via :func:`verify_int8_operators`) both build
    on this rather than loading the graph a second time, so "how many nodes of each
    type" is answered once per call, not once per question asked of it.
    """
    import onnx

    model = onnx.load(str(model_path))
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def read_operator_types(model_path: Path) -> frozenset[str]:
    """The distinct ONNX operator types ``model_path``'s graph contains — the
    node counts :func:`read_operator_counts` already read, reduced to their keys."""
    return frozenset(read_operator_counts(model_path))


def quantized_operators_present(operator_types: frozenset[str]) -> tuple[str, ...]:
    """Which of :data:`_QUANTIZATION_REPLACEMENT_OPERATORS` are actually present in
    ``operator_types`` — sorted, so the result is deterministic regardless of set
    iteration order.

    A pure function of a set of operator-type names, so it is testable against a
    hand-built set and needs no real graph (mirrors
    spikes/quantization/operators.py's ``classify_matmul_conversion``, reimplemented
    rather than imported — see :class:`QuantizedOperatorMismatchError` and this
    module's docstring for why nothing here imports ``spikes/``).
    """
    return tuple(
        sorted(op for op in _QUANTIZATION_REPLACEMENT_OPERATORS if op in operator_types)
    )


def matmul_conversion_census(
    operator_counts: Mapping[str, int],
) -> Mapping[str, MatmulConversion]:
    """How many of each candidate-for-quantization operator type
    (:data:`_MATMUL_CANDIDATE_REPLACEMENTS`'s keys) converted, and how many
    remain fp32, given one graph's own live-read ``operator_counts`` (issue
    #68's stated purpose: "how many of the graph's matmuls were converted and
    how many remain in fp32").

    A pure function of a single graph's operator-type counts — mirrors
    :func:`quantized_operators_present`'s own "pure function of a set of
    operator-type names" shape, testable against a hand-built mapping and
    needing no real graph. A candidate or replacement type absent from
    ``operator_counts`` counts as zero, not a missing key.
    """
    census = {}
    for candidate, replacement in _MATMUL_CANDIDATE_REPLACEMENTS.items():
        converted = operator_counts.get(replacement, 0)
        remaining = operator_counts.get(candidate, 0)
        census[candidate] = MatmulConversion(
            converted_count=converted,
            fp32_count=converted + remaining,
            int8_count_remaining=remaining,
        )
    return census


def verify_int8_operators(
    model_path: Path,
) -> tuple[tuple[str, ...], Mapping[str, MatmulConversion]]:
    """Read ``model_path``'s graph once and confirm both its quantized-operator
    census and its matmul-conversion census equal what RFC-0004's spike
    recorded for the exact artefact :data:`INT8_MODEL_SHA256` checksums —
    :data:`INT8_QUANTIZED_OPERATORS` and :data:`INT8_MATMUL_CONVERSION`
    respectively (issue #68's third bullet).

    Returns the freshly-read ``(operator types present, matmul-conversion
    census)`` pair on a match — this, not either pinned constant, is what a
    caller persists as the run record's ``quantized_operators`` value (issue
    #68's second bullet: read at run time, not copied from the spike record).
    Raises :class:`QuantizedOperatorMismatchError` on any other census for
    either check, including a subset or superset of what was expected:
    equality is checked, not containment, because a graph with a different
    census is a different graph, not a partially-matching one. The
    operator-type check runs first — a graph missing ``MatMulInteger``
    entirely is the more fundamental mismatch, and its message names the
    missing type directly rather than folding it into a conversion count of
    zero that would read as "nothing converted" rather than "this type is
    absent".
    """
    operator_counts = read_operator_counts(model_path)
    operator_types = frozenset(operator_counts)
    found = quantized_operators_present(operator_types)
    if found != INT8_QUANTIZED_OPERATORS:
        raise QuantizedOperatorMismatchError(
            f"int8 graph at {model_path} has quantized-operator census "
            f"{list(found)}, expected {list(INT8_QUANTIZED_OPERATORS)} "
            "(benchmarks/20260812T181752Z-quantization-spike.json's recorded "
            "operators)"
        )

    census = matmul_conversion_census(operator_counts)
    if census != INT8_MATMUL_CONVERSION:
        raise QuantizedOperatorMismatchError(
            f"int8 graph at {model_path} has matmul-conversion census "
            f"{dict(census)}, expected {dict(INT8_MATMUL_CONVERSION)} "
            "(benchmarks/20260812T181752Z-quantization-spike.json's recorded "
            "operators.matmul_conversion)"
        )
    return found, census


def _build_scorer(
    *, name: str, model_path: Path, tokenizer_path: Path
) -> EmbeddingScorer:
    """Load the tokenizer, open the session, wrap both in an
    :class:`EmbeddingScorer` named ``name`` — the construction
    :func:`load_fp32_scorer` and :func:`load_int8_scorer` share: RFC-0001's "the
    fp32 and int8 arms are the same class with different model artefacts" holds
    here structurally, since nothing about this function reads either arm's
    name beyond the one it is given to attach.

    ``onnxruntime`` and ``tokenizers`` are imported here, not at module level (ADR-0014,
    ADR-0017) — see the module docstring for why that boundary matters.

    Tokenizer construction and session construction are timed separately, right
    around each call, and the two durations travel on the returned
    :class:`EmbeddingScorer` (issue #108) — RFC-0002's Metrics table names them as
    two distinct phases ("tokenizer construction from the artefact" and "inference
    session construction from the artefact"), so they are captured as two distinct
    intervals here rather than one combined duration a caller would have to guess
    how to split.
    """
    import onnxruntime
    from tokenizers import Tokenizer

    _before_tokenizer = time.perf_counter()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_padding(pad_token=_PAD_TOKEN, pad_id=_PAD_ID)
    tokenizer.enable_truncation(max_length=_MAX_SEQUENCE_LENGTH)
    _after_tokenizer = time.perf_counter()

    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    _after_session = time.perf_counter()

    return EmbeddingScorer(
        name=name,
        session=session,
        tokenizer=tokenizer,
        tokenizer_load_seconds=_after_tokenizer - _before_tokenizer,
        session_creation_seconds=_after_session - _after_tokenizer,
    )


def load_fp32_scorer(environ: Mapping[str, str] | None = None) -> EmbeddingScorer:
    """Build the fp32 arm for real: resolve paths, then :func:`_build_scorer`.

    Assumes :func:`probe_fp32` has already returned ``None`` for the same ``environ`` —
    exactly how :func:`joinless.scoring.get_scorer` calls it, probe then factory, never
    the reverse — so artefact verification is not repeated here: hashing the graph a
    second time on every construction would cost real wall-clock time for a check
    already passed.
    """
    env = os.environ if environ is None else environ
    model_path, tokenizer_path = resolve_model_paths(env)
    return _build_scorer(
        name=_ARM_NAME, model_path=model_path, tokenizer_path=tokenizer_path
    )


def load_int8_scorer(environ: Mapping[str, str] | None = None) -> EmbeddingScorer:
    """Build the int8 arm for real: resolve paths, then :func:`_build_scorer`.

    Mirrors :func:`load_fp32_scorer` exactly, over
    :func:`resolve_int8_model_paths` instead — assumes :func:`probe_int8` has
    already returned ``None`` for the same ``environ``.
    """
    env = os.environ if environ is None else environ
    model_path, tokenizer_path = resolve_int8_model_paths(env)
    return _build_scorer(
        name=_INT8_ARM_NAME, model_path=model_path, tokenizer_path=tokenizer_path
    )


class _Encoding(Protocol):
    """The three fields this module reads off a ``tokenizers.Encoding`` — not the whole
    real type, so a test double needs to supply exactly this much to stand in for one."""

    @property
    def ids(self) -> list[int]:
        """Token ids, including any special tokens the tokenizer added."""

    @property
    def attention_mask(self) -> list[int]:
        """``1`` for a real token, ``0`` for padding."""

    @property
    def type_ids(self) -> list[int]:
        """Segment ids the graph's ``token_type_ids`` input expects."""


class _TokenizerLike(Protocol):
    """What :class:`EmbeddingScorer` needs from a tokenizer: batch encoding. Padding and
    truncation are configured once, by :func:`load_fp32_scorer`, before construction —
    this protocol does not need to know that happened, only that ``encode_batch`` already
    reflects it."""

    def encode_batch(self, texts: Sequence[str]) -> Sequence[_Encoding]:
        """One encoding per text, in the same order."""


class _SessionLike(Protocol):
    """What :class:`EmbeddingScorer` needs from an inference session: run the graph.
    Matches ``onnxruntime.InferenceSession.run``'s shape exactly, so the real session
    satisfies this without adaptation."""

    def run(
        self, output_names: list[str] | None, input_feed: Mapping[str, object]
    ) -> Sequence[Any]:
        """``None`` for ``output_names`` means every output the graph declares."""


def _clean(name: str | None) -> str | None:
    """``None`` for a record with no name to embed — ``None`` itself, or a string with
    nothing but whitespace in it — and ``name`` unchanged otherwise. Shared by
    ``prepare`` and ``prepare_all`` so the two routes can only ever disagree about which
    names are blank if this function itself is wrong, never because one route re-derived
    the rule differently. Returning the cleaned value (rather than a bare ``bool``) is
    what lets a caller narrow with a plain ``is None`` check afterwards, matching how the
    rest of this module is typed."""
    if name is None or not name.strip():
        return None
    return name


def _mean_pool(
    token_embeddings: Sequence[Sequence[float]], attention_mask: Sequence[int]
) -> list[float]:
    """Average per-token embeddings over the positions ``attention_mask`` marks real.

    Padding tokens carry an embedding but no meaning; including them would bias the
    pooled vector toward the padding scheme rather than the text (mirrors
    ``spikes/quantization/smoke.py``'s already-validated logic — reimplemented, not
    imported, since that tooling sits behind a different install extra).

    Raises if ``attention_mask`` selects no tokens at all, rather than dividing by zero:
    every encoding this module ever builds includes at least a `[CLS]` and a `[SEP]`
    token with ``attention_mask=1``, so this is unreachable from :meth:`EmbeddingScorer`'s
    own callers — :func:`_is_blank` filters out the only input that could otherwise reach
    it — and is exercised directly, as the edge case it guards against.
    """
    dim = len(token_embeddings[0]) if len(token_embeddings) else 0
    sums = [0.0] * dim
    count = 0
    for vector, mask in zip(token_embeddings, attention_mask, strict=True):
        if mask:
            for i, value in enumerate(vector):
                sums[i] += float(value)
            count += 1
    if count == 0:
        raise ValueError("attention mask selects no tokens to pool")
    return [total / count for total in sums]


def _l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """Scale ``vector`` to unit length. A zero vector has no direction to scale to, so
    it is returned unchanged rather than dividing by zero — :func:`_cosine_similarity`
    treats a zero vector as carrying no information, the same "no information" rule
    :func:`_is_blank` states for an absent name."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(x / norm for x in vector)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Cosine similarity of two vectors, in ``[-1.0, 1.0]`` — or ``None`` when either
    vector has zero magnitude, rather than raising or returning ``0.0``.

    ``None`` here, not ``0.0``: a zero vector's cosine similarity is undefined, not
    coincidentally equal to the value two genuinely *orthogonal* vectors would produce.
    :meth:`EmbeddingScorer.score` rescales an ordinary ``0.0`` (orthogonal) to ``0.5``;
    collapsing "undefined" into that same ``0.0`` would have scored a degenerate
    embedding as a half-match instead of "no information," which is the same
    ``None``-is-not-``0.0`` distinction ADR-0013 states for a metric's denominator. A
    genuine name never mean-pools to the zero vector (real token embeddings are never
    all zero), so this guards a defensive edge :meth:`EmbeddingScorer.score` should
    never actually need to reach for a real name.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)


class EmbeddingScorer:
    """Tokenize, embed, mean-pool, compare — one arm of RFC-0001's ``Scorer`` protocol.

    Constructed with an already-built tokenizer and inference session, never a path
    (:func:`load_fp32_scorer` resolves paths and builds both) — RFC-0001's "the fp32 and
    int8 arms are the same class with different model artefacts" holds structurally
    because nothing about this class's own logic names ``embed-fp32`` or ``embed-int8``;
    only the ``name`` a caller supplies and the session/tokenizer pair it is handed decide
    which arm an instance is.

    ``prepare`` returns ``None`` for a blank name and an L2-normalised mean-pooled
    embedding — a plain ``tuple[float, ...]``, not a NumPy array (see the module
    docstring for why) — otherwise. ``score`` returns ``0.0`` whenever either side is
    ``None`` or a degenerate zero-magnitude embedding (:func:`_cosine_similarity`'s
    ``None``), and otherwise the cosine similarity of the two embeddings rescaled from
    ``[-1, 1]`` to ``[0, 1]`` (RFC-0001's "Comparability"), clamped to that range to
    absorb the floating-point rounding a rescaled dot product can occasionally overshoot
    it by.

    ``tokenizer_load_seconds`` and ``session_creation_seconds`` carry how long
    :func:`_build_scorer` spent on each of its two construction steps (issue #108,
    RFC-0002's cold-start decomposition) — ``None`` for an instance built any other
    way, such as every other test in this module, which construct this class
    directly from an already-built double and never claim a duration they did not
    measure.
    """

    def __init__(
        self,
        *,
        name: str,
        session: _SessionLike,
        tokenizer: _TokenizerLike,
        tokenizer_load_seconds: float | None = None,
        session_creation_seconds: float | None = None,
    ) -> None:
        self._name = name
        self._session = session
        self._tokenizer = tokenizer
        self._tokenizer_load_seconds = tokenizer_load_seconds
        self._session_creation_seconds = session_creation_seconds

    @property
    def name(self) -> str:
        return self._name

    @property
    def tokenizer_load_seconds(self) -> float | None:
        """How long :func:`_build_scorer` spent constructing this instance's
        tokenizer — see the class docstring for why this is ``None`` outside
        that one construction path."""
        return self._tokenizer_load_seconds

    @property
    def session_creation_seconds(self) -> float | None:
        """How long :func:`_build_scorer` spent constructing this instance's
        inference session — see the class docstring for why this is ``None``
        outside that one construction path."""
        return self._session_creation_seconds

    def prepare_all(
        self, names: Sequence[str | None]
    ) -> list[tuple[float, ...] | None]:
        """Batched preparation (ADR-0009, issue #61): every distinct non-blank name in
        ``names`` is embedded exactly once, in a single tokenizer/session call, no
        matter how many times it recurs in the batch.

        Deduplication is on exact string equality of the raw name — a name is not
        casefolded or otherwise normalised before this comparison, so two names that
        differ only in case or whitespace are treated as distinct and each embedded in
        its own right, matching how the tokenizer would treat them if asked separately.
        """
        order: list[str] = []
        index_by_name: dict[str, int] = {}
        for raw in names:
            cleaned = _clean(raw)
            if cleaned is None:
                continue
            if cleaned not in index_by_name:
                index_by_name[cleaned] = len(order)
                order.append(cleaned)

        embeddings = self._embed_texts(order) if order else []

        results: list[tuple[float, ...] | None] = []
        for raw in names:
            cleaned = _clean(raw)
            if cleaned is None:
                results.append(None)
            else:
                results.append(embeddings[index_by_name[cleaned]])
        return results

    def prepare(self, name: str | None) -> tuple[float, ...] | None:
        """The naive, per-record control ADR-0009's hoist is measured against: one
        tokenizer/session call for exactly this one name, independent of whatever else
        is being prepared elsewhere. Not a second production path (RFC-0001) — a caller
        measuring this in a loop reproduces the per-comparison cost ``prepare_all``
        removes.
        """
        cleaned = _clean(name)
        if cleaned is None:
            return None
        return self._embed_texts([cleaned])[0]

    def score(self, a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
        if a is None or b is None:
            return 0.0
        cosine = _cosine_similarity(a, b)
        if cosine is None:
            return 0.0
        rescaled = (cosine + 1.0) / 2.0
        return max(0.0, min(1.0, rescaled))

    def _embed_texts(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Tokenize and embed a batch of already-known-non-blank texts in one call."""
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = [list(encoding.ids) for encoding in encodings]
        attention_masks = [list(encoding.attention_mask) for encoding in encodings]
        token_type_ids = [list(encoding.type_ids) for encoding in encodings]

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "token_type_ids": token_type_ids,
            },
        )
        token_embeddings_batch = outputs[0]

        return [
            _l2_normalize(_mean_pool(token_embeddings_batch[i], attention_masks[i]))
            for i in range(len(texts))
        ]
