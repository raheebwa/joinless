# SPDX-License-Identifier: MIT
"""The fp32 embedding arm: tokenize, embed, mean-pool, compare (RFC-0001, ADR-0002).

**Model, pooling, similarity — the arm in one paragraph.** The graph is
``sentence-transformers/all-MiniLM-L6-v2`` at revision :data:`MODEL_REVISION`, exported
to ONNX and executed on ONNX Runtime's CPU provider (ADR-0002). Tokenization is
``tokenizers.Tokenizer.from_file`` over a locally-fetched ``tokenizer.json`` — never
``Tokenizer.from_pretrained``, which is the one member of that package that would put a
network client on the path a comparison takes (ADR-0017). :func:`_mean_pool` averages the
graph's per-token ``last_hidden_state`` over the positions ``attention_mask`` marks real,
excluding padding; the pooled vector is then L2-normalised. :func:`_cosine_similarity`
compares two such vectors and :meth:`EmbeddingScorer.score` rescales the result from
``[-1, 1]`` to ``[0, 1]`` (RFC-0001's "Comparability"), so this arm's score sits on the
same scale as ``overlap`` and ``fuzzy`` without being the same quantity.

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
each imported exactly once, inside :func:`probe_fp32` and :func:`load_fp32_scorer` — never
at this module's top level. :class:`EmbeddingScorer` itself imports neither: it is
constructed with an already-built tokenizer and inference session (RFC-0001's
"``EmbeddingScorer`` is constructed with a model path and a runtime session"), so its own
``prepare_all``/``prepare``/``score`` logic has no import of its own to hide behind a
boundary — there is nothing there for a test to fake but two small objects exposing
``encode_batch`` and ``run``, and no reason for those tests to need the real packages
installed at all. :mod:`joinless.scoring` reaches this module only from inside its own
lazy ``embed-fp32`` registration functions, mirroring exactly how it reaches ``rapidfuzz``
from inside :class:`~joinless.scoring.FuzzyScorer` — so a classical-only run never imports
this module, and this module never imports the runtime, until an embedding arm is actually
requested and its :func:`probe_fp32` has already said yes.

**Fail closed (ADR-0013, issue #59).** Three independent things must hold before this arm
will run at all, checked cheapest-first by :func:`probe_fp32` so a missing dependency is
never misreported as a missing artefact: the dependencies import, ``JOINLESS_MODEL_CACHE_DIR``
names a directory, and both artefact files it should contain — the ONNX graph and the
tokenizer configuration alongside it — exist and hash to the value recorded for this
revision. Verification is :func:`joinless.measurement.verify_artifact`, called once per
file and never reimplemented here (RFC-0017's consequences: "one checksum-verification
mechanism for every artefact file an arm depends on, not a second one specific to the
tokenizer"). Nothing on this path fetches a replacement for a missing or mismatched file;
refusing, with a reason that names the setup command, is the correct response (ADR-0013),
and :func:`resolve_model_paths` is the one place this module reads the cache-directory
environment variable, so a caller supplying its own mapping (every test in this module)
never touches the process environment to do it.

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from joinless.measurement import ArtifactRequirement, verify_artifact
from joinless.runrecord import ModelIdentity

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

# tokenizer_config.json's own recorded "max_length" for this artefact — truncation is
# configured to match what the artefact already declares for itself, not a value chosen
# independently of it.
_MAX_SEQUENCE_LENGTH = 128
_PAD_TOKEN = "[PAD]"
_PAD_ID = 0

_ARM_NAME = "embed-fp32"

_SETUP_HINT = (
    f"set {CACHE_DIR_ENV_VAR} to a writable directory and fetch the model artefact by "
    "following spikes/quantization/README.md's Setup section, then running "
    "`python -m spikes.quantization.model` and `python -m spikes.quantization.export_fp32`"
)


class CacheDirNotSetError(RuntimeError):
    """``CACHE_DIR_ENV_VAR`` is unset or empty.

    Issue #59: "a user who has not run setup must get an error that names the command to
    run" — :data:`_SETUP_HINT` is folded into this exception's message rather than left
    for a caller to append, so the one place this error can be raised is also the one
    place its remedy is stated.
    """


def resolve_model_paths(environ: Mapping[str, str]) -> tuple[Path, Path]:
    """The fp32 model and tokenizer paths under the configured cache directory.

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
    fp32_dir = Path(value) / _FP32_SUBDIRECTORY
    return fp32_dir / _MODEL_FILENAME, fp32_dir / _TOKENIZER_FILENAME


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


def probe_fp32(environ: Mapping[str, str] | None = None) -> str | None:
    """``None`` when the fp32 arm can be constructed; otherwise the reason it cannot.

    This is the check :func:`joinless.scoring.get_scorer` runs, through
    :mod:`joinless.scoring`'s own lazy ``embed-fp32`` registration, before it will call
    :func:`load_fp32_scorer` (ADR-0013). Checks run cheapest-first and stop at the first
    failure: dependency importability, then configuration, then the artefact files
    themselves — hashing a 90 MB graph is the most expensive check here, so it never runs
    when a cheaper check already explains why the arm is unavailable.

    ``environ`` defaults to ``os.environ`` — this is the one function in this module that
    reads it, and only when a caller has not supplied its own mapping.
    """
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

    env = os.environ if environ is None else environ
    try:
        requirements = artifact_requirements_fp32(env)
    except CacheDirNotSetError as exc:
        return str(exc)

    for requirement in requirements:
        reason = verify_artifact(requirement)
        if reason is not None:
            return f"{reason}; {_SETUP_HINT}"

    return None


def load_fp32_scorer(environ: Mapping[str, str] | None = None) -> EmbeddingScorer:
    """Build the fp32 arm for real: resolve paths, load the tokenizer, open the session.

    Assumes :func:`probe_fp32` has already returned ``None`` for the same ``environ`` —
    exactly how :func:`joinless.scoring.get_scorer` calls it, probe then factory, never
    the reverse — so artefact verification is not repeated here: hashing the graph a
    second time on every construction would cost real wall-clock time for a check
    already passed.

    ``onnxruntime`` and ``tokenizers`` are imported here, not at module level (ADR-0014,
    ADR-0017) — see the module docstring for why that boundary matters.
    """
    env = os.environ if environ is None else environ
    model_path, tokenizer_path = resolve_model_paths(env)

    import onnxruntime
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_padding(pad_token=_PAD_TOKEN, pad_id=_PAD_ID)
    tokenizer.enable_truncation(max_length=_MAX_SEQUENCE_LENGTH)

    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )

    return EmbeddingScorer(name=_ARM_NAME, session=session, tokenizer=tokenizer)


class _Encoding(Protocol):
    """The three fields this module reads off a ``tokenizers.Encoding`` — not the whole
    real type, so a test double needs to supply exactly this much to stand in for one."""

    @property
    def ids(self) -> list[int]:
        """Token ids, including any special tokens the tokenizer added."""
        ...

    @property
    def attention_mask(self) -> list[int]:
        """``1`` for a real token, ``0`` for padding."""
        ...

    @property
    def type_ids(self) -> list[int]:
        """Segment ids the graph's ``token_type_ids`` input expects."""
        ...


class _TokenizerLike(Protocol):
    """What :class:`EmbeddingScorer` needs from a tokenizer: batch encoding. Padding and
    truncation are configured once, by :func:`load_fp32_scorer`, before construction —
    this protocol does not need to know that happened, only that ``encode_batch`` already
    reflects it."""

    def encode_batch(self, texts: Sequence[str]) -> Sequence[_Encoding]:
        """One encoding per text, in the same order."""
        ...


class _SessionLike(Protocol):
    """What :class:`EmbeddingScorer` needs from an inference session: run the graph.
    Matches ``onnxruntime.InferenceSession.run``'s shape exactly, so the real session
    satisfies this without adaptation."""

    def run(
        self, output_names: list[str] | None, input_feed: Mapping[str, object]
    ) -> Sequence[Any]:
        """``None`` for ``output_names`` means every output the graph declares."""
        ...


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
    """

    def __init__(
        self, *, name: str, session: _SessionLike, tokenizer: _TokenizerLike
    ) -> None:
        self._name = name
        self._session = session
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        return self._name

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
