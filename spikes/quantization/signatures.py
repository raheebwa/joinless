# SPDX-License-Identifier: MIT
"""RFC-0004 step 4: confirm both graphs load with equivalent signatures (issue #9).

Two graphs that do not accept the same inputs are not two arms of one benchmark. A
signature difference is a finding to record, not a detail to work around silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class IOSignature:
    """One input or output: name, shape (dynamic dims may be ``str`` or ``None``), and
    the runtime's own type string (e.g. ``"tensor(float)"``)."""

    name: str
    shape: tuple[int | str | None, ...]
    dtype: str


class NodeArgLike(Protocol):
    """The subset of ``onnxruntime.NodeArg`` this step reads."""

    @property
    def name(self) -> str: ...

    @property
    def shape(self) -> Sequence[int | str | None]: ...

    @property
    def type(self) -> str: ...


class SessionLike(Protocol):
    """The subset of ``onnxruntime.InferenceSession`` this step reads."""

    def get_inputs(self) -> Sequence[NodeArgLike]: ...

    def get_outputs(self) -> Sequence[NodeArgLike]: ...


@dataclass(frozen=True, slots=True)
class SignatureComparison:
    equivalent: bool
    differences: tuple[str, ...]


def extract_io_signature(
    session: SessionLike,
) -> tuple[tuple[IOSignature, ...], tuple[IOSignature, ...]]:
    """Read ``(inputs, outputs)`` off a loaded session."""

    def convert(nodes: Sequence[NodeArgLike]) -> tuple[IOSignature, ...]:
        return tuple(
            IOSignature(name=node.name, shape=tuple(node.shape), dtype=node.type)
            for node in nodes
        )

    return convert(session.get_inputs()), convert(session.get_outputs())


def _diff_signature_lists(
    fp32: Sequence[IOSignature], int8: Sequence[IOSignature], kind: str
) -> list[str]:
    fp32_by_name = {sig.name: sig for sig in fp32}
    int8_by_name = {sig.name: sig for sig in int8}
    diffs: list[str] = []

    for name in fp32_by_name.keys() - int8_by_name.keys():
        diffs.append(f"{kind} '{name}' present in fp32 graph, absent from int8 graph")
    for name in int8_by_name.keys() - fp32_by_name.keys():
        diffs.append(f"{kind} '{name}' present in int8 graph, absent from fp32 graph")
    for name in fp32_by_name.keys() & int8_by_name.keys():
        a, b = fp32_by_name[name], int8_by_name[name]
        if a.shape != b.shape:
            diffs.append(
                f"{kind} '{name}' shape differs: fp32={a.shape} int8={b.shape}"
            )
        if a.dtype != b.dtype:
            diffs.append(
                f"{kind} '{name}' dtype differs: fp32={a.dtype} int8={b.dtype}"
            )

    return diffs


def compare_signatures(
    fp32_inputs: Sequence[IOSignature],
    fp32_outputs: Sequence[IOSignature],
    int8_inputs: Sequence[IOSignature],
    int8_outputs: Sequence[IOSignature],
) -> SignatureComparison:
    """Diff both graphs' input and output signatures.

    Equivalence, not identity of load order — names anchor the comparison so a
    difference is reported for the field that actually changed rather than for every
    entry after the first mismatch.
    """
    diffs = _diff_signature_lists(
        fp32_inputs, int8_inputs, "input"
    ) + _diff_signature_lists(fp32_outputs, int8_outputs, "output")
    return SignatureComparison(equivalent=not diffs, differences=tuple(sorted(diffs)))


def main(argv: list[str] | None = None) -> int:
    """Load both graphs on this machine and record their signature comparison."""
    import argparse
    import os

    import onnxruntime

    from spikes.quantization.cli_common import (
        read_fragment,
        resolve_cache_dir,
        write_fragment,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    cache_dir = resolve_cache_dir(os.environ)
    quantize_record = read_fragment(cache_dir, "step3_quantize")
    call = quantize_record["call"]
    assert isinstance(call, dict)
    fp32_path = str(call["model_input"])
    int8_path = str(call["model_output"])

    fp32_session = onnxruntime.InferenceSession(
        fp32_path, providers=["CPUExecutionProvider"]
    )
    int8_session = onnxruntime.InferenceSession(
        int8_path, providers=["CPUExecutionProvider"]
    )

    fp32_inputs, fp32_outputs = extract_io_signature(fp32_session)
    int8_inputs, int8_outputs = extract_io_signature(int8_session)
    comparison = compare_signatures(
        fp32_inputs, fp32_outputs, int8_inputs, int8_outputs
    )

    def serialize(sigs: tuple[IOSignature, ...]) -> list[dict[str, object]]:
        return [
            {"name": s.name, "shape": list(s.shape), "dtype": s.dtype} for s in sigs
        ]

    write_fragment(
        cache_dir,
        "step4_signatures",
        {
            "fp32_inputs": serialize(fp32_inputs),
            "fp32_outputs": serialize(fp32_outputs),
            "int8_inputs": serialize(int8_inputs),
            "int8_outputs": serialize(int8_outputs),
            "equivalent": comparison.equivalent,
            "differences": list(comparison.differences),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
