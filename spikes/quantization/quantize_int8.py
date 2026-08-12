# SPDX-License-Identifier: MIT
"""RFC-0004 step 3: produce the int8 graph via ``quantize_dynamic`` (issue #8).

Every parameter is passed explicitly rather than left to the library's default, so the
record states what was *requested* independently of whatever a given
``onnxruntime`` version's default would have resolved to (RFC-0004 open question 1) —
and so a later diff of what was actually converted (step 5) is checked against a
declared intent, not an implicit one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

OP_TYPES_TO_QUANTIZE = ("MatMul", "Gemm")
"""Requested for conversion: the operator families that dominate a transformer
encoder's cost. If step 5's operator diff shows these untouched, the int8 arm is
mislabelled (RFC-0004 step 5) — this constant is what "untouched" is checked against.
"""


def build_quantize_kwargs(model_input: Path, model_output: Path) -> dict[str, Any]:
    """The exact call this step makes, as keyword arguments to ``quantize_dynamic``."""
    from onnxruntime.quantization import QuantType

    return {
        "model_input": str(model_input),
        "model_output": str(model_output),
        "weight_type": QuantType.QInt8,
        "per_channel": False,
        "reduce_range": False,
        "op_types_to_quantize": list(OP_TYPES_TO_QUANTIZE),
        "extra_options": {},
    }


def serialize_quantize_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """A JSON-safe copy of :func:`build_quantize_kwargs`'s output for the record.

    ``QuantType`` is an enum ``quantize_dynamic`` accepts directly but ``json.dumps``
    cannot; every other value already round-trips.
    """
    serialized = dict(kwargs)
    if "weight_type" in serialized:
        serialized["weight_type"] = str(serialized["weight_type"])
    return serialized


def run_quantization(
    kwargs: Mapping[str, Any], quantize_fn: Callable[..., None]
) -> None:
    """Call ``quantize_fn`` with ``kwargs`` verbatim.

    A one-line pass-through, injectable so the parameter-building logic above can be
    tested against the real ``quantize_dynamic`` without this module importing it at
    module level.
    """
    quantize_fn(**kwargs)


def main(argv: list[str] | None = None) -> int:
    """Quantize the fp32 artefact step 2 produced and record the call (issue #8)."""
    import argparse
    import os

    from onnxruntime.quantization import quantize_dynamic

    from spikes.quantization.cli_common import (
        read_fragment,
        resolve_cache_dir,
        write_fragment,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    cache_dir = resolve_cache_dir(os.environ)
    export_record = read_fragment(cache_dir, "step2_export")
    model_input = Path(str(export_record["output_dir"])) / "model.onnx"
    model_output = cache_dir / "int8" / "model.onnx"
    model_output.parent.mkdir(parents=True, exist_ok=True)

    kwargs = build_quantize_kwargs(model_input, model_output)
    run_quantization(kwargs, quantize_dynamic)

    write_fragment(
        cache_dir,
        "step3_quantize",
        {
            "call": serialize_quantize_kwargs(kwargs),
            "op_types_requested": list(OP_TYPES_TO_QUANTIZE),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
