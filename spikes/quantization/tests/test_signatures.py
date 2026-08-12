# SPDX-License-Identifier: MIT
"""RFC-0004 step 4: confirm both graphs load with equivalent signatures (issue #9).

``extract_io_signature`` is exercised against a real ``onnxruntime.InferenceSession``
over a hand-built graph — not the spike's model, no network — so the adapter is checked
against the runtime's actual ``NodeArg`` shape rather than a guessed one.
"""

from __future__ import annotations

import onnx
import onnxruntime
from onnx import TensorProto, helper

from spikes.quantization.signatures import (
    IOSignature,
    compare_signatures,
    extract_io_signature,
)


def _toy_identity_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "toy-identity", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_extract_io_signature_reads_a_real_session(tmp_path) -> None:
    path = tmp_path / "m.onnx"
    onnx.save(_toy_identity_model(), str(path))
    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )

    inputs, outputs = extract_io_signature(session)

    assert inputs == (IOSignature(name="X", shape=(1, 3), dtype="tensor(float)"),)
    assert outputs == (IOSignature(name="Y", shape=(1, 3), dtype="tensor(float)"),)


def test_compare_signatures_equivalent_when_identical() -> None:
    sig = (IOSignature(name="X", shape=(1, 3), dtype="tensor(float)"),)

    comparison = compare_signatures(sig, sig, sig, sig)

    assert comparison.equivalent is True
    assert comparison.differences == ()


def test_compare_signatures_reports_a_missing_input() -> None:
    fp32_inputs = (
        IOSignature(name="X", shape=(1, 3), dtype="tensor(float)"),
        IOSignature(name="mask", shape=(1, 3), dtype="tensor(int64)"),
    )
    int8_inputs = (IOSignature(name="X", shape=(1, 3), dtype="tensor(float)"),)

    comparison = compare_signatures(fp32_inputs, (), int8_inputs, ())

    assert comparison.equivalent is False
    assert any("mask" in diff for diff in comparison.differences)


def test_compare_signatures_reports_a_shape_mismatch() -> None:
    fp32_outputs = (IOSignature(name="Y", shape=(1, 3), dtype="tensor(float)"),)
    int8_outputs = (IOSignature(name="Y", shape=(1, 4), dtype="tensor(float)"),)

    comparison = compare_signatures((), fp32_outputs, (), int8_outputs)

    assert comparison.equivalent is False
    assert any("shape differs" in diff for diff in comparison.differences)


def test_compare_signatures_reports_a_dtype_mismatch() -> None:
    fp32_outputs = (IOSignature(name="Y", shape=(1, 3), dtype="tensor(float)"),)
    int8_outputs = (IOSignature(name="Y", shape=(1, 3), dtype="tensor(uint8)"),)

    comparison = compare_signatures((), fp32_outputs, (), int8_outputs)

    assert comparison.equivalent is False
    assert any("dtype differs" in diff for diff in comparison.differences)
