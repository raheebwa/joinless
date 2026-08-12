# SPDX-License-Identifier: MIT
"""Skip the graph-level tests when the export profile is not installed.

Three modules here build and inspect real ONNX graphs, so they need `onnx`, which
is declared under the `export` extra rather than `dev` or `neural`. Without this,
collecting them under a base or neural install is an error rather than a skip, and
every test in the repository fails to run because collection never completes.

Skipping is not the whole answer: a test that no environment runs proves nothing,
so CI installs the export profile in one matrix cell and these run there.
"""

from __future__ import annotations

import importlib.util

_REQUIRES_ONNX = (
    "test_operators.py",
    "test_quantize_int8.py",
    "test_signatures.py",
)

collect_ignore = [] if importlib.util.find_spec("onnx") else list(_REQUIRES_ONNX)
