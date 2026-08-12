# SPDX-License-Identifier: MIT
"""Invariant: importing joinless never initialises ONNX Runtime.

ADR-0014 fixes the boundary as an invariant that holds regardless of which
install profile is active: no module reachable from ``import joinless`` may
import the inference runtime at module level. This is deliberately not an
assertion that ``onnxruntime`` is unimportable — that state depends on which
profile is installed and is checked by the CI matrix instead (dev vs.
dev,neural), not by this suite.
"""

import sys


def test_import_never_touches_onnxruntime() -> None:
    sys.modules.pop("onnxruntime", None)

    import joinless  # noqa: F401

    assert "onnxruntime" not in sys.modules
