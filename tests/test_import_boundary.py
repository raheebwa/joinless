# SPDX-License-Identifier: MIT
"""Invariant: importing joinless never initialises ONNX Runtime.

ADR-0014 fixes the boundary as an invariant that holds regardless of which
install profile is active: no module reachable from ``import joinless`` may
import the inference runtime at module level. PRD FR-12a requires this to be an
invariant with a test rather than a convention, because a classical arm that
inherited the runtime's import cost would make "the classical arm is cheap"
unmeasurable.

This is deliberately not an assertion that ``onnxruntime`` is unimportable —
that state depends on which profile is installed and is checked by the CI matrix
instead (dev vs. dev,neural), not by this suite.

**The import happens in a child interpreter, and that is load-bearing.** Python
caches modules in ``sys.modules``, so a second ``import joinless`` in a process
that already imported it is served from cache and never re-executes the module
body. A same-process version of this test asserts against an import it did not
perform, and passes whether or not the invariant holds.
"""

import subprocess
import sys

_PROBE = """
import sys
import joinless  # noqa: F401
offenders = sorted(m for m in sys.modules if m == "onnxruntime" or m.startswith("onnxruntime."))
print("\\n".join(offenders))
sys.exit(1 if offenders else 0)
"""


def test_importing_joinless_never_initialises_the_runtime() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        "importing joinless pulled in the inference runtime at module level: "
        f"{result.stdout.strip() or result.stderr.strip()}"
    )
