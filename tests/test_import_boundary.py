# SPDX-License-Identifier: MIT
"""Invariant: importing any module of joinless never initialises the inference
runtime or its tokenizer.

ADR-0014 fixes the boundary as an invariant that holds regardless of which install
profile is active: no module reachable from ``import joinless`` may import
``onnxruntime`` at module level. ADR-0017 draws the identical boundary around
``tokenizers``. PRD FR-12a requires this to be an invariant with a test rather than
a convention, because a classical arm that inherited either import's cost would
make "the classical arm is cheap" unmeasurable.

This is deliberately not an assertion that ``onnxruntime`` or ``tokenizers`` are
unimportable — that state depends on which profile is installed and is checked by
the CI matrix instead (dev vs. dev,neural), not by this suite.

**Importing the package is not enough to observe the invariant, and that is why
this covers every module separately.** ``joinless/__init__.py`` holds only a
docstring, so a bare ``import joinless`` loads no submodule at all — the set of
``joinless.*`` names in ``sys.modules`` afterwards is empty. A probe performing
only that import therefore cannot see a module-level ``import onnxruntime``
anywhere else in the package: nothing it runs ever reaches those files. The
invariant PRD FR-12a states is about every module a caller can reach, so the test
has to reach them itself.

**Coverage is by construction, not by a hand-maintained list.**
:func:`_discovered_module_names` enumerates every module
:mod:`joinless` actually ships — via :func:`importlib.util.find_spec`, which
locates the package without executing it, and :func:`pkgutil.iter_modules`, which
lists its contents from the filesystem — so a module added to the package later is
covered the moment it exists, with nothing here to remember to extend. Each
discovered name becomes its own parametrized case, so a failure names the
importing module directly in the test id and in the assertion message, rather than
requiring a reader to guess which of several imports inside one shared probe was
the offender.

**The import happens in a child interpreter, and that is load-bearing.** Python
caches modules in ``sys.modules``, so a second import of a module already
imported in this process is served from cache and never re-executes the module
body — this file's own collection, and pytest's own import machinery, have
already imported parts of :mod:`joinless` in the *test* process by the time any
test here runs. A same-process probe would assert against an import it did not
perform, and pass whether or not the invariant holds. ``_run_probe`` instead
spawns a fresh ``sys.executable`` interpreter per module, mirroring
:func:`joinless.measurement._run_in_child`'s own reasoning for the same
mechanism.
"""

from __future__ import annotations

import importlib.util
import pkgutil
import subprocess
import sys

import pytest

# The packages the `neural` extra installs, and the whole of what may not be reachable
# from a base-profile import (ADR-0014 for the runtime, ADR-0017 for the tokenizer).
# Named once: the probe below and the failure message both read this, so a package added
# here cannot be covered by one and not reported by the other.
_OFFENDING_PREFIXES = ("onnxruntime", "tokenizers", "onnx")

_PROBE_TEMPLATE = """
import sys
import {module_name}  # noqa: F401
_offending_prefixes = {offending_prefixes!r}
offenders = sorted(
    m for m in sys.modules
    if m in _offending_prefixes or m.startswith(tuple(p + "." for p in _offending_prefixes))
)
print("\\n".join(offenders))
sys.exit(1 if offenders else 0)
"""


def _discovered_module_names() -> list[str]:
    """Every module :mod:`joinless` ships, plus the package itself.

    ``find_spec`` locates the package without executing ``joinless/__init__.py``
    (the same distinction :func:`joinless.cli._cmd_doctor` relies on for
    ``onnxruntime`` itself) — this function never imports the very thing its
    caller is about to test in isolation, in this process or any other.
    """
    spec = importlib.util.find_spec("joinless")
    assert spec is not None, "the joinless package is not installed"
    locations = spec.submodule_search_locations
    assert locations is not None, "joinless has no submodule search path"
    submodules = sorted(
        f"joinless.{info.name}" for info in pkgutil.iter_modules(locations)
    )
    return ["joinless", *submodules]


def _run_probe(module_name: str) -> subprocess.CompletedProcess[str]:
    probe = _PROBE_TEMPLATE.format(
        module_name=module_name, offending_prefixes=_OFFENDING_PREFIXES
    )
    return subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("module_name", _discovered_module_names())
def test_importing_a_joinless_module_never_initialises_the_runtime_or_tokenizer(
    module_name: str,
) -> None:
    result = _run_probe(module_name)

    assert result.returncode == 0, (
        f"importing {module_name!r} pulled one of {_OFFENDING_PREFIXES} into "
        f"sys.modules: {result.stdout.strip() or result.stderr.strip()}"
    )
