# SPDX-License-Identifier: MIT
"""Fresh-process workers for RFC-0004 step 7 (issue #12).

Each worker is spawned once per arm by :mod:`spikes.quantization.measure`, via
``python -m``, so it starts with a genuinely fresh interpreter rather than one that
already imported the other arm's runtime.
"""
