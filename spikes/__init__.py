# SPDX-License-Identifier: MIT
"""Spikes: bounded, scripted feasibility checks that run outside the package.

A spike's output is a record in ``benchmarks/``, not library code (issue #6). Nothing
under ``joinless/`` imports from this tree, and nothing here is part of the package's
public surface.
"""
