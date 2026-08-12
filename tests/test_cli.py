# SPDX-License-Identifier: MIT
"""The console entry point exits cleanly and names no commands that do not exist."""

import pytest


def test_help_exits_zero() -> None:
    from joinless.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0


def test_no_commands_are_declared() -> None:
    from joinless.cli import main

    # No commands exist at this point in the project. A word that would be a
    # subcommand name in a later version must be rejected as an unrecognized
    # argument today, which is only true if no subcommand group is wired up.
    with pytest.raises(SystemExit) as excinfo:
        main(["resolve"])

    assert excinfo.value.code != 0
