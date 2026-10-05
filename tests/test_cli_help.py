import subprocess
import sys

import pytest

SUBCOMMANDS = ["export", "fetch", "convert", "detect"]


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "deploy_ultralytics", *args],
        capture_output=True,
        text=True,
    )


def test_help_lists_the_four_subcommands():
    result = run("--help")
    assert result.returncode == 0
    for name in SUBCOMMANDS:
        assert name in result.stdout


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_subcommand_help_exits_zero(name):
    result = run(name, "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
