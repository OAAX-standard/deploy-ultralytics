import pytest
from helpers import run

SUBCOMMANDS = ["export", "fetch", "convert", "detect"]


def test_help_lists_the_four_subcommands(tmp_path):
    result = run(tmp_path, "--help")
    assert result.returncode == 0
    for name in SUBCOMMANDS:
        assert name in result.stdout


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_subcommand_help_exits_zero(tmp_path, name):
    result = run(tmp_path, name, "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
