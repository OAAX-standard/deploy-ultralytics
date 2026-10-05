import pytest


@pytest.fixture(scope="session")
def weights_dir(tmp_path_factory):
    """A working directory shared by all tests so Ultralytics downloads each model once."""
    return tmp_path_factory.mktemp("weights")
