import subprocess
import pytest


@pytest.fixture(scope="session")
def weights_dir(tmp_path_factory):
    """A working directory shared by all tests so Ultralytics downloads each model once."""
    return tmp_path_factory.mktemp("weights")


TOOLCHAIN_IMAGE = "oaax-cpu-toolchain:2.0.0"


@pytest.fixture(scope="session")
def toolchain_image():
    """The CPU toolchain image, built from the `oaax-v2` branch of reference-implementation."""
    try:
        found = subprocess.run(
            ["docker", "image", "inspect", TOOLCHAIN_IMAGE], capture_output=True
        ).returncode == 0
    except FileNotFoundError:
        found = False
    if not found:
        pytest.skip(
            f"toolchain image {TOOLCHAIN_IMAGE} is not in Docker: build it with "
            "conversion-toolchain/build-toolchain.sh from the oaax-v2 branch of reference-implementation"
        )
    return TOOLCHAIN_IMAGE
