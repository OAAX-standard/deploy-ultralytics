import os
import subprocess
from pathlib import Path

import pytest
from helpers import run


@pytest.fixture(scope="session")
def source_models_dir(tmp_path_factory):
    """A working directory shared by all tests so Ultralytics downloads each model once."""
    return tmp_path_factory.mktemp("source-models")


@pytest.fixture(scope="session")
def runtime():
    path = os.environ.get("DEPLOY_ULTRALYTICS_RUNTIME")
    if not path:
        pytest.skip("DEPLOY_ULTRALYTICS_RUNTIME is not set: no runtime to run detect with")
    assert Path(path).is_file(), f"DEPLOY_ULTRALYTICS_RUNTIME={path} is not a file"
    return Path(path)


@pytest.fixture(scope="session")
def exported_model(source_models_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("exported")
    result = run(source_models_dir, "export", "yolov8n", "--output", str(out))
    assert result.returncode == 0, result.stderr
    return out / "yolov8n.onnx"


TOOLCHAIN_IMAGE = "oaax-cpu-toolchain:2.0.0"


@pytest.fixture(scope="session")
def toolchain_image():
    """The CPU toolchain image, loaded by `deploy-ultralytics fetch cpu`."""
    try:
        found = subprocess.run(
            ["docker", "image", "inspect", TOOLCHAIN_IMAGE], capture_output=True
        ).returncode == 0
    except FileNotFoundError:
        found = False
    if not found:
        pytest.skip(
            f"toolchain image {TOOLCHAIN_IMAGE} is not in Docker: run `deploy-ultralytics fetch cpu`"
        )
    return TOOLCHAIN_IMAGE
