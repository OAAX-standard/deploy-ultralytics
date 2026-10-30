import os
import subprocess
from pathlib import Path

import pytest
from helpers import CONVERTED_MODEL, TARGET, TARGET_NAME, run

from deploy_ultralytics.detect import RUNTIME_ENV_VAR


@pytest.fixture(scope="session")
def source_models_dir(tmp_path_factory):
    """A working directory shared by all tests so Ultralytics downloads each model once."""
    return tmp_path_factory.mktemp("source-models")


@pytest.fixture(scope="session")
def runtime():
    path = os.environ.get(RUNTIME_ENV_VAR)
    if not path:
        pytest.skip(f"{RUNTIME_ENV_VAR} is not set: no runtime to run detect with")
    assert Path(path).is_file(), f"{RUNTIME_ENV_VAR}={path} is not a file"
    return Path(path)


@pytest.fixture(scope="session")
def exported_model(source_models_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("exported")
    result = run(source_models_dir, "export", "yolov8n", "--output", str(out))
    assert result.returncode == 0, result.stderr
    return out / "yolov8n.onnx"


@pytest.fixture(scope="session")
def toolchain_image():
    """The target's toolchain image, loaded by `deploy-ultralytics fetch <target>`."""
    image = TARGET.toolchain_image
    try:
        found = subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0
    except FileNotFoundError:
        found = False
    if not found:
        pytest.skip(f"toolchain image {image} is not in Docker: run `deploy-ultralytics fetch {TARGET_NAME}`")
    return image


@pytest.fixture(scope="session")
def converted_model(exported_model, toolchain_image, tmp_path_factory):
    """The exported model converted for the target, the only model its runtime loads."""
    out = tmp_path_factory.mktemp("converted")
    result = run(out, "convert", "--target", TARGET_NAME, "--output", str(out), str(exported_model))
    assert result.returncode == 0, result.stderr
    return out / CONVERTED_MODEL
