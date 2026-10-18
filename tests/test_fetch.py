import io
import os
import platform
import subprocess
import tarfile

import pytest
from helpers import run

from deploy_ultralytics.fetch import ARTIFACTS_ENV_VAR
from deploy_ultralytics.target import load_target

CPU = load_target("cpu")


@pytest.fixture
def artifacts(tmp_path):
    """A local stand-in for the OAAX release artifact store, with the CPU runtime only."""
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        pytest.skip("the local artifact layout in this test is for the Linux x86_64 runtime")
    store = tmp_path / "store"
    archive = store / "runtimes" / CPU.version / CPU.vendor / "x86_64" / "Ubuntu" / "library.tar.gz"
    archive.parent.mkdir(parents=True)
    content = b"not really a shared library"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("./libRuntimeLibrary.so")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return store


def with_store(store):
    return {**os.environ, ARTIFACTS_ENV_VAR: str(store)}


def test_fetch_places_the_runtime_and_loads_the_toolchain_image(tmp_path, artifacts, toolchain_image):
    image_archive = artifacts / "conversion-toolchain" / CPU.version / CPU.vendor / "oaax-cpu-toolchain.tar"
    image_archive.parent.mkdir(parents=True)
    with image_archive.open("wb") as out:
        subprocess.run(["docker", "save", toolchain_image], stdout=out, check=True)
    subprocess.run(["docker", "rmi", toolchain_image], check=True, capture_output=True)

    result = run(tmp_path, "fetch", "cpu", "--output", str(tmp_path / "out"), env=with_store(artifacts))

    assert result.returncode == 0, result.stderr
    library = tmp_path / "out" / "runtimes" / "cpu" / "libRuntimeLibrary.so"
    assert library.read_bytes() == b"not really a shared library"
    assert str(library) in result.stdout
    loaded = subprocess.run(["docker", "image", "inspect", toolchain_image], capture_output=True)
    assert loaded.returncode == 0


def test_fetch_names_the_missing_artifact(tmp_path, artifacts):
    result = run(tmp_path, "fetch", "cpu", "--output", str(tmp_path / "out"), env=with_store(artifacts))

    assert result.returncode != 0
    assert "oaax-cpu-toolchain.tar" in result.stderr
    assert ARTIFACTS_ENV_VAR in result.stderr


def test_fetch_of_an_unknown_target_lists_the_known_ones(tmp_path):
    result = run(tmp_path, "fetch", "no-such-target")

    assert result.returncode != 0
    assert "unknown target 'no-such-target'" in result.stderr
    assert "cpu" in result.stderr
