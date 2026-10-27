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
RUNTIME_ARCHIVE = {
    "lib/libRuntimeLibrary.so": b"not really a shared library",
    "lib/libdependency.so": b"a library the runtime needs",
}


@pytest.fixture
def artifacts(tmp_path):
    """A local stand-in for the OAAX release artifact store, with the CPU runtime only.

    The runtime archive keeps the runtime in a subdirectory, next to a library it depends on.
    """
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        pytest.skip("the local artifact layout in this test is for the Linux x86_64 runtime")
    store = tmp_path / "store"
    archive = store / "runtimes" / CPU.version / CPU.vendor / "x86_64" / "Ubuntu" / "library.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as tar:
        for name, content in RUNTIME_ARCHIVE.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return store


def with_store(store):
    return {**os.environ, ARTIFACTS_ENV_VAR: str(store)}


def test_fetch_places_the_runtime_and_tags_the_loaded_toolchain_image(tmp_path, artifacts, toolchain_image):
    # The archive carries the image under the vendor's own tag, not the name in the target config.
    vendor_tag = "oaax-fetch-test-vendor:latest"
    image_archive = artifacts / "conversion-toolchain" / CPU.version / CPU.vendor / "oaax-cpu-toolchain.tar"
    image_archive.parent.mkdir(parents=True)
    subprocess.run(["docker", "tag", toolchain_image, vendor_tag], check=True)
    with image_archive.open("wb") as out:
        subprocess.run(["docker", "save", vendor_tag], stdout=out, check=True)
    # Untag the image so fetch has to load it, but keep its layers under a second tag so the
    # developer's image comes back whatever fetch does.
    backup = f"{toolchain_image}-fetch-test"
    subprocess.run(["docker", "tag", toolchain_image, backup], check=True)
    subprocess.run(["docker", "rmi", toolchain_image, vendor_tag], check=True, capture_output=True)
    try:
        result = run(tmp_path, "fetch", "cpu", "--output", str(tmp_path / "out"), env=with_store(artifacts))
        loaded = subprocess.run(["docker", "image", "inspect", toolchain_image], capture_output=True)
    finally:
        subprocess.run(["docker", "tag", backup, toolchain_image], check=True)
        subprocess.run(["docker", "rmi", backup], check=True, capture_output=True)
        subprocess.run(["docker", "rmi", vendor_tag], capture_output=True)

    assert result.returncode == 0, result.stderr
    placed = tmp_path / "out" / "runtimes" / "cpu"
    for name, content in RUNTIME_ARCHIVE.items():
        assert (placed / name).read_bytes() == content
    assert str(placed / "lib" / "libRuntimeLibrary.so") in result.stdout
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
