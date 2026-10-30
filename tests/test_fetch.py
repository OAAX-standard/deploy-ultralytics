import io
import os
import platform
import subprocess
import tarfile
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from helpers import TARGET, TARGET_NAME, run

from deploy_ultralytics.fetch import ARTIFACTS_ENV_VAR

TOOLCHAIN_ARCHIVE = f"oaax-{TARGET.vendor.lower()}-toolchain.tar"
RUNTIME_ARCHIVE = {
    "lib/libRuntimeLibrary.so": b"not really a shared library",
    "lib/libdependency.so": b"a library the runtime needs",
}


@pytest.fixture
def artifacts(tmp_path):
    """A local stand-in for the OAAX release artifact store, with the target's runtime only.

    The runtime archive keeps the runtime in a subdirectory, next to a library it depends on.
    """
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        pytest.skip("the local artifact layout in this test is for the Linux x86_64 runtime")
    store = tmp_path / "store"
    archive = store / "runtimes" / TARGET.version / TARGET.vendor / "x86_64" / "Ubuntu" / "library.tar.gz"
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
    image_archive = artifacts / "conversion-toolchain" / TARGET.version / TARGET.vendor / TOOLCHAIN_ARCHIVE
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
        result = run(tmp_path, "fetch", TARGET_NAME, "--output", str(tmp_path / "out"), env=with_store(artifacts))
        loaded = subprocess.run(["docker", "image", "inspect", toolchain_image], capture_output=True)
    finally:
        subprocess.run(["docker", "tag", backup, toolchain_image], check=True)
        subprocess.run(["docker", "rmi", backup], check=True, capture_output=True)
        subprocess.run(["docker", "rmi", vendor_tag], capture_output=True)

    assert result.returncode == 0, result.stderr
    placed = _assert_runtime_placed(tmp_path)
    assert str(placed / "lib" / "libRuntimeLibrary.so") in result.stdout
    assert loaded.returncode == 0


def test_fetch_names_the_missing_artifact(tmp_path, artifacts):
    result = run(tmp_path, "fetch", TARGET_NAME, "--output", str(tmp_path / "out"), env=with_store(artifacts))

    assert result.returncode != 0
    assert TOOLCHAIN_ARCHIVE in result.stderr
    assert ARTIFACTS_ENV_VAR in result.stderr


def test_fetch_resumes_a_download_the_store_cuts_short(tmp_path, artifacts):
    with _store_that_cuts_the_first_response_short(artifacts) as url:
        result = run(tmp_path, "fetch", TARGET_NAME, "--output", str(tmp_path / "out"), env=with_store(url))

    # This store has no toolchain archive: fetch gets that far only if it got the whole runtime.
    assert TOOLCHAIN_ARCHIVE in result.stderr
    _assert_runtime_placed(tmp_path)


def test_fetch_refuses_a_resume_the_store_answers_from_the_start(tmp_path, artifacts):
    with _store_that_cuts_the_first_response_short(artifacts, honours_range=False) as url:
        result = run(tmp_path, "fetch", TARGET_NAME, "--output", str(tmp_path / "out"), env=with_store(url))

    assert result.returncode != 0
    assert "ignored the range request" in result.stderr


def test_fetch_of_an_unknown_target_lists_the_known_ones(tmp_path):
    result = run(tmp_path, "fetch", "no-such-target")

    assert result.returncode != 0
    assert "unknown target 'no-such-target'" in result.stderr
    assert "cpu" in result.stderr


class _CutShortHandler(SimpleHTTPRequestHandler):
    """Serves the store, but ends the first response for each file after half of its bytes. A store
    that does not honour ranges answers a range request with the whole file."""

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        start = 0
        if "Range" in self.headers and self.server.honours_range:
            start = int(self.headers["Range"].removeprefix("bytes=").rstrip("-"))
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        if path in self.server.cut:
            self.wfile.write(data[start:])
        else:
            self.server.cut.add(path)
            self.wfile.write(data[start : len(data) // 2])
        self.close_connection = True

    def log_message(self, *args):
        pass


@contextmanager
def _store_that_cuts_the_first_response_short(store, honours_range=True):
    handler = partial(_CutShortHandler, directory=str(store))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        server.cut = set()
        server.honours_range = honours_range
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def _assert_runtime_placed(tmp_path):
    """Every file of the runtime archive is under the target's runtime directory; returns that directory."""
    placed = tmp_path / "out" / "runtimes" / TARGET_NAME
    for name, content in RUNTIME_ARCHIVE.items():
        assert (placed / name).read_bytes() == content
    return placed
