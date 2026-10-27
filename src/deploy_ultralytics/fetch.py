import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO

from deploy_ultralytics.target import load_target

ARTIFACTS_ENV_VAR = "DEPLOY_ULTRALYTICS_ARTIFACTS"
RUNTIME_FILE_NAME = "libRuntimeLibrary.so"

RUNTIME_ARCHITECTURE = "x86_64"
RUNTIME_OPERATING_SYSTEM = "Ubuntu"

# How many times a download resumes after the store ends a response before its Content-Length.
RESUME_ATTEMPTS = 5


class FetchError(Exception):
    pass


def fetch_target(target_name: str, output_dir: str | Path) -> Path:
    """Download a target's runtime and toolchain image from the OAAX release artifact store.

    The runtime is placed in `<output_dir>/runtimes/<target>/` and the toolchain image is
    loaded into Docker. `DEPLOY_ULTRALYTICS_ARTIFACTS` replaces the store with a local directory
    or a URL laid out the same way. Returns the path of the runtime.
    """
    target = load_target(target_name)
    if (platform.system(), platform.machine()) != ("Linux", RUNTIME_ARCHITECTURE):
        raise FetchError("OAAX runtimes are only available for Linux on x86_64")
    store = os.environ.get(ARTIFACTS_ENV_VAR) or target.store_url
    vendor, version = target.vendor, target.version

    runtime_path = f"runtimes/{version}/{vendor}/{RUNTIME_ARCHITECTURE}/{RUNTIME_OPERATING_SYSTEM}/library.tar.gz"
    toolchain_path = f"conversion-toolchain/{version}/{vendor}/oaax-{vendor.lower()}-toolchain.tar"

    library = _place_runtime(store, runtime_path, Path(output_dir) / "runtimes" / target_name)
    _load_toolchain(store, toolchain_path, target.toolchain_image)
    return library


def _place_runtime(store: str, path: str, directory: Path) -> Path:
    """Extract the whole runtime archive, so the runtime finds the libraries shipped beside it."""
    with _download(store, path) as archive:
        directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive.name) as tar:
            tar.extractall(directory, filter="data")
    libraries = list(directory.rglob(RUNTIME_FILE_NAME))
    if len(libraries) != 1:
        raise FetchError(f"{path} does not hold exactly one {RUNTIME_FILE_NAME}")
    libraries[0].chmod(0o755)
    return libraries[0]


def _load_toolchain(store: str, path: str, image: str) -> None:
    """Load the toolchain image and tag it with the target config's name, whatever tag the vendor gave it."""
    with _download(store, path) as archive:
        loaded = subprocess.run(["docker", "load"], stdin=archive, capture_output=True, text=True)
    if loaded.returncode != 0:
        raise FetchError(f"docker load of {path} failed: {loaded.stderr.strip()}")
    # `docker load` prints "Loaded image: <name>", or "Loaded image ID: <id>" for an untagged image.
    names = [line.split(": ", 1)[1] for line in loaded.stdout.splitlines() if line.startswith("Loaded image")]
    if len(names) != 1:
        raise FetchError(f"{path} holds {len(names)} images; expected the one toolchain image")
    tagged = subprocess.run(["docker", "tag", names[0], image], capture_output=True, text=True)
    if tagged.returncode != 0:
        raise FetchError(f"cannot tag the image from {path} as {image}: {tagged.stderr.strip()}")


def _download(store: str, path: str) -> IO[bytes]:
    """Return a temporary file holding `path` under `store`, a local directory or an URL."""
    source = f"{store.rstrip('/')}/{path}"
    destination = tempfile.NamedTemporaryFile()
    try:
        if store.startswith(("http://", "https://")):
            _download_url(source, destination)
        else:
            with open(Path(store) / path, "rb") as local:
                shutil.copyfileobj(local, destination)
    except (urllib.error.URLError, OSError) as error:
        destination.close()
        raise FetchError(
            f"cannot get {source}: {error}; to use another store, "
            f"point {ARTIFACTS_ENV_VAR} at a local directory or URL holding {path}"
        ) from error
    destination.flush()
    destination.seek(0)
    return destination


def _download_url(url: str, destination: IO[bytes]) -> None:
    """Copy `url` into `destination`. The store can end a response early without an error, so a
    download shorter than its Content-Length resumes with a range request."""
    with urllib.request.urlopen(url) as response:
        size = response.length
        shutil.copyfileobj(response, destination)
    for _ in range(RESUME_ATTEMPTS):
        if size is None or destination.tell() >= size:
            return
        request = urllib.request.Request(url, headers={"Range": f"bytes={destination.tell()}-"})
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise OSError(f"the store ignored the range request and answered {response.status}")
            shutil.copyfileobj(response, destination)
    if destination.tell() < size:
        raise OSError(f"the store ended the download at {destination.tell()} of {size} bytes")
