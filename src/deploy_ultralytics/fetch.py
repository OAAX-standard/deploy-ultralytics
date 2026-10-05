import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from deploy_ultralytics.target import load_target

ARTIFACTS_ENV_VAR = "DEPLOY_ULTRALYTICS_ARTIFACTS"
RUNTIME_LIBRARY_NAME = "libRuntimeLibrary.so"

_ARCHITECTURES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
_OPERATING_SYSTEMS = {"Linux": "Ubuntu", "Windows": "Windows"}


class FetchError(Exception):
    pass


def fetch_target(target_name, output_dir):
    """Download a target's runtime and toolchain image from the OAAX release artifact store.

    The runtime library is placed in `<output_dir>/runtimes/<target>/` and the toolchain image is
    loaded into Docker. `DEPLOY_ULTRALYTICS_ARTIFACTS` replaces the store with a local directory
    or a URL laid out the same way. Returns the path of the runtime library.
    """
    target = load_target(target_name)
    store = os.environ.get(ARTIFACTS_ENV_VAR) or target.store_url
    vendor, version = target.vendor, target.version

    runtime_path = f"runtimes/{version}/{vendor}/{_architecture()}/{_operating_system()}/library.tar.gz"
    toolchain_path = f"conversion-toolchain/{version}/{vendor}/oaax-{vendor.lower()}-toolchain.tar"

    library = _place_runtime(store, runtime_path, Path(output_dir) / "runtimes" / target_name)
    _load_toolchain(store, toolchain_path, target.toolchain_image)
    return library


def _place_runtime(store, path, directory):
    with _download(store, path) as archive:
        directory.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive.name) as tar:
            members = [m for m in tar.getmembers() if Path(m.name).name == RUNTIME_LIBRARY_NAME]
            if len(members) != 1:
                raise FetchError(f"{path} does not hold exactly one {RUNTIME_LIBRARY_NAME}")
            library = directory / RUNTIME_LIBRARY_NAME
            with tar.extractfile(members[0]) as source, library.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    library.chmod(0o755)
    return library


def _load_toolchain(store, path, image):
    with _download(store, path) as archive:
        loaded = subprocess.run(["docker", "load"], stdin=archive, capture_output=True, text=True)
    if loaded.returncode != 0:
        raise FetchError(f"docker load of {path} failed: {loaded.stderr.strip()}")
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        raise FetchError(f"{path} did not provide the image {image} the target expects")


def _download(store, path):
    """Return a temporary file holding `path` under `store`, a local directory or an URL."""
    source = f"{store.rstrip('/')}/{path}"
    destination = tempfile.NamedTemporaryFile()
    try:
        if store.startswith(("http://", "https://")):
            with urllib.request.urlopen(source) as response:
                shutil.copyfileobj(response, destination)
        else:
            with open(Path(store) / path, "rb") as local:
                shutil.copyfileobj(local, destination)
    except (urllib.error.URLError, OSError) as error:
        destination.close()
        raise FetchError(
            f"cannot get {source}: {error}; until OAAX 2.0 is published, "
            f"point {ARTIFACTS_ENV_VAR} at a local directory or URL holding {path}"
        ) from error
    destination.flush()
    destination.seek(0)
    return destination


def _architecture():
    machine = platform.machine().lower()
    if machine not in _ARCHITECTURES:
        raise FetchError(f"no OAAX runtime for the architecture {machine}")
    return _ARCHITECTURES[machine]


def _operating_system():
    system = platform.system()
    if system not in _OPERATING_SYSTEMS:
        raise FetchError(f"no OAAX runtime for the operating system {system}")
    return _OPERATING_SYSTEMS[system]
