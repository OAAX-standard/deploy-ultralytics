import io
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from deploy_ultralytics.target import load_target

CONTAINER_INPUT_DIR = "/app/input"
CONTAINER_OUTPUT_DIR = "/app/output"


class ConvertError(Exception):
    pass


def convert_model(target_name: str, model_path: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Run a target's toolchain on an exported model, or on a bundle of it if the target config asks for one.

    Writes `<model stem>.<target>.<extension>` and the model manifest as
    `<model stem>.<target>.manifest.json` into `output_dir` and returns their paths. The model
    and the toolchain's output are moved through `docker cp` rather than bind mounts, so the
    paths need not be visible to the Docker daemon.
    """
    target = load_target(target_name)
    model_path = Path(model_path)
    manifest_path = model_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ConvertError(f"no model manifest {manifest_path} next to the model: run export to write it")
    if _docker("image", "inspect", target.toolchain_image).returncode != 0:
        raise ConvertError(
            f"toolchain image {target.toolchain_image} is not in Docker: run `deploy-ultralytics fetch {target_name}`"
        )

    if target.toolchain_input == "bundle":
        input_name, input_data = f"{model_path.stem}.zip", _bundle(model_path, target.bundle_dir)
    else:
        input_name, input_data = model_path.name, model_path.read_bytes()
    produced = _run_toolchain(target.toolchain_image, input_name, input_data)
    candidates = [name for name in produced if name.endswith(f".{target.converted_model_extension}")]
    if len(candidates) != 1:
        raise ConvertError(
            f"the toolchain wrote {sorted(produced)}; expected exactly one .{target.converted_model_extension} file"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_path.stem}.{target_name}"
    converted_path = output_dir / f"{stem}.{target.converted_model_extension}"
    converted_path.write_bytes(produced[candidates[0]])
    shutil.copyfile(manifest_path, output_dir / f"{stem}.manifest.json")
    return converted_path, output_dir / f"{stem}.manifest.json"


def _run_toolchain(image: str, input_name: str, input_data: bytes) -> dict[str, bytes]:
    """Return the files the toolchain wrote to its output directory, by name."""
    created = _docker("create", image, f"{CONTAINER_INPUT_DIR}/{input_name}", CONTAINER_OUTPUT_DIR)
    if created.returncode != 0:
        raise ConvertError(f"cannot create a container from {image}: {_text(created.stderr)}")
    container = _text(created.stdout)
    try:
        _docker_checked(
            "cp",
            "-",
            f"{container}:/app",
            stdin=_input_tar(input_name, input_data),
            failure=f"cannot copy {input_name} in",
        )
        started = _docker("start", "--attach", container)
        if started.returncode != 0:
            raise ConvertError(f"the toolchain failed:\n{_text(started.stdout + started.stderr)}")
        archive = _docker_checked(
            "cp", f"{container}:{CONTAINER_OUTPUT_DIR}", "-", failure="cannot copy the output out"
        )
        return _tar_files(archive.stdout)
    finally:
        _docker("rm", "--force", container)


def _bundle(model_path: Path, bundle_dir: Path) -> bytes:
    """A zip of the exported model under its own name and the files of the target's bundle directory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(model_path, model_path.name)
        for path in sorted(bundle_dir.iterdir()):
            bundle.write(path, path.name)
    return buffer.getvalue()


def _input_tar(name: str, data: bytes) -> bytes:
    """A tar of `input/<name>`; the directory is world-writable because the toolchain writes next to its input."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        directory = tarfile.TarInfo("input")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o777
        tar.addfile(directory)
        file = tarfile.TarInfo(f"input/{name}")
        file.size = len(data)
        file.mode = 0o644
        tar.addfile(file, io.BytesIO(data))
    return buffer.getvalue()


def _tar_files(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {
            Path(member.name).name: tar.extractfile(member).read() for member in tar.getmembers() if member.isfile()
        }


def _docker(*args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["docker", *args], input=stdin, capture_output=True)


def _docker_checked(*args: str, stdin: bytes | None = None, failure: str) -> subprocess.CompletedProcess[bytes]:
    result = _docker(*args, stdin=stdin)
    if result.returncode != 0:
        raise ConvertError(f"{failure}: {_text(result.stderr)}")
    return result


def _text(output: bytes) -> str:
    return output.decode(errors="replace").strip()
