import io
import shutil
import subprocess
import tarfile
from pathlib import Path

from deploy_ultralytics.target import load_target

CONTAINER_INPUT_DIR = "/app/input"
CONTAINER_OUTPUT_DIR = "/app/output"


class ConvertError(Exception):
    pass


def convert_model(target_name, model_path, output_dir):
    """Run a target's toolchain on an exported model.

    Writes `<model stem>.<target>.<extension>` and the model manifest as
    `<model stem>.<target>.manifest.json` into `output_dir` and returns their paths. The model
    and the toolchain's output are moved through `docker cp` rather than bind mounts, so the
    paths need not be visible to the Docker daemon.
    """
    target = load_target(target_name)
    model_path = Path(model_path)
    manifest_path = model_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ConvertError(
            f"no model manifest {manifest_path} next to the model: run export to write it"
        )
    if _docker("image", "inspect", target.toolchain_image).returncode != 0:
        raise ConvertError(
            f"toolchain image {target.toolchain_image} is not in Docker: "
            f"run `deploy-ultralytics fetch {target_name}`"
        )

    produced = _run_toolchain(target.toolchain_image, model_path)
    candidates = [name for name in produced if name.endswith(f".{target.converted_model_extension}")]
    if len(candidates) != 1:
        raise ConvertError(
            f"the toolchain wrote {sorted(produced)}; expected exactly one "
            f".{target.converted_model_extension} file"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_path.stem}.{target_name}"
    converted_path = output_dir / f"{stem}.{target.converted_model_extension}"
    converted_path.write_bytes(produced[candidates[0]])
    shutil.copyfile(manifest_path, output_dir / f"{stem}.manifest.json")
    return converted_path, output_dir / f"{stem}.manifest.json"


def _run_toolchain(image, model_path):
    """Return the files the toolchain wrote to its output directory, by name."""
    created = _docker("create", image, f"{CONTAINER_INPUT_DIR}/{model_path.name}", CONTAINER_OUTPUT_DIR)
    if created.returncode != 0:
        raise ConvertError(f"cannot create a container from {image}: {_text(created.stderr)}")
    container = _text(created.stdout)
    try:
        _docker_checked(
            "cp", "-", f"{container}:/app", stdin=_input_tar(model_path), failure="cannot copy the model in"
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


def _input_tar(model_path):
    """A tar of `input/<model>`; the directory is world-writable because the toolchain writes next to its input."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        directory = tarfile.TarInfo("input")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o777
        tar.addfile(directory)
        tar.add(model_path, arcname=f"input/{model_path.name}", filter=_world_readable)
    return buffer.getvalue()


def _world_readable(info):
    info.mode = 0o644
    return info


def _tar_files(data):
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {
            Path(member.name).name: tar.extractfile(member).read()
            for member in tar.getmembers()
            if member.isfile()
        }


def _docker(*args, stdin=None):
    return subprocess.run(["docker", *args], input=stdin, capture_output=True)


def _docker_checked(*args, stdin=None, failure):
    result = _docker(*args, stdin=stdin)
    if result.returncode != 0:
        raise ConvertError(f"{failure}: {_text(result.stderr)}")
    return result


def _text(output):
    return output.decode(errors="replace").strip()
