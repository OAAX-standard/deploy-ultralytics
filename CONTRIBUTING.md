# Contributing to deploy-ultralytics

Thank you for your interest in contributing! Before you start, please read the
[OAAX general contributing guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
for project-wide conventions (commit message format, code style, etc.).

## How to contribute

1. **Open an issue** to discuss the bug or feature before writing code.
2. **Fork the repository** and create a branch in your fork from `main`.
3. **Make your changes**, keeping each PR focused on one thing.
4. **Run the tests** locally (see below) to confirm nothing is broken.
5. **Push your branch** to your fork and open a pull request against `main`.

CI runs on each pull request. On your first pull request, CI starts only after a maintainer
approves the run.

Sign off every commit (`git commit -s`): the Developer Certificate of Origin is required.

Never copy code from the `ultralytics` package. It is AGPL-3.0, which is incompatible with this
repository's Apache-2.0 licence; call it as a dependency only.

## Development environment

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv). The OAAX runtimes are Linux
x86_64 shared libraries, so the `detect`, `convert` and `fetch` tests need x86_64 Linux (WSL works)
and Docker.

```sh
uv sync --locked --extra test
uv run pytest
```

Tests that need a runtime or a toolchain image skip when it is missing. To run them, fetch both,
as the CI workflow does, and point the tests at the runtime:

```sh
uv run deploy-ultralytics fetch cpu
export DEPLOY_ULTRALYTICS_RUNTIME=$PWD/runtimes/cpu/libRuntimeLibrary.so
```

The tests use the cpu target. To test another target, fetch it and set
`DEPLOY_ULTRALYTICS_TARGET` to its name, as the CI job for that target does.

CI fails when any test is skipped.

CI also runs the lint hooks: file hygiene, and ruff for lint and format. Run them before you push:

```sh
uvx pre-commit run --all-files
```
