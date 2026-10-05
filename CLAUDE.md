# deploy-ultralytics

## Workflow (phase 1)

Commit and push straight to `main`; no PRs until phase 1 ships. Every commit carries
`Signed-off-by` (DCO): always pass `-s` to `git commit` (Git has no setting that adds it
automatically) and check the trailer with `git log --format=%(trailers)` before pushing. Never copy code from
the `ultralytics` package (AGPL-3.0, incompatible with Apache-2.0); call it as a dependency only.

## Runtimes during development

No OAAX 2.0 runtime binaries are published yet (`runtimes/2.0.0/` on S3 returns 403).
Until the 2.0 release, build runtimes locally from the `oaax-v2` branches:

- CPU: `reference-implementation` → `runtime-library/build-runtimes.sh`
- Intel: `intel-acceleration` → `runtime-library/build-runtimes.sh` (needs `OPENVINO_DIR`)
- DeepX: `deepx-acceleration` → `runtime-library/build-runtimes.sh` (Docker, needs DX-RT)

Local clones live in `C:\Users\mpods\develop\oaax\<repo>`. Point the example at the built library
instead of downloading one. Do not commit runtime binaries to this repo.

The repo itself is designed for the post-release state: the runtime is fetched from the release
artifacts. Do not redesign that around the current gap.

## Agent skills

### Issue tracker

GitHub Issues of this repository, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
`wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
