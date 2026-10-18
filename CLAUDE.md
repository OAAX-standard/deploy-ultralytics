# deploy-ultralytics

## Workflow (phase 1)

Commit and push straight to `main`; no PRs until phase 1 ships. Every commit carries
`Signed-off-by` (DCO): always pass `-s` to `git commit` (Git has no setting that adds it
automatically) and check the trailer with `git log --format=%(trailers)` before pushing. Never copy code from
the `ultralytics` package (AGPL-3.0, incompatible with Apache-2.0); call it as a dependency only.

## Runtimes

Runtimes and toolchain images come from the OAAX release artifact store through
`deploy-ultralytics fetch <target>`; the 2.0.0 CPU and Intel ones are published. Do not commit
runtime binaries to this repo. To try an unreleased runtime, build it from its vendor repository
and pass it to `detect` with `--runtime` or `DEPLOY_ULTRALYTICS_RUNTIME`.

## Agent skills

### Issue tracker

GitHub Issues of this repository, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
`wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
