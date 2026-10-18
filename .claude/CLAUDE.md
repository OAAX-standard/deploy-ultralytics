# deploy-ultralytics

## Workflow

Work on a branch from `main`. Batch related changes into one pull request with one commit per
concern; open a separate one only for a change that must merge on its own. Every commit carries
`Signed-off-by` (DCO): always pass `-s` to `git commit` (Git has no setting that adds it
automatically) and check the trailer with `git log --format=%(trailers)` before pushing. Never put
`Co-Authored-By` or `Claude-Session` lines in a commit message. Never copy code from
the `ultralytics` package (AGPL-3.0, incompatible with Apache-2.0); call it as a dependency only.

## Runtimes

Runtimes and toolchain images come from the OAAX release artifact store through
`deploy-ultralytics fetch <target>`; the 2.0.0 CPU and Intel ones are published. Do not commit
runtime binaries to this repo. To try an unreleased runtime, build it from its vendor repository
and pass it to `detect` with `--runtime` or `DEPLOY_ULTRALYTICS_RUNTIME`.

## Agent skills

### Issue tracker

GitHub Issues of this repository, via the `gh` CLI. See `.claude/rules/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
`wontfix`). See `.claude/rules/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `.claude/rules/domain.md`.
