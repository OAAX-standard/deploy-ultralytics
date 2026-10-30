---
status: accepted
date: 2026-10-27
---

# Targets are data: a target config with a closed set of variants, no per-target code

The claim of this repository is that one inference program serves every target and only the
conversion step differs. Intel, the second target, differs from CPU in its toolchain input (a
bundle, not the bare exported model), its runtime archive (many libraries, not one) and its
toolchain image tag. We decided that a target is described only by its target config and the files
in its directory. Shared code supports a closed set of named variants, for example
`toolchain.input` is `model` or `bundle`, and a target selects one. Shared code has no branch on
the target name and loads no code from a target directory. Thus the difference between two targets
is the difference between their two directories, which a reader can compare directly.

## Considered options

- **A Python module for each target**, with hooks that shared code calls (prepare the toolchain
  input, unpack the runtime). This is more flexible, but each target becomes a separate program
  again, and the claim of the repository is then no longer visible.
- **Config first, with a module as an escape route.** We rejected it because the escape route
  becomes the usual route at the first difficult target.

## Consequences

- A target that no variant covers adds a new variant to shared code. After that, every target can
  select it.
- If two targets differ by accident and not by need, we change shared code for all targets and do
  not add a config field. Examples: `fetch` extracts the full runtime archive for every target, and
  it tags each loaded toolchain image with the name in the target config, whatever tag the vendor
  gave the image.
- We add a variant only when a target needs it. The DeepX needs that the current variants do not
  cover (a toolchain config that depends on the model, calibration data, a different runtime path
  in the artifact store) wait for the DeepX target.
