---
status: accepted
date: 2026-10-05
---

# Python via PythonBindings instead of the inherited C++ example

The repository was seeded from the `ultralytics-inference-example` branch of `examples`: ~1200
lines of C++ written against the OAAX v1 API and `tools/c-utilities`. We decided to drop that code
from `main` (it stays in history) and write the example in Python on top of PythonBindings, which
already target the 2.0 API. The reader we serve is evaluating OAAX across accelerators and lives in
Python because Ultralytics does; porting the C++ to 2.0 meant rewriting its runtime and tensor
layers and asking the reader for OpenCV, CMake and four vendored dependencies, and every change to
the still-unreleased `interface.h` would break it again, whereas the bindings absorb such changes
in one place.

## Consequences

- The example depends on a private repository with no release; publishing `deploy-ultralytics`
  therefore depends on publishing PythonBindings.
  *Amended 2026-10-30:* this consequence no longer holds. PythonBindings is public, and the example
  depends on its release `oaax` 0.1.1 from PyPI. The decision is unchanged.
- A C++ example, if wanted later, is a second client of the same export/convert pipeline, not a
  revival of the deleted code.
