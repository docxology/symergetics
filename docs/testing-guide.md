# Testing exact geometry, plotting resources and integration boundaries

[Contributor workflow](contributor-development-guide.md) · [Test responsibilities](test-architecture-summary.md)

## Overview

Tests live in [`tests/`](../tests/); the [test index](../tests/TEST_INDEX.md) is
a navigation aid, not a live test-count receipt. Current collection and executed
results depend on the source revision and environment. Historical outcomes
belong in [STATUS.md](../STATUS.md), clearly dated, never advertised as current
green status.

## Running Tests

Prepare the scientific Python 3.13 environment without changing repository pins:

```bash
uv sync --python 3.13 --extra scientific --extra test
```

Full package gate:

```bash
MPLBACKEND=Agg uv run --python 3.13 --extra scientific --extra test python -m pytest tests/ -q
```

Targeted behavioral diagnostic:

```bash
MPLBACKEND=Agg uv run --python 3.13 --extra scientific --extra test python -m pytest tests/test_core_coordinates.py -q --no-cov
```

Discover current cases without claiming that collection executes them:

```bash
uv run --python 3.13 --extra scientific --extra test python -m pytest tests/ --collect-only -q --no-cov
```

Commands here are instructions, not receipts that they were run in this docs
pass. Coordinated implementation assignments may explicitly defer all suites
until integration; do not launch redundant gates during those assignments.

### Coverage Analysis

[`pyproject.toml`](../pyproject.toml) enables strict markers/configuration,
package-wide coverage, terminal and HTML reports, an 85% floor and fail-fast.
A single-file run can pass its tests yet fail the **package-wide** floor.
`--no-cov` is appropriate for a focused diagnostic, not a substitute for the
full gate. Read the full pytest result, including skips and the coverage exit
reason. Default HTML output goes to `htmlcov/`.

## Test Architecture

- Core: represented rational values, constructor snapping, gauge aliases,
  exact metrics and nearest-lattice selection.
- Geometry: indexed edges/fans, mutable vertices, rational volume, solid versus
  vertex centers, degenerate geometry and shell semantics.
- Computation/utilities: finite prime/digit inputs, exact results, conversion
  boundaries and report/format behavior.
- Visualization: actual files and metadata, physical geometry, supported
  backends, optional sections, error paths and resource closure.
- Integration/orchestration: real consumer processes, interpreter selection,
  requested-stage outcomes and cross-project conventions.

The [architecture summary](test-architecture-summary.md) links representative
modules. It intentionally avoids saved category totals and invented case names.

## Test Patterns

Use real computations and temporary directories. For exact methods, compare
against independent exact arithmetic or meaningful invariants; for floating
results, declare the tolerance and physical scale. A useful regression
separates the broken behavior from the corrected behavior, rather than testing
only a source string, default word, export existence or the test's own copy of
the implementation. Do not substitute mocked success for real scientific data.

Plot tests should isolate output/configuration and restore it, inspect the
consumer-visible resource/error behavior at issue and avoid writing into
repository `output/`. A generated file's existence alone does not establish
correct geometry, labels, units or readability. Rendering inspection complements,
not replaces, numerical assertions.

## Performance Testing

Use bounded workloads and record input size, environment and repeat samples.
No fixed timing threshold or speedup in this guide is a package guarantee.
See [measurement methodology](performance-optimization-guide.md). The optional
benchmark plugin does not imply every test is a benchmark or that
`--benchmark-only` exercises correctness coverage.

## Continuous Integration

Consult actual [package](../.github/workflows/verify.yml) and
[suite](../.github/workflows/suite.yml) workflows for interpreter matrices,
extras and commands. A local targeted pass does not establish all CI profiles.
The [suite guide](synergetics-suite.md) documents sibling/tool prerequisites.

## Troubleshooting

For a failure, preserve its traceback and the selected interpreter, revision,
backend and dependency versions. Do not rerun a reported failure merely to
confirm it; investigate the source contract and failing input. Distinguish a
coverage-floor failure, optional dependency skip, environmental failure and
mathematical assertion failure. No saved count proves that a new tree is green.
