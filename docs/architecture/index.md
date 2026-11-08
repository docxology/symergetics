# Package architecture: arithmetic, indexed surfaces and file-producing views

[Documentation map](../index.md) · [API contracts](../api/index.md) · [Contributor guide](../contributor-development-guide.md)

## System Overview

```text
symergetics/
  core/           rational scalars, quadray coordinates, reference constants
  geometry/       indexed polyhedra, solid measurements, quantized transforms
  computation/    primorials, decimal patterns, finite analysis, mnemonics
  utils/          conversion, number formatting, reports
  visualization/  configuration, geometry/number/math/advanced rendering
```

The package root re-exports a selected API in
[`symergetics/__init__.py`](../../symergetics/__init__.py), not every helper.
NumPy is a core dependency. Scientific backends are optional extras declared in
[`pyproject.toml`](../../pyproject.toml). There is no runtime import of sibling
QuadMath or QuadCraft packages; cross-project checks belong to the examples and
suite verifier.

## Core Module Architecture

### SymergeticsNumber Class

[`core/numbers.py`](../../symergetics/core/numbers.py) wraps `Fraction` and
promotes integral fields before native arithmetic. Rational operations and
perfect rational roots stay exact; irrational powers use finite-precision
approximation. Display and float conversion are separate boundaries. There is
no general symbolic-expression engine or custom arithmetic exception hierarchy.

### Quadray Coordinate System

[`core/coordinates.py`](../../symergetics/core/coordinates.py) stores integer
labels for a rank-three quotient. Constructor snapping is distinct from the
XYZ nearest-lattice inverse. Minimum-zero normalization chooses a gauge;
`add`/`sub` retain raw labels. Equality/hash use stored labels, so mutable
coordinates need care in hashed containers. The embedding, exact metrics,
nearest projection and IVM shell enumeration share this module. See the
[canonical coordinate contract](../api/index.md#quadraycoordinate).

## Geometry Module Architecture

### Polyhedral Classes

[`geometry/polyhedra.py`](../../symergetics/geometry/polyhedra.py) declares
face cycles over mutable vertex lists. Edges are deduplicated by index;
faces become triangle fans. Volume, area, diagnostics and both exact centroids
are recomputed from current vertices rather than cached reference constants.

Closed orientable nonintersecting shells are a precondition for solid measures.
Multiple shells use even-odd enclosure to model cavities. Face nonplanarity,
rank, closure and convexity diagnostics are not an intersection certificate.
Vertex mean, floor-snapped centroid and solid mass center are separate APIs;
see [solid analysis](../api/index.md#indexed-surfaces-and-live-solid-analysis).

### Transformation System

[`geometry/transformations.py`](../../symergetics/geometry/transformations.py)
implements translation, scale, reflection, coordinate matrices, XYZ rotations
and plane projection. Operations leaving the lattice project back; they are
not generally invertible or metric preserving. Gauge-compatible 4 by 4 maps
and continuous 4 by 3 pseudoinverses have different contracts from discrete
nearest conversion.

## Computation Module Architecture

Primorials use a prime upper bound, Scheherazade values are powers of 1001,
and palindrome analysis concerns finite decimal strings. Analysis metrics and
geometric mnemonic matches describe their selected inputs; they are not
automatically scientific evidence for a natural system. Functions return
scalars, lists or dictionaries as declared in the source. See
[computation reference](../api/index.md#computation-module).

## Visualization and output lifecycle

[`visualization/__init__.py`](../../symergetics/visualization/__init__.py)
owns process-global configuration, organized path creation, export/batch and
animation helpers. `set_config` merges a dictionary and validates dimensions
before replacing configuration. It is not a thread-local context manager.

Plot functions produce files plus metadata; they do not expose live figures as
the public return contract. Geometry displays anchor-relative scale-one XYZ
coordinates and records exact origins and live analysis. The same declared
fans underpin measurement and rendering. Backend support and output format
handling are function-specific, not universal promises attached to global
`formats`. See [plotting contracts](../api/index.md#visualization-module).

The unreleased main mega API retains its original three positional parameters
and adds keyword-only topic selection. Its focused default, all-topic option,
computed payload and overwrite behavior are documented in
[poster sections](../api/index.md#scientific-poster-sections-unreleased-main).

## Error Handling Architecture

Public methods use standard exceptions such as `ValueError`, `TypeError`,
`ZeroDivisionError` and optional-backend `ImportError`. Source validation is
specific to each function. No `SynergeticsError`, `GeometryError` or
`ConversionError` class tree ships here. A validation failure must not be
silently reclassified as a successful plot or a different computation.

## Orchestration (run.py)

[`run.py`](../../run.py) selects the running interpreter unless `--python`
is supplied. Default setup installs scientific/test extras into that selected
environment; tests and examples are real subprocess stages. `--examples-only`
skips setup and tests. Requested stages continue after failures and the final
exit status records failure. Reports under `output/orchestration_results.json`
include commands, outcomes and an existing-file inventory; an existing output
is not proof that this run produced it. The orchestrator writes `logs/` and
`output/`, so do not run it against user artifacts casually.

## Deployment Architecture

Use a checkout or GitHub release wheel, not PyPI. See
[installation](../../README.md#install-and-run). The scientific workflow uses
Python 3.13 explicitly without changing the repository pin. Main documentation
can describe unreleased changes while package metadata still says 1.3.0;
pin the source revision when reproducing them.

## Testing Architecture

[Test responsibilities](../test-architecture-summary.md) map behavior to
current modules. [Test commands](../testing-guide.md) explain package-wide
coverage, focused diagnostics and receipts. Do not infer readiness or test
inventory from saved category counts or an unexecuted command.

## Performance Considerations

Exact integer/rational cost depends on operand bit length. Multi-shell geometry
and general nearest-point search depend on input shape; rendering depends on
canvas size and primitives. No package GPU/distributed acceleration, general
read-thread-safety guarantee or automatic geometry cache is promised. See
[measurement guidance](../performance-optimization-guide.md).
