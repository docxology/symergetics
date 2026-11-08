# First calculations: rational values, quadray metrics and rendered solids

[Documentation map](../index.md) · [Cookbook](../api-cookbook-recipes.md) · [API contracts](../api/index.md)

## Getting Started

### Installation

Follow [checkout or release-wheel installation](../../README.md#install-and-run).
The package is not on PyPI. Scientific commands explicitly select Python 3.13;
the repository pin remains unchanged. From a development checkout:

```bash
uv sync --python 3.13 --extra scientific
uv run --python 3.13 --extra scientific python -c "from symergetics import SymergeticsNumber as N; print((N(1, 3) + N(1, 6)).value)"
```

The result represents exactly 1/2. Importing `Fraction` through this API does
not make a floating input an exact intended decimal or make pi rational.

## Core Concepts

### 1. Exact Rational Arithmetic

```python
from fractions import Fraction
from symergetics import SymergeticsNumber as N

assert (N(1, 3) + N(1, 6)).value == Fraction(1, 2)
assert N("0.1").value == Fraction(1, 10)
assert N.sqrt(Fraction(9, 16)).value == Fraction(3, 4)
```

`value` exposes the underlying rational. Irrational roots/powers are rational
approximations; see [the numeric contract](../api/index.md#symergeticsnumber).

### 2. Quadray Coordinate System

```python
from symergetics import QuadrayCoordinate as Q

q = Q(3, 2, 2, 1)
assert q.as_tuple() == (2, 1, 1, 0)
assert q.to_xyz() == (0.0, 2.0, 2.0)
assert q.squared_norm_exact() == 8
assert Q.from_xyz(*q.to_xyz()).as_tuple() == q.as_tuple()
```

Four integer labels represent three spatial dimensions modulo common offsets.
The constructor snaps noninteger labels; `from_xyz` solves a different problem,
nearest spatial lattice projection. The example round-trips an actual lattice
point, not arbitrary XYZ data. Euclidean distance can be irrational even when
its squared value is exact.

### 3. Volume Calculations

```python
from symergetics import Tetrahedron, Cube

assert Tetrahedron().volume() == 1
cube = Cube()
assert cube.volume() == 3
assert cube.analyze_geometry()["volume_xyz"] == 8
```

Native IVM volume differs from scale-one XYZ volume by 8/3. The default cube
has edge two; the default regular tetrahedron has edge `2*sqrt(2)`. A vertex
mean is not a uniform-solid mass center. Review
[solid measurement contracts](../api/index.md#indexed-surfaces-and-live-solid-analysis)
before constructing or deforming custom surfaces.

## Intermediate Tutorials

### Working with Scheherazade Numbers

```python
from symergetics import primorial, scheherazade_power, is_palindromic
from symergetics.computation.primorials import scheherazade_pascal_coefficients

assert primorial(13).value == 30030
assert scheherazade_power(2).value == 1002001
assert is_palindromic(scheherazade_power(2))
assert scheherazade_pascal_coefficients(6) == [1, 6, 15, 20, 15, 6, 1]
```

The primorial argument is a prime **upper bound**, not a count. Three-digit
Pascal blocks are recognized only without carries. These are finite arithmetic
observations, not evidence of a universal natural pattern.

### Render a solid without changing existing outputs

Use the complete [isolated rendering recipe](../api-cookbook-recipes.md#write-a-solid-graphical-abstract-to-an-isolated-directory).
`set_config` takes a dictionary. Returned `files` identifies outputs, while
`metadata` describes the live geometry and exact display origin. Do not treat
the return value as a figure. For modular main-only scientific posters use
[section selection](../api/index.md#scientific-poster-sections-unreleased-main).

### Orchestration with run.py

```bash
MPLBACKEND=Agg uv run --python 3.13 --extra scientific python run.py --examples-only
```

This intentionally regenerates repository `output/` and writes logs/reports;
run it only when those side effects are wanted. It skips setup and tests,
not the examples themselves. For individual demo entry points see
[the examples map](../examples/index.md).

## Best Practices

Keep rational inputs until a numerical consumer needs floats. Subtract exact
anchors before embedding very large translated shapes. Never mutate a
coordinate while it is a hashed key. Keep plotting configuration/output
directories isolated and restore configuration after application-owned work.
No mnemonic formatting step reduces the arithmetic size of an integer.

## Next Steps

- [Cookbook](../api-cookbook-recipes.md): short source-bound recipes.
- [Mathematical foundations](../mathematical-foundations-synergetics.md): metric and volume derivations.
- [Advanced boundaries](../advanced-topics-deep-dive.md): skew embeddings, shells and approximation.
- [Research](../research/index.md): conceptual applications, not installed API promises.
- [Troubleshooting](../troubleshooting-faq.md): common numerical and environment surprises.
