# API cookbook: exact values, solid measurements and focused figures

[Install first](../README.md#install-and-run) · [Canonical API contracts](api/index.md)

Each Python block is an independent consumer example for a development checkout.
Run it with the scientific Python 3.13 environment described in the root guide.
Expected exact values below follow the source contracts; they are not benchmark
receipts. The new poster selector requires unreleased main, not the 1.3.0 wheel.

## Basic Operations

### Recipe 1: Getting Started with Exact Arithmetic

```python
from fractions import Fraction
from symergetics import SymergeticsNumber as N

assert (N(1, 3) + N(1, 6)).value == Fraction(1, 2)
assert N("0.125").value == Fraction(1, 8)
assert N.nth_root(Fraction(-8, 27), 3).value == Fraction(-2, 3)
```

Use strings for intended decimal inputs. A rational approximation to pi or an
irrational root is not an exact irrational value.

### Recipe 2: Converting Between Coordinate Systems

```python
from symergetics import QuadrayCoordinate as Q

q = Q(2, 1, 1, 0)
assert q.to_xyz() == (0.0, 2.0, 2.0)
assert Q.from_xyz(*q.to_xyz()).as_tuple() == q.as_tuple()
assert q.squared_distance_exact(Q(0, 0, 0, 0)) == 8
```

This on-lattice example round-trips. Arbitrary XYZ inputs project to the
nearest integer site, not necessarily FCC. See [inverse semantics](api/index.md#quadraycoordinate).

## Geometric Operations

### Recipe 3: Volume Calculations for Polyhedra

```python
from fractions import Fraction
from symergetics import QuadrayCoordinate as Q, Tetrahedron, Cube, Octahedron, Cuboctahedron
from symergetics.geometry.polyhedra import integer_tetra_volume

assert [m().volume() for m in (Tetrahedron, Cube, Octahedron, Cuboctahedron)] == [1, 3, 4, 20]
corner = [Q(0, 0, 0, 0), Q(1, 0, 0, 0), Q(0, 1, 0, 0), Q(0, 0, 1, 0)]
assert integer_tetra_volume(*corner) == Fraction(1, 4)
```

These are native IVM units. At scale one, multiply by `Fraction(8,3)` for XYZ
cubic volume. The default cube and tetrahedron do not have equal edge lengths.

### Compare a vertex mean with a uniform-solid mass center

```python
from fractions import Fraction
from symergetics import Cube, QuadrayCoordinate as Q

shape = Cube()
shape.vertices[0] = Q(2, 0, 0, 0)
analysis = shape.analyze_geometry()
assert analysis["volume_ivm"] == Fraction(9, 2)
assert shape.centroid_exact() == (Fraction(1, 8), 0, 0, 0)
assert shape.volume_centroid_exact() == (Fraction(1, 4), 0, 0, 0)
local_xyz = shape.to_xyz_vertices(relative_to=shape.vertices[0])
```

Do not pass these rational centroid tuples through `Q(...)`: it snaps to
integers. Solid measurement requires the [declared surface preconditions](api/index.md#indexed-surfaces-and-live-solid-analysis).

## Pattern Recognition

### Recipe 4: Analyzing Scheherazade Number Patterns

```python
from symergetics import primorial, scheherazade_power, is_palindromic
from symergetics.computation.primorials import scheherazade_pascal_coefficients

assert primorial(13).value == 30030
assert scheherazade_power(2).value == 1002001
assert is_palindromic(scheherazade_power(2))
assert scheherazade_pascal_coefficients(6) == [1, 6, 15, 20, 15, 6, 1]
```

`primorial` uses a prime upper bound. Pascal coefficients are recognized only
when three-digit blocks do not carry; an empty result must not be replaced with
an asserted discovery. Decimal symmetry is not a physical symmetry claim.

## Write a solid graphical abstract to an isolated directory

```python
from tempfile import TemporaryDirectory
from symergetics import Cube
from symergetics.visualization import get_config, set_config
from symergetics.visualization.geometry import plot_polyhedron_graphical_abstract

previous = get_config()
try:
    with TemporaryDirectory(prefix="symergetics-cookbook-") as directory:
        set_config({"output_dir": directory, "formats": ["png", "svg"], "dpi": 120})
        result = plot_polyhedron_graphical_abstract(Cube(), backend="matplotlib")
        print(result["files"])
        print(result["metadata"]["geometry_analysis"])
        # Copy wanted files elsewhere before TemporaryDirectory removes them.
finally:
    set_config(previous)
```

Plotting returns paths and metadata, not a Matplotlib figure. Metadata contains
`Fraction` and tuple values; define an explicit serialization policy before
writing JSON. The figure uses an exact anchor-relative XYZ frame. See
[output contracts](api/index.md#geometry-visualizations).

## Select a scientific poster topic on unreleased main

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from symergetics.visualization.advanced import create_mega_graphical_abstract

with TemporaryDirectory(prefix="symergetics-posters-") as directory:
    for section in ("overview", "geometry", "arithmetic", "patterns", "all"):
        result = create_mega_graphical_abstract(
            output_path=str(Path(directory) / f"poster_{section}.png"),
            backend="matplotlib", section=section,
        )
        print(section, result["metadata"]["panels"], result["files"])
```

Explicit paths prevent sections overwriting the same canonical filename.
`overview` is the six-panel default; `all` retains all seventeen computed topics.
See [section metadata](api/index.md#scientific-poster-sections-unreleased-main).
