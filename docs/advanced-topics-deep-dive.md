# Advanced implementation boundaries: exactness, topology and projection

[API contracts](api/index.md) · [Cookbook](api-cookbook-recipes.md) · [Performance](performance-optimization-guide.md)

## Large integer coordinates and floating display

Python integers and `Fraction` preserve native arithmetic beyond floating
range. XYZ conversion remains a floating boundary: enormous absolute positions
can overflow or lose small separations. Subtract a quadray anchor **before**
embedding, using `to_xyz_vertices(relative_to=anchor)`. Subtracting two already
rounded XYZ positions cannot recover lost information. Exact metric methods
operate before final float conversion; use them for invariant comparisons.
Source: [coordinates.py](../symergetics/core/coordinates.py) and
[polyhedra.py](../symergetics/geometry/polyhedra.py).

## Binary-rational embedding coefficients

Custom matrices must be finite, rank-three 3 by 4 maps with the all-ones gauge
in their kernel. Exact methods interpret finite stored coefficients as binary
rationals, not as ideal real numbers. Only the first three columns define the
effective d-zero map; the fourth is constrained by the kernel. See
[the canonical coordinate contract](api/index.md#quadraycoordinate).

## Nearest lattice projection is not component rounding

The regular tetrahedral embedding has a BCC fast path. General skew embeddings
use bounded closest-vector enumeration after lattice reduction, in three
dimensions. Exact-distance ties select canonical tuple order. The result is on
the full integer-quadray lattice, not necessarily FCC. A continuous
pseudoinverse has different semantics; applying it and independently rounding
labels is not a substitute for the nearest-point API.

## Exact roots versus rational approximations

```python
from fractions import Fraction
from symergetics import SymergeticsNumber as N

assert N.sqrt(10**400).value == 10**200
assert N.nth_root(Fraction(-8, 27), 3).value == Fraction(-2, 3)
assert (N(-8) ** Fraction(1, 3)).value == -2
```

Perfect rational roots stay exact regardless of the approximation denominator
cap. Irrational roots use finite-precision Decimal evaluation followed by a
bounded rational approximation, not symbolic algebra or certified interval
arithmetic. Tiny irrational results may approximate to zero. `1/3` is a float;
use `Fraction(1,3)` when an exact cube-root exponent matters.

## Nonconvex, warped and multi-shell solids

The surface contract is indexed triangle fans, not convex-hull reconstruction.
Nonplanar faces are measured and rendered as their declared fans. Closed
orientable surfaces without self- or mutual intersections are required.
Disconnected solids add; nested shells alternate solid, cavity and island by
even-odd enclosure. Input winding does not choose mass sign. Closure and
convexity diagnostics do not prove nonintersection. Zero-volume solids have no
defined mass centroid. See [solid analysis](api/index.md#indexed-surfaces-and-live-solid-analysis).

## Mutability and cache invalidation

`vertices` is public and mutable; volume and geometry analysis are recomputed.
A coordinate's fields are also mutable even though it implements `__hash__`.
Do not mutate coordinates stored as dictionary keys or set members. If an
application caches a result, use immutable tuple keys and include the embedding
and other parameters in its own cache policy. No global read-thread-safety
promise follows from rational arithmetic or NumPy usage.

## Modular scientific abstracts on unreleased main

The existing `create_mega_graphical_abstract` API now selects `overview`,
`geometry`, `arithmetic`, `patterns` or `all` using keyword-only `section`.
The focused default is not the old all-17-panel poster. The 1.3.0 release wheel
does **not** acquire this interface from main documentation. Use
[the visualization API](api/index.md#scientific-poster-sections-unreleased-main)
for selection, paths and metadata.

## Conceptual extensions are not installed APIs

Higher-dimensional algebras, quantum error correction, cosmological scaling,
VR, genetic-code analysis and consciousness models need independent algorithms,
dependencies and empirical validation. No working package recipes for those
systems are provided here. Their discussion belongs in
[research](research/index.md), not in executable implementation examples.
