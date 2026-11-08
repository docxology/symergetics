# Exact arithmetic and three-dimensional quadray geometry

[Documentation map](index.md) · [First calculation](tutorials/index.md) · [API reference](api/index.md)

## What the implementation provides

`symergetics` combines rational scalar arithmetic, integer quadray coordinates,
indexed polyhedra, finite number-pattern analysis and file-producing plots.
Fuller's terminology motivates these tools; it is not evidence that the package
implements a general theory of natural, biological or cognitive systems.
Conceptual applications belong in the [research guides](research/index.md).

## Rational values are not exact irrational constants

```python
from fractions import Fraction
from symergetics import SymergeticsNumber as N

result = N(1, 3) + N(1, 6)
assert result.value == Fraction(1, 2)
assert N("0.1").value == Fraction(1, 10)
```

Addition, subtraction, multiplication and division preserve represented rational
values. A float input represents its stored binary value, not the decimal the
caller may have intended. Integer powers and perfect rational roots stay exact;
irrational roots are bounded rational approximations. `N(22, 7)` is exactly
22/7, **not** exactly pi. See [numeric contracts](api/index.md#symergeticsnumber).

## Four labels describe three spatial dimensions

A common shift of `(a,b,c,d)` does not change its physical position. The default
constructor rounds components to integers and subtracts their minimum; it does
not constrain their sum. Four labels are not four independent Euclidean axes.

```python
from symergetics import QuadrayCoordinate as Q

q = Q(1, 0, 0, 0)
alias = Q(8, 7, 7, 7, normalize=False)
assert q.to_xyz() == alias.to_xyz() == (1.0, 1.0, 1.0)
assert q.squared_norm_exact() == 3
```

Tuple equality compares stored components, so unnormalized aliases can compare
unequal despite occupying the same point. Use canonical tuples for identity;
use exact displacement metrics for physical comparison. The nearest-XYZ inverse
projects to a lattice, not to arbitrary continuous coordinates. See
[coordinates and inverse projection](api/index.md#quadraycoordinate).

## IVM centers and all integer quadrays are different sets

The full integer-quadray image is BCC under the default embedding. The IVM/FCC
subset uses twelve neighbor moves. `enumerate_ivm_sites(1)` contains the origin
and its twelve contacts, not all integer quadrays in a four-index box. See
[the geometric neighborhood guide](geometric-patterns-spatial-relationships.md).

## Geometry measurements name their units and their surface

`volume()` returns exact rational IVM volume; it is not necessarily an integer.
At default scale one, XYZ volume equals `8/3` times IVM volume. Surface area and
Euclidean distances use the XYZ embedding. The exact mean of declared vertices
and the uniform-solid mass centroid are generally different; neither should be
silently replaced by the floor-snapped `centroid()` lattice approximation.

Polyhedral faces are indexed triangle fans. Solid measures require closed,
orientable shells without self- or mutual intersections. This is an input
precondition, not a promised intersection detector. See
[indexed surface and centroid contracts](api/index.md#indexed-surfaces-and-live-solid-analysis).

## Choose the next guide by task

- Derive a metric or volume: [mathematical foundations](mathematical-foundations-synergetics.md).
- Execute a short recipe: [cookbook](api-cookbook-recipes.md).
- Understand floating boundaries: [advanced implementation topics](advanced-topics-deep-dive.md).
- Render geometry or a modular scientific poster: [visualization API](api/index.md#visualization-module).
- Compare companion projects: [suite conventions](synergetics-suite.md).
