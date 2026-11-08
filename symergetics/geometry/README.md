# Indexed polyhedra: native volumes, centroids and lattice transforms

`polyhedra.py` defines `SymergeticsPolyhedron`, concrete solids and constructors;
`transformations.py` provides coordinate/polyhedron transforms. See the
[API reference](../../docs/api/index.md) and [suite units](../../docs/synergetics-suite.md).

`integer_tetra_volume(p0, p1, p2, p3)` and `ace_tetra_volume_5x5(...)`
return `Fraction` values: native volume is `abs(det)/4` in the `(a-d,b-d,c-d)`
projection. A primitive corner tetrahedron has volume 1/4; the regular unit
reference tetrahedron has volume 1 and default XYZ edge `2*sqrt(2)`.
Default scale-one Cartesian volume is native volume times `8/3`.

`volume()` recomputes from current vertices and declared indexed face fans,
not a convex hull. Closed orientable, non-self-intersecting and mutually
nonintersecting shells are preconditions. Disjoint solids add; nested shells
alternate solid/cavity by even-odd enclosure independently of input winding.
This is not a general intersection detector. Surface area is floating-point
and uses the same declared fan triangles in embedding-squared units.

- `centroid()` is an intentional floor-snapped lattice approximation of the
  vertex mean, returning a `QuadrayCoordinate` (origin for empty vertices).
- `centroid_exact()` returns a canonical rational four-tuple for the exact
  vertex mean; empty input raises `ValueError`.
- `volume_centroid_exact()` returns the uniform-solid mass centroid under the
  same enclosure contract; zero volume raises `ValueError`.

Vertex mean and mass centroid generally differ. Rotations, projections and
noninteger scaling can cross floating-point and integer-snapping boundaries;
do not promise arbitrary transform round trips on the integer lattice.
