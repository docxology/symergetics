# Quadray quotient, exact metrics and tetrahedral volume

[Concept guide](core-concept-guide.md) · [API contracts](api/index.md) · [Suite units](synergetics-suite.md)

## The rational arithmetic boundary

`SymergeticsNumber.value` is a `fractions.Fraction`. Rational field operations
are exact for the represented inputs. Integer and perfect rational-root results
stay rational; irrational roots and powers are approximated. A bounded-denominator
result does not certify an error bound for the original real quantity. Decimal
strings, rational pairs and binary floats express different input commitments.
For implementation details see [numbers.py](../symergetics/core/numbers.py).

## The rank-three quotient

For integer labels $q=(a,b,c,d)$, identify $q$ with $q+k(1,1,1,1)$.
Minimum-zero normalization chooses a canonical representative. The scale-one
Urner map is the **3 by 4** matrix

$$U=\begin{pmatrix}1&-1&-1&1\\1&1&-1&-1\\1&-1&1&-1\end{pmatrix},$$

so $(X,Y,Z)=(a-b-c+d,a+b-c-d,a-b+c-d)$. Its kernel is the
common-offset direction. The four columns have norm $\sqrt3$, not one,
and distinct columns have dot product $-1$.

The induced bilinear form is

$$\langle q,r\rangle=4\sum_i q_i r_i-(\sum_i q_i)(\sum_i r_i).$$

Thus subtracting any common offset from either argument leaves the physical
inner product unchanged. `dot_exact`, `squared_norm_exact` and
`squared_distance_exact` return `Fraction`; `dot`, `magnitude`, `distance_to`
and `to_xyz` return floats. Do not call a generally irrational distance rational
merely because its square is rational.

## Continuous inverse versus discrete nearest point

There is no unique continuous four-label inverse without a gauge choice.
`xyz_to_quadray_matrix()` supplies a continuous Moore–Penrose inverse.
`QuadrayCoordinate.from_xyz()` instead chooses the Euclidean-nearest integer
quadray and resolves exact ties by the lexicographically smallest normalized
tuple. The default integer image consists of XYZ triples whose coordinates
are all even or all odd: a BCC lattice. This nearest-point API does not restrict
its result to the IVM/FCC subset and is not QuadMath's half-up inverse.

A custom embedding must be finite, rank three and annihilate the all-ones
gauge. Stored finite float coefficients are treated as binary rationals in
exact calculations; that does not make an intended irrational coefficient exact.

## IVM tetrahedral units and XYZ cubic units

Project each vertex to $(a-d,b-d,c-d)$. For the three displacements from one
vertex, the native volume is $|\det M|/4$. The XYZ embedding multiplies this
projected determinant by four, and Euclidean tetrahedron volume divides by six:

$$V_{\mathrm{XYZ}}=\frac{8}{3}V_{\mathrm{IVM}}.$$

```python
from fractions import Fraction
from symergetics import QuadrayCoordinate as Q, Tetrahedron
from symergetics.geometry.polyhedra import integer_tetra_volume, ace_tetra_volume_5x5

corner = [Q(0, 0, 0, 0), Q(1, 0, 0, 0), Q(0, 1, 0, 0), Q(0, 0, 1, 0)]
assert integer_tetra_volume(*corner) == Fraction(1, 4)
assert ace_tetra_volume_5x5(*corner) == Fraction(1, 4)
assert Tetrahedron().volume() == 1
```

The default regular unit tetrahedron has edge $2\sqrt2$ and IVM volume one.
Under a uniform embedding scale $s$, lengths scale by $|s|$, areas by $s^2$
and XYZ volumes by $|s|^3$; native IVM volume does not change. Custom skew
embeddings need their own determinant conversion, not the scale-one factor.

## Solid geometry is defined by the declared surface

`triangulated_faces()` fans each cyclic face from its first index. Warped or
concave polygons mean exactly those triangles, not a best-fit plane or convex
hull. Closed orientable nonintersecting shells are required for mass measures.
Disconnected solids add; nested shells alternate solid/cavity by even-odd
enclosure, independently of input winding. `centroid_exact()` averages vertices;
`volume_centroid_exact()` integrates uniform enclosed mass. See the
[canonical solid API](api/index.md#indexed-surfaces-and-live-solid-analysis).

## Number patterns are finite, representation-dependent observations

`primorial(n)` multiplies primes **at most n**, not the first n primes.
`scheherazade_power(n)` computes $1001^n$. Decimal palindromes concern decimal
digits, not coordinate symmetry. Three-digit block readings of binomial
coefficients fail when carries cross block boundaries; use the real
`scheherazade_pascal_coefficients` result rather than assuming every power
contains an uncarried Pascal row. These arithmetic observations alone establish
no biological, cosmological or optimization law.
