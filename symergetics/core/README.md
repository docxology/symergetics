# Core arithmetic, constants and rank-three quadray coordinates

`numbers.py` implements `SymergeticsNumber` over `fractions.Fraction`;
`constants.py` supplies named values; `coordinates.py` implements
`QuadrayCoordinate`, embedding validation and lattice helpers.
See the [canonical API](../../docs/api/index.md) and
[suite units and gauges](../../docs/synergetics-suite.md).

Rational arithmetic, integer powers and perfect rational roots are exact.
Genuinely irrational roots/powers return bounded-denominator approximations.
Float inputs retain their binary rational value; use strings or fractions when
an exact decimal/rational value is intended. `to_float()` is a float boundary.

Quadrays are integer tuples modulo adding the same value to all four labels.
The constructor rounds nonintegers with exact halves away from zero, then
normally subtracts the minimum. Raw tuple equality is not physical gauge equality.
Default scale-one XYZ is `(a-b-c+d, a+b-c-d, a-b+c-d)`.
`to_xyz(embedding=None)` returns floats after exact native combinations.
`from_xyz(x, y, z, embedding=None)` finds the nearest full integer-quadray
image (BCC for the default embedding), not the FCC/IVM subset; exact ties select
the lexicographically smallest normalized tuple. Custom embeddings must be finite,
rank-three 3×4 matrices annihilating the all-ones gauge. Skew embeddings use a
bounded closest-vector search, not component rounding. Exact dot/distance-squared
methods and floating norms have distinct contracts.
