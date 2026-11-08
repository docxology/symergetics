# Research Applications: Implemented Mathematics and Evidence Boundaries

This guide describes the development checkout. It is not a report of empirical
applications, a physics theory or a set of planned APIs disguised as examples.
Install from a checkout or release wheel using the [root guide](../../README.md);
Symergetics is not published on PyPI. Use Python 3.13 for the documented scientific
workflow without changing the repository's interpreter pin.

## Exact Arithmetic and Approximation Boundaries

| Component | Source-bound behavior | Limit |
| --- | --- | --- |
| `SymergeticsNumber` | Wraps `fractions.Fraction`; rational arithmetic and integer powers preserve rational values | It is not a symbolic algebraic/transcendental-number system |
| Float conversion | Constructor represents a float's binary rational value; `from_float` limits its denominator | Neither recovers an unknown intended measurement exactly |
| Roots and constants | Perfect rational roots stay exact; irrational roots/powers are approximated; `PI`, `E`, `PHI` are rational approximations | Exact storage of an approximation is not exact π, e or φ |
| `QuadrayCoordinate` | Four integer slots modulo a common additive offset; normalization subtracts the minimum | Rank three, not four independent Euclidean axes; fractional inputs are rounded |
| Default embedding | XYZ = (a−b−c+d, a+b−c−d, a−b+c−d) at scale 1 | Float-returning views/rendering have representability limits |
| Inverse embedding | `from_xyz` selects a nearest integer quadray lattice site | Not an exact inverse on arbitrary continuous inputs; not the sibling QuadMath snapping policy |
| Integer image versus IVM | Default integer image is BCC; the defined 12-neighbor FCC/IVM sublattice is distinct | Do not call every integer tuple an FCC site |
| Tetrahedral volume | `integer_tetra_volume` returns `Fraction(abs(det), 4)` in quotient coordinates (a−d,b−d,c−d) | Despite the function name, volume need not be an integer; primitive corner tetrahedron = 1/4 |
| Solid measurements | Declared indexed triangle fans define the surface; live vertices determine volume and moments | Closed orientable nonintersecting shells are a precondition, not a promise of intersection detection |

Sources: [numbers.py](../../symergetics/core/numbers.py),
[constants.py](../../symergetics/core/constants.py),
[coordinates.py](../../symergetics/core/coordinates.py),
[polyhedra.py](../../symergetics/geometry/polyhedra.py).
The [API reference](../api/index.md) is the canonical callable contract.

### Volume units and centers

The default regular tetrahedron has native volume 1 and edge 2√2 in scale-1
Cartesian coordinates. For that embedding, Cartesian cubic volume is
(8/3) × native volume; a uniform embedding scale s adds a factor |s|³.
Default constructors give tetrahedron/cube/octahedron/cuboctahedron native
volumes 1/3/4/20, respectively, at their declared relative sizes. This is not an
equal-edge volume table for arbitrary solids, and the cuboctahedron is not a
Platonic solid. Constants involving φ are not evidence of implemented exact
icosahedron or dodecahedron geometry.

`centroid_exact` is the rational vertex mean; `volume_centroid_exact` is the
uniform-solid mass centroid under the declared shell semantics. They generally
differ. Disjoint shells add mass; nested shells alternate solid/cavity by
even–odd enclosure, independently of input winding. Surface area and Euclidean
lengths are generally irrational and exposed through approximate views.

## Active Inference and Cognitive Science

**Implemented component:** rational arithmetic, not an active-inference engine.
For rational residual r and precision λ, λr² is a quadratic cost. It is not a
complete variational free energy. For a model p(o,s) and distribution q(s), that
quantity is an expectation of log q(s) − log p(o,s); logarithms are not generally
rational. Fisher information additionally requires a specified likelihood,
derivatives and an expectation, not just matrix inversion.

For a finite Bayesian table, posterior weights can remain rational when the
inputs are rational and the evidence denominator is nonzero. The denominator
must sum likelihood × prior over **all** hypotheses; a constant denominator of
one is not a generic Bayes update. No belief-propagation or natural-gradient API
is claimed here. Rational bookkeeping does not establish cognitive validity.

## Entomological Pattern Analysis

A hexagonal planar tiling and a rhombic-dodecahedral spatial partition are
different models. Neither alone determines biological material savings or
strength. Measured wingbeat frequencies have uncertainty; rationalizing them
cannot make them exact. A low-denominator frequency ratio requires an uncertainty
model and null comparison before being called a harmonic relationship. Decimal
palindromes in concatenated measurements are not a validated biological statistic.

## Crystallographic Applications

Ideal lattice geometry is a plausible mathematical input to crystallography.
State the cell convention, occupancy, embedding and length scale before computing
coordination or volume. A model FCC coordination count does not establish a
material's structure, stability or properties. No space-group refinement,
quasicrystal generator or material optimizer is supplied by this guide.

## Cognitive Security Applications

The [digit-pattern utilities](../../symergetics/computation/palindromes.py) inspect
numbers and representations; they are not validated anomaly detectors or threat
classifiers. Repetition alone does not show maliciousness, authenticity or
information content. Evaluate a separately defined detector on labeled data with
false-positive/false-negative rates and leakage controls before making such claims.
Entropy H = −Σp log₂p generally requires nonrational values. Wrapping floating
logarithms in fractions does not turn their result into exact entropy.

## Advanced Research Topics

### Quantum Geometry Applications

No quantum simulator, derivation of physical constants or empirical quantum
result is established here. 1/137 is an approximation to the fine-structure
constant, not its exact value. Natural-unit formulas must not silently combine
normalized constants with SI masses. A proposed physical calculation needs a
consistent unit system, measured inputs, uncertainty and an independently
specified physical model.

### Fractal Dimension Analysis

A dimension estimator needs a defined statistic across scales, a sampling model
and uncertainty. Dividing a point-pair count by a logarithmic distance range is
not a justified sandbox dimension estimator. Returning a default dimension for
insufficient data would fabricate a result. Integer-lattice snapping can also
collapse distances and alter inferred scaling. No dimension API is claimed here.

## Research Methodology

Classify each claim before interpreting it:

| Claim type | Evidence required | What does not establish it |
| --- | --- | --- |
| Mathematical identity | Stated assumptions and derivation/proof | A decorative plot |
| Implemented behavior | Source contract and exercised consumer behavior for a revision | A proposed function name |
| Numerical performance | Dataset/input sizes, environment, comparator, timing/memory protocol and uncertainty | An unmeasured percentage or test count |
| Predictive association | Identified data, held-out evaluation, uncertainty and null comparisons | Synthetic data presented as observations |
| Causal or practical benefit | Appropriate comparison/intervention design, confound controls and replication limits | A plausible analogy or domain vocabulary |

Record source revision, input provenance, units, gauge, lattice selection,
approximation policy, dependencies and output provenance. Separate exact
arithmetic correctness from model adequacy and measurement error. Consider the
alternative explanation most likely to overturn a result and identify the next
discriminating observation. A suite receipt supports its executed revision only;
see [test evidence](../../paper/TEST_EVIDENCE.md). This guide reports source review,
not a newly executed study or benchmark.

## Related Research Modules

- [Application scenarios and withdrawn success stories](../case-studies-real-world-applications.md)
- [Interdisciplinary component map](../interdisciplinary-applications-synergetics.md)
- [Future questions](../future-implications-research-directions.md)
- [Learning activities and study design](../learning-theory-synergetics.md)
- [Philosophical interpretation](../synergetics-philosophy-education.md)
- [Editable paper sources](../../paper/markdown/README.md)
- [Documentation map](../index.md)
