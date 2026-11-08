## Abstract

Symergetics is an open-source Python package for selected mathematical topics
inspired by Buckminster Fuller's Synergetics: rational arithmetic, integer
quadray coordinates, tetrahedral/polyhedral volume and integer digit-pattern
analysis. `SymergeticsNumber` wraps Python's `fractions.Fraction`; rational
operations preserve exact values, while irrational roots, constants and
float-derived inputs have explicit approximation boundaries. Quadrays represent
three-dimensional space with four components modulo a common offset, not four
independent spatial dimensions.

The geometry implementation computes native tetrahedral volumes as exact
fractions and measures declared indexed polyhedral surfaces. Default
constructors yield tetrahedron/cube/octahedron/cuboctahedron native volumes
1/3/4/20 at their specified relative sizes. The computational modules include
prime products, powers of 1001 and representation-dependent palindrome analysis.
Visualization converts computed quantities to finite-precision display data;
a figure is not independent mathematical validation.

This manuscript describes software and mathematical examples, not empirical
success in cognition, biology, materials, climate, finance or quantum physics.
Proposed applications are separated from implemented interfaces. Test inventories,
collected cases and historical execution receipts are distinguished in
[Test evidence and count semantics](../TEST_EVIDENCE.md); none proves universal
correctness or a speed advantage. The development source is available in the
[repository](https://github.com/docxology/symergetics), under its Apache 2.0
[license](../../LICENSE). These editable sources are not a revision of an
immutable published deposit.
