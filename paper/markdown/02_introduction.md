## Introduction

### The Precision Problem in Geometric Computing

Finite binary floating-point cannot represent every rational number exactly.
For example, 3/4 + 1/6 has exact value 11/12, while a float calculation returns
a nearby representable value. Error propagation depends on the algorithm and
conditioning; it is not universally exponential or always consequential.
Exact rational arithmetic preserves rational algebra but can require growing
numerators and denominators. It does not remove measurement error or make
irrational quantities rational.

### Research Gap and Motivation

[Fuller and Applewhite's Synergetics](https://www.rwgrayprojects.com/synergetics/)
provides historical motivation for tetrahedral geometry and ratio-based
accounting. This does not establish that natural systems are universally
rational, nor that Cartesian geometry cannot support exact calculations.
Python's `fractions.Fraction` and symbolic systems already provide exact tools;
no first-of-its-kind or comparative superiority claim is made here.

Symergetics combines selected arithmetic, coordinate, volume and number-pattern
operations in one package. Related [QuadMath](https://github.com/docxology/QuadMath)
uses shared quadray ideas, but consumers must check each package's embedding,
rounding and API contracts rather than assuming interchangeable behavior.

### Main Questions Asked

- Which operations preserve rational values, and where do approximations enter?
- How does a four-component gauge representation encode rank-three geometry?
- Which volume ratios follow from the explicitly chosen vertices and units?
- Which digit patterns follow from a generating identity rather than a novel
  empirical discovery?
- What evidence would a proposed domain application need beyond software output?

### Research Objectives

Describe implemented interfaces and source boundaries, derive the relevant unit
and coordinate conventions, and distinguish software examples from research
hypotheses. Performance comparisons and domain efficacy require separate
experiments; they are not established by this manuscript.

### Key Contributions

The contribution is a package-level integration and exposition of selected
mathematical computations, not a new foundation for physics or a claim to have
invented exact arithmetic. The [API reference](../../docs/api/index.md) is the
canonical callable contract. The [research guide](../../docs/research/index.md)
records approximation, modeling and evidence limits.

### Paper Organization

[Mathematical foundations](03_mathematical_foundations.md) define arithmetic,
gauge and units. [Architecture](04_system_architecture.md) maps real modules;
[methods](05_computational_methods.md) describe algorithms and limitations;
[results](06_results.md) separates identities from measurements.
[Geometry](07_geometric_applications.md) and [number patterns](08_pattern_discovery.md)
explain examples. [Research applications](09_research_applications.md) are
proposals, followed by [open questions](10_ongoing_questions_inquiries.md) and
[conclusions](11_conclusion.md).
