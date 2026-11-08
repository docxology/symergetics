# Application Scenarios: Geometry Tools and Evidence Requirements

## Introduction

These are **proposed uses**, not reports of deployments or measured outcomes.
Earlier versions described a Montreal Biosphere restoration, bridge design,
bee-colony warning system, climate optimization, protein prediction, neural
training and portfolio management without identifiable studies or reproducible
results. Those narratives and their percentage improvements are withdrawn;
randomly generated inputs and incomplete example classes were not evidence.

Use the [research boundary guide](research/index.md) to distinguish implemented
primitives from a domain model. Runnable interfaces belong in the
[API reference](api/index.md), not in hypothetical application classes here.

## Architecture and Structural Engineering

A dome or tensegrity study could use quadray coordinates to encode a chosen
lattice geometry and exact determinants to measure tetrahedra. The current
[polyhedra implementation](../symergetics/geometry/polyhedra.py) measures declared
surfaces; it does not solve structural equilibrium, buckling, wind loading,
fatigue, connection design or building-code compliance.

A nonzero tetrahedral volume only establishes noncoplanarity. Enumerating every
four-vertex combination does **not** produce a nonoverlapping decomposition:
summing those volumes generally double-counts space. A defensible engineering
study needs a specified mesh, material law, boundary conditions, independently
checked solver and measured comparison design before claiming savings or safety.
No project-specific engineering outcome is established here.

## Environmental Science and Climate Modeling

Rational arithmetic can preserve ratios of recorded counts, and geometry can
measure an explicitly constructed spatial model. Neither defines an ecosystem
health indicator or a carbon-sequestration model. A proposed colony-collapse
study would need identified observations, sampling intervals, stressor labels,
held-out temporal/site validation, uncertainty and false-alarm rates. A
reforestation study additionally needs biomass/allometry, carbon accounting,
land-use constraints and a declared counterfactual. No early-warning lead time,
survival improvement or climate benefit has been measured in these examples.

## Biological Research and Medical Applications

Exact distances squared or volumes can be descriptors of a discretized molecular
model. `QuadrayCoordinate.from_xyz` snaps to an integer lattice, so measured atom
positions are not retained exactly by that conversion. Preserve input units,
resolution and quantization error before comparing structures.

A geometric descriptor does not predict folding energy, protein stability,
drug response or clinical benefit. Those claims require domain datasets,
physical/statistical models, appropriate baselines and external validation.
The package is not documented here as a medical decision-support system.

## Machine Learning Experiments

A geometric regularizer is a proposed experiment, not a supplied optimizer.
Specify the loss, invariances, data split, parameter count, optimizer and compute
budget, then compare against matched baselines over repeated runs. Golden-ratio
layer widths do not by themselves establish better learning or interpretability.
There is no training-speed or generalization benchmark on this page.

## Financial Modeling and Risk Analysis

Exact arithmetic can account for rational cash amounts or allocation weights;
it cannot make uncertain returns exact. A portfolio experiment would need an
explicit objective, dated price data, transaction costs, walk-forward evaluation,
selection-bias controls and risk reporting. Decimal digit patterns and
Scheherazade identities are not evidence of market predictability. No Sharpe
ratio improvement or investment recommendation is supported here.

## References and Case Study Sources

The sources below establish software behavior, not real-world efficacy:

- [Arithmetic and conversion limits](research/index.md#exact-arithmetic-and-approximation-boundaries)
- [Current implementation contracts](api/index.md)
- [Geometric source](../symergetics/geometry/polyhedra.py)
- [Evidence and study-design checklist](research/index.md#research-methodology)
- [Interdisciplinary scope](interdisciplinary-applications-synergetics.md)
- [Documentation map](index.md)
