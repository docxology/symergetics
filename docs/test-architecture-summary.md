# Test responsibilities and evidence boundaries

[Commands and coverage](testing-guide.md) · [Package architecture](architecture/index.md)

## Executive Summary

The test suite covers exact arithmetic, integer geometry, finite sequence
analysis, plotting and real orchestration/integration. This page maps
responsibilities, not test totals or a current pass receipt. Use pytest
collection for current cases; use an executed gate for results. Historical
receipts are explicitly dated in [STATUS.md](../STATUS.md).

## Test Architecture Overview

| Responsibility | Representative source of checks |
|---|---|
| Coordinate snapping, gauge, embedding and inverse | [test_core_coordinates.py](../tests/test_core_coordinates.py) |
| Rational number behavior | [test_core_numbers.py](../tests/test_core_numbers.py) |
| Primorials and finite decimal patterns | [test_computation_primorials.py](../tests/test_computation_primorials.py), [test_computation_palindromes.py](../tests/test_computation_palindromes.py) |
| Geometry and visual integration | [tests directory](../tests/), [test index](../tests/TEST_INDEX.md) |
| Scientific poster selection and computed output | [test_mega_graphical_abstract.py](../tests/test_mega_graphical_abstract.py) |
| Actual cross-project behavior | [suite_interoperability.py](../examples/suite_interoperability.py), [suite guide](synergetics-suite.md) |

These are navigation examples, not an exhaustive inventory. Parametrization,
optional libraries and newly added behavior can change collected and executed
case counts without changing this map.

## Test Quality Metrics

Coverage is a measured property of a specific executed command and environment,
not a consequence of the number of files or assertions. The configured package
floor is 85%; targeted runs can fail it despite passing all selected tests.
See [coverage semantics](testing-guide.md#coverage-analysis).

An assertion on default wording, source text, export presence or file existence
alone does not prove a scientific result. Prefer consumer-visible geometry,
resource lifetime, error transitions and independent exact calculations. Real
data and computations are required; do not mock away the behavior under review.

## Scientific boundaries worth preserving

- Exact rational operations versus approximated irrational roots.
- Common-offset quadray aliases versus tuple equality and canonical identity.
- Full BCC nearest projection versus FCC/IVM shell enumeration.
- Native IVM volume versus scale-one XYZ cubic volume.
- Indexed topology versus unique physical vertices and nonintersection assumptions.
- Vertex mean versus uniform-solid mass center versus snapped lattice centroid.
- Anchor-relative plotting, actual declared edges/fans and bounded labels with
  complete exact metadata.
- Unreleased-main poster selection versus immutable release-wheel signatures.

These distinctions are specified in the [API reference](api/index.md), not
redefined in each test description.

## Test Execution Examples

Use the [testing guide](testing-guide.md#running-tests) for complete Python 3.13
commands, headless rendering and the `--no-cov` diagnostic caveat. This document
contains no copied “all passing” outputs or machine-dependent timing receipts.

## Continuous Integration

Actual [verification workflows](../.github/workflows/verify.yml) and
[suite workflow](../.github/workflows/suite.yml) determine the exercised gates.
Report skipped/unexecuted checks and source revisions alongside executed ones;
a green subset is not a universal readiness or scientific-validity claim.
