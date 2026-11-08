# Core maintenance: preserve exactness and coordinate gauge

Read [core contracts](README.md) and the [API reference](../../docs/api/index.md).
`numbers.py`, `constants.py` and `coordinates.py` are shared foundations: trace
callers before changing exports or numeric semantics.

Keep exact integer/rational inputs out of intermediate floats. Do not conflate
constructor snapping with nearest-lattice XYZ inversion, or the full integer
quadray image with the FCC/IVM sublattice. Respect independent common-offset
gauges and custom embedding validation. Irrational approximations must remain
explicit; rational storage alone is not a proof of exact irrational results.
Tests are top-level `../../tests/test_core_*.py`, not `tests/core/`.
Follow [test guidance](../../tests/AGENTS.md) and the active assignment's gates.
