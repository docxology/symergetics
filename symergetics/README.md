# Symergetics Python package: modules and numerical boundaries

This directory is the importable `symergetics` package, not the command-line
orchestration layer. Start with the [API reference](../docs/api/index.md) and
[suite contracts](../docs/synergetics-suite.md).

- [core](core/README.md): rational numbers, constants and rank-three quadrays.
- [geometry](geometry/README.md): indexed polyhedra and lattice transforms.
- [computation](computation/README.md): integer experiments and pattern analysis.
- [utils](utils/README.md): conversion, mnemonic formatting and reports.
- [visualization](visualization/README.md): renderer configuration and backends.

Exact rational storage does not make every operation exact: irrational roots,
float conversion, lattice projection and rendering have explicit boundaries.
NumPy and psutil are base dependencies; plotting libraries are optional in the
`scientific` extra. Checkout installation is documented in the [root guide](../README.md).

The 1.4.0 interface adds [scientific poster sections](../docs/api/index.md#scientific-poster-sections)
and the geometry `with_vertices` ownership protocol. Plane transforms preserve
represented rational inputs before nearest-site snapping; continuous inverse
matrices remain a separate floating output boundary.
