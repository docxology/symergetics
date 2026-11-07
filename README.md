# Symergetics

Python implementation of Buckminster Fuller's Synergetics: exact rational
arithmetic, quadray/IVM coordinates, polyhedra, number-pattern experiments
(palindromes, primorials, scheherazade), and multi-backend visualization.
Paper pipeline under `paper/`; demo outputs under `output/`.

- Agents: start at [AGENTS.md](AGENTS.md) (orientation, verification commands).
- Current state: [STATUS.md](STATUS.md) · Backlog: [TODO.md](TODO.md)
- Documentation hub: [docs/index.md](docs/index.md)
- Quick check: see [Install and run](#install-and-run).

## Install and run

```bash
uv sync --python 3.13
uv run --python 3.13 python -c "from symergetics import QuadrayCoordinate; print(QuadrayCoordinate(1, 0, 0, 0).to_xyz())"
```

NumPy is a required dependency of the core coordinate API. Scientific and test
tools remain available through their extras:

```bash
uv run --python 3.13 --extra scientific --extra test pytest tests/test_core_coordinates.py -q --no-cov
```

## Synergetics suite

[QuadMath](https://github.com/docxology/QuadMath) provides the mathematical
specification, lattice algorithms and formalization;
[QuadCraft](https://github.com/docxology/QuadCraft) applies the geometry in
native and browser games. See the [suite guide](docs/synergetics-suite.md) for
coordinate/volume conventions and exact-data exchange boundaries.

With sibling checkouts, Node.js and a C++17 compiler:

```bash
uv run --python 3.13 --extra scientific --extra test python examples/suite_interoperability.py
```

This exercises the actual implementations across all three projects; no
production package depends on a sibling checkout.

For the complete reproducible gate, run from this repository:

```bash
python3 verify_suite.py --report /tmp/synergetics-suite-receipt.json
```

This requires `uv`, Node.js, CMake and a C++17 compiler. It uses frozen locks and
temporary Python 3.13 environments/native builds, without replacing project
environments. It stops on the first failed check and records unexecuted checks.
See [release notes](CHANGELOG.md), the [suite workflow](.github/workflows/suite.yml)
and [package verification](.github/workflows/verify.yml).
