# Synergetics

[![DOI](https://zenodo.org/badge/1053524519.svg)](https://doi.org/10.5281/zenodo.17114383)

**Symbolic Operations on All-Integer Accounting**: Exact rational arithmetic implementing Buckminster Fuller's Synergetics through geometric ratios from high-frequency shapes in Quadray/4D coordinates. Research-grade precision for pattern discovery and mathematical analysis.

**Paper DOI**: [10.5281/zenodo.17114390](https://doi.org/10.5281/zenodo.17114390)

## Core Approach

This package implements **symbolic operations on all-integer accounting based upon ratios (geometrically based upon high frequency shapes) of the Synergetics geometry and Quadray/4D coordinates**:

- **Symbolic Operations**: Exact mathematical manipulations without floating-point errors
- **All-Integer Accounting**: Rational relationships expressed as precise integer ratios
- **Geometric Ratios**: Proportions derived from high-frequency polyhedral shapes
- **Synergetics Geometry**: Fuller's comprehensive geometric framework
- **Quadray/4D Coordinates**: Tetrahedral coordinate system for spatial relationships

## Highlights

- Exact rational arithmetic via `SymergeticsNumber` (no floating-point drift)
- Quadray (Fuller.4D) coordinate system and IVM lattice utilities
- Polyhedral volumes (tetrahedron, octahedron, cube, cuboctahedron) in exact IVM ratios
- Scheherazade numbers (1001^n), palindromes, SSRCD analysis, primorials
- Multiple visualization backends (matplotlib, plotly, ASCII) with organized output
- Comprehensive test suite with 977 test functions and 90%+ coverage

## Install & Test

```bash
python -m pip install -e .[test]
pytest -q
```

## Quick Start

```python
from symergetics import SymergeticsNumber, QuadrayCoordinate, Tetrahedron
from symergetics.computation.primorials import primorial, scheherazade_power

# Exact arithmetic
a = SymergeticsNumber(3, 4)
b = SymergeticsNumber(1, 2)
print(a + b)  # 5/4 (1.25000000)

# Quadray to XYZ (simplified mapping)
q = QuadrayCoordinate(2, 1, 1, 0)
print(q.to_xyz())

# Polyhedral volume
print(Tetrahedron().volume())  # 1 (IVM units)

# Scheherazade and primorial
print(scheherazade_power(6))
print(primorial(13))  # 30030
```

## Number Formatting (Default: Ungrouped)

- Default representations are raw/ungrouped for clarity in symbolic workflows.
- Utilities:
  - `format_large_number(n, grouping=3|4|...)` – group digits (default 3 = right-based; others = left-based)
  - `ungroup_number("1,234,567") -> 1234567` – strip separators back to int
- Notes: use grouping when you need SSRCD/Scheherazade pattern alignment (e.g., grouping=4), otherwise prefer ungrouped for clean symbolic processing.

## Visualizations

```python
from symergetics.visualization import set_config, plot_polyhedron
set_config({'backend': 'matplotlib', 'output_dir': 'output'})
result = plot_polyhedron('tetrahedron')
print(result['files'])  # saved assets in output/
```

**Backends**: `matplotlib`, `plotly`, `ascii`  
**Output**: Organized by type (geometric, mathematical, numbers) under `output/`  
**Features**: 3D polyhedra, coordinate systems, pattern analysis, batch processing

## Documentation

- Start here: [`docs/`](docs/)
  - [API Reference](docs/api/) – core types, computation, utils
  - [Examples](docs/examples/) – recipes and advanced usage
  - [Tutorials](docs/tutorials/) – guided, step-by-step
  - [Architecture](docs/architecture/) – internals and design
  - [Research](docs/research/) – methods and validation

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Citation

If you use this software in your research, please cite:

```bibtex
@software{symergetics2024,
  title={Symergetics: Symbolic Operations on All-Integer Accounting},
  author={Daniel Ari Friedman (docxology)},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17114383},
  url={https://doi.org/10.5281/zenodo.17114383}
}
```
