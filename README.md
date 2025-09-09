# Synergetics

Exact, symbolic Synergetics: all-integer accounting, Quadray/IVM geometry, Scheherazade and primorial mathematics, and modular visualizations. Built for research-grade precision with a unified, well-tested Python package.

## Highlights

- Exact rational arithmetic via `SymergeticsNumber` (no floating-point drift)
- Quadray (Fuller.4D) coordinate system and IVM lattice utilities
- Polyhedral volumes (tetrahedron, octahedron, cube, cuboctahedron) in exact IVM ratios
- Scheherazade numbers (1001^n), palindromes, SSRCD analysis, primorials
- Configurable visualizations (matplotlib/ASCII) saving to `output/`
- Rigorously tested with a comprehensive test suite

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

Backends: `matplotlib`, `ascii`. All artifacts are saved under `output/`.

## Documentation

- Start here: [`docs/`](docs/)
  - [API Reference](docs/api/) – core types, computation, utils
  - [Examples](docs/examples/) – recipes and advanced usage
  - [Tutorials](docs/tutorials/) – guided, step-by-step
  - [Architecture](docs/architecture/) – internals and design
  - [Research](docs/research/) – methods and validation

## License

MIT — see [LICENSE](LICENSE).
