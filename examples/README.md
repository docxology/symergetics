# Runnable examples: numerical demos, renderers and suite interoperability

Run from the repository root using the Python 3.13 scientific environment;
installation and dependency setup are in the [root guide](../README.md).

```bash
uv run --python 3.13 --extra scientific python examples/coordinate_systems_demo.py
MPLBACKEND=Agg uv run --python 3.13 --extra scientific python examples/comprehensive_visualizations_demo.py
```

These are execution recipes, not receipts. Demos may create figures, reports and
logs under `output/` or other relative destinations; inspect each script before
running and preserve existing user artifacts. They are not uniformly read-only.

| Script | Purpose |
| --- | --- |
| `symergetics_core_demo.py` | Core arithmetic and Synergetics relationships |
| `coordinate_systems_demo.py` | Quadrays, embeddings and transforms |
| `number_theory_demo.py` | Primorials and Scheherazade numbers |
| `mathematical_patterns_demo.py` | Pattern-analysis workflows |
| `geometric_mnemonics_demo.py` | Geometric-ratio and mnemonic analysis |
| `integration_demo.py` | Cross-module workflows |
| `visualization_demo.py` | General renderer examples |
| `png_visualization_demo.py` | PNG-oriented rendering |
| `organized_output_demo.py` | Configured output organization |
| `comprehensive_visualizations_demo.py` | Geometry, numbers, reports and grouped posters |
| `suite_interoperability.py` | Actual sibling-package coordinate/volume/transport comparison |

The comprehensive demo's `demonstrate_mega_graphical_abstracts(output_directory=None)`
can be called with an external destination; it renders `overview`, `geometry`,
`arithmetic`, `patterns`, `all` and writes JSON sidecars with actual metadata.
The overview keeps `mega_graphical_abstract.png`; other groups append their
section name. Group selection is unreleased-main behavior, not 1.3.0 wheel behavior.
This helper's destination does not redirect every other phase of `main()`.

`suite_interoperability.py` requires sibling QuadMath/QuadCraft checkouts,
Node.js and a C++17 compiler; adapters compile in temporary storage. Options
include `--quadmath-root`, `--quadcraft-root`, `--node`, `--cxx`:

```bash
uv run --python 3.13 --extra scientific python examples/suite_interoperability.py --help
```

Consult [canonical APIs](../docs/api/index.md), [suite gauges and units](../docs/synergetics-suite.md)
and [example guides](../docs/examples/) before interpreting numerical output.
