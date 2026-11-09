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
section name. Group selection was introduced in 1.4.0, not retrofitted to older wheels.
This helper's destination does not redirect every other phase of `main()`.
JSON `files` are relative to the sidecar directory; runtime paths still identify
the actual generated files. `SOURCE_DATE_EPOCH` controls the scientific footer's
UTC date, otherwise the local date is used. This is not a byte-identity guarantee.

`suite_interoperability.py` requires sibling QuadMath/QuadCraft checkouts,
Node.js and a C++17 compiler; adapters compile in temporary storage. Options
include `--quadmath-root`, `--quadcraft-root`, `--node`, `--cxx`:

```bash
uv run --python 3.13 --extra scientific python examples/suite_interoperability.py --help
```

The interoperability report includes actual `finite_boundary_outcomes` and
derives `finite_boundary_cases` from the returned records. Cases include
degree-11/18 extreme stars with both signs and full/single/disabled masks,
isolated values, subnormal/maximum ties, tiny coupling and signed-zero
zero-coupling identity. Both production kernels are exercised; this is not
a frozen inventory count or proof that every finite input behaves identically.

Consult [canonical APIs](../docs/api/index.md), [suite gauges and units](../docs/synergetics-suite.md)
and [example guides](../docs/examples/) before interpreting numerical output.
