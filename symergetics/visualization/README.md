# Visualization modules: live geometry, computed patterns and output contracts

Renderers consume the actual coordinate, number and polyhedron APIs. Exact
integer/rational measurements, irrational approximations and floating display
coordinates are different quantities; **not all visualization calculations
are rational or exact**.

## Install the checkout's optional scientific dependencies

From the repository root:

```bash
uv sync --python 3.13 --extra scientific
```

The package is not on PyPI. Use the checkout or a versioned GitHub wheel for its
own interface. Poster selection was introduced in 1.4.0 and is not retrofitted
to older releases. See the [installation and API guide](../../docs/api/index.md).

## Choose a renderer by its actual mathematical input

| Module | Implemented purpose | Canonical contract |
|---|---|---|
| [`geometry.py`](geometry.py) | Coordinates, physical FCC shells, live indexed polyhedra, projections and solid abstracts | [Geometry API](../../docs/api/index.md#geometry-module) |
| [`numbers.py`](numbers.py) | Finite palindrome, Scheherazade and prime-bound primorial displays | [Number APIs](../../docs/api/index.md) |
| [`mathematical.py`](mathematical.py) | Continued fractions, base/digit analysis and related finite diagnostics | [Examples and cookbook](../../docs/api-cookbook-recipes.md) |
| [`advanced.py`](advanced.py) | Comparative/dashboard plots and computed scientific posters | [Poster selections](../../docs/api/index.md#scientific-poster-sections) |
| [`__init__.py`](__init__.py) | Process-global configuration, organized paths, serial batch dispatch and reporting | [Visualization API](../../docs/api/index.md#visualization-module) |

Backend support is function-specific. A shared `backend` parameter does not
mean every renderer implements Matplotlib, Plotly and ASCII. Solid and mega
graphical abstracts require Matplotlib; unsupported choices raise.

## Render a live solid without overwriting project output

```python
from symergetics.visualization import plot_polyhedron, set_config

set_config({
    "backend": "matplotlib",
    "output_dir": "/tmp/symergetics-visualizations",
    "formats": ["png", "svg"],
    "include_timestamps": False,
})
result = plot_polyhedron("tetrahedron")
print(result["files"])
```

Configuration takes one dictionary, not keyword arguments. It is global to the
process; use isolated processes for independent render configurations.
`get_config()` returns a shallow copy, not a deeply isolated nested snapshot.
Most renderers return file paths plus metadata; exact Python measurements can
include `Fraction` objects and are not a promise of universal JSON serialization.

## Interpret solid topology, units and centroids correctly

`plot_polyhedron_graphical_abstract` uses current vertices, declared indexed
edges and the surface's declared triangle fans. Its axes are anchor-relative,
scale-one XYZ with equal metric scale. The XY panel is an orthographic edge
projection, retaining coincident/hidden edges; it is not a projected convex hull.

Native IVM volume differs from XYZ cubic volume:
$V_{\mathrm{XYZ}}=(8/3)V_{\mathrm{IVM}}$ at this scale. Surface area uses XYZ
squared units. A canonical volume ratio is not a measured volume of a deformed
or translated display.

Vertex means and uniform-solid mass centroids are separate measurements.
The latter requires nonzero enclosed volume and the documented valid embedded
surface preconditions. Closure alone does not detect all self/intersections.
Nonplanar faces mean their declared triangle fan, not an unspecified surface.

`show_edges`, `show_faces` and `show_vertices` control geometry layers; the XY
view has no face fill. `show_volume_ratios` and `show_coordinates` remove their
optional panels when disabled. Solid `metadata.panels` is the actual panel ID
list. Full exact vertices, topology, anchor and measurements remain in metadata
even when enormous display annotations are explicitly omitted.

## Scientific poster sections

```python
from symergetics.visualization.advanced import create_mega_graphical_abstract

overview = create_mega_graphical_abstract(
    section="overview", output_path="/tmp/symergetics-overview.svg"
)
geometry = create_mega_graphical_abstract(
    section="geometry", output_path="/tmp/symergetics-geometry.svg"
)
```

Use `overview`, `geometry`, `arithmetic`, `patterns` or `all`. The default
overview is readable at ordinary width; `all` retains seventeen topics at
poster scale. An explicit title is preserved. Different selections otherwise
share the default primary filename, so use distinct explicit paths.

Mega results contain actual ordered panel IDs, titles and computed side data,
including exact integer/rational strings and units. Its `metadata.panels` is
an integer count, unlike the solid abstract's panel list. It writes its one
explicit/default primary path rather than automatically exporting every
configured format. The
[comprehensive demo](../../examples/comprehensive_visualizations_demo.py)
renders separate groups with JSON sidecars.

Scientific footers use the UTC date of `SOURCE_DATE_EPOCH` when set, otherwise
the local date. Sidecar paths are relative to the sidecar directory. Registry
export timestamps instead describe real UTC export time; these fields do not
guarantee byte-identical files across toolchains.

Raw digit strings retain leading zeros. Pattern matrices are dense by default;
ASCII `include_matrix=False` explicitly opts into linear text and `None` matrix
metadata. Graphical backends reject omission; radar coverage is always linear.
See [mathematical contracts](../../docs/api/index.md#mathematical-visualizations).

## Use actual paths and batch outcomes rather than assumed capabilities

- Organized filenames follow `output_dir`, `organize_by_type` and timestamp
  settings unless an API documents an explicit path override.
- `formats` are consumed by applicable exporters; an HTML entry does not turn
  a static Matplotlib renderer into an interactive one.
- `batch_visualize` dispatches supported task names **serially**. Inspect each
  result for errors; batch completion is not proof every task succeeded.
- Reporting/export helpers consume their documented data shapes. Do not
  treat arbitrary analysis/report dictionaries as interchangeable.
- Irrational coordinates, trigonometry, floating plots and finite sampled
  patterns do not prove a universal rationality, frequency or physical law.

## Verification and cross-project references

Run the relevant behavior tests through the [test guide](../../docs/testing-guide.md).
Numerical geometry/topology, optional-panel behavior and figure ownership are
different from a wording or nonempty-file assertion. Inspect actual rendered
PNG/SVG output as well as numerical metadata.

- [Suite coordinate, unit and visualization boundaries](../../docs/synergetics-suite.md)
- [Package-level guide](../README.md)
- [Source-backed tutorials](../../docs/tutorials/index.md)
- [Module agent guidance](AGENTS.md)

## License

This module follows the repository's [Apache 2.0 license](../../LICENSE).

## Render resource boundary

Configuration updates reject non-finite/non-positive figure dimensions or DPI
and nominal canvases larger than 100,000,000 pixels with `ValueError`, before
any settings are changed. Matplotlib exporters check their actual requested figure dimensions,
including scaled and fixed-size plots, before constructing the figure. This
check uses the higher of export DPI and Matplotlib's configured draw DPI, so
a low export DPI cannot conceal a huge intermediate draw buffer. Numeric
strings and booleans are rejected rather than stored as rendering settings. Reduce figure
size or DPI after a rejection. Existing 32×24-inch publication figures at 300 DPI
and 10×8-inch figures at 600 DPI fit this limit.

The cap corresponds to 400 MB for a single nominal RGBA buffer. Tight bounding
boxes, padding and encoder overhead can require additional memory; this is a
nominal canvas boundary, not a total process-memory guarantee. SVG/PDF exports
also pass through this check because Matplotlib can rasterize artists during
vector export. No image is silently downsampled and no limit override is exposed.
