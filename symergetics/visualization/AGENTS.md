# Visualization maintenance: optional backends and owned artifacts

Read [renderer documentation](README.md), the root [maintenance guide](../../AGENTS.md)
and [API reference](../../docs/api/index.md). Geometry, numbers, mathematical and
advanced modules have different backend support: do not promise that every
function supports matplotlib, Plotly and ASCII equally.

Plotting dependencies are optional (`scientific` extra); NumPy is a base
dependency. Keep optional plotting imports lazy and report missing dependencies
at use. `set_config` accepts one dictionary, not configuration keywords. Relative
output paths resolve from the working directory, not from this source directory;
use explicit temporary destinations for probes and never overwrite user `output/`.
Close owned figures on success and failure; inspect actual returned artifacts.

Introduced in 1.4.0, `create_mega_graphical_abstract(title=None, output_path=None,
backend=None, *, section="overview")` accepts `overview`, `geometry`,
`arithmetic`, `patterns`, `all`. The default is a focused overview; `all` retains
all 17 computed topics. Earlier release artifacts retain their own interfaces.
Preserve selected-section/panel metadata and exact native-vs-XYZ units. Solid
abstracts must distinguish vertex means, snapped centers and mass centroids;
respect display flags without inventing missing geometry. Follow the current
assignment's verification policy.
