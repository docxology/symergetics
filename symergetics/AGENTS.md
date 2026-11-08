# Package maintenance: shared contracts and module ownership

Read the root [maintenance guide](../AGENTS.md), each module's `AGENTS.md`,
and the [API reference](../docs/api/index.md) before changes. The package map is
in [README.md](README.md); scripts and orchestration live outside this directory.

Preserve exact native arithmetic and explicitly name approximation boundaries.
Four quadray labels have a common-offset gauge and rank three, not four
independent spatial axes. Geometry depends on indexed surfaces, not convex hulls.
Check exported callers before changing interfaces; avoid importing optional
plotting libraries eagerly. NumPy is a required dependency, not an optional
renderer. Do not write into repository `output/` for inspection or validation.
Current poster section selection belongs to unreleased main, not the immutable
1.3.0 wheel. Verification follows the current assignment; do not infer a green
suite from historical receipts.
