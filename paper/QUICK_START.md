# Paper Quick Start: Python 3.13 and Direct Rendering

## Overview

Run these commands from the repository root in a **disposable checkout** with
its own generated artifacts. They are instructions, not a claim they were
executed in this documentation pass. Use the direct renderer to avoid the
broader test/example/cleanup orchestration in `generate_paper.py`.

## Quick Commands

### Set up an isolated environment

```bash
python3.13 -m venv .venv-paper
.venv-paper/bin/python -m pip install -e '.[scientific]'
.venv-paper/bin/python -m pip install -r paper/render/requirements.txt
```

This installs the checkout, not a nonexistent PyPI release. The requirements
file also includes development/test tools; installing them does not run them.
The scientific dependencies come from [pyproject.toml](../pyproject.toml).
Renderer requirements include Matplotlib's portable DejaVu font bundle and the
PDF parser used by regression checks; no machine-specific TeX fonts are needed
for this ReportLab path.
Do not change `.python-version` to run this workflow.

### Validate Paper

```bash
.venv-paper/bin/python paper/validation/validate_paper.py --paper-dir paper --output /tmp/symergetics-paper-validation.md
.venv-paper/bin/python paper/validation/check_images.py --paper-dir paper --output /tmp/symergetics-paper-images.md
```

The structural validator checks source sections and root `output/` references;
it does not prove mathematics, follow every external link or establish current
empirical evidence. The image helper is a separate diagnostic. See
[validation scope](validation/README.md).

### Generate a PDF without regenerating examples

```bash
.venv-paper/bin/python paper/render/run_render.py --paper-dir paper --output-dir /tmp/symergetics-paper-preview --output preview.pdf --verbose
```

This consumes existing figures and writes a PDF to the selected directory;
it does not refresh their data. Renderer initialization creates its selected
output directory. Mermaid processing can additionally write under the paper
tree, which is another reason to use a disposable checkout. Missing, unreadable
or unsupported referenced images fail rendering. Inspect the PDF as well as
the exit code; Mermaid's optional text representation is a separate boundary.

## Common Workflows

### Updating Paper Content

Edit the twelve numbered files linked from [paper structure](README.md#paper-structure).
Keep actual APIs in the [implementation reference](../docs/api/index.md), and
keep proposed uses visibly separate from tested behavior. Generate visualizations
only through the real example/renderer that owns them, in isolation; do not
hand-edit or overwrite user output to make a validation check pass.

### Adding New Visualizations

Use the [integration guide](INTEGRATION_GUIDE.md) to locate the actual producer,
preserve provenance. Image paths are relative to the source Markdown file,
for example `../../output/geometric/polyhedra/tetrahedron_3d_enhanced.png`,
with a matching detailed caption and concrete authored figure number.

## Authoring Limits and PDF Review

[Style guide](STYLE_GUIDE.md) covers topic-specific headings, numeric domains,
units and captions. Four quadray slots encode rank-three geometry; “exact
rational approximation” means an exact stored rational approximating a different
value, not an exact irrational constant or exact measurement.

Follow the [heading/code/table rules](STYLE_GUIDE.md#heading-levels-literal-code-and-pipe-tables):
H1–H4 stay with subsequent content, but long bodies still split. Use valid
three-dash table delimiters and escape literal cell pipes, including in code.
In the isolated PDF, check headings near page feet, unchanged literal identifiers,
empty columns and wrapped table cells as well as figures. H5/H6, merged table
cells, full LaTeX math and bibliography processing are not equivalent to a full
publishing engine; see the [renderer limits](render/README.md#pdf-heading-pagination-unicode-and-pipe-tables).

## Troubleshooting

- **Imports:** use the same `.venv-paper/bin/python` used for installation.
- **Missing image:** compare the full source-relative path with the producer's
  real filename. The renderer no longer guesses by basename.
- **SVG/PDF image:** discovery is not conversion support; use a generated raster
  counterpart when ReportLab cannot embed the source.
- **Wrong source/output directory:** keep `--paper-dir paper` explicit from root.
- **Old report:** regenerate to a disposable report path for the reviewed revision.
- **Unexpected regeneration:** stop using the orchestration command for PDF-only
  work; inspect [generate_paper.py](generate_paper.py) before invoking it.

[Paper overview](README.md) · [Renderer contract](render/README.md)
