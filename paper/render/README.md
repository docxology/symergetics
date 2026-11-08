# Paper PDF Renderer: Entry Points and Presentation Limits

## Overview

[ScientificPaperRenderer](pdf_renderer.py) reads the twelve numbered source
modules under `paper/markdown/` and creates a ReportLab PDF. Source figures are
resolved from existing artifacts. Rendering is presentation, not mathematical
proof, empirical validation or automatic regeneration of all examples.

## Installation

Use the [Python 3.13 environment setup](../QUICK_START.md#set-up-an-isolated-environment)
from repository root. [requirements.txt](requirements.txt) includes renderer and
development dependencies. The renderer is part of a checkout, not a separately
published PyPI package. Do not change the repository interpreter pin.

## Usage

From repository root in a disposable checkout:

```bash
.venv-paper/bin/python paper/render/run_render.py --paper-dir paper --output-dir /tmp/symergetics-paper-preview --output preview.pdf --verbose
```

`run_render.py` supports `--paper-dir`, `--output`, `--output-dir` and `--verbose`.
Keep the paper path explicit; relative paths supplied by the caller are relative
to that caller's working directory. The CLI in `pdf_renderer.py` also supports
paper/output paths but not the wrapper's verbose flag. Default output is under
`paper/output/`; choose an isolated directory to preserve existing artifacts.

Do not substitute `generate_paper.py` for a render-only command: that orchestration
can clear outputs, run tests and regenerate examples.

## File Structure

- [pdf_renderer.py](pdf_renderer.py): section loading, formatting, figure lookup
  and PDF construction.
- [run_render.py](run_render.py): command-line wrapper.
- [config.yaml](config.yaml): renderer configuration; inspect actual consumers
  before assuming every setting is applied.
- [figure_manager.py](figure_manager.py): separate figure registry utilities.
- [scan_and_register_figures.py](scan_and_register_figures.py): registry scanner.
- [test_renderer.py](test_renderer.py): source tests, not a current pass receipt.

## PDF Heading Pagination, Unicode and Pipe Tables

Use [manuscript style](../STYLE_GUIDE.md) for headings, real code examples,
references and bounded captions. The renderer embeds DejaVu Sans and Mono fonts
bundled with Matplotlib rather than discarding non-ASCII text. Coordinate minus
signs, square roots, inequalities and cubic units must survive in the PDF.

Ordinary H1–H4 headings are printed at distinct sizes and kept with subsequent
content. Spacing belongs to heading styles, not an intervening spacer that could
leave the heading alone at a page foot. Long paragraphs can still split, and each
numbered source module begins a new page. The title module uses its own metadata
layout; H5/H6 remain body-sized text.

Simple pipe tables wrap within the page frame, preserve empty columns, omit valid
delimiter rows and repeat headers across pages. Header/delimiter widths must
match; delimiters need at least three dashes and may have edge colons. Colons do
not currently set column alignment. Outer pipes are optional. Escape literal
cell pipes as `\|`, even inside code spans; use blank lines around a table.
Rows are padded or trimmed to the header width; merged cells and nested blocks
are unsupported. Inline code stays literal and monospaced rather than becoming
emphasis or links. Cell links otherwise use body inline formatting. Relative
links retain their source context; use the Markdown sources for navigation.

This is not a full scholarly Markdown, LaTeX or bibliography processor. Long code
blocks can be truncated; a printed excerpt is not the complete implementation.
Inspect mathematical glyphs, tables, links and page layout in the actual output.

## Image Integration

References resolve exactly from `paper/markdown/`, not through recursive
basename search. Both validators use the same source-relative path convention;
publication assets must remain under repository `output/`. See the
[integration contract](../INTEGRATION_GUIDE.md#image-discovery-process).

Raster figures fit within the display box without changing aspect ratio.
Authored figure numbers and detailed captions survive optional separating blank
lines, and a repeated reference is rendered rather than silently discarded.
The paper itself cross-references its repeated sequence summary instead of
printing the same panel twice.

Missing, unreadable or unsupported image inputs fail rendering; there are no
image placeholders or guessed same-basename substitutes. SVG/PDF need a real
conversion stage or raster counterpart; filename recognition is not conversion.
Mermaid remains separate: `mmdc` can create `paper/mermaid_images/`, and missing
diagram conversion can yield an explicitly textual representation.

## Real-PDF and Image Regression Checks

From repository root after installing the renderer requirements:

```bash
.venv-paper/bin/python -m unittest discover -s paper/render -p test_renderer.py
```

Regressions build real PDFs and inspect mathematical text, heading/body page
co-location, distinct heading sizes, literal code and table-column placement,
alongside image identity, aspect, repetition and failure controls.
Run the structural/image diagnostics linked from
[quick start](../QUICK_START.md#validate-paper), then inspect an isolated render.
These checks do not establish every mathematical interpretation or publication
quality, and this guide does not pin a timeless pass count.

## Troubleshooting

Use the same environment for installation and execution, explicit source/output
paths and logs from the actual failed command. Inspect duplicate filenames,
unsupported image formats and final-page clipping before changing source claims.
Do not create fake images or suppress warnings to obtain a nominal success.

## License

This repository uses [Apache 2.0](../../LICENSE), not MIT.

[Paper overview](../README.md) · [Figure integration](../INTEGRATION_GUIDE.md)
