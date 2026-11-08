# Figure Integration: Source References, Lookup and Provenance

## Overview

The manuscript references generated figures; it does not make every file under
`output/` a paper figure or validate its content. Keep source prose editable and
leave generated/user-owned artifacts untouched except through an authorized,
isolated generation workflow. See [direct rendering](QUICK_START.md).

## Output Directory Structure

The existing convention groups package figures under root `output/geometric/`
and `output/mathematical/`, with topic subdirectories. Rendered papers normally
go under `paper/output/` unless an output directory is explicitly selected.
Do not invent a filename from a documentation template: locate the actual
producer and its returned paths. Generated artifacts can predate source edits.

## Image Discovery Process

[ScientificPaperRenderer._find_image_path](render/pdf_renderer.py) resolves the
reference against `paper/markdown/`, exactly as ordinary Markdown does.
It does not discard directory prefixes, recursively choose a basename match or
substitute a different file when the requested path is absent.
Absolute image references are rejected. The publication validators additionally
require the resolved asset to remain under repository `output/`.

## Reference Patterns

The structural [validator](validation/validate_paper.py) and image diagnostic
use source-relative image paths and matching caption paragraphs. Existing
concrete source example:

```markdown
![Figure 1: Quadray Coordinate System Origin](../../output/geometric/coordinates/quadray_coordinate_0_0_0_0.png)

**Figure 1**: Quadray origin in a rank-three embedding. The four slots represent a common-offset gauge, not four independent spatial dimensions.
```

The reference does not prove regeneration. Authored figure numbers and detailed
captions are preserved, including captions after blank lines. Both
`**Figure N**:` and `**Figure N:**` forms are recognized. An intentional repeated
reference is rendered again; prefer a cross-reference when the same idea need
not be drawn twice.

### Formats and Mermaid

The renderer fits raster images proportionally within its display box, preserving
geometry. Missing, unreadable or unsupported images fail rather than becoming
placeholders. ReportLab still has no general SVG/PDF conversion stage: produce
a real PNG/JPEG counterpart from the owning renderer when required.

Mermaid blocks invoke `mmdc` when available and may write generated PNGs under
`paper/mermaid_images/`. If conversion fails, the renderer can display text
instead. A conceptual flowchart depicts a proposed relationship, not a causal
result. Text fallback is not evidence of a diagram image. Image-reference
failures are hard failures and must not be hidden by a successful partial PDF.

## Figure Management System

[figure_manager.py](render/figure_manager.py) and
[scan_and_register_figures.py](render/scan_and_register_figures.py) provide a
separate registry/scan workflow. Registry metadata is not independently measured
evidence and is not a substitute for the renderer's actual caption handling.
Link the implementation instead of maintaining copied dataclasses or private
function bodies in documentation.

## Validation and Testing

Use the [quick-start commands](QUICK_START.md#validate-paper) for structural and
image diagnostics in a disposable checkout. Record the source revision, producer,
inputs, embedding/units, dependency environment and output path for regenerated
figures. Then inspect the PDF at ordinary reading width for legibility, clipping,
geometry and caption agreement. A file-existence check cannot establish those
properties or mathematical accuracy.

The checked-in [validation_report.md](validation_report.md) is a historical
generated artifact with old section names and paths. It is preserved, not a
current receipt. [Test evidence](TEST_EVIDENCE.md) has separate revision semantics.

## Troubleshooting

- **Wrong figure:** compare the full source-relative path and actual loaded file.
- **Missing figure:** inspect the producer and source-relative path; do not
  create a fake file or alter the caption to conceal missing input.
- **Distorted image:** check the actual raster aspect and rendered PDF; ordinary
  raster insertion now preserves that aspect.
- **Unsupported format:** generate a supported raster export from real data.
- **Caption mismatch:** compare authored numbering and detailed caption text.
- **Mermaid fallback:** inspect `mmdc` availability and logs in the isolated
  environment; text fallback is not evidence of a successful diagram render.

## Best Practices

Captions state what is computed, units, scale/gauge where relevant and whether
values are exact or approximate. They do not claim new discoveries or empirical
benefit from visual appearance. Keep detailed mathematical definitions in
[foundations](markdown/03_mathematical_foundations.md) and writing conventions in
[style guidance](STYLE_GUIDE.md).
