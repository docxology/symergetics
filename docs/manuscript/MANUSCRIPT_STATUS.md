# Manuscript Status: Editable Sources and Immutable Publications

## Repo type

Symergetics is a mathematical software project with a paper source/render
pipeline. The canonical editable manuscript is
[paper/markdown/](../../paper/markdown/README.md), not a parallel top-level
`manuscript/` tree. This directory holds navigation and status only.

## Evidence checked

This status describes source organization, not a freshly executed paper build:

- [Paper overview](../../paper/README.md) maps all twelve numbered sections.
- [Renderer](../../paper/render/README.md) documents direct ReportLab rendering.
- [Validation](../../paper/validation/README.md) defines structural/image gates.
- [Test evidence](../../paper/TEST_EVIDENCE.md) preserves historical source hashes
  and execution receipts, not current suite results.
- [Research scope](../research/index.md) separates implemented mathematics from
  proposed applications and empirical evidence.

## Why no canonical `manuscript/` tree exists today

Duplicating numbered sections would create conflicting sources of truth.
Source prose lives under `paper/markdown/`; generated paper outputs live under
`paper/output/` and package-generated figures under root `output/`. Do not
hand-edit those artifacts to reflect a source-only correction.

## Publication boundaries

The repository records [paper DOI 10.5281/zenodo.17114390](https://doi.org/10.5281/zenodo.17114390)
and [software DOI 10.5281/zenodo.17114383](https://doi.org/10.5281/zenodo.17114383).
Use their landing metadata for the deposited versions. Local source edits,
new documentation or a regenerated PDF do not revise an immutable deposit or
release. The title page's 2025 date/version are historical metadata.
No peer-review outcome or updated publication is asserted here.

## What would trigger creating one

A separately authorized pipeline migration would need to move canonical sections,
update consumers and links, and retire the old source path. It is a possible
maintenance decision, not a scheduled migration or an invitation to add stubs.

[Manuscript navigation](README.md) · [Documentation map](../index.md)
