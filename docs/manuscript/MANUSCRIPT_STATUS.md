# MANUSCRIPT_STATUS

## Repo type

Publication-track research package whose paper content lives under `paper/`,
not in a top-level `manuscript/` directory.

## Evidence checked

- `README.md` (package + Paper DOI 10.5281/zenodo.17114390; repo DOI 10.5281/zenodo.17114383)
- `paper/` (generate_paper.py, markdown/, render/, output/, STYLE_GUIDE.md, validation/)
- `docs/` (23 guides; no manuscript sections)

## Why no canonical `manuscript/` tree exists today

The published paper is generated and maintained by the project's own
`paper/` pipeline with its own style guide and validation. Duplicating it as
template-format `manuscript/` section stubs would create a second,
conflicting source of truth for the same content.

## What would trigger creating one

Adopting the shared Research Project Template render pipeline for the paper:
migrate `paper/markdown/` sections into `manuscript/00..99` section files with
`config.yaml`, `preamble.md`, and `references.bib`, then retire the parallel
pipeline.
