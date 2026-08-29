# Agent guide — docs/ (symergetics)

## Layout

- `index.md` — human entry; links the major areas below.
- Topic guides at the top level (concept, math foundations, glossary, testing,
  performance, troubleshooting, contributor guide, case studies, and more).
- Subdirectories: `api/`, `examples/`, `research/`, `architecture/` (each with
  its own `index.md`), plus `tutorials/`.

## Conventions observed

- Docs are hand-written prose guides; no generated pages live here.
- The paper pipeline lives outside docs/, under `../paper/` with its own
  `STYLE_GUIDE.md` and validation.
- CI snippets inside guide pages (e.g. GitHub Actions YAML with `${{ }}`)
  are code-fence examples, not template tokens.

## Maintenance

- Keep `index.md` as the single navigation hub; when adding a new guide, add
  it to the map there.
- Verify any relative link target exists before committing (use the same
  link-audit pass used across the fleet).
