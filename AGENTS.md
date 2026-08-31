# symergetics — Agent Notes

Python implementation of Buckminster Fuller's Synergetics: exact rational
arithmetic, quadray/IVM coordinates, polyhedra, number-pattern experiments
(palindromes, primorials, scheherazade), and multi-backend visualization.

## Orientation ladder (cold start)

1. **What is this?** This file, plus `docs/index.md` (DOIs, doc map).
2. **Current state?** `STATUS.md` — every claim names the command that
   verifies it. Do not trust prose status; re-run.
3. **What next?** `TODO.md` — the single backlog file (Minor/Medium/Major).
4. **History of doc passes?** `REVIEW_LOG_2026-08-31.md` (append, never rewrite).

## Layout

- `symergetics/` — package: core/computation/geometry/utils/visualization
- `tests/` — pytest suite (no mocks); index at `tests/TEST_INDEX.md`
- `examples/` — demo scripts; each regenerates the matching `output/` subtree
- `docs/` — guides; entry point `docs/index.md`
- `output/` — generated artifacts. **Never hand-edit; regenerable.**
- `paper/` — markdown sections → render → timestamped PDF in `paper/output/`

## Verification (read the gotchas before running)

- Quick check, one module: `uv run pytest tests/core/test_coordinates.py -q --no-cov`
  (measured 2026-08-31: 25 passed in 371 s on external-drive I/O).
- **Gotcha 1:** a single-file run WITHOUT `--no-cov` fails pyproject's 85%
  package-wide coverage floor (measured: 25/25 passed, "Total coverage: 8.19%",
  exit nonzero). Failing exit ≠ failing tests.
- **Gotcha 2:** the full suite is slow here — full collection exceeded 300 s on
  external-drive cold I/O. Run in background with a generous timeout:
  `uv run pytest tests/ -q`. Full-suite receipt not yet recorded (TODO.md Major).
- Demos: `python run.py --examples-only` regenerates `output/`.

## Decision points

- Adding code → `symergetics/` package subdirs; see `docs/architecture/index.md`.
- Adding tests → `tests/`; keep no-mocks policy; real data/computation only.
- Updating status/backlog → edit `STATUS.md` / `TODO.md` here, not in docs/.
- Never commit `output/` artifacts or anything you didn't author.

## Doc-surface status

Coverage of every subfolder with README.md + AGENTS.md completed 2026-08-29
(2026-08-29 docs fleet). Relative links in root + docs entry files verified
2026-08-31 (0 broken).
