# symergetics — Agent Notes

Python implementation of Buckminster Fuller's Synergetics: exact rational
arithmetic, quadray/IVM coordinates, polyhedra, number-pattern experiments
(palindromes, primorials, scheherazade), and multi-backend visualization.

## Orientation ladder (cold start)

1. **What is this?** This file, plus `docs/index.md` (DOIs, doc map).
2. **Current state?** `STATUS.md` — every claim names the command that
   verifies it. Do not trust prose status; re-run.
3. **What next?** `TODO.md` — the single backlog file (Minor/Medium/Major).
4. **History of doc passes?** Point-in-time review logs were removed from the
   tree (commit 2c7edcf); receipts live in `STATUS.md` and git history.

## Layout

- `symergetics/` — package: core/computation/geometry/utils/visualization
- `tests/` — pytest suite (no mocks); index at `tests/TEST_INDEX.md`
- `examples/` — demo scripts; each regenerates the matching `output/` subtree
- `docs/` — guides; entry point `docs/index.md`
- `output/` — generated artifacts. **Never hand-edit; regenerable.**
- `paper/` — markdown sections → render → timestamped PDF in `paper/output/`

## Verification (read the gotchas before running)

- Quick check, one module: `uv run pytest tests/test_core_coordinates.py -q --no-cov`
  (the `tests/core/` twins were merged into the `tests/` root on 2026-09-05;
  measured 2026-09-05 after merge: 41 passed).
- **Gotcha 1:** a single-file run WITHOUT `--no-cov` fails pyproject's 85%
  package-wide coverage floor (failing exit ≠ failing tests).
- **Gotcha 2:** the full suite is slow here on cold external-drive I/O. Run in
  background with a generous timeout: `uv run pytest tests/ -q`. Receipt:
  2026-09-05 → 993 passed, 20 skipped, coverage 86.14% (STATUS.md).
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
