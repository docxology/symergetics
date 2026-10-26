# STATUS — symergetics (as of 2026-09-05)

Executable truth over prose truth: every claim below names the command that
verifies it. Re-run instead of trusting this file.

- **State:** the 2026-09-05 review fixes (package, tests, run.py, docs) are
  committed; `output/` artifacts are regenerable and its `test_custom_dir`
  residue was removed. Verify: `git status --porcelain`.
- **Branch:** `main`, clean tree. Verify: `git status -sb`.
- **Tests:** full suite receipt 2026-09-05 (final tree): `uv run pytest tests/ -q`
  → **997 passed, 20 skipped**, coverage **85.85%** (floor 85% reached), 9 min.
  Pre-fix baseline (same machine): 954 passed, 20 skipped with `--no-cov`, 279 s.
  Plotly backends are implemented for every numbers/mathematical/geometry plot
  and covered by tests; `tests/core/` twins merged into the `tests/` root;
  1017 tests collected across 39 modules. Quick single-file checks still need
  `--no-cov` (per-file coverage floor gotcha below).
- **Demo outputs:** everything under `output/` is regenerable via `run.py` /
  `examples/` scripts; never hand-edit. Verify: `ls output/`.
- **Paper:** last build `paper/output/symergetics_paper_20250913_124855.pdf`.
  Verify: `ls paper/output/`.
- **DOIs:** repo 10.5281/zenodo.17114383, paper 10.5281/zenodo.17114390
  (per docs/index.md). Verify: `grep -n zenodo docs/index.md`.
