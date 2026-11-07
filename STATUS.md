# STATUS — symergetics

Executable truth over prose truth: every claim below names the command that
verifies it. Re-run instead of trusting this file.

## Suite review receipt — 2026-11-06

- Python 3.13.17 with the scientific/test extras: full suite **1,344 passed,
  12 skipped**, coverage **88.16%** (85% floor). Command in that environment:
  `MPLBACKEND=Agg python -m pytest tests/ -q -o 'addopts=--strict-markers --strict-config --cov=symergetics --cov-report=term-missing --cov-fail-under=85'`.
- `python examples/suite_interoperability.py`: **279 coordinate cases per
  implementation** across actual QuadMath, symergetics, QuadCraft Python,
  JavaScript and C++, plus **34 exact-volume cases** and **216 FCC site comparisons**.
- `python verify_suite.py --report /tmp/synergetics-suite.json`: all **13**
  real checks passed, with zero unexecuted checks, using frozen temporary environments.
- `python run.py --examples-only` in the prepared scientific environment:
  all **10** ordinary demos succeeded; setup/tests were explicitly skipped.
- Minimal isolated package installation imports the core successfully; NumPy
  is required rather than hidden in the scientific extra.
- The patched lockfile excludes all 98 captured open Dependabot affected ranges.
  The full coverage gate and all demos also passed with Pillow 13 and pytest 9.
- [Suite contracts and evidence](docs/synergetics-suite.md) document the exact
  integer/rational boundaries, nearest-point algorithms, embedding units and
  actual browser/native verification limits. Python pins were not changed.

## Earlier receipt — 2026-09-05

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
