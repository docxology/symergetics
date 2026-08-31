# STATUS — symergetics (as of 2026-08-31)

Executable truth over prose truth: every claim below names the command that
verifies it. Re-run instead of trusting this file.

- **State:** working tree has ~88 untracked files, all under `output/`
  (regenerable build artifacts + docs-fleet AGENTS/README residue in output
  subtrees). Verify: `git status --porcelain | grep -v '^?? output/' | wc -l`
  (expect 0 non-output dirty files).
- **Branch:** `main`, in sync with origin at time of writing. Verify:
  `git status -sb`.
- **Tests:** `uv run pytest tests/core/test_coordinates.py -q` → 25 passed in
  371 s (2026-08-31, external-drive I/O). NOTE: a single-file run fails the
  pyproject 85% coverage floor (8.19%) even when tests pass — use `--no-cov`
  for quick checks or run the full `uv run pytest tests/ -q` (background;
  full collection exceeds 5 min here). Full-suite receipt: not yet recorded —
  see TODO.md Major item.
- **Demo outputs:** everything under `output/` is regenerable via `run.py` /
  `examples/` scripts; never hand-edit. Verify: `ls output/`.
- **Paper:** last build `paper/output/symergetics_paper_20250913_124855.pdf`.
  Verify: `ls paper/output/`.
- **DOIs:** repo 10.5281/zenodo.17114383, paper 10.5281/zenodo.17114390
  (per docs/index.md). Verify: `grep -n zenodo docs/index.md`.
