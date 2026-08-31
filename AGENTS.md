# symergetics — Agent Notes

(Lane-level quick reference appended by the 2026-08-29 docs fleet; detailed
repo policy lives in existing docs.)

- **Layout:** `symergetics/` (package: core/computation/geometry/utils/visualization),
  `tests/` (comprehensive pytest suite — no mocks), `examples/` (demo scripts),
  `docs/` (hand-written guides: api/architecture/examples/research/tutorials index pages),
  `output/` (generated visualization/report products incl. a disposable
  `test_custom_dir/` test-residue tree), `paper/` (markdown sections → render →
  timestamped PDF + figure trees; validation tooling), plus `run.py`,
  `pyproject.toml`, `uv.lock`.
- **Invariants:** everything under `output/` is regenerable — never hand-edit or
  treat as source; `paper/output/` filenames are timestamped build artifacts.
- **Verify:** `uv run pytest tests/ -q` (unverified this pass — no run performed);
  demo scripts in `examples/` regenerate the corresponding `output/` subtrees.
- Coverage of this tree completed 2026-08-29: every subfolder now has
  README.md + AGENTS.md.
