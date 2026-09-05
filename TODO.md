# TODO — symergetics agent backlog

Created 2026-08-31 by the agent-ergonomics fleet pass (see REVIEW_LOG_2026-08-31.md).
One line per entry + file path(s). Mark done items `x`.

## Minor

- [x] tests/TEST_INDEX.md claimed "977 test functions across 41 test modules" with no
  verification command — updated 2026-09-05 to 1013 functions / 42 modules, verified by
  `uv run pytest tests/ --collect-only -q` (1013 collected); per-file counts regenerated;
  added missing entries for `*_real.py`, `tests/core/` twins, and
  `test_review_regressions.py`. (tests/TEST_INDEX.md, tests/README.md)
- [x] `output/test_custom_dir/` residue: the generating test
  (`tests/test_visualization_methods.py::test_output_directory_configuration`)
  now renders into `tmp_path` and resets config, so it never creates the
  directory; existing residue deleted 2026-09-05. (tests/test_visualization_methods.py)

## Medium

- [x] Document the per-file coverage-floor gotcha: `uv run pytest
  tests/core/test_coordinates.py` alone FAILS ("Required test coverage of 85% not
  reached. Total coverage: 8.19%") even when all 25 tests pass, because pyproject
  sets an 85% floor against the whole package. Run full `uv run pytest tests/ -q`
  (measured 2026-08-31: full collection exceeds 5 min on external-drive cold I/O)
  or pass `--no-cov` for quick checks. (AGENTS.md, STATUS.md)
- [x] No backlog file existed; TODO.md created 2026-08-31. (TODO.md)
- [x] No status surface existed; STATUS.md created with verification commands. (STATUS.md)

- [x] Full test suite had no recorded pass/fail receipt — recorded 2026-09-05:
  `uv run pytest tests/ -q` → 993 passed, 20 skipped, coverage 86.13%
  (85% floor reached), 536 s. Pre-fix baseline: 954 passed, 20 skipped
  (`--no-cov`, 279 s). See STATUS.md. (STATUS.md)
