# TODO — symergetics agent backlog

Created 2026-08-31 by the agent-ergonomics fleet pass (see REVIEW_LOG_2026-08-31.md).
One line per entry + file path(s). Mark done items `x`.

## Minor

- [ ] tests/TEST_INDEX.md claims "977 test functions across 41 test modules" with no
  verification command — add "as of <date>, verified by `uv run pytest tests/
  --collect-only -q | tail -1`" or link to a generator. (tests/TEST_INDEX.md)
- [ ] `output/test_custom_dir/` is disposable test residue documented as such in
  AGENTS.md; consider deleting it locally so the tree reads clean. (output/test_custom_dir/)

## Medium

- [x] Document the per-file coverage-floor gotcha: `uv run pytest
  tests/core/test_coordinates.py` alone FAILS ("Required test coverage of 85% not
  reached. Total coverage: 8.19%") even when all 25 tests pass, because pyproject
  sets an 85% floor against the whole package. Run full `uv run pytest tests/ -q`
  (measured 2026-08-31: full collection exceeds 5 min on external-drive cold I/O)
  or pass `--no-cov` for quick checks. (AGENTS.md, STATUS.md)
- [x] No backlog file existed; TODO.md created 2026-08-31. (TODO.md)
- [x] No status surface existed; STATUS.md created with verification commands. (STATUS.md)

## Major

- [ ] Full test suite has no recorded pass/fail receipt from this pass — only
  tests/core/test_coordinates.py (25 passed) was run 2026-08-31; full collection
  exceeded the 300 s agent timeout on external-drive cold I/O. Run
  `uv run pytest tests/ -q` in the background with a generous timeout and record
  the result in STATUS.md. (STATUS.md)
