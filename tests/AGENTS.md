# tests/ — Agent Notes

Comprehensive pytest suite (existing README: 977 test functions across 41
modules, all real-method TDD — no mocks). Layout: `test_computation_*.py`,
`test_*` modules at top level, plus `core/` (`test_constants.py`,
`test_coordinates.py`, `test_numbers.py`). Run from repo root:
`uv run pytest tests/ -q` (unverified this pass — no run performed).
