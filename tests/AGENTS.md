# tests/ — Agent Notes

Pytest modules live directly under this directory. `conftest.py` gives each
test a temporary working directory and resets visualization configuration before
and after it, including tests that call `reset_config()` themselves. Never
delete files from repository `output/` during teardown or mock renderer results.
Use real APIs and test-owned paths. Run from the repository root:
`uv run --python 3.13 --extra scientific --extra test python -m pytest tests/ -q`.
Focused runs need `--no-cov` to avoid the package-wide coverage floor.
Current full-suite receipts live in `../STATUS.md`.
