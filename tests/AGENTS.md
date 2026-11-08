# Test maintenance: consumer behavior and temporary artifacts

Pytest modules live directly here, not in `tests/core/`. Read the
[test guide](README.md) and [semantic index](TEST_INDEX.md). `conftest.py` gives
every test a `tmp_path` working directory and resets visualization configuration
before and in a `finally` teardown, including tests calling `reset_config()`.
Never recursively clean repository `output/` or mock successful renderer results.
Use real numerical oracles, actual artifacts and test-owned destinations.

From the repository root, a focused diagnostic recipe is:

```bash
MPLBACKEND=Agg uv run --python 3.13 --extra scientific --extra test python -m pytest tests/test_core_coordinates.py -q --no-cov
```

The full suite omits `--no-cov`; the package-wide floor is 85%, not a per-file
requirement. Current receipts belong in [STATUS](../STATUS.md), not hard-coded
counts here. Follow the active assignment when it defers verification; these
commands do not authorize mid-flight runs.
