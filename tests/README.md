# Test suite: execution, isolation and numerical evidence

The [test index](TEST_INDEX.md) is a navigation map, not a live collection or
correctness certificate. [Testing guidance](../docs/testing-guide.md) defines
canonical commands; dated receipts live in [STATUS](../STATUS.md) and the
[suite guide](../docs/synergetics-suite.md).

Run from the repository root with Python 3.13 and scientific/test extras:

```bash
# Full package coverage gate
MPLBACKEND=Agg uv run --python 3.13 --extra scientific --extra test python -m pytest tests/ -q
# Focused diagnostic: disable the package-wide coverage requirement
MPLBACKEND=Agg uv run --python 3.13 --extra scientific --extra test python -m pytest tests/test_core_coordinates.py -q --no-cov
# Current collected case IDs (collection is not test execution)
MPLBACKEND=Agg uv run --python 3.13 --extra scientific --extra test python -m pytest tests/ --collect-only -q --no-cov
```

These are recipes, not evidence they were exercised. The configured coverage
floor is 85% across the package; no per-module percentage or complete correctness
is guaranteed. Parametrization and skips mean function counts, collected case
counts and executed pass counts are different measurements. Do not use a saved
index count as a current gate result.

## Test isolation and assertions

`conftest.py` changes each test's working directory to `tmp_path`, resets
visualization config before it and resets again in `finally`. This contains
default relative renderer output even if a test itself calls `reset_config()`.
No fixture should scan or delete repository `output/`. Explicit output paths
must also be test-owned; assert actual returned containment and generated files.

Prefer exact numerical oracles, meaningful geometry/resource/error behavior and
real rendered artifacts. Failure injection may exercise an explicit error path;
mock echoes and incidental default/source-text assertions are not consumer proof.
Test names containing `coverage`, `comprehensive` or `real` describe historical
organization, not a guarantee that every branch or mathematical claim is covered.

Core/geometry cases must distinguish common-offset gauges, full integer quadray
versus FCC/IVM images, exact fractions versus floating boundaries, native versus
XYZ volume, and vertex mean versus solid mass centroid. Renderer cases must
respect optional dependency/backend requirements and figure ownership on errors.
Do not infer visual legibility from file existence alone.

Consult [API contracts](../docs/api/index.md) when adding or updating cases.
The active coordinated assignment may defer all verification until integration;
do not run these commands automatically.
