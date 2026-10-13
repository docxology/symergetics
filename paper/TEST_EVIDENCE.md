# Test evidence and count semantics

Source review base: `8ab949e4932537ceee1af93b08196fe21bf4ee85` (2026-10-13).

Reproduce the inventory without importing test modules:

```bash
python paper/validation/test_inventory.py > /tmp/test_inventory.json
diff -u paper/validation/test_inventory.json /tmp/test_inventory.json
```

The checked inventory counts **1,013 test-prefixed AST function definitions
across 39 Python files** under `tests/**/test_*.py`. It includes nested
functions; it does not claim that every definition is discoverable by pytest.
Its input digest is `54da74ac354e2c205ccf4888ff787b37d3b40c073bfcfcab5b4d018118450e34`.
Each file has its own SHA-256 in [the generated inventory](validation/test_inventory.json).
Counts change with source revisions; regenerate rather than replacing prose
with another floating count.

Pytest collection is a separate measurement: parametrization can create
multiple collected cases per definition, while skips and collection rules can
exclude definitions. Use:

```bash
uv run --python 3.12 --extra test --extra scientific pytest tests/ --collect-only -q --no-cov
uv run --python 3.12 --extra test --extra scientific pytest tests/ -q
```

Current collection receipt (2026-10-13, Python 3.12): **1,017 collected cases**
using the collection command above at the recorded test-source digest.

Collection does not demonstrate execution, pass rate, numerical accuracy or
coverage. The existing `STATUS.md` records a historical 2026-09-05 execution:
997 passed, 20 skipped, 85.85% coverage. That is a dated receipt, not a current
result and not evidence that all mathematical operations preserve precision.

The manuscript's older 757/953 claims lack revision-bound source inventories
and used "functions" and "cases" interchangeably. They are superseded in the
editable sections by these explicit semantics. No fresh full-suite pass or
coverage result is asserted by this correction. Historical rendered PDFs and
Zenodo deposits are preserved; they do not contain this source correction.
