# Test evidence and count semantics

Source review base: `8ab949e4932537ceee1af93b08196fe21bf4ee85` (2026-10-13).

Reproduce the inventory without importing test modules:

```bash
python paper/validation/test_inventory.py > /tmp/test_inventory.json
diff -u paper/validation/test_inventory.json /tmp/test_inventory.json
```

The checked inventory counts **1,017 test-prefixed AST function definitions
across 39 Python files** under `tests/**/test_*.py`. It includes nested
functions; it does not claim that every definition is discoverable by pytest.
Its input digest is `84ccefc608ec85b1d3ab5273440e3b25970d34523b4be9800d2fee73e97fba7d`.
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

Current collection receipt (2026-10-13, Python 3.12): **1,030 collected cases**
using the collection command above at the recorded test-source digest.

Collection does not demonstrate execution, pass rate, numerical accuracy or
coverage. The existing `STATUS.md` records a historical 2026-09-05 execution:
997 passed, 20 skipped, 85.85% coverage. That is a dated receipt, not a current
result and not evidence that all mathematical operations preserve precision.

The manuscript's older 757/953 claims lack revision-bound source inventories
and used "functions" and "cases" interchangeably. They are superseded in the
editable sections by these explicit semantics. The first unrestricted run on the original test-source digest was interrupted
in the pre-existing `test_invalid_dpi_handling` after 549 passes: its
10,000-DPI image request spent minutes encoding an extremely large image.
A bounded original-source run yielded 996 passed, 20 skipped, 1 deselected,
85.91% coverage. Those receipts belong to the earlier digest
`54da74ac354e2c205ccf4888ff787b37d3b40c073bfcfcab5b4d018118450e34`.

The resource boundary now rejects oversized nominal canvases before figure
construction, checks draw and export DPI, and applies configuration atomically.
The formerly stalled test requires early `ValueError`; it is no longer excluded.
Additional controls cover scaled figures, low export DPI with huge draw buffers,
non-finite dimensions, numeric strings and accepted publication sizes.
Fresh unrestricted execution receipt (2026-10-13, Python 3.12, current test-source
digest): **1,010 passed, 20 skipped, no deselections**, **85.97% coverage**
(85% floor reached), 55.89 seconds, using the full-suite command above.
Test-generated output artifacts were restored after verification.

Historical rendered PDFs and
Zenodo deposits are preserved; they do not contain this source correction.
