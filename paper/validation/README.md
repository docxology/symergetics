# Paper Validation: Structural Gates and Historical Inventories

## Structural and image diagnostics

Run from repository root in the Python 3.13 environment described by
[paper quick start](../QUICK_START.md):

```bash
.venv-paper/bin/python paper/validation/validate_paper.py --paper-dir paper --output /tmp/symergetics-paper-validation.md
.venv-paper/bin/python paper/validation/check_images.py --paper-dir paper --output /tmp/symergetics-paper-images.md
```

[validate_paper.py](validate_paper.py) requires the twelve numbered sections and
source-relative image paths resolving into repository `output/`. Absolute or
escaping paths, missing files, directories used as images and malformed/absent
captions are errors. Both `**Figure N**:` and `**Figure N:**` forms accept
optional blank lines. The renderer now resolves the same exact source-relative
path, without a basename fallback. See [figure integration](../INTEGRATION_GUIDE.md).

These are structural/style checks, not mathematical proof, a complete external
link audit or empirical validation. Warnings about decorative link style or
technical-term formatting do not establish a scientific defect. Conversely, a
clean report cannot establish correct units, readable images or defensible claims.

## Source inventory versus execution

[test_inventory.py](test_inventory.py) parses source definitions without test
imports or execution. Its checked [JSON snapshot](test_inventory.json) belongs
to its recorded input hashes. See [test evidence](../TEST_EVIDENCE.md) for the
distinction between definitions, collected cases and historical run results.
Do not relabel old counts or reports as current verification.

## Existing test controls

[test_validate_paper.py](test_validate_paper.py) uses real temporary files and
negative controls, including an actual missing-image CLI failure followed by a
valid source-relative target. Run it with
`python -m unittest discover -s paper/validation -p test_validate_paper.py`.
Combine structural diagnostics, reference review and isolated PDF inspection;
per-run CI reports, not a timeless count here, record successful execution.

[Paper source modules](../markdown/README.md) · [Writing contract](../STYLE_GUIDE.md)
