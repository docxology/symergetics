# validation/

Validation tooling for the paper build (checks run before/with rendering).
Run from the repository root:

```bash
python paper/validation/validate_paper.py --output /tmp/validation_report.md
python -m unittest discover -s paper/validation -p test_validate_paper.py
python paper/validation/test_inventory.py
```

The manuscript gate requires the current twelve numbered sections and project-rooted
`output/` image files. Paths that escape `output/`, missing files, directories used
as images, and absent/malformed captions remain errors. Captions accept the style
guide's `**Figure N**:` form and the legacy `**Figure N:**` form after optional blank
lines. The focused tests use real temporary files and include negative controls.

The inventory is a separate source-definition count, with no test imports or
execution. See [test evidence](../TEST_EVIDENCE.md) for its hash-bound snapshot and
the distinction between definitions, collected cases and execution results.
