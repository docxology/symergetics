# paper/ — Agent Notes

Synergetics research-paper pipeline (verified 2026-08-29):

- `markdown/` — section sources (`00_title.md` …)
- `render/` — PDF renderer (integrates visualizations from `output/`)
- `validation/` — paper validation tooling; `validation_report.md` at this level
- `output/` — timestamped rendered PDF + figure trees (build artifacts)
- `generate_paper.py`, `improve_paper_visualizations.py` — pipeline scripts
- `STYLE_GUIDE.md`, `QUICK_START.md`, `INTEGRATION_GUIDE.md`, `README.md` — contracts

**Invariants:** `output/` contents are regenerable build artifacts (timestamped
filenames); markdown sections are the editable source. Run validation before
rendering.
