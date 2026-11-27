# Pinned Template render

Use a clean checkout with the tracked publication images under `output/publication/1.4.0/`. The preparation script checks exact arithmetic and quadray results against fixed examples, copies the ten retained images, and stages the canonical authored sections for both formats. It writes to fresh output directories, leaving the native ReportLab source and historical deposits intact.

```sh
uv sync --locked --python 3.13 --extra scientific
uv run --locked --no-sync python docs/paper-tools/stage_template.py
```

Use a separate [Template checkout](https://github.com/docxology/template/tree/87704feae0a4490ef7dad9c931d268e8c30273c2) at `87704feae0a4490ef7dad9c931d268e8c30273c2`, with Pandoc, pandoc-crossref and LaTeX from the [Template run guide](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/RUN_GUIDE.md). Set `symergetics_checkout` to the absolute project checkout path. Run from Template with an unused slot.

```sh
uv sync --locked --no-dev
mkdir -p projects/working
ln -s "$symergetics_checkout" projects/working/symergetics
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/symergetics --skip-manuscript-hydration
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/symergetics
```

Open the PDF under `output/pdf/`, the HTML at `output/web/index.html`, and the reports under `output/reports/` in the Symergetics checkout. Consult the [native quick start](QUICK_START.md) separately for ReportLab and source diagnostics.
