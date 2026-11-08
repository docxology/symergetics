# Paper Work: Source, Rendering and Evidence Contracts

- [markdown/](markdown/README.md): twelve numbered editable sections.
- [render/](render/README.md): ReportLab renderer and figure lookup.
- [validation/](validation/README.md): structural/image diagnostics and source inventory.
- `output/`: generated paper artifacts; never hand-edit.
- [Style](STYLE_GUIDE.md), [quick start](QUICK_START.md) and
  [integration](INTEGRATION_GUIDE.md): authoring and operational contracts.

Use explicit Python 3.13 for a fresh scientific workflow without changing the
interpreter pin. `generate_paper.py` is broad orchestration that can clear
outputs, run tests and regenerate examples, not a PDF-only entry point.
Render/check only when assigned, in isolation, preserving user outputs.
Historical reports and immutable deposits are not current verification.
