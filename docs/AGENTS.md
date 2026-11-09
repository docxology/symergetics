# Maintaining implementation and conceptual documentation

## Layout

[index.md](index.md) is the navigation hub. Implementation contracts live in
[api/index.md](api/index.md); architecture, examples and tutorials link to those
contracts. Mathematical guides define the quotient/metric/units. Research,
application and learning/philosophy pages discuss context, not extra APIs.
The separate paper pipeline lives under `../paper/`; manuscript status is under
`manuscript/`. Generated `output/` and `paper/output/` files are not hand-edited docs.

## Source-bound examples

Read actual source signatures and callers before editing recipes. Complete
imports and real return shapes matter. Do not invent methods, exception trees,
performance results or scientific validity. Use checkout/downloaded-wheel
installation, not PyPI; select Python 3.13 for scientific commands without
changing repository pins. Document 1.4.0 interfaces without extending older artifacts.

## Canonical contracts and navigation

Link to the API reference instead of repeating incompatible unit, inverse,
centroid or output contracts. Preserve explicit anchors and update inbound
fragment links after heading changes. Keep READMEs linked to their area index.
If an inbound link belongs to another owner, report the needed target change.
Dates/counts in receipts are historical, not live collection or green status.

## Evidence discipline

Report only exercised checks and actual artifact paths. Follow the current
assignment's verification restrictions; do not run parallel build/test/lint
passes during coordinated edits. A successful arithmetic example does not
validate a conceptual application or an entire backend matrix.
