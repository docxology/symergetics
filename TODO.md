# TODO — backlog scope and historical maintenance

## Current backlog policy

No new feature backlog is inferred from an old receipt. Track an actionable
issue with its affected paths and observable acceptance criterion; completion
requires evidence for that issue, not a saved package-wide test count.
[STATUS.md](STATUS.md) records dated evidence and
[the testing guide](docs/testing-guide.md) supplies current commands.

## Historical completed maintenance — 2026-08/09

These entries describe the earlier maintenance pass, not unfinished work or
current gate status:

- [x] Added root status/backlog surfaces and orientation guidance.
- [x] Documented why a focused pytest run can fail package-wide coverage and
  when to use `--no-cov` for diagnostics.
- [x] Updated test navigation after the earlier module consolidation. Its old
  per-file totals are historical; collect current cases when an inventory is needed.
- [x] Changed the output-directory test to isolate its generated files rather
  than leaving repository output residue.
- [x] Recorded the then-current full-suite result. Consult dated status/history
  rather than interpreting that outcome as evidence for the present checkout.

## Unreleased documentation and visualization scope

Main documentation distinguishes focused scientific-poster selections from the
all-topic view, exact native measurements from floating display, and implemented
APIs from conceptual applications. Published 1.3.0 artifacts remain immutable.
The active integration process owns current gates and receipts; this page does
not mark an unexecuted gate complete or advertise historical counts as current.
