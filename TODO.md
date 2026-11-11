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

## 1.4.0 documentation and visualization scope

Documentation distinguishes focused scientific-poster selections introduced in
1.4.0 from the all-topic view, exact native measurements from floating display,
and implemented APIs from conceptual applications. Earlier assets are preserved by policy.
The active integration process owns current gates and receipts; this page does
not mark an unexecuted gate complete or advertise historical counts as current.

## Workflowz review — 2026-11-10

- [ ] **SYMERGETICS-20261110-01 — P1: Stop returning unrelated or empty analysis as completed results.** Replace the placeholder Scheherazade value and empty prime-factor result with analysis derived from the supplied `SymergeticsNumber`, or omit/ explicitly mark unavailable fields so callers cannot mistake them for valid findings. Evidence: `symergetics/computation/palindromes.py:407` branches on the input type, while `symergetics/computation/palindromes.py:408-409` hard-code `analyze_scheherazade_ssrcd(1)` and `{}`. Acceptance probe: from the repository root, run a focused Python behavioral check with the project’s declared dependencies that analyzes at least two distinct `SymergeticsNumber` inputs and verifies every returned Scheherazade/prime-factor field is either correctly derived from that input or absent/explicitly unavailable; include the focused palindrome tests with `--no-cov` as described by `AGENTS.md`. Dependencies/constraints: preserve the existing public result contract where possible, retain exact native arithmetic, and do not invent prime-factor semantics without checking the enclosing function and its callers; no Python-version or dependency changes are required by this finding.
