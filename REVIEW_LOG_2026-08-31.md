# Review Log — 2026-08-31 (agent-ergonomics fleet pass)

Preflight: on `main`, in sync with origin; 88 pre-existing untracked files, all
under `output/` — treated as pre-existing, never staged.

Cold-start audit (read entry docs only first):
- (a) current status: FAIL before — no status file anywhere; now PASS via
  STATUS.md (commands included).
- (b) what to do next: FAIL before — no backlog; now PASS via TODO.md; AGENTS.md
  links it.
- (c) how to run verification: PARTIAL before — AGENTS.md said
  `uv run pytest tests/ -q` but did not warn that single-file runs fail the 85%
  coverage floor or that the suite is slow on this volume; now PASS with
  gotchas documented in AGENTS.md + STATUS.md.

Doc-surface sweep: relative-link check over README.md, AGENTS.md, docs/index.md,
docs/README.md, docs/AGENTS.md → 0 broken links (python script, this pass).
docs/architecture/AGENTS.md is drifted boilerplate from the template repo
(references infrastructure/, projects/, _generated/ — none exist here); fixed
this pass.

Changes: TODO.md, STATUS.md, REVIEW_LOG_2026-08-31.md (new); AGENTS.md,
docs/architecture/AGENTS.md (edited). No source code touched.
