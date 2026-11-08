# symergetics — source and documentation maintenance

## Orientation ladder (cold start)

1. Read [README.md](README.md) for installation and scope, then [docs/index.md](docs/index.md).
2. [STATUS.md](STATUS.md) separates present capabilities from dated receipts;
   it is not a current green-main claim.
3. [TODO.md](TODO.md) is the root backlog/history surface, not a test inventory.
4. Consult [architecture](docs/architecture/index.md) and scoped `AGENTS.md`
   before editing. Follow the current assignment's verification/ownership rules.

## Layout

- `symergetics/`: core, geometry, computation, utils and visualization source.
- `tests/`: real-data/computation tests; [commands](docs/testing-guide.md).
- `examples/`: actual demo scripts, many of which generate `output/` files.
- `docs/`: hand-written implementation and conceptual guides.
- `output/`: generated/user artifacts; never hand-edit or overwrite casually.
- `paper/`: separate manuscript and rendering pipeline; `paper/output/` is generated.

## Verification (read the gotchas before running)

Scientific commands explicitly select Python 3.13, without changing the
repository `.python-version`. The complete command and coverage policy live in
[the testing guide](docs/testing-guide.md#running-tests). In brief:

```bash
MPLBACKEND=Agg uv run --python 3.13 --extra scientific --extra test python -m pytest tests/test_core_coordinates.py -q --no-cov
MPLBACKEND=Agg uv run --python 3.13 --extra scientific --extra test python -m pytest tests/ -q
```

A targeted diagnostic uses `--no-cov` because the default 85% floor covers the
whole package. Neither command above is a receipt that it ran in this pass.
If a coordinated assignment defers builds/tests/lint/formatters until integration,
do not run them mid-flight. Report actual exercised probes separately.

`run.py --examples-only` skips setup and tests but regenerates `output/` and
writes logs/reports. Default orchestration setup installs extras into its
selected interpreter. Never use demos as harmless read-only inspection.

## Decision points

- Preserve exact native arithmetic; name float and lattice-projection boundaries.
- Keep four-label rank-three coordinates distinct from independent Euclidean axes.
- Preserve indexed triangle-fan surfaces and vertex-versus-solid centroid contracts.
- API changes require reference searches and migration of every affected caller.
- Maintain canonical contracts in [docs/api/index.md](docs/api/index.md), linking
  from recipes instead of duplicating incompatible definitions.
- New scientific poster sections are unreleased main, not 1.3.0 wheel behavior.
- Do not edit user-generated outputs, immutable releases or unrelated changes.
- Do not claim a green gate, current count or performance result from old prose.
