# STATUS — capabilities and dated evidence

## Current capability scope, not a gate receipt

The development checkout implements rational scalar arithmetic, rank-three
integer quadrays, nearest-lattice conversion, exact metrics, indexed live solid
analysis and file-producing visualization. See [API contracts](docs/api/index.md)
and [suite conventions](docs/synergetics-suite.md).

Version 1.4.0 introduces focused scientific poster sections, state-preserving
polyhedron transforms, exact plane arithmetic and source-bound verification.
Earlier release wheels retain their own interfaces. A version string is not a
source revision identifier. Release preparation does **not** assert that the
current tree is clean, on a particular branch, published or green; integrated
verification belongs to the revision and environment that produced a receipt.

Current commands are centralized in the [testing guide](docs/testing-guide.md)
and [suite guide](docs/synergetics-suite.md). Scientific work explicitly uses
Python 3.13; existing repository pins remain unchanged. Collection counts,
coverage and demo outcomes must be measured when needed, not inferred below.

## Suite review receipt — 2026-11-06 (historical)

The preceding review recorded the following outcomes in a prepared Python
3.13.17 scientific/test environment. These are archived reports, **not rerun
results for the current edits**:

- Full pytest suite: 1,344 passed, 12 skipped, 88.16% coverage against the 85%
  floor. Recorded command in that environment:
  `MPLBACKEND=Agg python -m pytest tests/ -q -o 'addopts=--strict-markers --strict-config --cov=symergetics --cov-report=term-missing --cov-fail-under=85'`.
- `python examples/suite_interoperability.py`: recorded 279 coordinate cases per
  implementation, 34 exact-volume cases and 216 FCC site comparisons.
- `python verify_suite.py --report /tmp/synergetics-suite.json`: recorded 13
  checks executed successfully in frozen temporary environments.
- `python run.py --examples-only`: recorded all ten ordinary demos succeeding;
  setup/tests were intentionally skipped by that command.
- The earlier review additionally reported a minimal core import and targeted
  Python 3.11 compatibility work. Those observations do not establish present
  optional-dependency compatibility or a universal supported interpreter matrix.

See [suite evidence boundaries](docs/synergetics-suite.md) and
[actual workflows](.github/workflows/verify.yml) for reproduction context.
The commands above intentionally preserve the historical invocation spelling;
use the linked current guides for explicitly selected Python 3.13 commands.

## Earlier receipt — 2026-09-05 (historical)

An earlier review recorded 997 passed, 20 skipped and 85.85% coverage for its
final tree. Other intermediate receipts in history used different counts and
coverage totals; they are not a current inventory. Past clean-tree/branch and
rendered-paper statements apply only to those snapshots. Generated `output/`
and `paper/output/` files are not current-run evidence merely because they exist.
