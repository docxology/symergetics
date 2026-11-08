# Changelog

## 1.3.0 — 2026-11-07

### Exact metrics and live solid analysis

- Add exact coordinate dot products, squared norms and squared distances, with native cancellation before floating conversion and range-safe floating norms/angles.
- Share stable indexed faces and declared triangle fans across volume, surface measures, Matplotlib, Plotly and animation. Analyze live affine rank, physical vertices, closure, nonplanarity and triangle-surface convexity.
- Add exact uniform-solid mass centroids, distinct from declared vertex means. Integrate concave surfaces and disjoint/nested shells using even-odd enclosure; self/mutual intersections remain an explicit input precondition.
- Preserve recoverable subnormal surface totals and use anchor-relative XYZ display coordinates for huge translations.

### Computed scientific visualization

- Apply real Plotly camera and surface/wireframe controls; preserve physical scale and report local-origin/live-geometry metadata.
- Replace synthetic poster curves, stale test/coverage claims and misleading rationality/frequency statements with computed exact arithmetic, integer Fibonacci ratios, prime-bound primorials, finite palindrome enumeration and bounded rational approximations.
- Distinguish tetrahedral layer populations, FCC graph shells, counts and uniform dilation. Report live edge/area/volume scaling and actual numerical-boundary examples.
- Close poster figures on both successful and failed saves. The mega-abstract API now accepts only its declared `title`, `output_path` and `backend`; previously ignored extra keywords raise rather than pretending to work.
- Keep comparative/dashboard box plots compatible with both the declared Matplotlib 3.5 floor and current 3.11 releases; avoid the removed `boxplot(labels=...)` keyword.

### Suite integration

- Extend the five-runtime interoperability example with 280 exact native-metric pairs and 15 conservative Python/JavaScript graph steps, using an independently enumerated shared FCC graph.
- Document native/default/QuadCraft units, vertex versus mass centroids, hard versus soft field methods, graph/clock ownership, formal proof scope and actual browser capability limits.

Verification: **1,467 passed, 12 skipped; 91.07% statement coverage**, above the existing 85% floor, on the exercised Python 3.13 scientific environment. Installed Python 3.11 consumers also rendered both comparative/dashboard plots with Matplotlib 3.11.2. See the [suite guide](docs/synergetics-suite.md) for cross-runtime evidence and limitations.

## 1.2.0 — 2026-11-07

### Exact numbers and live geometry

- Add `SymergeticsNumber.nth_root(value, degree, max_denominator=1000000)` and reuse it for `sqrt`.
- Preserve integer `Fraction` exponents, perfect rational roots and powers without a binary-float conversion, including huge integers, tiny fractions and negative odd roots.
- Promote integral scalars and NumPy-backed `Fraction` fields before arithmetic, comparison, powers/roots, sign normalization and coordinate snapping.
- Bound only irrational approximations by `max_denominator`; exact roots may exceed that cap. Decimal approximation is finite precision, not certified symbolic algebra or rounding.
- Add polyhedron `centroid_exact()` compatible with QuadMath's canonical rational centroid tuple, without integer coordinate snapping.
- Remove stale volume caching: public vertex mutations immediately affect exact volume calculations. Existing floor-snapped centroids remain distinct.

Exact-root compatibility: use `Fraction(1, 3)`, not the binary float `1/3`,
for an exact cube-root exponent. Non-real roots raise domain errors; tiny
irrational approximations may round to zero. Denominator limits apply only
to approximations, never to exact perfect roots.

### Coordinated verification and provenance

- Version execution receipts and record the three declared package versions, actual Git revisions, selected core/full profile and truthful source/worktree dirty state.
- Replace repository, scratch, tool and explicit interpreter paths with portable aliases in shared receipts.
- Reject enclosing-repository provenance and optionally require clean source while retaining generated-output dirtiness.
- Include the actual 30-game Chromium gate with `--browser`; missing browser tooling is a failure, never an implicit skip.
- Accept explicit refs for all three CI checkouts and upload the complete receipt plus browser diagnostics.
- Extend real interoperability to 37 exact-centroid cases, alongside 279 coordinates per implementation, 34 exact volumes and 216 independent FCC site comparisons.
- Document exact-root approximation limits, centroid transport, mutation semantics and the coordinated QuadMath 0.5.0 / QuadCraft v0.4.0 release.

## 1.1.0 — 2026-11-06

### Exact and spatial methods

- Preserve integral/Fraction/SymergeticsNumber precision through coordinate snapping, with halves away from zero; parse scientific rational strings without a lossy float conversion.
- Find the true nearest full integer-quadray point, using the regular BCC fast path or exact-rational LLL reduction and bounded closest-first enumeration for compatible skew embeddings. Canonical tuple order resolves exact ties; there is no heuristic candidate cap.
- Validate finite rank-three gauge-compatible embeddings with scaled arithmetic. Keep integer scaling, origin inversion and compatible integer maps exact; fractional spatial operations quantize explicitly.
- Correct named reflection planes and the continuous 4×3 pseudoinverse, including extreme finite unit scales.
- Add `enumerate_ivm_sites(size)` with graph-shell-radius semantics, canonical unique FCC sites and deterministic shell ordering. Radii 0/1/2/3 contain 1/13/55/147 sites.
- Reuse the actual IVM sites across ASCII, Matplotlib, Plotly, coordinate backgrounds and demo density/counts, replacing raw tuple aliases and non-FCC points.
- Propagate transformation callback errors instead of saving a misleading success image; reject unknown backends and close figures on exceptional exits. Initialize advanced dashboard dependencies on a cold first call.
- Render numeric XYZ annotations in graphical abstracts, remove the unused overlapping 2D axes and reserve space for the overall title.
- Use the existing exact denominator-based integrality convention in geometric-mnemonic analysis, preserving Python 3.11 support without newer `Fraction.is_integer()` methods or floating conversion.
- Present exact demo volume ratios as rational strings and request floating presentation explicitly for decimal primorial ratios, avoiding newer `Fraction` formatting APIs. Exercise all ten ordinary demos in both scientific CI matrix jobs.

### Orchestration and real tests

- Keep one selected interpreter for setup, installation verification, tests and demos. `--python` selects an existing interpreter; setup uses declared extras without bypassing externally managed environment protections.
- Continue requested stages after failures while retaining truthful stage/command results and a nonzero final status. Handle empty demo discovery and label existing output inventory honestly.
- Discover ordinary `*_demo.py` files without implicitly requiring sibling repositories, Node or a C++ compiler. Required failures in PNG, comprehensive and geometric-mnemonic demos now reach their process exit status.
- Correct exact-Fraction integration output, batch visualization result handling and high-radix positional demonstrations; register base-conversion matrices for batch visualization.
- Isolate tests with temporary working directories and visualization configuration reset before/after each test, including mid-test resets. Remove teardown that deleted unrelated output files and replace renderer/path mocks with real isolated behavior checks.
- Add `verify_suite.py`: frozen temporary Python environments, existing coverage gates, game checks, Release native tests, independent FCC enumeration, five actual mathematical implementations and documentation checks. It stops on failure and records unexecuted checks in an optional JSON receipt.

### Distribution, CI and compatibility

- Make NumPy a base dependency because public coordinate imports require it; scientific/test tools remain extras. Exercise minimally installed wheels without plotting dependencies.
- Exclude generated visualization/paper output, logs and local PDFs from source distributions.
- Align `__version__`, project/lock metadata and the versioned release at 1.1.0, following the legacy GitHub `1` release. Align license metadata with the existing Apache-2.0 LICENSE; this is not a new license grant.
- Align tooling/classifiers with the existing Python ≥3.11 requirement; keep repository interpreter pins unchanged. CI exercises Python 3.11/3.13 plus cross-repository integration.
- Document gauge equivalence, orientation reversal and scale conversion to QuadCraft, rational volume units, transport precision and distinct inverse contracts.

### Dependency security

- Selectively update the eleven packages implicated by all 98 open `uv.lock` Dependabot alerts, plus resolver-required dependencies. Every newly resolved version was checked against every captured affected range; none remained affected.
- Require Pillow ≥13.0.0 in the scientific extra, pytest ≥9.0.3 in test tools and Black ≥26.10.1 in development tools. Other patched lock entries cover certifi, filelock, fonttools, idna, Pygments, requests, urllib3 and virtualenv.
- Advisory evidence includes [certifi GHSA-6q48-6c7w-9gw2](https://github.com/advisories/GHSA-6q48-6c7w-9gw2), [pytest GHSA-6w46-j5rx-g56g](https://github.com/advisories/GHSA-6w46-j5rx-g56g) and [Black GHSA-cg8m-r9f2-5wm2](https://github.com/advisories/GHSA-cg8m-r9f2-5wm2). This is a bounded lockfile advisory check, not a claim of complete vulnerability absence.

From sibling checkouts, run `python3 verify_suite.py --report /tmp/synergetics-suite.json`. To regenerate ordinary demonstrations in a prepared scientific environment, run `uv run --python 3.13 --extra scientific --extra test python run.py --examples-only`. See [suite contracts and receipts](docs/synergetics-suite.md).

Release verification: **1,344 tests passed, 12 skipped, 88.16% coverage**; all **10 ordinary demos** and all **13 coordinated checks** passed with the patched scientific lock. A minimal installed wheel exercised exact coordinate inversion and FCC enumeration without Matplotlib, SciPy or Plotly installed.
