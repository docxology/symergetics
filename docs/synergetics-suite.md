# Synergetics suite: coordinates, units and interoperability

The projects remain independently usable. Integration exchanges mathematical
values, not imports between production packages or a shared dependency cycle.

| Project | Responsibility | Implementation entry points |
|---|---|---|
| [QuadMath](https://github.com/docxology/QuadMath) | Mathematical specification, exact determinants, lattice search, optimization and Lean formalization | [`core/quadray.py`](https://github.com/docxology/QuadMath/blob/main/src/quadmath/core/quadray.py), [`lattice/conversions.py`](https://github.com/docxology/QuadMath/blob/main/src/quadmath/lattice/conversions.py), [`SPEC.md`](https://github.com/docxology/QuadMath/blob/main/SPEC.md) |
| [symergetics](https://github.com/docxology/symergetics) | Rational arithmetic, integer quadray geometry, polyhedra, transformations and visualization | [`core/coordinates.py`](../symergetics/core/coordinates.py), [`geometry/polyhedra.py`](../symergetics/geometry/polyhedra.py), [API reference](api/index.md) |
| [QuadCraft](https://github.com/docxology/QuadCraft) | Interactive games and native/browser/Python coordinate consumers | [shared JavaScript](https://github.com/docxology/QuadCraft/blob/main/games/4d_generic/quadray.js), [native C++](https://github.com/docxology/QuadCraft/blob/main/src/core/coordinate/Quadray.h), [Python](https://github.com/docxology/QuadCraft/blob/main/games/src/space/quadrays.py) |

## One spatial point, multiple coordinate representatives

Four quadray components describe **three** spatial dimensions:

$$q \sim q+t(1,1,1,1),\qquad
  \operatorname{canonical}(q)=q-\min(q)(1,1,1,1).$$

A fixed component sum is not required. Canonical tuples have nonnegative
components and at least one zero. Preserve this equivalence when computing
lengths, distances, centroids, transforms, keys and physical grid boundaries.
Raw board labels are a separate contract: a game may intentionally distinguish
four-index labels even when their geometric projections coincide. Do not
silently normalize those labels during save/load or integer construction.

The full integer-quadray image is also not the single FCC/IVM sublattice.
QuadMath's IVM centers satisfy `sum(canonical(q)) % 4 == 0`; their twelve
neighbor moves are permutations of `(2,1,1,0)`. These moves preserve the residue
class modulo four. An arbitrary integer quadray grid need not be connected
under that neighbor relation.

## Embedding and unit conversion

QuadMath's `DEFAULT_EMBEDDING` and symergetics' `urner_embedding()` use

$$X=a-b-c+d,\qquad Y=a+b-c-d,\qquad Z=a-b+c-d.$$

QuadCraft's native/shared-game embedding uses

$$(x,y,z)_\mathrm{craft}=(X,Z,Y)/\sqrt{2}.$$

The Y/Z swap **reverses orientation**. It is not merely a rotation. Lengths
scale by $1/\sqrt{2}$, squared distances by $1/2$, and unsigned Cartesian
volumes by $1/(2\sqrt{2})$. Do not compare XYZ arrays across projects without
this mapping. Historical browser experiments can expose additional raw-XYZ
units; their explicit conversion APIs remain distinct from the shared games.

For either Python project's scale-1 embedding,

$$\|q\|^2=4\sum_i q_i^2-\left(\sum_i q_i\right)^2.$$

Use QuadMath `dot_exact`, `squared_norm_exact`, `squared_distance_exact` and
symergetics' corresponding coordinate methods for exact rational metrics.
Native combinations are formed before floating conversion; finite embedding
floats mean their stored binary rationals. This resolves cancellation such as
the `2**60` one-count cases that a float-first inner product erases.
Range-safe floating norms/angles remain distinct from exact squared measures.
Floating XYZ inputs cannot recover information already lost before API entry.

### Exact volume versus Euclidean volume

Both Python projects return `Fraction(abs(det), 4)` from their projected and
Ace tetrahedron determinants. The origin plus three coordinate basis vectors
has IVM volume `Fraction(1, 4)`; the regular tetrahedron with vertices
`(0,0,0,0)`, `(2,1,0,1)`, `(2,1,1,0)`, `(2,0,1,1)` has IVM volume `1`.
Not every integer-coordinate tetrahedron has an integer volume.

At the scale-1 Python embedding, Euclidean volume is `(8/3) * IVM_volume`.
QuadCraft's corresponding Euclidean volume is `4/(3*sqrt(2)) * IVM_volume`.
QuadMath's Cayley–Menger IVM helper applies `S3 = sqrt(9/8)` for the
**sphere-radius-one** convention: the unit tetrahedron edge is `2`. Rescale
QuadMath's default embedding by `1/sqrt(2)` before using that conversion to
compare with the exact IVM determinant. Uniform unit changes must scale volume
cubically and circumradius/inradius linearly.

## Choose the appropriate inverse

| API | Contract |
|---|---|
| QuadMath `xyz_to_quadray_canonical` | Strict exact lattice membership using rational arithmetic. Floats mean their exact binary rational values. Off-image points raise; no tolerance or snapping. |
| QuadMath `quadray_from_xyz` | Float inverse with component-wise half-up snapping. Not necessarily the XYZ-nearest site. Keep this behavior distinct from spatial nearest search. |
| QuadMath `nearest` / `within_radius` | Queries on the IVM-center lattice, in default embedding units and within the documented shell limit. |
| symergetics `QuadrayCoordinate.from_xyz` / `xyz_to_quadray` | Nearest integer-quadray point in the supplied compatible embedding; canonical tuple order resolves equal-distance ties. |
| symergetics `xyz_to_quadray_matrix` | Continuous zero-sum minimum-norm 4×3 inverse of the effective embedding, using rational cofactors before final float entries; not exact nearest search, integer quantization or min-zero normalization. |
| QuadCraft `fromCartesian` / Python `from_xyz` | Continuous inverse modulo the common-offset gauge; not an FCC nearest-neighbor operation. |
| QuadCraft `toIVM` | Canonical component rounding; not a closest-point search on the FCC/CCP sublattice. |

symergetics' fractional scaling and named plane reflections perform the
Euclidean operation and then quantize to an integer quadray. Plane projection
uses exact represented rational normal/point values (including NumPy floating
scalars) before a single nearest-lattice snap. A snapped result can leave the
plane. `reflect` accepts `origin`/`a`/`b`/`c`/`d`, not an arbitrary normal.
Quantization can change distances and make a transform non-invertible; it is
not exact continuous geometry. Integer scaling and origin inversion remain exact.
Polyhedron transforms call `with_vertices`: concrete type/topology survive,
the new vertex list is owned, and other state is shallow-shared unless the
subclass overrides that protocol.

## Exchange exact data without losing precision

Use canonical integer tuples and explicitly identify the embedding and units.
For JavaScript consumers, numeric coordinates must fit `Number`'s safe-integer
range. Exchange larger integers as decimal strings, and exact rational volumes
as decimal-string numerator/denominator pairs. This is transport guidance, not
a new serializer API; the shared JavaScript geometry uses `Number`, not
arbitrary-precision geometric arithmetic. Do not convert large strings to
`Number` and call the result exact.

### Exact vertex means and uniform-solid mass centroids

QuadMath `centroid_exact(points)` and symergetics
`SymergeticsPolyhedron.centroid_exact()` return the same minimum-zero tuple of
four `Fraction` components. They average the physical points exactly, including
independent common-offset representatives and integers beyond floating range.
The primitive tetrahedron's centroid is `(1/4, 1/4, 1/4, 0)`. Empty exact
centroids raise `ValueError`.

Exchange each rational component as decimal-string numerator/denominator pairs.
Do not feed these fractions through symergetics' integer-snapping coordinate
constructor. Existing snapped centroids remain distinct: QuadMath uses half-up
component rounding; symergetics uses floor snapping.

These are **vertex means**, not generic solid centers of mass. For example,
moving one Cube vertex to `(2,0,0,0)` gives vertex mean `(1/8,0,0,0)` but
uniform-solid centroid `(1/4,0,0,0)`. symergetics
`volume_centroid_exact()` integrates consistently oriented triangle-surface
moments. `analyze_geometry()` reports both centers and live rank, closure,
nonplanarity, triangle-surface convexity, area and volume in both unit systems.
The common indexed mesh also drives Matplotlib, Plotly and animation.

Nonintersecting disconnected shells add; nested shells use even-odd enclosure
(cavities subtract, islands add). Closure does not certify absence of
self/mutual intersections: valid embedded triangle surfaces are a precondition.
Zero-volume solids have no mass centroid. Nonplanar polygon faces mean their
declared triangle fan, not an unspecified alternative surface.

### Simulation operators are not interchangeable

For an undirected graph with Laplacian `L` and full maximum degree `d_max`,
QuadMath's conservative heat and QuadCraft's `ConservativeGraphField` use
`u_next = u - alpha * L*u/d_max`, with `0 <= alpha <= 1`.
Each edge contributes equal/opposite flux. In exact arithmetic this preserves
ordinary mass and constants, obeys the maximum principle and dissipates
`0.5 * sum_edges((u_i-u_j)**2)`. Floating implementations add rounding error.
Symmetric-normalized heat instead preserves the `sqrt(degree)`-weighted mode,
not ordinary mass on irregular boundary graphs. Neither operator's stationary
distribution should be relabeled as the other's.

The finite binary64 implementation keeps full-graph maximum degree under masks
and reserves headroom for accumulated incident flux **and** the old state,
not merely individual edge differences. Larger-magnitude updates use exact
accumulation before rounding; no exact-JavaScript or universal bit-identity
claim follows. The boundary consumer reports its actual case outcomes rather
than a timeless case-count constant.

| Consumer | Deliberate contract |
|---|---|
| QuadMath `IVMField.harmonic_extension` | Hard Dirichlet values; every component must be anchored, with failure atomicity and separate component scales |
| Regularized `IVMField.learn` | Soft observation penalty via observation-response weights, not hard Dirichlet data; zero regularization evaluates the kernel target, every component needs an observation, and nonfinite/nonconvex/backward-inaccurate solves fail atomically |
| Majority trajectory fitting | Exact integer/Fraction residuals, binary-float alpha decisions and one final mean-square conversion; noisy fractional observations stay fractional |
| SimAnt terrain graph | Eight raw reversible axis neighbors, terrain masks and distinct occupancy aliases; not the twelve-neighbor physical FCC graph |
| SimAnt pheromones | Conservative transport separated from deposition, evaporation, caps and Float32 storage; HUD accounts for the nonconservative terms |
| Shared continuous game loop | Fixed callback seconds; interval configuration and browser timestamps remain milliseconds |
| Doom | Sixty-Hz simulation; input changes settle elapsed time first, held fire belongs to captured input and respawn starts a fresh clock |
| Asteroids | Swept moving-target contacts in the shared physical metric, including periodic images, lifetime and fragment remainder |
| Minecraft/Mahjong | Actual projected footprint and reverse stable paint order; preserve raw cell/tile identity |

The interoperability example supplies an explicit FCC graph to the generic
JavaScript field kernel before comparing it with QuadMath. It does not pretend
SimAnt's raw terrain graph is that FCC graph.

Dense learning response weights remain double precision: a backward-error
check is not a forward-accuracy guarantee for ill-conditioned systems. Preserve
physical mixed-magnitude targets rather than treating small observations as
zero after global normalization.

### Formal scope and graphical evidence

For integer coordinates, QuadMath's dependency-free Lean development proves the
common-offset physical quotient and canonical uniqueness, exact distance positivity/congruence,
independent-gauge determinant invariance, the signed XYZ/native volume bridge,
and universal shell lower bounds. `lake build` and `leanchecker` are separate
CI gates. A universal shell-count formula and BFS equivalence remain outside
the proved scope; floating simulation and browser gameplay are not Lean proofs.

Scientific figures declare their physical scale and use live geometry.
The mega abstract computes finite Fibonacci ratios, prime-bound primorials,
palindrome counts and bounded rational approximations; it contains no copied
test counts or universal rationality claim. Irrational quantities remain
approximations. Sparse lattice planes preserve exact native indices while
displaying local-origin cells, rather than allocating a dense huge-span image.

## Scientific poster sections

These graphical-abstract interfaces are introduced in the coordinated
0.7.0 / 1.4.0 / v0.6.0 releases. Earlier assets retain their release-specific APIs.

| Surface | Appropriate use | Important boundary |
|---|---|---|
| QuadMath `quadmath.viz.abstracts.render_quadray_abstract` | Vector/raster explanation of the rank-three quotient, native basis and all twelve central FCC contacts | Scale-one XYZ: basis rays have length $\sqrt3$, contact centers are $2\sqrt2$ apart and sphere radius is $\sqrt2$ |
| symergetics `create_mega_graphical_abstract` | Readable computed overview or a focused geometry/arithmetic/patterns view | `all` retains seventeen topics and needs poster-scale viewing; ordinary reading should use a focused selection |
| symergetics `plot_polyhedron_graphical_abstract` | Live solid topology, XY edge projection, units and vertex-versus-mass centroids | Exact metadata survives bounded display annotations; embedded, non-self-intersecting surfaces remain an input precondition |
| QuadCraft browser portfolio | Actual input/state/rendering evidence | Static scientific figures do not establish gameplay behavior; use the live Chromium gate |

From a current symergetics checkout environment with scientific extras:

```python
from symergetics.visualization.advanced import create_mega_graphical_abstract

overview = create_mega_graphical_abstract(
    section="overview", output_path="overview.svg"
)
geometry = create_mega_graphical_abstract(
    section="geometry", output_path="geometry.svg"
)
poster = create_mega_graphical_abstract(
    section="all", output_path="complete_poster.svg"
)
```

Selections are `overview`, `geometry`, `arithmetic`, `patterns` and `all`.
An omitted title follows the section; an explicit custom title is retained.
Returned metadata names the actual selected panels; returned data contains
their computed quantities and units. The
[comprehensive demo](../examples/comprehensive_visualizations_demo.py) writes
separate group outputs and JSON sidecars with paths relative to each sidecar.
Use a checkout at the recorded revision or the explicitly versioned GitHub
wheel; the package is not published on PyPI. Scientific footers use the UTC
date of `SOURCE_DATE_EPOCH` when supplied, otherwise the local date. Registry
`generated_at` instead records actual UTC export time; filename-inferred
generator/scale/resolution hints are not verified scientific provenance.

Solid abstracts honor `show_edges`, `show_faces`, `show_vertices`,
`show_volume_ratios` and `show_coordinates`. XY projection preserves declared
edges, including overlaps, rather than inventing a projected convex hull.
Disabling optional sections removes their panels; it does not leave empty
axes. Exact Python metadata and display strings have different contracts:
bounded annotations do not truncate exact measurements or vertex/edge data.
Serialize rational metadata with an explicit transport policy, not a blind
float conversion.

QuadMath's regression diagnostics are separate mathematical models, not four
biological states or a physical Quadray trajectory. They distinguish sample
gradient products from likelihood-score Fisher information, report recorded
states and stopping reasons, and mask infeasible categorical probabilities.
A sampled minimum, ideal entropy-Hessian trace and numerically evaluated
variational objective are different quantities and are labeled accordingly.

## Statistical samples and scaling diagnostics

QuadMath's `sample_mean(x)` accepts a nonempty finite one-dimensional sample,
using compensated accumulation with an exact overflow-boundary fallback.
Resampling callbacks must return finite scalars; the default bootstrap and
jackknife statistic is this mean. Singleton descriptive standard deviation
remains undefined (`NaN`), and truly unrepresentable descriptive spread can be
infinite. Student-t tails, representable confidence intervals and sample
moments are numerical contracts, not universal inference validity.

`scaling_fit(sizes, times)` still returns `(slope, intercept, r2)`.
Use `scaling_predict(sizes, times, query_sizes)` for predictions from that same
fit: paired observed references, centered/scaled log-ratio OLS, a residual
relative intercept and binary-exponent evaluation retain narrow extreme-size
variation. Do not reconstruct predictions with
`exp(intercept + slope*log(query))`; the global intercept is a reporting
coefficient. Observed references do not force a noisy fit through one sample,
and fitted coefficients describe measurements rather than proving complexity.
See [QuadMath statistics](https://github.com/docxology/QuadMath/blob/main/src/quadmath/stats/statistics.py)
for finite/positive/vector input and representability errors.

## Coordinated software releases

| Project | Version | Release notes |
|---|---|---|
| QuadMath | 0.7.0 | [Range-safe statistics and scientific diagnostics](https://github.com/docxology/QuadMath/releases/tag/0.7.0) |
| symergetics | 1.4.0 | [Stateful geometry, faithful figures and bound verification](https://github.com/docxology/symergetics/releases/tag/1.4.0) |
| QuadCraft | v0.6.0 | [Owned simulation, native capture and gameplay evidence](https://github.com/docxology/QuadCraft/releases/tag/v0.6.0) |

Release preparation is dated 2026-11-09. These are the intended versioned release
destinations, not evidence that publication or this revision's CI has completed.
Earlier GitHub assets are preserved by project policy; GitHub hosting alone
does not make them platform-immutable. Consult the
[suite workflow index](https://github.com/docxology/symergetics/actions/workflows/suite.yml)
and [package workflow index](https://github.com/docxology/symergetics/actions/workflows/verify.yml)
for actual current runs. Dated evidence below is historical.

## Complete suite verification

Prepare QuadCraft's [locked Chromium tooling](https://github.com/docxology/QuadCraft/blob/main/games/tests/browser/README.md),
then run from the symergetics checkout:

```bash
python3 verify_suite.py --browser --report /tmp/synergetics-suite.json
```

The [verifier](../verify_suite.py) requires git, bash, uv, Node.js/npm, CMake,
a C++17 compiler and the media prerequisites of the repository tests. It creates
temporary locked Python environments and a native Release build, preserving
existing project environments and interpreter pins. `--browser` includes the
actual 30-game Chromium gate; it requires prepared tooling and never silently
skips a missing browser. Without that flag, the explicitly labelled `core`
profile runs the numerical/native/documentation checks only.

The schema-2 JSON receipt records the selected profile, commands, durations,
return codes and unexecuted checks. Each repository binding records
`requested_ref`, its `requested_ref_revision`, optional `expected_revision`,
and `before`/`after` states with actual `revision`, declared `version`,
`source_dirty`, `worktree_dirty`, `source_paths`, `input_count` and opaque
`input_sha256`. `--quadmath-ref`, `--symergetics-ref`, `--quadcraft-ref` must
resolve to their checkout HEAD; the corresponding `--*-expected-revision`
arguments require a full commit object ID equal to HEAD.

Raw integrity and canonical Git cleanliness are different contracts.
`input_sha256` binds raw relative filenames, file kinds, executable bits,
contents and staged index entries; symlinks bind link text, not target bytes.
Cleanliness uses Git's clean filters/object format and `core.filemode`,
checking tracked contents even when hidden by assume-unchanged. Inspection
does not refresh the on-disk index or write Git objects. CRLF/filter-equivalent
bytes or mode differences ignored by Git policy can be clean, but any raw
input or HEAD drift during execution still fails. Fingerprints are opaque
within one implementation/execution, not cross-version content-equivalence IDs.

**Every tracked file is an input**, including tracked generated manuscript
assets under output directories. Exclusions apply only to nonignored
untracked products: QuadMath `quadmath/output/`; symergetics `output/` and
`paper/output/`; no QuadCraft output prefix. A selected untracked report path
is also excluded; ignored untracked builds are not inputs. Actual exclusions
appear in `untracked_output_prefixes` and `untracked_report_exclusions`.
`worktree_dirty` still exposes output changes; an enclosing repository cannot
stand in for a missing independent checkout. Shared command/cwd fields use
logical root/tool aliases, not personal absolute paths.

The verifier rejects empty plans and empty/duplicate check names, then stops
at the first failed command. A successful receipt has no unexecuted checks
**within its selected profile** and unchanged source integrity. Native CTest
uses `--no-tests=error`, so an empty test inventory fails rather than passing.
`--require-clean` requires clean inputs before **and after** execution, not
merely a clean start. Run release verification in separate exact-ref clean
checkouts: do not discard local user outputs to satisfy the gate. Ordinary
stable dirty checkouts remain usable without this flag but cannot certify a
clean release. Optional `--quadmath-root`, `--quadcraft-root` and `--python`
support non-sibling checkouts and explicit interpreters.

These are **editable-source checks**, not proof of installed wheel/sdist behavior.
The package workflows produce per-Python artifacts
`quadmath-distributions-python-3.11` / `-3.13` and
`symergetics-distributions-python-3.11` / `-3.13`. Their layout is:

- `original/`: source-built wheel and source archive;
- `rebuilt/`: wheel built from the extracted source archive;
- `proof/`: runner/build-tool versions, resolved installed packages/runtime
  constraints, SHA256 checksums and JSON results from both isolated installed
  consumers outside checkouts.

The symergetics consumers install only core runtime dependencies and use
isolated imports; optional visualization behavior is covered separately.
Frozen runtime locks do not pin all PEP517/publication backends: resolved
versions are captured, not universally fixed. Checksums identify actual files;
neither passing consumers nor `SOURCE_DATE_EPOCH` promise byte-identical builds
across arbitrary toolchains. Release evidence must bind the final refs,
receipts and distributed assets after those gates actually run.

The [suite workflow](../.github/workflows/suite.yml) runs the full profile with
clean-source enforcement and uploads the receipt plus browser diagnostics.
Manual dispatch accepts `quadmath_ref`, `symergetics_ref` and `quadcraft_ref`
branches, tags or commit hashes, so a release can verify a fixed three-repository
combination instead of moving `main` branches.

The documentation stage scans authored Markdown across all three repositories,
including internal module/game guides and manuscript sources. Dependency,
build and generated-output directories are pruned from discovery; linked
targets there still resolve on demand. Missing/empty roots fail, as do dangling
same-page or cross-file fragments. This is not an external-URL crawler or
proof of mathematical claims.

Separate suite stages validate symergetics manuscript structure, source-relative
figure assets and captions. Its package CI also exercises real-image renderer
regressions and produces a current PDF artifact. Those gates preserve raster
aspect, authored captions and exact file identity; they do not establish
empirical validity or replace visual inspection.


Ordinary scientific demonstrations remain independent:

```bash
uv run --python 3.13 --extra scientific --extra test python run.py --examples-only
```

`run.py` discovers only standalone `*_demo.py` scripts. Its selected existing
interpreter is used for setup, verification, tests and demos; `--python` may
select another interpreter. Requested stages continue after failure, but the
final report and exit status retain failed work. Generated output is an
inventory, not proof that every file was created by the latest run.

QuadCraft's [separate real Chromium gate](https://github.com/docxology/QuadCraft/blob/main/games/tests/browser/README.md)
uses a directory-local Playwright lock. Ordinary unit-test discovery does not
implicitly install or execute browser tooling.

## Run the cross-repository example

With sibling checkouts and Node.js plus a C++17 compiler available, run from
`symergetics/`:

```bash
uv run --python 3.13 --extra scientific --extra test python examples/suite_interoperability.py
```

For non-sibling checkouts or alternative executable names:

```bash
uv run --python 3.13 --extra scientific --extra test python examples/suite_interoperability.py \
  --quadmath-root /path/to/QuadMath --quadcraft-root /path/to/QuadCraft \
  --node node --cxx clang++
```

The [example](../examples/suite_interoperability.py) imports the actual Python
implementations, invokes the actual shared JavaScript module and compiles a
small adapter against the actual native C++ header in a temporary directory.
It checks signed/gauge-equivalent representatives, normalization, inversion,
Cartesian norms, distinct-point distances, explicit unit mapping, independently
enumerated FCC graph balls, agreement between exact volume formulations and
continuous rational centroids beyond binary64's range and integer precision,
plus NumPy-backed rational ingress before fixed-width arithmetic can wrap.
It also checks 280 native exact-metric pairs against an independent integer
XYZ oracle and 15 conservative steps on the same explicit FCC graph in the
Python/JavaScript kernels. A historical exercised run observed maximum mass
drift `1.4210854715202004e-14`; that is not a current receipt or exact-JavaScript claim.
The current consumer additionally returns `finite_boundary_outcomes` and derives
`finite_boundary_cases` from those records: extreme degree-11/18 stars, both
signs, full/single/disabled masks, isolated values, subnormal/maximum ties,
tiny coupling and signed-zero zero-coupling identity.
Missing runtimes and mismatches fail the command; they are not silently skipped.
It neither launches the OpenGL game nor proves browser rendering or the entire
mathematical library.

Python 3.13 is the selected scientific environment for current reproduction;
the historical receipts below identify previously exercised environments.
Existing repository pins are unchanged. Locked scientific dependencies can
require source builds under newer interpreters.

For graphics-free native verification:

```bash
cmake -S /path/to/QuadCraft -B /tmp/quadcraft-core -DQUADCRAFT_BUILD_GAME=OFF -DBUILD_TESTING=ON
cmake --build /tmp/quadcraft-core
ctest --test-dir /tmp/quadcraft-core --output-on-failure --no-tests=error
```

The core/full suite profiles do not run native OpenGL gameplay.
[QuadCraft's native workflow](https://github.com/docxology/QuadCraft/blob/main/.github/workflows/native-build.yml)
separately builds the actual game and exercises real Mesa/Xvfb tests.
Bounded `--frames N --capture output.ppm` runs must render through their frame
target or fail; a separate application smoke consumer checks the exact P6
payload and visible pixel variation. Its `native-mesa-application` artifact is
separate from `native-mesa-framebuffers`. These current real-GL gates and
captures remain pending verification; older build successes below do not
establish current gameplay, capture integrity or allocation-failure coverage.

See [testing](testing-guide.md) for symergetics' coverage and focused-run
commands, [QuadMath's workflow](https://github.com/docxology/QuadMath/blob/main/WORKFLOW.md)
and [QuadCraft's game index](https://github.com/docxology/QuadCraft/blob/main/games/GAMES_INDEX.md)
for each project's own checks.

## Review evidence and verification

The review challenged load-bearing assumptions about exactness, finite-scale
arithmetic, cache validity and unvisited browser surfaces. Reproductions,
independent code review and runtime gates establish confidence in the exercised
contracts, not universal correctness.

The following review summaries and run links belong to earlier releases,
including the 2026-11-07 methods release. They are retained as historical
evidence, not as successful gates for 0.7.0 / 1.4.0 / v0.6.0.

| Boundary | Implemented correction and evidence |
|---|---|
| Gauge and precision | QuadMath promotes centroid inputs before exact accumulation and widens low-precision floats before gauge subtraction. Reconstructed inverse candidates must replay the actual forward XYZ floats, so large axes cannot hide resolved off-lattice errors. Covered by its core/lattice regressions. |
| Nearest-point projection | symergetics uses the regular BCC fast path or exact-rational LLL reduction and closest-first bounded enumeration, retaining canonical tie-breaking. Anisotropic, sheared and permuted-basis oracle regressions exercised the general path; the earlier continuous pseudoinverse was separately scale-normalized. See above for the new rational-cofactor inverse. |
| Geometry versus labels | QuadCraft keeps physical distances separate from raw occupancy. Doom traverses raw four-index cell boundaries; Connect Four retains independent `(0,0,0,0)` and `(1,1,1,1)` slots through drop/undo. Physical boards enumerate canonical cells, and weighted pathfinding uses Dijkstra when no positive cost bound is supplied. |
| Runtime integration | Seven renderers/controllers use the current shared projection/lifecycle APIs. Doom's ESM bridge resolves the canonical class. Tower Defense draws indexed twelve-neighbor IVM edges. The CLI discovers nested shared JS/Python suites and treats nonzero process exits as failures, even after a passing summary. |
| Exact integer ingress | Integral scalars and `Fraction` fields, including NumPy-backed numerators/denominators, are promoted to Python integers before arithmetic and snapping. Bareiss elimination and boxed-object count inputs cannot wrap fixed-width products. Displacements are embedded before floating conversion; range-safe norms/angles and scaled huge-integer products retain finite geometry. |
| Physical enumeration | symergetics enumerates canonical FCC graph balls and reuses them across actual rendering backends; the interoperability example checks them independently against QuadMath. |
| Live Snake geometry | The perspective camera stays outside the raw board's Cartesian bounding sphere, preventing a negative Canvas radius that halted the first frame. The browser gate exercises actual movement, turning, rotated rendering and pause/resume. |
| Execution ownership | Local orchestration retains failed stages, demos fail truthfully, tests own temporary output, HTTP requests cannot block server shutdown, and native test invocation cannot pass an empty cached-OFF inventory. |
| Exact roots and centers | Perfect rational roots/powers stay exact; canonical `Fraction` centroids agree across both Python projects, including huge integers and independent gauges. Mutable vertex updates immediately affect volume. |
| Information geometry | Integral/large/mixed-scale sample gradients retain finite second moments; statistical Fisher interpretation requires likelihood scores. Log-domain probability arithmetic preserves positive support and recoverable rare-state contributions. |
| Complete browser inventory | All 30 actual games exercise real input/state/rendering, including 2048 pause/merge/reset, Life paused redraw/full-domain fitting, Reversi reset history and Mahjong tile fitting. |
| Release provenance | Earlier versioned full/core receipts identified three Git revisions/versions, distinguished generated/source changes and redacted host/interpreter paths. CI accepted fixed refs and enforced its then-current clean-source policy; see above for the schema-2 raw-input contract. |

Exact rational enumeration has no heuristic candidate cap. Its cost still
depends on basis/input bit lengths; no constant-time claim is made for
arbitrary custom embeddings.

Historical 0.5.0 / 1.2.0 / v0.4.0 release evidence, not current test-count pins:

| Check | Observed result |
|---|---|
| QuadMath pytest/coverage | 1,077 cases; 100% statement/branch coverage on local Python 3.13 |
| symergetics full pytest/coverage | 1,415 passed, 12 skipped; 88.28% coverage, above the 85% floor |
| QuadCraft CLI, including nested shared suites | 2,395 reported checks passed; zero failures |
| QuadCraft structural validation | All 30 games passed |
| Graphics-free native build | Both CTest targets passed in Release |
| Cross-repository example | 279 coordinate cases per implementation, five real implementations; 34 exact volumes, 37 exact-centroid cases, 216 FCC site comparisons and two fixed-width rational-ingress cases |
| Full suite command | All 14 selected checks passed; no unexecuted checks; private interpreter/repository paths absent from the receipt |
| Ordinary scientific demos | All 10 completed successfully through the actual examples-only orchestrator |
| Locked Chromium gate | All 30 catalog games passed with actual rendering/input and initial/final screenshots; local mouse capture was denied and the keyboard-control notice was visible |
| Installed distributions | Both new wheels exercised outside checkouts on Python 3.11.18/NumPy 2.4.6: exact roots, rational centroids, live volumes and stable information methods; symergetics without plotting/scientific extras |
| Previous release advisory snapshot | All 98 then-captured symergetics lockfile alert ranges excluded the resolved affected-package versions; not a new advisory audit |

The earlier numerical smoke exercised 196 scaled inverse roundtrips, 21
volume/radius scale-covariance checks and 120 nearest queries against a finite
exhaustive oracle. The methods-pass Chromium gate again passed all 30 games;
its Doom scenario additionally exercises late input and respawn timing, and
held automatic fire when native capture is available. The methods-pass CLI
reported 2,341 passing checks, zero failures. This does not claim every gameplay
path or every browser was tested.

The repeatable browser command and artifact/trace policy are in
[QuadCraft's browser gate](https://github.com/docxology/QuadCraft/blob/main/games/tests/browser/README.md).
Mouse capture capability is recorded, not fabricated; keyboard movement remains
available when the browser rejects native pointer lock. The patched scientific
environment, including Pillow 13 and pytest 9, passed the full suite and all demos.
The [previous-release Linux Chromium run](https://github.com/docxology/QuadCraft/actions/runs/42339885685)
passed all 30 surfaces with native Doom capture active. The current methods
pass's local Chromium run recorded actual denial, with keyboard controls
available. A capture-denied run does not exercise the capture-only held-fire branch.
The [merged methods-pass Linux run](https://github.com/docxology/QuadCraft/actions/runs/42431463709)
passed all 30 scenarios with native capture active. It exercised held automatic
fire, late input/release ownership and fresh respawn timing on the actual game.

On the local macOS workstation, the normal graphical native configuration
could not complete because system GLEW was missing. The independent headless
build and native coordinate adapter were exercised successfully. The
[merged Linux native workflow](https://github.com/docxology/QuadCraft/actions/runs/42431463605)
succeeded for the graphical build and Debug/Release headless jobs. Interactive
native gameplay was not exercised.

The local manuscript PDF build did not complete: XeTeX could not load the required
DejaVu Serif / DejaVu Sans Mono fonts. A process-local Fontconfig configuration
found Matplotlib's bundled font files but did not make them available to
XeTeX's font loader. Release PDF/TeX are built by the
[Linux manuscript workflow](https://github.com/docxology/QuadMath/blob/main/.github/workflows/verify.yml)
after regenerating and validating the canonical manuscript. Release notes link
the source-specific successful run; published assets carry SHA256 checksums.
The manuscript's visible date is derived from the source commit's UTC date.
