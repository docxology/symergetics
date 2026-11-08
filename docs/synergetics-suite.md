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
| symergetics `xyz_to_quadray_matrix` | Continuous 4×3 pseudoinverse; not integer quantization or min-zero normalization. |
| QuadCraft `fromCartesian` / Python `from_xyz` | Continuous inverse modulo the common-offset gauge; not an FCC nearest-neighbor operation. |
| QuadCraft `toIVM` | Canonical component rounding; not a closest-point search on the FCC/CCP sublattice. |

symergetics' fractional scaling and named plane reflections perform the
Euclidean operation and then quantize to an integer quadray. Quantization can
change distances and make a transform non-invertible; it is not exact
continuous geometry. Integer scaling and origin inversion can remain exact.

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

| Consumer | Deliberate contract |
|---|---|
| QuadMath `IVMField.harmonic_extension` | Hard Dirichlet values; every component must be anchored, with failure atomicity and separate component scales |
| Existing regularized `IVMField.learn` | Soft observation penalty; not a hard-boundary solver |
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

### Formal scope and graphical evidence

QuadMath's dependency-free Lean development proves the common-offset physical
quotient and canonical uniqueness, exact distance positivity/congruence,
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

## Coordinated software releases

| Project | Version | Release notes |
|---|---|---|
| QuadMath | 0.6.0 | [Exact methods, conservative simulation and formal quotient proofs](https://github.com/docxology/QuadMath/blob/main/CHANGELOG.md) |
| symergetics | 1.3.0 | [Live solid moments, indexed geometry and computed scientific figures](../CHANGELOG.md) |
| QuadCraft | v0.5.0 | [Time-owned simulation, continuous collisions and conservative gameplay transport](https://github.com/docxology/QuadCraft/blob/main/CHANGELOG.md) |

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

The version-1 JSON receipt records the selected profile, all three declared
versions and actual Git revisions, source/worktree dirty state, executed
commands, durations, return codes and unexecuted checks. Repository, temporary
and executable paths use logical aliases rather than personal absolute paths.
`source_dirty` excludes only the listed generated-output prefixes;
`worktree_dirty` still reports those changes. An enclosing repository cannot
stand in for a missing independent checkout.

The verifier stops at the first failure. A successful receipt has no unexecuted
checks **within its selected profile**. `--require-clean` rejects uncommitted
source changes before running stages; generated output remains visible in the
receipt. Optional `--quadmath-root`, `--quadcraft-root` and `--python` arguments
support non-sibling checkouts and explicit interpreters.

The [suite workflow](../.github/workflows/suite.yml) runs the full profile with
clean-source enforcement and uploads the receipt plus browser diagnostics.
Manual dispatch accepts `quadmath_ref`, `symergetics_ref` and `quadcraft_ref`
branches, tags or commit hashes, so a release can verify a fixed three-repository
combination instead of moving `main` branches.

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
Python/JavaScript kernels. The exercised run observed maximum mass drift
`1.4210854715202004e-14`; this is floating evidence, not an exact-JavaScript claim.
Missing runtimes and mismatches fail the command; they are not silently skipped.
It neither launches the OpenGL game nor proves browser rendering or the entire
mathematical library.

Python 3.13 is the exercised scientific environment for this review. The
repositories' existing Python pins are unchanged. Their older locked
scientific dependencies can require source builds under newer interpreters.

For graphics-free native verification:

```bash
cmake -S /path/to/QuadCraft -B /tmp/quadcraft-core -DQUADCRAFT_BUILD_GAME=OFF
cmake --build /tmp/quadcraft-core
(cd /tmp/quadcraft-core && ctest --output-on-failure)
```

See [testing](testing-guide.md) for symergetics' coverage and focused-run
commands, [QuadMath's workflow](https://github.com/docxology/QuadMath/blob/main/WORKFLOW.md)
and [QuadCraft's game index](https://github.com/docxology/QuadCraft/blob/main/games/GAMES_INDEX.md)
for each project's own checks.

## Review evidence and verification

The review challenged load-bearing assumptions about exactness, finite-scale
arithmetic, cache validity and unvisited browser surfaces. Reproductions,
independent code review and runtime gates establish confidence in the exercised
contracts, not universal correctness.

| Boundary | Implemented correction and evidence |
|---|---|
| Gauge and precision | QuadMath promotes centroid inputs before exact accumulation and widens low-precision floats before gauge subtraction. Reconstructed inverse candidates must replay the actual forward XYZ floats, so large axes cannot hide resolved off-lattice errors. Covered by its core/lattice regressions. |
| Nearest-point projection | symergetics uses the regular BCC fast path or exact-rational LLL reduction and closest-first bounded enumeration, retaining canonical tie-breaking. Anisotropic, sheared and permuted-basis oracle regressions exercise the general path; continuous pseudoinverses are scale-normalized separately. |
| Geometry versus labels | QuadCraft keeps physical distances separate from raw occupancy. Doom traverses raw four-index cell boundaries; Connect Four retains independent `(0,0,0,0)` and `(1,1,1,1)` slots through drop/undo. Physical boards enumerate canonical cells, and weighted pathfinding uses Dijkstra when no positive cost bound is supplied. |
| Runtime integration | Seven renderers/controllers use the current shared projection/lifecycle APIs. Doom's ESM bridge resolves the canonical class. Tower Defense draws indexed twelve-neighbor IVM edges. The CLI discovers nested shared JS/Python suites and treats nonzero process exits as failures, even after a passing summary. |
| Exact integer ingress | Integral scalars and `Fraction` fields, including NumPy-backed numerators/denominators, are promoted to Python integers before arithmetic and snapping. Bareiss elimination and boxed-object count inputs cannot wrap fixed-width products. Displacements are embedded before floating conversion; range-safe norms/angles and scaled huge-integer products retain finite geometry. |
| Physical enumeration | symergetics enumerates canonical FCC graph balls and reuses them across actual rendering backends; the interoperability example checks them independently against QuadMath. |
| Live Snake geometry | The perspective camera stays outside the raw board's Cartesian bounding sphere, preventing a negative Canvas radius that halted the first frame. The browser gate exercises actual movement, turning, rotated rendering and pause/resume. |
| Execution ownership | Local orchestration retains failed stages, demos fail truthfully, tests own temporary output, HTTP requests cannot block server shutdown, and native test invocation cannot pass an empty cached-OFF inventory. |
| Exact roots and centers | Perfect rational roots/powers stay exact; canonical `Fraction` centroids agree across both Python projects, including huge integers and independent gauges. Mutable vertex updates immediately affect volume. |
| Information geometry | Integral/large/mixed-scale Fisher scores retain finite curvature; log-domain probability arithmetic preserves positive support and recoverable rare-state contributions. |
| Complete browser inventory | All 30 actual games exercise real input/state/rendering, including 2048 pause/merge/reset, Life paused redraw/full-domain fitting, Reversi reset history and Mahjong tile fitting. |
| Release provenance | Versioned full/core receipts identify three Git revisions/versions, distinguish generated/source changes and redact host/interpreter paths. CI accepts fixed refs and enforces clean source. |

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

On the local macOS workstation, the normal graphical native configuration
could not complete because system GLEW was missing. The independent headless
build and native coordinate adapter were exercised successfully. The
[Linux native workflow](https://github.com/docxology/QuadCraft/actions/runs/42339885598)
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
