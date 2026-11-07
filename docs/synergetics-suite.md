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

The identity is exact algebraically. Numerically, subtracting those two large
quadratic terms can erase a small distance. Gauge-free Cartesian differences
and a sum of nonnegative squares avoid that cancellation. Floating XYZ inputs
still cannot recover integer information already lost before entering an API.

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

## Coordinated software releases

| Project | Version | Release notes |
|---|---|---|
| QuadMath | 0.4.0 | [Mathematical methods and manuscript pipeline](https://github.com/docxology/QuadMath/blob/main/CHANGELOG.md) |
| symergetics | 1.1.0 | [Exact geometry, visualization and orchestration](../CHANGELOG.md) |
| QuadCraft | v0.3.0 | [Interactive/native/browser applications](https://github.com/docxology/QuadCraft/blob/main/CHANGELOG.md) |

## Complete suite verification

From the symergetics checkout:

```bash
python3 verify_suite.py --report /tmp/synergetics-suite.json
```

The [verifier](../verify_suite.py) requires bash, uv, Node.js, CMake and a
C++17 compiler. It creates temporary locked Python environments and a native
Release build, preserving existing project environments and interpreter pins.
It invokes the actual repository gates, stops at the first failure, and records
executed commands, return codes and unexecuted checks. A successful receipt has
no unexecuted checks. Optional `--quadmath-root`, `--quadcraft-root` and
`--python` arguments support non-sibling checkouts and explicit interpreters.

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
enumerated FCC graph balls and agreement between exact volume formulations
beyond binary64's integer precision.
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

The review used Claim–Evidence Audit: mathematical identities, implementation
claims and exercised behavior were checked separately.

| Boundary | Implemented correction and evidence |
|---|---|
| Gauge and precision | QuadMath promotes centroid inputs before exact accumulation and widens low-precision floats before gauge subtraction. Reconstructed inverse candidates must replay the actual forward XYZ floats, so large axes cannot hide resolved off-lattice errors. Covered by its core/lattice regressions. |
| Nearest-point projection | symergetics uses the regular BCC fast path or exact-rational LLL reduction and closest-first bounded enumeration, retaining canonical tie-breaking. Anisotropic, sheared and permuted-basis oracle regressions exercise the general path; continuous pseudoinverses are scale-normalized separately. |
| Geometry versus labels | QuadCraft keeps physical distances separate from raw occupancy. Doom traverses raw four-index cell boundaries; Connect Four retains independent `(0,0,0,0)` and `(1,1,1,1)` slots through drop/undo. Physical boards enumerate canonical cells, and weighted pathfinding uses Dijkstra when no positive cost bound is supplied. |
| Runtime integration | Seven renderers/controllers use the current shared projection/lifecycle APIs. Doom's ESM bridge resolves the canonical class. Tower Defense draws indexed twelve-neighbor IVM edges. The CLI discovers nested shared JS/Python suites and treats nonzero process exits as failures, even after a passing summary. |
| Exact integer ingress | Integral scalar components are stored as Python integers before normalization/arithmetic. Bareiss elimination and boxed-object count inputs cannot wrap fixed-width products. Displacements are embedded before floating conversion; range-safe norms/angles and scaled huge-integer products retain finite geometry. |
| Physical enumeration | symergetics enumerates canonical FCC graph balls and reuses them across actual rendering backends; the interoperability example checks them independently against QuadMath. |
| Live Snake geometry | The perspective camera stays outside the raw board's Cartesian bounding sphere, preventing a negative Canvas radius that halted the first frame. The browser gate exercises actual movement, turning, rotated rendering and pause/resume. |
| Execution ownership | Local orchestration retains failed stages, demos fail truthfully, tests own temporary output, HTTP requests cannot block server shutdown, and native test invocation cannot pass an empty cached-OFF inventory. |

Exact rational enumeration has no heuristic candidate cap. Its cost still
depends on basis/input bit lengths; no constant-time claim is made for
arbitrary custom embeddings.

Verified with Python 3.13.17, Node.js and the C++17 toolchain:

| Check | Observed result |
|---|---|
| QuadMath full pytest/coverage run | 988 passed; 100% statement/branch coverage |
| symergetics full pytest/coverage run | 1,344 passed, 12 skipped; 88.16% coverage, above the 85% floor |
| QuadCraft CLI, including nested shared suites | 2,395 reported checks passed; zero failures |
| QuadCraft structural validation | All 30 games passed |
| Graphics-free native build | Both CTest targets passed in Release; cached testing-OFF → enabled transition also exercised |
| Cross-repository example | 279 coordinate cases per implementation, five real implementations; 34 exact-volume cases; 216 FCC site comparisons |
| Complete suite command | All 13 checks executed successfully; no unexecuted checks |
| Ordinary scientific demos | All 10 completed successfully through the actual examples-only orchestrator |
| Locked Chromium gate | All 13 selected surfaces passed with actual rendering/input; native mouse capture was denied locally and the keyboard-control notice was visible |
| Installed distributions | Both wheels exercised outside the checkouts; QuadMath generated data/PNG output in a path containing spaces, and symergetics imported/inverted/enumerated FCC sites without plotting dependencies |
| Dependency advisory snapshot | All 98 open symergetics lockfile alert ranges excluded every newly resolved affected package version |

The separate numerical smoke exercised 196 scaled inverse roundtrips, 21
volume/radius scale-covariance checks and 120 nearest queries against a finite
exhaustive oracle. Actual HTTP browser checks exercised Go, Hex, Lights Out,
Memory, Rogue, Sokoban, Snake, Sudoku, Doom, Minesweeper, Connect Four, Tower Defense
and Minecraft: picking, keyboard actions, rotation/zoom, raw-slot drop/undo,
and displayed mathematical verification where applicable. This is not a claim
that every gameplay path or every browser was tested.

The repeatable browser command and artifact/trace policy are in
[QuadCraft's browser gate](https://github.com/docxology/QuadCraft/blob/main/games/tests/browser/README.md).
Mouse capture capability is recorded, not fabricated; keyboard movement remains
available when the browser rejects native pointer lock. The patched scientific
environment, including Pillow 13 and pytest 9, passed the full suite and all demos.
The [merged Linux Chromium run](https://github.com/docxology/QuadCraft/actions/runs/42328466873)
also passed every selected surface and recorded Doom mouse capture as active,
so both actual native-capture outcomes have been exercised.

On the local macOS workstation, the normal graphical native configuration
could not complete because system GLEW was missing. The independent headless
build and native coordinate adapter were exercised successfully. The
[Linux native workflow](https://github.com/docxology/QuadCraft/actions/runs/42328466897)
succeeded for the graphical build and Debug/Release headless jobs. Interactive
native gameplay was not exercised.

The local manuscript PDF build did not complete: XeTeX could not load the required
DejaVu Serif / DejaVu Sans Mono fonts. A process-local Fontconfig configuration
found Matplotlib's bundled font files but did not make them available to
XeTeX's font loader. Local published PDFs remain unchanged. The
[Linux manuscript run](https://github.com/docxology/QuadMath/actions/runs/42327721786)
successfully regenerated and validated the manuscript, producing the
119-page combined PDF and TeX. The downloaded PDF text, metadata and rendered
cover were inspected; its visible date uses the source commit's UTC date.
