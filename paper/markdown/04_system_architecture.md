## System Architecture

### Design Principles

The package separates rational data, integer geometry, numerical analysis and
visualization. Exactness is a property of particular operations and inputs, not
of every component. Float plotting, irrational quantities and lattice snapping
are explicit boundaries. Modularity does not guarantee compatibility or
correctness of an arbitrary extension.

### Package Structure

| Area | Source | Responsibility |
| --- | --- | --- |
| Arithmetic | [core/numbers.py](../../symergetics/core/numbers.py) | Fraction-backed numbers, roots and rational approximations |
| Coordinates | [core/coordinates.py](../../symergetics/core/coordinates.py) | Integer quadrays, embedding, nearest-site inversion and exact metrics |
| Constants | [core/constants.py](../../symergetics/core/constants.py) | Named rational values and approximation tables |
| Solid geometry | [geometry/polyhedra.py](../../symergetics/geometry/polyhedra.py) | Declared topology, live volume, area and centers |
| Prime products | [computation/primorials.py](../../symergetics/computation/primorials.py) | Prime-bounded primorials, powers of 1001 and binomial-block checks |
| Digit patterns | [computation/palindromes.py](../../symergetics/computation/palindromes.py) | Representation-dependent number-pattern utilities |
| Analysis | [computation/analysis.py](../../symergetics/computation/analysis.py) | Mathematical analysis helpers; inspect individual numerical methods |
| Visualization | [visualization](../../symergetics/visualization/) | Geometric and mathematical plots and output handling |
| Conversion | [utils/conversion.py](../../symergetics/utils/conversion.py) | Conversion helpers and their own numeric policies |

This replaces descriptions of unimplemented `patterns`, `complexity`,
`optimization`, engineering-analysis and automatic analyzer-registration
frameworks. The [architecture guide](../../docs/architecture/index.md) and
[API reference](../../docs/api/index.md) own detailed interfaces.

### Core Module: Mathematical Foundation

The core stores rational numbers and integer quadray representatives. Do not
infer symbolic π/φ, continuous rational coordinates or exact float exports from
those types. The [mathematical foundations](03_mathematical_foundations.md)
define their domains and units.

### Geometry Module: Spatial Analysis

The four concrete solid classes are `Tetrahedron`, `Cube`, `Octahedron` and
`Cuboctahedron`. Their mutable vertices and declared indexed fans determine
measurements. A vertex mean is not a uniform-solid mass centroid. Valid
nonintersecting closed surfaces are a precondition of solid measurements.

### Computation Module: Pattern Discovery

Prime products and digit-pattern operations act on specified mathematical
inputs. A string pattern is not a general scientific classifier. No distributed
analysis service or benchmarked large-data performance is established by this
architecture description.

### Visualization Module: Output Generation

Visualization uses numerical projections of computed data. Supported output
formats depend on the selected renderer and installed optional dependencies;
a format recognized by discovery is not necessarily embeddable by the paper
renderer. Captions must state units and approximation boundaries.

### Testing and Quality Assurance

[Test evidence](../TEST_EVIDENCE.md) distinguishes source-definition inventories,
collected cases and historical execution/coverage receipts. A passing assertion
supports that case, not universal exactness or domain efficacy. This document
reports source review, not a fresh execution receipt.

### Integration and Workflow

The [checkout guide](../../README.md) owns package installation. The
[paper quick start](../QUICK_START.md) uses the renderer directly; the larger
`generate_paper.py` orchestration can run tests, regenerate examples and clear
outputs, so it must not be described as a PDF-only command.

### Performance and Scalability

Rational arithmetic and integer pattern analysis are limited by bit growth,
algorithmic search and memory. No automatic load balancing, memory-budget
calculator or distributed runtime is claimed here. Record problem sizes and
measured conditions before claiming efficiency.

### Documentation and Maintenance

Keep callable examples in the [API reference](../../docs/api/index.md),
mathematical conventions in [foundations](03_mathematical_foundations.md), and
proposed domain uses in [research applications](09_research_applications.md).
