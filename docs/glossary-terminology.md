# Glossary: implemented geometry and conceptual research vocabulary

## Introduction

Terms below include both package contracts and wider research vocabulary.
Unless an entry names an implemented API, it is context, not a claim of an
installed model or a validated application. See the [API reference](api/index.md)
for executable contracts and [research](research/index.md) for conceptual uses.

## A

### Active Inference
A family of probabilistic models of perception and action using generative models and variational objectives. The package does not implement an active-inference engine; rational arithmetic alone does not validate such a model.

### All-Integer Accounting
Fuller's geometric accounting terminology. Here it motivates integer labels and rational calculations; it is not a demonstrated law that all natural quantities are rational or that all lattice volumes are integers.

### ASCII Art
Text-based visual representations used for geometric objects when graphical backends are unavailable. The package supports ASCII visualization of polyhedra, coordinates, and mathematical patterns.

### Attractor State
In dynamical systems theory, a state toward which a system tends to evolve. In learning contexts, attractor states represent stable patterns of knowledge organization.

## B

### Bareiss Algorithm
An exact method for calculating determinants of matrices, used in the computation of tetrahedral volumes. This algorithm maintains integer precision throughout calculations.

### Belief Propagation
An algorithm for marginal inference on graphical models. It is research context here, not a package inference API or a guarantee that rational arithmetic prevents all instability.

### Bifurcation
A qualitative change in system behavior that occurs at a critical parameter value. In learning theory, bifurcations represent choice points leading to different learning trajectories.

### Biomimicry
Design informed by biological systems. Package geometry can support exploratory calculations; it does not provide a validated biological model.

## C

### Cartesian Coordinates
The traditional (x, y, z) coordinate system in three-dimensional Euclidean space. The package provides conversion functions between Cartesian and Quadray coordinate systems.

### Closest Packing
Maximum-density packing of congruent spheres in Euclidean space. FCC and HCP are familiar close-packed arrangements. This package enumerates an FCC/IVM subset; a rendered local cluster is not a packing-optimality proof.

### Cognitive Security
Protection against manipulation of information and reasoning. The package's finite digit-pattern metrics do not constitute a validated anomaly detector or cognitive-security system.

### Collaborative Intelligence
Collective problem-solving by people or systems. This is conceptual application vocabulary, not an implemented collaboration framework.

### Complex Adaptive System
A system of interacting parts whose behavior adapts. Applying geometry to one requires a separate model and evidence; no general benefit follows from the coordinate choice.

### Computational Geometry
The study of algorithms for solving geometric problems. The package implements computational geometry algorithms for exact volume calculations and coordinate transformations.

### Continued Fraction
An expression of a number as a sequence of fractions. Used for finding optimal rational approximations to irrational numbers like π and φ (golden ratio).

### Coordinate System
A system for specifying positions in space. Synergetics introduces the Quadray coordinate system as an alternative to traditional Cartesian coordinates.

### Cuboctahedron
A polyhedron with twelve vertices, twenty-four edges, eight triangular faces and six square faces. The default `Cuboctahedron()` has IVM volume twenty. “Vector equilibrium” is geometric terminology, not a simulated force-balance result.

### Curvilinear Perspective
Fuller's term for the natural coordinate system that follows the curvature of spherical surfaces, as opposed to the straight-line perspective of traditional geometry.

## D

### Determinant
A scalar value that can be computed from the elements of a square matrix. Used in volume calculations to determine the signed volume of tetrahedra.

### Dodecahedron
A regular dodecahedron has twelve regular pentagonal faces. It is a Platonic solid, but no dodecahedron geometry class is exported by this package; its volume depends on the chosen size.

### Dynamical System
A mathematical system that changes over time according to fixed rules. Learning processes and biological systems can be modeled as dynamical systems.

## E

### Emergent Property
A property that arises from the interactions of simpler components, not predictable from individual components alone. Collaborative intelligence and self-organization are emergent properties.

### Entropy
A measure of disorder or randomness in a system. Shannon entropy quantifies information content, while thermodynamic entropy relates to energy distribution.

### Epistemology
The theory of knowledge and understanding. Synergetic epistemology emphasizes holistic understanding and interconnected knowledge domains.

### Exact Arithmetic
Operations preserving the represented integer or rational values. This does not make irrational quantities rational, remove input/model error or make floating XYZ/display exact. See [numeric contracts](api/index.md#symergeticsnumber).

### Exponential Growth
Growth that increases at a rate proportional to its current value. Seen in phenomena like population growth, compound interest, and algorithmic complexity.

## F

### Feedback Loop
A process where the output of a system is fed back as input, creating self-regulating behavior. Positive feedback amplifies changes, while negative feedback dampens them.

### Fisher Information
A measure of the amount of information that an observable random variable carries about an unknown parameter. Used in statistical inference and machine learning.

### Floating-Point Arithmetic
A computer representation of real numbers using a fixed number of significant digits. Subject to rounding errors that can accumulate in iterative calculations.

### Fractal
A complex geometric shape that can be subdivided into parts similar to the whole. Characterized by self-similarity and non-integer dimension.

### Free Energy Principle
A theoretical framework relating self-organizing systems to variational objectives. It is conceptual context here; the package does not establish that exact rational arithmetic improves inference or implement a free-energy engine.

## G

### Geodesic
The shortest path between two points on a curved surface. In architecture, geodesic domes use this principle for structural efficiency.

### Geodesic Dome
A dome-like triangulated structure. Structural efficiency depends on geometry, materials, loading and constraints; constructing a polyhedron is not a validated dome-design analysis.

### Geometric Algebra
A mathematical system that extends vector algebra to include geometric operations like rotations and reflections. Related to Clifford algebra and Grassmann algebra.

### Golden Ratio (φ)
The irrational number `(1 + sqrt(5))/2`. Continued fractions supply rational approximations, not an exact rational representation of the golden ratio.

## H

### Hexagonal Close Packing
A close-packed sphere arrangement with ABAB layer stacking, distinct from FCC's ABCABC stacking. This package's IVM enumeration is FCC, not a general HCP generator.

### Holistic Understanding
Comprehension that considers the whole system rather than isolated parts. A key principle of synergetic epistemology and education.

### Honeycomb Conjecture
The mathematical problem of finding the most efficient way to divide a plane into regions of equal area with minimal boundary length. Solved by the hexagonal honeycomb pattern.

## I

### Icosahedron
A regular polyhedron with twenty triangular faces. It is a Platonic solid, but no icosahedron geometry class is exported here; its volume depends on its scale.

### Information Geometry
The study of statistical manifolds and their geometric properties. Provides a framework for understanding probability distributions and inference processes.

### Integer Linear Programming
A mathematical optimization technique where variables are restricted to integers. Used in some geometric optimization problems.

### Interdisciplinary Integration
The synthesis of knowledge from multiple academic disciplines. Synergetics provides a framework for understanding connections between mathematics, science, art, and humanities.

### Isotropic Vector Matrix (IVM)
In this implementation, the FCC sublattice generated by the twelve permutations of `(2,1,1,0)`. Canonical sites have component sum divisible by four. It is distinct from the full integer-quadray BCC image. See [FCC neighborhoods](geometric-patterns-spatial-relationships.md).

### Iterative Method
An algorithm that refines an approximation through repeated application. Used in numerical analysis and optimization problems.

## J

### Jacobian Matrix
A matrix of all first-order partial derivatives of a vector-valued function. Used in coordinate transformations and optimization algorithms.

## K

### Kepler Conjecture
The mathematical problem of finding the densest packing of spheres in three dimensions. Solved by Thomas Hales using computer-assisted proof methods.

### Knowledge Network
An interconnected system of concepts and relationships. Synergetic education aims to develop well-connected knowledge networks rather than isolated facts.

## L

### Lattice
A discrete additive subgroup generated by independent spatial vectors. The full default integer-quadray image is BCC; the package's twelve-neighbor IVM subset is FCC.

### Learning Analytics
The measurement, collection, analysis, and reporting of data about learners and their contexts. Used to optimize learning processes and educational outcomes.

### Linear Algebra
The branch of mathematics dealing with vectors, matrices, and linear transformations. Fundamental to coordinate system conversions and geometric transformations.

### Local Optimum
A solution that is optimal within a neighborhood of points, but not necessarily the global optimum. A challenge in optimization problems.

## M

### Mnemonic Encoding
Human-facing formatting and associations for large numbers. It is not numeric compression, and does not remove the cost of computing or storing the original integer.

### Modular Arithmetic
A system of arithmetic for integers where numbers "wrap around" after reaching a certain value. Important in number theory and cryptography.

### Morphogenesis
Development of biological form. Geometric analogy alone is not a validated morphogenesis model; this package does not implement one.

## N

### Natural Gradient
A gradient defined with respect to a chosen information-geometric metric. Its convergence depends on the model and algorithm; it is not an implemented package optimizer.

### Network Theory
The study of graphs and networks. Applied to social networks, biological systems, and information flow in collaborative systems.

### Nonlinear Dynamics
The study of systems that do not follow linear relationships. Important for understanding complex systems like weather patterns and population dynamics.

### Normalization
For quadrays, subtracting the minimum component to select a minimum-zero representative of a common-offset class. It does not impose a fixed component sum or unit Euclidean norm. Constructor integer snapping is a separate step.

### Numerical Stability
The property of an algorithm to produce accurate results despite small perturbations in input data. Critical for geometric calculations.

## O

### Octahedron
A regular polyhedron with eight triangular faces. The default `Octahedron()` has IVM volume four at the package's declared size; this is not a scale-independent volume for every octahedron.

### Ontology
The study of being and existence. In synergetics, ontology addresses the fundamental nature of geometric and mathematical relationships.

### Optimization
Selecting a best solution relative to a specified objective and constraints. The package does not provide a general multiobjective optimizer.

### Orbital Mechanics
The application of celestial mechanics to artificial satellites and spacecraft. Uses geometric principles for trajectory optimization.

## P

### Palindromic Number
A number that reads the same forwards and backwards. The package provides functions for detecting and analyzing palindromic patterns in mathematical sequences.

### Parameter Space
The set of all possible parameter values for a mathematical model. Used in optimization and sensitivity analysis.

### Pattern Recognition
Finding regularities in selected data. Implemented decimal-pattern analysis describes finite inputs; a match alone is not evidence of a universal mathematical, physical or biological law.

### Penrose Tiling
A non-periodic tiling of the plane discovered by Roger Penrose. Demonstrates aperiodic order and fivefold symmetry.

### Phase Transition
A sudden change in the behavior of a system as a parameter is varied. Seen in learning breakthroughs and physical systems like water freezing.

### Platonic Solid
One of the five convex regular polyhedra: tetrahedron, cube, octahedron, dodecahedron and icosahedron. Their volumes require a size convention; the package's cuboctahedron is not a Platonic solid.

### Primorial
The product of prime numbers up to a given limit. Similar to factorial but using only prime numbers. Used in number theory and cryptography.

### Probability Distribution
A mathematical function that describes the likelihood of different outcomes. Exact probability calculations are important for inference and decision-making.

### Projective Geometry
A type of geometry that studies properties invariant under projection. Related to perspective drawing and computer graphics.

## Q

### Quadray Coordinate System
Four redundant tetrahedral labels for three spatial dimensions modulo common offsets. `QuadrayCoordinate` stores integer labels; its nearest inverse projects onto the full integer lattice, not only IVM. See [coordinate contracts](api/index.md#quadraycoordinate).

### Quantum Geometry
The study of geometric structures in quantum mechanics. Synergetic principles may provide insights into quantum information and computation.

### Quasicrystal
A solid material with long-range order but no translational periodicity. Discovered by Dan Shechtman, with geometric properties studied using synergetic methods.

## R

### Rational Approximation
A rational value used to approximate another quantity. An exact representation of the rational is not an exact representation of an irrational target; a denominator cap alone is not a universal error bound.

### Recursive Function
A function that calls itself during execution. Used in algorithms for pattern generation and fractal construction.

### Rhombic Dodecahedron
A polyhedron with twelve rhombic faces; a suitable rhombic dodecahedron is an FCC Voronoi cell. No general “most efficient space-filling shape” claim follows without a specified optimization criterion. It is not an exported geometry class here.

### Rhombic Triacontahedron
A polyhedron with thirty rhombic faces, dual to the icosidodecahedron and a Catalan solid, not an Archimedean solid. It is not an exported geometry class here.

### Round-Off Error
The difference between an approximation and the true value due to rounding. Accumulates in floating-point calculations but avoided in exact arithmetic.

## S

### Scheherazade Number
A power of 1001. Three-digit blocks can match a Pascal row only while coefficients fit without carries. Use `scheherazade_pascal_coefficients` rather than assuming that every power has an uncarried reading.

### Self-Organization
The process by which complex systems organize themselves without external control. A key principle in synergetic learning and biological systems.

### Self-Similarity
The property of having the same structure at different scales. Characteristic of fractals and some natural patterns.

### Shannon Entropy
A measure of information content or uncertainty in a probability distribution. Named after Claude Shannon, founder of information theory.

### Sierpinski Tetrahedron
A three-dimensional fractal constructed by recursively subdividing a tetrahedron and removing the central portion.

### Sphere Packing
Arranging nonoverlapping spheres, often to study density or contact structure. This package visualizes declared FCC neighborhoods; it neither discovers nor proves the general optimal-packing theorem.

### Statistical Mechanics
The application of probability theory and statistics to systems with many particles. Uses concepts like entropy and phase transitions.

### Stochastic Process
A mathematical model for systems that evolve randomly over time. Important in modeling learning processes and decision-making.

### Symbolic Computation
Manipulation of symbolic expressions. `SymergeticsNumber` provides rational arithmetic, not a general symbolic-expression engine; irrational roots remain approximations.

### Symmetry Group
A set of transformations that preserve the structure of an object. Important in crystallography and geometric pattern analysis.

### Synergetic Geometry
Fuller's geometric and philosophical framework. The Python implementation supplies specific arithmetic and geometry tools, not a demonstrated unification of physics and philosophy.

### Synergetics
The study of systems in transformation, emphasizing synergy (cooperative action) and geometry. Buckminster Fuller's lifelong work to understand fundamental patterns.

### Systems Thinking
An approach to problem-solving that views problems as part of an overall system. Essential for understanding complex, interconnected phenomena.

## T

### Tensegrity
A structural principle based on isolated compression elements within a network of tension elements. Creates lightweight, strong structures.

### Tetrahedron
A polyhedron with four triangular faces. The package's default regular `Tetrahedron()` defines IVM volume one, with edge `2*sqrt(2)` at scale one. Other tetrahedra can have fractional IVM volume.

### Topological Invariant
A property that remains unchanged under continuous deformations. Important in understanding the fundamental structure of geometric objects.

### Transformation Matrix
A matrix that describes a linear transformation in vector space. Used for coordinate system conversions and geometric transformations.

### Translational Symmetry
The property of a pattern that looks the same when shifted by a certain amount. Characteristic of periodic structures like crystals.

## U

### Unit Vector
A vector of Euclidean length one. The default Urner basis columns have length `sqrt(3)`, so quadray basis directions are not unit vectors at scale one.

### Universal Pattern
A proposed relation holding across a stated domain. A finite sequence plot or geometric analogy cannot establish universality without a precise proposition and independent evidence.

## V

### Variational Free Energy
A quantity that measures the difference between an internal model and sensory input. Minimized in active inference processes.

### Vector Equilibrium
Fuller's name for the cuboctahedral arrangement of twelve equal contact directions. The package constructs its geometry; it does not solve a force-balance model.

### Volume Element
An infinitesimal volume used in integration. In synergetic geometry, the tetrahedral volume element is fundamental.

## W

### Wave Function
In quantum mechanics, a mathematical description of the quantum state of a system. May have geometric interpretations in higher-dimensional spaces.

### World Line
In spacetime geometry, the path of an object through four-dimensional spacetime. Related to concepts in relativistic physics.

## X

### XYZ Coordinates
Standard Cartesian coordinates in three-dimensional space. The package provides conversion between XYZ and Quadray coordinate systems.

### Vertex mean and solid mass centroid
`centroid_exact()` returns the exact mean of declared vertices.
`volume_centroid_exact()` returns the uniform-solid mass center of the declared
triangle-fan surface; zero volume has no defined mass centroid.
`centroid()` is a separate floor-snapped lattice approximation.

### XYZ and IVM units
At the default scale-one embedding, XYZ volume equals `8/3` times IVM volume.
Surface area uses XYZ-squared units. Custom embedding scales change Euclidean
measures, not native IVM volume. See [unit derivation](mathematical-foundations-synergetics.md#ivm-tetrahedral-units-and-xyz-cubic-units).

## Z

### Zero-Point Energy
The lowest possible energy that a quantum mechanical system can have. Related to vacuum fluctuations in quantum field theory.

## Mathematical Symbols

### φ (Phi)
The golden ratio, approximately 1.618. Expressible as (1 + √5)/2.

### π (Pi)
The ratio of a circle's circumference to its diameter, approximately 3.14159. Irrational number with infinite decimal expansion.

### √2 (Square Root of 2)
An irrational number representing the diagonal of a unit square. Cannot be expressed as a finite fraction.

### ∑ (Sigma)
Summation symbol. Used to denote sums of sequences or series.

### ∏ (Pi)
Product symbol. Used to denote products of sequences, like primorials.

### ∇ (Nabla)
Gradient operator. Used in vector calculus and optimization.

### ∂ (Partial)
Partial derivative symbol. Used in multivariable calculus.

### ∈ (Element Of)
Set membership symbol. Indicates that an element belongs to a set.

### ⊂ (Subset)
Subset symbol. Indicates that one set is contained within another.

### ∪ (Union)
Set union symbol. Combines elements from multiple sets.

### ∩ (Intersection)
Set intersection symbol. Contains elements common to multiple sets.

### ∅ (Empty Set)
The set containing no elements.

### ∀ (For All)
Universal quantifier. Indicates a statement that holds for all elements.

### ∃ (There Exists)
Existential quantifier. Indicates that there is at least one element for which a statement holds.

### ⇒ (Implies)
Logical implication. If A is true, then B must be true.

### ⇔ (If and Only If)
Logical equivalence. A is true if and only if B is true.

### ¬ (Not)
Logical negation. The opposite of a statement.

## References and Further Reading

### Primary Sources
- Fuller, R. Buckminster. *Synergetics: Explorations in the Geometry of Thinking*
- Fuller, R. Buckminster. *Synergetics 2: Further Explorations in the Geometry of Thinking*

### Mathematical References
- Conway, John H. and Sloane, N.J.A. *Sphere Packings, Lattices and Groups*
- Khinchin, A.Y. *Continued Fractions*

### Related Fields
- Capra, Fritjof. *The Web of Life: A New Scientific Understanding of Living Systems*
- Senge, Peter M. *The Fifth Discipline: The Art and Practice of the Learning Organization*

This glossary will be updated as new terms and concepts are introduced to the Synergetics package. For the most current definitions and additional context, consult the relevant documentation sections.

