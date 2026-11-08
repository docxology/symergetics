# Geometry maintenance: indexed surfaces and exact solid moments

Read [geometry contracts](README.md) before editing `polyhedra.py` or
`transformations.py`. Preserve declared positional face indices and triangle
fans; do not substitute coordinate-based topology or a convex hull. Public
vertices are mutable, so volumes and geometry must reflect current values.

Keep native `Fraction` volumes distinct from XYZ volumes, and exact vertex
means distinct from solid mass centers and lattice approximations. Valid
nonintersecting shells are a precondition, not a detected guarantee. Transform
code must respect common-offset gauge and document snapping/float boundaries.
Relevant tests are `../../tests/test_geometry_*.py` and
`../../tests/test_ivm_visualization_regressions.py`; apply the active assignment's
verification policy rather than running checks automatically.
