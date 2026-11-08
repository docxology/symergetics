# Test index: source modules and collection semantics

This is a source-file navigation map, not a saved pytest collection, pass count,
coverage report or proof of complete correctness. Parametrization expands test
functions into cases; skips depend on the environment. A filename containing
`comprehensive`, `coverage` or `real` does not certify completeness or all-backend
execution. See [execution and isolation](README.md), [canonical testing guide](../docs/testing-guide.md)
and dated [receipts](../STATUS.md).

## Source-file map

### Core arithmetic and coordinates

- [`test_core_comprehensive.py`](test_core_comprehensive.py)
- [`test_core_constants.py`](test_core_constants.py)
- [`test_core_coordinates.py`](test_core_coordinates.py)
- [`test_core_extended.py`](test_core_extended.py)
- [`test_core_numbers.py`](test_core_numbers.py)

### Computation and descriptive analysis

- [`test_computation_analysis.py`](test_computation_analysis.py)
- [`test_computation_analysis_comprehensive.py`](test_computation_analysis_comprehensive.py)
- [`test_computation_geometric_mnemonics.py`](test_computation_geometric_mnemonics.py)
- [`test_computation_palindromes.py`](test_computation_palindromes.py)
- [`test_computation_primorials.py`](test_computation_primorials.py)
- [`test_computation_primorials_coverage.py`](test_computation_primorials_coverage.py)

### Geometry and transforms

- [`test_geometry_polyhedra.py`](test_geometry_polyhedra.py)
- [`test_geometry_real.py`](test_geometry_real.py)
- [`test_geometry_transformations.py`](test_geometry_transformations.py)
- [`test_geometry_transformations_coverage.py`](test_geometry_transformations_coverage.py)

### Visualization and artifact behavior

- [`test_ivm_visualization_regressions.py`](test_ivm_visualization_regressions.py)
- [`test_mega_graphical_abstract.py`](test_mega_graphical_abstract.py)
- [`test_png_visualizations.py`](test_png_visualizations.py)
- [`test_solid_graphical_abstract_behavior.py`](test_solid_graphical_abstract_behavior.py)
- [`test_visualization.py`](test_visualization.py)
- [`test_visualization_advanced.py`](test_visualization_advanced.py)
- [`test_visualization_comprehensive.py`](test_visualization_comprehensive.py)
- [`test_visualization_geometry_comprehensive.py`](test_visualization_geometry_comprehensive.py)
- [`test_visualization_geometry_coverage.py`](test_visualization_geometry_coverage.py)
- [`test_visualization_init_comprehensive.py`](test_visualization_init_comprehensive.py)
- [`test_visualization_init_coverage.py`](test_visualization_init_coverage.py)
- [`test_visualization_mathematical_comprehensive.py`](test_visualization_mathematical_comprehensive.py)
- [`test_visualization_mathematical_coverage.py`](test_visualization_mathematical_coverage.py)
- [`test_visualization_methods.py`](test_visualization_methods.py)
- [`test_visualization_numbers_comprehensive.py`](test_visualization_numbers_comprehensive.py)
- [`test_visualization_numbers_coverage.py`](test_visualization_numbers_coverage.py)
- [`test_visualization_real.py`](test_visualization_real.py)

### Conversion, mnemonics and reporting

- [`test_utils_conversion.py`](test_utils_conversion.py)
- [`test_utils_extended.py`](test_utils_extended.py)
- [`test_utils_mnemonics.py`](test_utils_mnemonics.py)
- [`test_utils_real.py`](test_utils_real.py)
- [`test_utils_reporting.py`](test_utils_reporting.py)

### Integration, orchestration and regression cases

- [`test_coverage_improvement.py`](test_coverage_improvement.py)
- [`test_edge_cases_comprehensive.py`](test_edge_cases_comprehensive.py)
- [`test_integration_comprehensive.py`](test_integration_comprehensive.py)
- [`test_orchestration.py`](test_orchestration.py)
- [`test_review_regressions.py`](test_review_regressions.py)
- [`test_suite_orchestration.py`](test_suite_orchestration.py)

## Contract-focused entrypoints

- `test_core_coordinates.py`: constructor/inverse boundaries, embedding validation,
  gauge behavior and exact coordinate metrics.
- `test_geometry_polyhedra.py`: declared surfaces, native volume and centroid contracts.
- `test_geometry_transformations.py`: lattice transforms and projection behavior.
- `test_mega_graphical_abstract.py`: computed poster topics, section selection and resources.
- `test_solid_graphical_abstract_behavior.py`: topology-aware solid abstracts,
  display flags and figure ownership.
- `test_ivm_visualization_regressions.py`: lattice/geometry rendering regressions.
- `test_orchestration.py` and `test_suite_orchestration.py`: orchestration behavior,
  distinct from mathematical unit tests.
- `conftest.py`: temporary working-directory and visualization-config isolation.

For current case IDs, run the collection recipe in [README.md](README.md).
Collection alone does not execute assertions or establish a green suite. Add
new modules to the appropriate category without pinning a timeless total.
