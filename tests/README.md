# Symergetics Test Suite

## Overview

The 2026-11-06 scientific-environment receipt is **1,344 passed, 12 skipped, 88.16% coverage**. Category descriptions below are not a live test inventory; use pytest collection for current case names. See [suite contracts and receipts](../docs/synergetics-suite.md).

```bash
export UV_PYTHON=3.13
uv sync --frozen --extra scientific --extra test
uv run --frozen python -m pytest tests/
```

## Test Architecture

### Core Philosophy
- **Consumer Behavior**: Prefer exact numerical oracles and actual renderer/artifact checks; keep failure injection confined to explicit error paths.
- **Mathematical Validation**: Tests verify mathematical correctness and implementation accuracy
- **Coverage Gate**: The configured package-wide floor is 85%; coverage is evidence of execution, not a correctness proof.
- **Integration Focus**: Tests validate cross-module interactions and workflows

### Test Categories

#### 1. Core Module Tests (`test_core_*`)
- **`test_core_numbers.py`** - SymergeticsNumber class functionality
- **`test_core_constants.py`** - SymergeticsConstants class and mathematical constants
- **`test_core_coordinates.py`** - QuadrayCoordinate class and coordinate systems
- **`test_core_comprehensive.py`** - Comprehensive core module testing
- **`test_core_extended.py`** - Extended core functionality testing

#### 2. Computation Module Tests (`test_computation_*`)
- **`test_computation_analysis.py`** - Mathematical pattern analysis
- **`test_computation_analysis_comprehensive.py`** - Comprehensive analysis testing
- **`test_computation_palindromes.py`** - Palindromic pattern analysis
- **`test_computation_primorials.py`** - Primorial and Scheherazade number calculations
- **`test_computation_primorials_coverage.py`** - Primorial coverage improvements
- **`test_computation_geometric_mnemonics.py`** - Geometric mnemonic analysis

#### 3. Geometry Module Tests (`test_geometry_*`)
- **`test_geometry_polyhedra.py`** - Polyhedra volume calculations and classes
- **`test_geometry_transformations.py`** - Coordinate transformations
- **`test_geometry_transformations_coverage.py`** - Transformation coverage improvements

#### 4. Visualization Module Tests (`test_visualization_*`)
- **`test_visualization.py`** - Core visualization functionality
- **`test_visualization_comprehensive.py`** - Comprehensive visualization testing
- **`test_visualization_advanced.py`** - Advanced visualization features
- **`test_visualization_geometry_comprehensive.py`** - Geometric visualizations
- **`test_visualization_geometry_coverage.py`** - Geometry visualization coverage
- **`test_visualization_mathematical_comprehensive.py`** - Mathematical visualizations
- **`test_visualization_mathematical_coverage.py`** - Math visualization coverage
- **`test_visualization_numbers_comprehensive.py`** - Number pattern visualizations
- **`test_visualization_numbers_coverage.py`** - Number visualization coverage
- **`test_visualization_init_comprehensive.py`** - Visualization initialization
- **`test_visualization_init_coverage.py`** - Init coverage improvements
- **`test_visualization_methods.py`** - Visualization method testing
- **`test_png_visualizations.py`** - PNG-specific visualization testing

#### 5. Utils Module Tests (`test_utils_*`)
- **`test_utils_conversion.py`** - Number and coordinate conversion utilities
- **`test_utils_extended.py`** - Extended utility functionality
- **`test_utils_mnemonics.py`** - Mnemonic encoding and memory aids
- **`test_utils_reporting.py`** - Statistical reporting and analysis

#### 6. Integration Tests (`test_integration_*`)
- **`test_integration_comprehensive.py`** - Cross-module integration testing

#### 7. Edge Case and Coverage Tests
- **`test_edge_cases_comprehensive.py`** - Edge case and error handling
- **`test_coverage_improvement.py`** - Coverage improvement tests

#### 8. Specialized Tests
- **`test_mega_graphical_abstract.py`** - Mega graphical abstract functionality
- **`test_geometry_real.py`**, **`test_utils_real.py`**, **`test_visualization_real.py`** - Real-data module coverage
- **`test_review_regressions.py`** - Regressions pinned by the 2026-09-05 review

## Test Patterns

### 1. Real Method Validation
All tests use actual Symergetics methods:

```python
# Example from test_computation_analysis.py
def test_analyze_palindromic_number(self):
    result = analyze_mathematical_patterns(121)  # Real method call
    assert result['is_palindromic'] == True
    assert 'pattern_complexity' in result
```

### 2. Mathematical Correctness
Tests verify mathematical accuracy:

```python
# Example from test_core_numbers.py
def test_arithmetic_operations(self):
    a = SymergeticsNumber(3, 4)
    b = SymergeticsNumber(1, 6)
    result = a + b
    assert result.numerator == 11
    assert result.denominator == 12
```

### 3. Integration Testing
Tests validate cross-module interactions:

```python
# Example from test_integration_comprehensive.py
def test_complete_analysis_workflow(self):
    # Step 1: Convert to SymergeticsNumber
    symergetics_numbers = [SymergeticsNumber(num) for num in test_numbers]
    
    # Step 2: Analyze patterns
    analyses = [analyze_mathematical_patterns(str(num.numerator)) for num in symergetics_numbers]
    
    # Step 3: Test reporting
    summary = generate_statistical_summary(symergetics_numbers)
```

### 4. Edge Case Handling
Tests cover boundary conditions and error cases:

```python
# Example from test_edge_cases_comprehensive.py
def test_zero_handling(self):
    zero_num = SymergeticsNumber(0)
    assert zero_num.numerator == 0
    assert zero_num.denominator == 1
    assert zero_num.to_float() == 0.0
```

## Running Tests

### Basic Test Execution
```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_core_numbers.py

# Run specific test class
python -m pytest tests/test_core_numbers.py::TestSymergeticsNumber

# Run specific test function
python -m pytest tests/test_core_numbers.py::TestSymergeticsNumber::test_arithmetic_operations
```

### Coverage Analysis
```bash
# Run with coverage
python -m pytest tests/ --cov=symergetics --cov-report=html

# Coverage with specific modules
python -m pytest tests/ --cov=symergetics.core --cov-report=html
```

### Performance Testing
```bash
# Run with performance profiling
python -m pytest tests/ --benchmark-only

# Run specific performance tests
python -m pytest tests/ -k "performance" --benchmark-only
```

### Test Data and Fixtures
`tests/conftest.py` gives every test its own `tmp_path` working directory and
resets visualization configuration before the test and in a `finally` teardown.
Default relative output therefore remains temporary even when a test calls
`reset_config()` midway through rendering. Tests that configure a custom output
directory must use temporary storage and check actual returned file containment;
passing an unused `output_dir` keyword does not isolate a renderer.

Visualization tests use actual renderers and inspect generated files. A child
pytest-session regression preserves pre-existing output sentinels through full
teardown, including filenames formerly targeted by the removed recursive
cleanup. No fixture scans or deletes the repository's `output/` artifacts.
A fresh-interpreter dashboard regression also checks lazy matplotlib/NumPy
initialization without relying on a previous visualization to warm global state.

## Test Coverage

### Current Coverage
- **Observed receipt**: 1,344 passed, 12 skipped on 2026-11-06
- **Code Coverage**: 88.16% observed; 85% required
- **Modules Covered**: All major Symergetics modules
- **Integration Coverage**: Cross-module interactions validated

### Coverage by Module
Run `uv run --frozen coverage report --show-missing` after the full coverage command for current per-module results. There are no separate per-module percentage guarantees.

## Test Quality Assurance

### Validation Criteria
1. **Consumer Behavior**: Regressions must observe meaningful results rather than mock echoes or incidental implementation details.
2. **Mathematical Accuracy**: Tests verify correct mathematical results
3. **Error Handling**: Tests cover edge cases and error conditions
4. **Integration**: Tests validate cross-module interactions
5. **Performance**: Tests include performance validation where appropriate

### Test Documentation
- **Docstrings**: All test functions have descriptive docstrings
- **Comments**: Complex test logic is well-commented
- **Examples**: Tests include clear examples of expected behavior

## Continuous Integration

No CI workflow or pre-commit configuration is currently checked in (there is no
`.github/` directory and no `.pre-commit-config.yaml`). Tests are run locally:

```bash
# Quick single-module check (per-file runs need --no-cov, see pyproject floor)
uv run pytest tests/test_core_coordinates.py -q --no-cov

# Full suite with the 85% coverage floor
uv run pytest tests/ -q
```

### Test Maintenance
- **Regular Updates**: Tests updated with new functionality
- **Coverage Monitoring**: Coverage tracked and improved
- **Performance Tracking**: Performance benchmarks maintained
- **Documentation Updates**: Test documentation kept current

## Contributing to Tests

### Adding New Tests
1. **Follow Naming Conventions**: Use descriptive test function names
2. **Use Real Methods**: Never mock Symergetics functionality
3. **Validate Mathematics**: Ensure mathematical correctness
4. **Document Thoroughly**: Include clear docstrings and comments
5. **Test Edge Cases**: Include boundary condition testing

### Test Structure
```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_functionality(self):
        """Test basic feature functionality."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = new_feature_function(input_data)
        
        # Assert
        assert result.expected_property == expected_value
        assert 'expected_key' in result
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all Symergetics modules are properly installed
2. **Test Failures**: Check that test data matches expected Symergetics behavior
3. **Coverage Issues**: Add tests for uncovered code paths
4. **Performance Issues**: Optimize test data size for performance tests

### Debugging Tests
```bash
# Run with detailed output
python -m pytest tests/ -v -s

# Run with debugging
python -m pytest tests/ --pdb

# Run specific failing test
python -m pytest tests/test_specific.py::test_failing_function -v -s
```

## References

### Documentation
- **API Documentation**: [docs/api/](../docs/api/)
- **Core Concepts**: [docs/core-concept-guide.md](../docs/core-concept-guide.md)
- **Mathematical Foundations**: [docs/mathematical-foundations-synergetics.md](../docs/mathematical-foundations-synergetics.md)

### Related Files
- **Package + test + coverage configuration**: `pyproject.toml`
  (`[tool.pytest.ini_options]`, `[tool.coverage.*]`)

---

Receipts identify exercised behavior and skipped cases; they do not establish complete correctness of every package path.
