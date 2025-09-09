# Synergetics Visualization Module

The visualization module provides comprehensive visualization capabilities for geometric objects, mathematical patterns, and symbolic computations in the Synergetics framework.

## Features

- **Modular Design**: Separate modules for geometry, numbers, and mathematical visualizations
- **Multiple Backends**: Support for matplotlib, plotly, and ASCII art
- **Configurable Output**: Customizable colors, sizes, formats, and output directories
- **Batch Processing**: Execute multiple visualizations simultaneously
- **Export Capabilities**: Save results in JSON, text, and other formats

## Installation

The visualization module requires matplotlib for 3D plotting:

```bash
pip install matplotlib numpy
```

For enhanced 3D plotting with plotly:

```bash
pip install plotly
```

## Quick Start

```python
from symergetics.visualization import plot_polyhedron, set_config

# Configure output
set_config({'backend': 'matplotlib', 'output_dir': 'my_visualizations'})

# Create a 3D visualization of a tetrahedron
result = plot_polyhedron('tetrahedron')
print(f"Files created: {result['files']}")
```

## Modules Overview

### Geometry Visualizations (`geometry.py`)

Visualize geometric objects and coordinate systems from Synergetics.

#### Polyhedron Visualization

```python
from symergetics.visualization import plot_polyhedron

# Visualize regular polyhedra
plot_polyhedron('tetrahedron')      # Unit volume = 1
plot_polyhedron('octahedron')       # Unit volume = 4
plot_polyhedron('cube')            # Unit volume = 3
plot_polyhedron('cuboctahedron')   # Unit volume = 20
```

#### Quadray Coordinate System

```python
from symergetics.visualization import plot_quadray_coordinate
from symergetics.core.coordinates import QuadrayCoordinate

# Visualize a Quadray coordinate
coord = QuadrayCoordinate(1, 2, 0, 1)
plot_quadray_coordinate(coord, show_lattice=True, lattice_size=3)
```

#### IVM Lattice

```python
from symergetics.visualization import plot_ivm_lattice

# Visualize the Isotropic Vector Matrix lattice
plot_ivm_lattice(size=5)
```

### Number Pattern Visualizations (`numbers.py`)

Analyze and visualize patterns in numbers, especially those from Fuller's work.

#### Palindromic Patterns

```python
from symergetics.visualization import plot_palindromic_pattern

# Analyze palindromic patterns in numbers
plot_palindromic_pattern(12321)      # Palindrome
plot_palindromic_pattern(1001)       # Scheherazade base
```

#### Scheherazade Number Patterns

```python
from symergetics.visualization import plot_scheherazade_pattern

# Visualize powers of 1001 (Scheherazade numbers)
plot_scheherazade_pattern(6)  # Contains Pascal's triangle row
```

#### Primorial Distribution

```python
from symergetics.visualization import plot_primorial_distribution

# Show primorial growth patterns
plot_primorial_distribution(max_n=20)
```

### Mathematical Visualizations (`mathematical.py`)

Visualize mathematical concepts and algorithms.

#### Continued Fractions

```python
from symergetics.visualization import plot_continued_fraction
import math

# Analyze continued fraction expansions
plot_continued_fraction(math.pi, max_terms=10)
plot_continued_fraction(math.e, max_terms=8)
```

#### Base Conversions

```python
from symergetics.visualization import plot_base_conversion

# Visualize number base conversions
plot_base_conversion(42, 10, 2)     # Decimal to binary
plot_base_conversion(1001, 10, 7)   # Decimal to base 7
```

#### Pattern Analysis

```python
from symergetics.visualization import plot_pattern_analysis

# Analyze digit patterns
plot_pattern_analysis(12321, 'palindrome')
plot_pattern_analysis(112233, 'repeated')
```

## Configuration

Customize visualization behavior with the configuration system:

```python
from symergetics.visualization import set_config, get_config

# Set configuration
set_config({
    'backend': 'matplotlib',        # 'matplotlib', 'plotly', 'ascii'
    'output_dir': 'output',         # Output directory
    'figure_size': (10, 8),         # Figure dimensions
    'dpi': 150,                     # Resolution
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'accent': '#2ca02c'
    }
})

# Get current configuration
config = get_config()
```

## Batch Processing

Execute multiple visualizations efficiently:

```python
from symergetics.visualization import batch_visualize

# Define visualization tasks
tasks = [
    {'function': 'plot_polyhedron', 'args': ['tetrahedron']},
    {'function': 'plot_scheherazade_pattern', 'args': [6]},
    {'function': 'plot_continued_fraction', 'args': [3.14159], 'kwargs': {'max_terms': 8}}
]

# Execute batch
results = batch_visualize(tasks, backend='ascii')

# Create summary report
from symergetics.visualization import create_visualization_report
report_path = create_visualization_report(results, "Batch Visualization Report")
```

## Export and Reporting

Export visualization results and create reports:

```python
from symergetics.visualization import export_visualization

# Export results to different formats
export_visualization(results, format='json', filename='my_results')
export_visualization(results, format='txt', filename='summary')
```

## Backend Options

### Matplotlib Backend
- High-quality 2D and 3D plots
- Publication-ready output
- Requires: `pip install matplotlib`

### Plotly Backend
- Interactive 3D visualizations
- Web-based output
- Requires: `pip install plotly`

### ASCII Backend
- Text-based visualizations
- No dependencies required
- Great for terminals and documentation

## File Organization

All visualizations are saved to the configured output directory:

```
output/
├── geometric_demo/
│   ├── tetrahedron_3d.png
│   ├── octahedron_3d.png
│   └── ivm_lattice_size_3.png
├── number_demo/
│   ├── palindrome_pattern_121.png
│   └── scheherazade_pattern_1001_power_6.png
├── math_demo/
│   ├── continued_fraction_3.1416_terms_10.png
│   └── base_conversion_42_base_10_to_2.png
└── visualization_config.json
```

## Synergetics Context

The visualizations are designed to support Fuller's Synergetics principles:

- **Exact Arithmetic**: All calculations use rational numbers for precision
- **Geometric Relationships**: Visualizations show volume ratios and coordinate transformations
- **Pattern Recognition**: Analysis of palindromic and symmetrical patterns
- **Scale Relationships**: Support for cosmic hierarchy scaling

## Examples

See `examples/visualization_demo.py` for comprehensive usage examples covering:

- All visualization types
- Configuration management
- Batch processing
- Export capabilities
- Error handling

## Dependencies

Core dependencies:
- `numpy`: Numerical operations
- `matplotlib`: Plotting (optional, for matplotlib backend)
- `plotly`: Interactive plotting (optional, for plotly backend)

## API Reference

### Configuration Functions
- `set_config(config)`: Update configuration
- `get_config()`: Get current configuration
- `reset_config()`: Reset to defaults

### Visualization Functions
- `plot_polyhedron(polyhedron, **kwargs)`: Visualize polyhedra
- `plot_quadray_coordinate(coord, **kwargs)`: Visualize Quadray coordinates
- `plot_ivm_lattice(size, **kwargs)`: Visualize IVM lattice
- `plot_palindromic_pattern(number, **kwargs)`: Analyze palindromes
- `plot_scheherazade_pattern(power, **kwargs)`: Visualize Scheherazade numbers
- `plot_continued_fraction(value, **kwargs)`: Analyze continued fractions
- `plot_base_conversion(number, from_base, to_base, **kwargs)`: Show base conversions

### Utility Functions
- `batch_visualize(tasks, **kwargs)`: Execute multiple visualizations
- `export_visualization(data, **kwargs)`: Export results
- `create_visualization_report(results, **kwargs)`: Create summary reports

## Contributing

When adding new visualizations:

1. Follow the modular structure (geometry, numbers, mathematical)
2. Support all backend types (matplotlib, plotly, ascii)
3. Include comprehensive error handling
4. Add tests in `tests/test_visualization.py`
5. Update this documentation

## License

This visualization module is part of the Synergetics package and follows the same MIT license.
