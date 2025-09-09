#!/usr/bin/env python3
"""
Visualization Demo for Synergetics Package

This script demonstrates the comprehensive visualization capabilities
of the Synergetics package, showcasing geometric, mathematical, and
number pattern visualizations.

Usage:
    python visualization_demo.py

Requirements:
    - matplotlib (for 3D plots and charts)
    - numpy (for numerical operations)
    - symergetics package
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import math
from symergetics.visualization import *
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.core.numbers import SymergeticsNumber
from symergetics.computation.primorials import scheherazade_power


def demo_geometric_visualizations():
    """Demonstrate geometric visualization capabilities."""
    print("\n" + "="*60)
    print("GEOMETRIC VISUALIZATIONS")
    print("="*60)

    # Configure for organized ASCII output
    set_config({
        'backend': 'ascii',
        'output_dir': 'output',
        'organize_by_type': True,
        'include_timestamps': False
    })
    
    # Initialize organized structure
    from symergetics.visualization import create_output_structure_readme
    create_output_structure_readme()

    print("\n1. Polyhedron Visualizations")
    print("-" * 30)

    # Visualize all regular polyhedra
    polyhedra = ['tetrahedron', 'octahedron', 'cube', 'cuboctahedron']

    for poly in polyhedra:
        print(f"\nVisualizing {poly}...")
        result = plot_polyhedron(poly)
        print(f"✓ Created: {result['files']}")
        print(f"  Volume: {result['metadata']['volume']} IVM units")

    print("\n2. Quadray Coordinate System")
    print("-" * 30)

    # Create some interesting Quadray coordinates
    coordinates = [
        QuadrayCoordinate(1, 0, 0, 1),  # Simple case
        QuadrayCoordinate(2, 1, 1, 0),  # IVM neighbor
        QuadrayCoordinate(3, 1, 1, 1),  # More complex
    ]

    for i, coord in enumerate(coordinates):
        print(f"\nVisualizing coordinate {i+1}: {coord}")
        result = plot_quadray_coordinate(coord, show_lattice=True, lattice_size=3)
        print(f"✓ Created: {result['files']}")
        print(f"  XYZ position: {coord.to_xyz()}")

    print("\n3. IVM Lattice")
    print("-" * 30)

    print("\nVisualizing IVM lattice (size ±3)...")
    result = plot_ivm_lattice(size=3)
    print(f"✓ Created: {result['files']}")
    print(f"  Total points: {result['metadata']['total_points']}")


def demo_number_visualizations():
    """Demonstrate number pattern visualization capabilities."""
    print("\n" + "="*60)
    print("NUMBER PATTERN VISUALIZATIONS")
    print("="*60)

    # Use organized structure (already configured above)
    print("Using organized output structure: output/numbers/")
    print("- Palindromes -> output/numbers/palindromes/")
    print("- Scheherazade -> output/numbers/scheherazade/") 
    print("- Primorials -> output/numbers/primorials/")

    print("\n1. Palindromic Patterns")
    print("-" * 30)

    # Test various palindromic numbers
    palindromes = [121, 12321, 123454321, SymergeticsNumber(1001)]

    for num in palindromes:
        print(f"\nAnalyzing palindrome: {num}")
        result = plot_palindromic_pattern(num)
        print(f"✓ Created: {result['files']}")
        print(f"  Is palindromic: {result['metadata']['is_palindromic']}")
        print(f"  Pattern count: {result['metadata']['pattern_count']}")

    print("\n2. Scheherazade Number Patterns")
    print("-" * 30)

    # Analyze powers of 1001 (Scheherazade numbers)
    for power in [1, 2, 3]:
        print(f"\nAnalyzing 1001^{power}...")
        result = plot_scheherazade_pattern(power)
        print(f"✓ Created: {result['files']}")
        print(f"  Number length: {result['metadata']['number_length']} digits")
        print(f"  Is palindromic: {result['metadata']['is_palindromic']}")

    print("\n3. Primorial Distribution")
    print("-" * 30)

    print("\nVisualizing primorials up to n=15...")
    result = plot_primorial_distribution(max_n=15)
    print(f"✓ Created: {result['files']}")
    print(f"  Max n value: {result['metadata'].get('max_n', 'N/A')}")


def demo_mathematical_visualizations():
    """Demonstrate mathematical visualization capabilities."""
    print("\n" + "="*60)
    print("MATHEMATICAL VISUALIZATIONS")
    print("="*60)

    # Use organized structure (already configured above)
    print("Using organized output structure: output/mathematical/")
    print("- Continued fractions -> output/mathematical/continued_fractions/")
    print("- Base conversions -> output/mathematical/base_conversions/")
    print("- Pattern analysis -> output/mathematical/pattern_analysis/")

    print("\n1. Continued Fractions")
    print("-" * 30)

    # Analyze continued fractions of famous constants
    constants = [
        ('π', math.pi),
        ('e', math.e),
        ('√2', math.sqrt(2)),
        ('φ (golden ratio)', (1 + math.sqrt(5)) / 2)
    ]

    for name, value in constants:
        print(f"\nAnalyzing continued fraction of {name}...")
        result = plot_continued_fraction(value, max_terms=10)
        print(f"✓ Created: {result['files']}")
        print(f"  Terms calculated: {result['metadata']['terms_calculated']}")
        if result['metadata'].get('final_error') is not None:
            print(f"  Final approximation error: {result['metadata']['final_error']:.2e}")

    print("\n2. Base Conversions")
    print("-" * 30)

    # Demonstrate base conversion visualizations
    conversions = [
        (42, 10, 2),    # Decimal to binary
        (1001, 10, 7),  # Decimal to base 7 (Scheherazade base)
        (255, 10, 16),  # Decimal to hexadecimal
    ]

    for number, from_base, to_base in conversions:
        print(f"\nConverting {number} from base {from_base} to base {to_base}...")
        result = plot_base_conversion(number, from_base, to_base)
        print(f"✓ Created: {result['files']}")
        print(f"  Result: {result['metadata']['result']}")

    print("\n3. Pattern Analysis")
    print("-" * 30)

    # Analyze different pattern types
    test_numbers = [12321, 112233, 24681357]
    pattern_types = ['palindrome', 'repeated', 'symmetric']

    for num in test_numbers:
        for pattern_type in pattern_types:
            print(f"\nAnalyzing {num} for {pattern_type} patterns...")
            result = plot_pattern_analysis(num, pattern_type)
            print(f"✓ Created: {result['files']}")
            print(f"  Number length: {result['metadata']['length']}")
            print(f"  Unique digits: {result['metadata'].get('unique_digits', 'N/A')}")


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n" + "="*60)
    print("BATCH PROCESSING DEMO")
    print("="*60)

    # Use organized structure (already configured above)
    print("Batch results will be organized by visualization type.")

    # Create a batch of visualization tasks
    tasks = [
        {'function': 'plot_polyhedron', 'args': ['tetrahedron']},
        {'function': 'plot_polyhedron', 'args': ['octahedron']},
        {'function': 'plot_palindromic_pattern', 'args': [121]},
        {'function': 'plot_scheherazade_pattern', 'args': [2]},
        {'function': 'plot_continued_fraction', 'args': [math.pi], 'kwargs': {'max_terms': 8}},
    ]

    print(f"\nProcessing {len(tasks)} visualization tasks...")
    results = batch_visualize(tasks)

    print("\nBatch Results:")
    successful = 0
    failed = 0

    for i, result in enumerate(results):
        if 'error' in result:
            failed += 1
            print(f"❌ Task {i+1}: {result['error']}")
        else:
            successful += 1
            vis_type = result['metadata']['type']
            files = len(result['files'])
            print(f"✅ Task {i+1}: {vis_type} ({files} files)")

    print(f"\nSummary: {successful} successful, {failed} failed")

    # Create a report
    report_path = create_visualization_report(results, "Batch Processing Demo Report")
    print(f"\n📊 Report saved to: {report_path}")


def demo_advanced_features():
    """Demonstrate advanced visualization features."""
    print("\n" + "="*60)
    print("ADVANCED FEATURES")
    print("="*60)

    # Demonstrate the new organized structure features
    from symergetics.visualization import list_output_structure, get_config
    
    print("\n1. Output Structure Organization")
    print("-" * 30)
    
    structure_info = list_output_structure()
    print(f"Output directory: {structure_info['base_path']}")
    print(f"Organized structure: {structure_info['organized']}")
    print("Categories created:")
    for category, info in structure_info['categories'].items():
        if info['exists']:
            total_files = sum(sc.get('file_count', 0) for sc in info['subcategories'].values())
            print(f"  {category}/: {total_files} files in {len(info['subcategories'])} subcategories")

    # Demonstrate configuration and export features
    print("\n1. Configuration Management")
    print("-" * 30)

    # Show current config
    config = get_config()
    print(f"Current backend: {config['backend']}")
    print(f"Output directory: {config['output_dir']}")

    # Modify configuration
    set_config({'dpi': 300, 'figure_size': (12, 9)})
    print("Updated configuration with higher DPI and larger figures")

    print("\n2. Export Capabilities")
    print("-" * 30)

    # Create some sample data
    sample_data = {
        'experiment': 'visualization_demo',
        'timestamp': '2024-01-01T00:00:00Z',
        'results': {
            'polyhedra_visualized': 4,
            'numbers_analyzed': 10,
            'patterns_found': 25
        }
    }

    # Export in different formats
    json_path = export_visualization(sample_data, format='json', filename='demo_results')
    txt_path = export_visualization(sample_data, format='txt', filename='demo_summary')

    print(f"✓ JSON export: {json_path}")
    print(f"✓ Text export: {txt_path}")

    print("\n3. Mnemonic Encoding Visualization")
    print("-" * 30)

    # This would normally use the numbers visualization module
    large_number = SymergeticsNumber(12345678901234567890)
    print(f"\nLarge number: {large_number}")

    # Show mnemonic representation (conceptual)
    print("Mnemonic strategies available:")
    print("- Grouped digits: 12,345,678,901,234,567,890")
    print("- Scientific: 1.234568e+19")
    print("- Words: twelve quintillion three hundred forty-five quadrillion...")


def main():
    """Run the complete visualization demo."""
    print("SYNERGETICS VISUALIZATION DEMO")
    print("==============================")
    print("\nThis demo showcases the comprehensive visualization capabilities")
    print("of the Synergetics package, focusing on geometric, mathematical,")
    print("and symbolic computation visualizations.")
    print("\nAll outputs will be saved to the 'output/' directory.")

    try:
        # Run all demos
        demo_geometric_visualizations()
        demo_number_visualizations()
        demo_mathematical_visualizations()
        demo_batch_processing()
        demo_advanced_features()

        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nSummary of organized visualizations created:")
        print("✓ Geometric: output/geometric/{polyhedra,coordinates,lattice}/")
        print("✓ Numbers: output/numbers/{palindromes,scheherazade,primorials}/")
        print("✓ Mathematical: output/mathematical/{continued_fractions,base_conversions,pattern_analysis}/")
        print("✓ Batch processing: Files organized by type automatically")
        print("✓ Export capabilities: JSON and text formats with proper structure")
        print("\n📁 Organized Output Structure:")
        print("output/")
        print("├── README.md (explains the structure)")
        print("├── geometric/")
        print("│   ├── polyhedra/     (3D polyhedron visualizations)")
        print("│   ├── coordinates/   (Quadray coordinate plots)")
        print("│   └── lattice/       (IVM lattice structures)")
        print("├── numbers/")
        print("│   ├── palindromes/   (Palindromic patterns)")
        print("│   ├── scheherazade/  (Scheherazade analysis)")
        print("│   └── primorials/    (Primorial distributions)")
        print("├── mathematical/")
        print("│   ├── continued_fractions/  (Continued fraction analysis)")
        print("│   ├── base_conversions/     (Number base conversions)")  
        print("│   └── pattern_analysis/     (Mathematical patterns)")
        print("└── reports/           (Summary reports and exports)")
        print("\nEach subdirectory contains relevant README files explaining the contents.")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Make sure all required dependencies are installed:")
        print("  pip install matplotlib numpy")
        raise

    finally:
        # Reset configuration
        reset_config()


if __name__ == "__main__":
    main()
