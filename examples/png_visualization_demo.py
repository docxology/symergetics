#!/usr/bin/env python3
"""
PNG Visualization Demo for Symergetics Package

This demo generates comprehensive PNG visualizations showcasing the
mathematical and geometric capabilities of the Symergetics package.
All outputs are saved as high-quality PNG files in the 'output/' directory.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import symergetics
sys.path.insert(0, str(Path(__file__).parent.parent))

from symergetics.visualization import (
    set_config, plot_polyhedron, plot_quadray_coordinate, plot_ivm_lattice,
    plot_palindromic_pattern, plot_scheherazade_pattern, plot_primorial_distribution,
    plot_continued_fraction, plot_base_conversion, plot_pattern_analysis
)
from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.computation.palindromes import is_palindromic
from symergetics.computation.primorials import scheherazade_power


def setup_png_config():
    """Configure for high-quality PNG output with organized structure."""
    config = {
        'output_dir': 'output',
        'backend': 'matplotlib',
        'organize_by_type': True,  # Enable organized structure
        'include_timestamps': False,  # Clean filenames for demo
        'dpi': 300,  # High resolution
        'figure_size': (12, 9),  # Larger figures for better detail
        'png_options': {
            'transparent': False,
            'facecolor': 'white',
            'bbox_inches': 'tight',
            'pad_inches': 0.2
        },
        'colors': {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'background': '#ffffff',
            'grid': '#e9ecef'
        },
        'fonts': {
            'title': {'size': 16, 'weight': 'bold'},
            'label': {'size': 14},
            'annotation': {'size': 12}
        }
    }
    set_config(config)
    
    # Create organized output structure with README files
    from symergetics.visualization import create_output_structure_readme
    create_output_structure_readme()
    
    print(f"✓ PNG configuration set - DPI: {config['dpi']}, Size: {config['figure_size']}")
    print(f"✓ Organized output structure enabled: output/{{category}}/{{subcategory}}/")


def demo_geometric_png_visualizations():
    """Generate geometric PNG visualizations."""
    print("\n" + "="*60)
    print("GEOMETRIC PNG VISUALIZATIONS")
    print("="*60)
    
    results = []
    
    # 1. Polyhedron visualizations
    print("\n1. Polyhedron Visualizations")
    print("-" * 30)
    
    polyhedra = ['tetrahedron', 'octahedron', 'cube', 'cuboctahedron']
    for poly in polyhedra:
        try:
            result = plot_polyhedron(poly, backend='matplotlib', 
                                   show_edges=True, show_faces=True, show_vertices=True)
            results.append(result)
            print(f"✓ {poly.capitalize()}: {result['files'][0]}")
            print(f"  Volume: {result['metadata']['volume']} IVM units")
        except Exception as e:
            print(f"❌ Error with {poly}: {e}")
            results.append({'error': str(e), 'type': f'polyhedron:{poly}'})
    
    # 2. Quadray coordinate system
    print("\n2. Quadray Coordinate System")
    print("-" * 30)
    
    coordinates = [
        (1, 0, 0, 1),  # Basic coordinate
        (2, 1, 1, 0),  # Asymmetric coordinate
        (3, 2, 1, 0),  # Larger coordinate
        (1, 1, 1, 1)   # Symmetric coordinate
    ]
    
    for i, coord in enumerate(coordinates):
        try:
            qc = QuadrayCoordinate(*coord)
            result = plot_quadray_coordinate(qc, backend='matplotlib', size=3)
            results.append(result)
            xyz = qc.to_xyz()
            print(f"✓ Quadray {coord}: {result['files'][0]}")
            print(f"  XYZ position: ({xyz[0]}, {xyz[1]}, {xyz[2]})")
        except Exception as e:
            print(f"❌ Error with coordinate {coord}: {e}")
            results.append({'error': str(e), 'type': f'coordinate:{coord}'})
    
    # 3. IVM lattice visualization
    print("\n3. IVM Lattice Structure")
    print("-" * 30)
    
    try:
        result = plot_ivm_lattice(size=3, backend='matplotlib')
        results.append(result)
        print(f"✓ IVM Lattice: {result['files'][0]}")
        print(f"  Lattice points: {result['metadata']['total_points']}")
    except Exception as e:
        print(f"❌ Error with IVM lattice: {e}")
        results.append({'error': str(e), 'type': 'ivm_lattice'})
    
    return results


def demo_number_pattern_png_visualizations():
    """Generate number pattern PNG visualizations."""
    print("\n" + "="*60)
    print("NUMBER PATTERN PNG VISUALIZATIONS")
    print("="*60)
    
    results = []
    
    # 1. Palindromic patterns
    print("\n1. Palindromic Pattern Analysis")
    print("-" * 30)
    
    palindromes = [
        121,
        12321,
        1234321,
        123454321,
        SymergeticsNumber(1001),
        SymergeticsNumber(1002001)
    ]
    
    for pal in palindromes:
        try:
            result = plot_palindromic_pattern(pal, backend='matplotlib')
            results.append(result)
            is_pal = result['metadata']['is_palindromic']
            print(f"✓ Palindrome {pal}: {result['files'][0]}")
            print(f"  Is palindromic: {is_pal}")
        except Exception as e:
            print(f"❌ Error with palindrome {pal}: {e}")
            results.append({'error': str(e), 'type': f'palindrome:{pal}'})
    
    # 2. Scheherazade number patterns
    print("\n2. Scheherazade Number Patterns")
    print("-" * 30)
    
    for power in range(1, 5):
        try:
            result = plot_scheherazade_pattern(power, backend='matplotlib')
            results.append(result)
            sch_num = scheherazade_power(power)
            print(f"✓ Scheherazade 1001^{power}: {result['files'][0]}")
            print(f"  Number: {sch_num} (palindromic: {is_palindromic(sch_num)})")
        except Exception as e:
            print(f"❌ Error with Scheherazade power {power}: {e}")
            results.append({'error': str(e), 'type': f'scheherazade:{power}'})
    
    # 3. Primorial distribution
    print("\n3. Primorial Distribution")
    print("-" * 30)
    
    try:
        result = plot_primorial_distribution(max_n=15, backend='matplotlib')
        results.append(result)
        print(f"✓ Primorial Distribution: {result['files'][0]}")
        print(f"  Maximum n: 15")
    except Exception as e:
        print(f"❌ Error with primorial distribution: {e}")
        results.append({'error': str(e), 'type': 'primorial_distribution'})
    
    return results


def demo_mathematical_png_visualizations():
    """Generate mathematical PNG visualizations."""
    print("\n" + "="*60)
    print("MATHEMATICAL PNG VISUALIZATIONS")
    print("="*60)
    
    results = []
    
    # 1. Continued fractions
    print("\n1. Continued Fraction Analysis")
    print("-" * 30)
    
    constants = [
        (np.pi, "π", "Pi"),
        (np.e, "e", "Euler's number"),
        ((1 + np.sqrt(5)) / 2, "φ", "Golden ratio"),
        (np.sqrt(2), "√2", "Square root of 2"),
        (np.sqrt(3), "√3", "Square root of 3")
    ]
    
    for value, symbol, name in constants:
        try:
            result = plot_continued_fraction(value, max_terms=10, backend='matplotlib')
            results.append(result)
            print(f"✓ Continued fraction {symbol} ({name}): {result['files'][0]}")
            print(f"  Value: {value:.10f}")
        except Exception as e:
            print(f"❌ Error with continued fraction {symbol}: {e}")
            results.append({'error': str(e), 'type': f'continued_fraction:{symbol}'})
    
    # 2. Base conversions
    print("\n2. Base Conversion Analysis")
    print("-" * 30)
    
    numbers_and_bases = [
        (1001, 2, "Scheherazade to binary"),
        (12321, 2, "Palindrome to binary"),
        (30030, 2, "Primorial 13# to binary"),
        (1001, 8, "Scheherazade to octal"),
        (12321, 16, "Palindrome to hexadecimal")
    ]
    
    for number, base, description in numbers_and_bases:
        try:
            result = plot_base_conversion(number, from_base=10, to_base=base, backend='matplotlib')
            results.append(result)
            print(f"✓ Base conversion {number} → base {base}: {result['files'][0]}")
            print(f"  Description: {description}")
        except Exception as e:
            print(f"❌ Error with base conversion {number} to base {base}: {e}")
            results.append({'error': str(e), 'type': f'base_conversion:{number}:{base}'})
    
    # 3. Pattern analysis
    print("\n3. Advanced Pattern Analysis")
    print("-" * 30)
    
    pattern_numbers = [
        (12321, "palindrome", "Perfect palindrome"),
        (1001001, "palindrome", "Scheherazade-based palindrome"),
        (123454321, "palindrome", "Symmetric palindrome"),
        (1234567890, "digit_sequence", "Sequential digits")
    ]
    
    for number, pattern_type, description in pattern_numbers:
        try:
            result = plot_pattern_analysis(number, pattern_type=pattern_type, backend='matplotlib')
            results.append(result)
            print(f"✓ Pattern analysis {number}: {result['files'][0]}")
            print(f"  Type: {pattern_type} - {description}")
        except Exception as e:
            print(f"❌ Error with pattern analysis {number}: {e}")
            results.append({'error': str(e), 'type': f'pattern_analysis:{number}'})
    
    return results


def generate_summary_report(all_results):
    """Generate a summary report of all PNG visualizations."""
    print("\n" + "="*60)
    print("PNG VISUALIZATION SUMMARY REPORT")
    print("="*60)
    
    total_files = sum(len(result.get('files', [])) for result in all_results)
    successful_results = [r for r in all_results if 'error' not in r and r.get('files')]
    
    print(f"\n📊 GENERATION STATISTICS:")
    print(f"   Total visualizations attempted: {len(all_results)}")
    print(f"   Successful generations: {len(successful_results)}")
    print(f"   Total PNG files created: {total_files}")
    print(f"   Success rate: {len(successful_results)/len(all_results)*100:.1f}%" if all_results else "   No visualizations attempted")
    
    # Check organized file structure
    output_dir = Path('output')
    if output_dir.exists():
        # Get all PNG files in organized structure
        png_files = list(output_dir.rglob('*.png'))
        
        if png_files:
            total_size = sum(f.stat().st_size for f in png_files)
            avg_size = total_size / len(png_files)
            print(f"\n📁 ORGANIZED FILE STATISTICS:")
            print(f"   Existing PNG files (may predate this run): {len(png_files)}")
            print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")
            print(f"   Average file size: {avg_size / 1024:.1f} KB")
            
            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)
            for png_file in png_files:
                if len(png_file.parts) >= 3:  # output/category/subcategory/file.png
                    category = png_file.parts[1]
                    by_category[category].append(png_file)
            
            print(f"\n📋 ORGANIZED PNG FILES BY CATEGORY:")
            for category, files in sorted(by_category.items()):
                print(f"\n   {category.upper()}/ ({len(files)} files):")
                by_subcat = defaultdict(list)
                for f in files:
                    if len(f.parts) >= 4:
                        subcat = f.parts[2]
                        by_subcat[subcat].append(f)
                
                for subcat, subfiles in sorted(by_subcat.items()):
                    print(f"     {subcat}/ ({len(subfiles)} files)")
                    for sf in sorted(subfiles)[:3]:  # Show first 3 files
                        size_kb = sf.stat().st_size / 1024
                        print(f"       • {sf.name} ({size_kb:.1f} KB)")
                    if len(subfiles) > 3:
                        print(f"       ... and {len(subfiles) - 3} more files")
    
    print(f"\n✅ PNG VISUALIZATION DEMO COMPLETE!")
    print(f"   All high-quality PNG files organized in: output/")
    print(f"   📂 Structure: output/{{category}}/{{subcategory}}/{{file}}.png")
    print(f"   📖 See output/README.md for detailed structure explanation")
    print(f"   🎨 Resolution: 300 DPI for print-quality output")
    print(f"   📐 Format: PNG with optimized compression and organized categorization")


def main():
    """Main demo function."""
    print("SYMERGETICS PNG VISUALIZATION DEMO")
    print("=" * 60)
    print("\nThis demo generates comprehensive PNG visualizations showcasing")
    print("the mathematical and geometric capabilities of the Symergetics package.")
    print("All outputs are saved as high-quality PNG files in the 'output/' directory.")
    
    # Setup configuration
    setup_png_config()
    
    # Generate all visualizations
    all_results = []
    
    try:
        # Geometric visualizations
        geometric_results = demo_geometric_png_visualizations()
        all_results.extend(geometric_results)
        
        # Number pattern visualizations
        number_results = demo_number_pattern_png_visualizations()
        all_results.extend(number_results)
        
        # Mathematical visualizations
        math_results = demo_mathematical_png_visualizations()
        all_results.extend(math_results)
        
        # Generate summary report
        generate_summary_report(all_results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nMake sure to check the 'output/' directory for all generated PNG files!")
    return bool(all_results) and all('error' not in result and result.get('files') for result in all_results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

