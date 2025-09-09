#!/usr/bin/env python3
"""
Integration Demonstration

This example showcases how different modules of the Symergetics package
work together in practical applications, demonstrating:

- Cross-module integration and workflow
- Real-world mathematical problem solving
- Combined geometric and numerical analysis
- Comprehensive visualization pipelines
- Research methodology examples

Perfect for understanding how to use the package for complex research.
"""

import sys
from pathlib import Path
import math

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from symergetics.core.numbers import SymergeticsNumber, rational_sqrt
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.core.constants import SymergeticsConstants
from symergetics.geometry.polyhedra import Tetrahedron, Octahedron, Cube, Cuboctahedron
from symergetics.geometry.transformations import scale
from symergetics.computation.palindromes import is_palindromic, analyze_scheherazade_ssrcd
from symergetics.computation.primorials import primorial, scheherazade_power
from symergetics.utils.conversion import continued_fraction_approximation, convert_between_bases
from symergetics.utils.mnemonics import create_memory_aid, format_large_number
from symergetics.visualization import (
    set_config, plot_polyhedron, plot_quadray_coordinate, plot_palindromic_pattern,
    plot_scheherazade_pattern, plot_continued_fraction, batch_visualize,
    create_visualization_report, create_output_structure_readme
)


def integrated_geometric_analysis():
    """Demonstrate integrated geometric and numerical analysis."""
    print("🔗 INTEGRATED GEOMETRIC ANALYSIS")
    print("="*40)
    
    print("\n1. Polyhedron-Coordinate Integration:")
    print("-" * 40)
    
    # Use polyhedron volumes as coordinate multipliers
    constants = SymergeticsConstants()
    polyhedra_data = []
    
    polyhedron_classes = [
        ("Tetrahedron", Tetrahedron),
        ("Octahedron", Octahedron),
        ("Cube", Cube),
        ("Cuboctahedron", Cuboctahedron),
    ]
    
    for name, poly_class in polyhedron_classes:
        poly = poly_class()
        volume = poly.volume()
        
        # Use volume as coordinate scaling factor
        base_coord = QuadrayCoordinate(1, 1, 1, 1)
        scaled_coord = scale(base_coord, float(volume))
        
        # Convert to XYZ for spatial analysis
        xyz = scaled_coord.to_xyz()
        
        polyhedra_data.append({
            'name': name,
            'volume': volume,
            'coordinate': scaled_coord,
            'xyz': xyz,
            'distance_from_origin': scaled_coord.magnitude()
        })
        
        print(f"{name:12}: Volume = {volume}, Coord = {scaled_coord}")
        print(f"              XYZ = ({xyz[0]:6.3f}, {xyz[1]:6.3f}, {xyz[2]:6.3f})")
        print(f"              Distance = {scaled_coord.magnitude():.6f}")
        
        # Generate visualization
        result = plot_polyhedron(name.lower(), backend='ascii')
        print(f"              Visual: {result['files'][0]}")
        print()
        
    print("2. Volume-Space Relationship Analysis:")
    print("-" * 40)
    
    # Analyze relationships between volumes and spatial positions
    volumes = [data['volume'] for data in polyhedra_data]
    distances = [data['distance_from_origin'] for data in polyhedra_data]
    
    print("Volume ratios (relative to tetrahedron):")
    tet_volume = volumes[0]  # Tetrahedron is first
    for i, data in enumerate(polyhedra_data):
        ratio = float(data['volume'] / tet_volume)
        print(f"  {data['name']:12}: {ratio:4.0f}:1")
        
    # Find mathematical relationships
    print(f"\nSpatial pattern analysis:")
    print(f"  Average distance: {sum(distances)/len(distances):.3f}")
    print(f"  Distance range: {min(distances):.3f} to {max(distances):.3f}")
    
    # Check for geometric progressions
    distance_ratios = []
    for i in range(1, len(distances)):
        if distances[i-1] != 0:  # Avoid division by zero
            ratio = distances[i] / distances[i-1]
            distance_ratios.append(ratio)
        else:
            distance_ratios.append(float('inf'))  # Indicate infinite ratio
        
    print(f"  Distance growth ratios: {[f'{r:.2f}' for r in distance_ratios]}")


def integrated_number_pattern_analysis():
    """Demonstrate integrated number pattern and geometric analysis."""
    print("\n🔢 INTEGRATED NUMBER PATTERN ANALYSIS")
    print("="*45)
    
    print("\n1. Scheherazade-Geometry Integration:")
    print("-" * 40)
    
    # Use Scheherazade numbers in geometric contexts
    for power in range(2, 6):
        sch_num = scheherazade_power(power)
        analysis = analyze_scheherazade_ssrcd(power)
        
        print(f"\n1001^{power} Analysis:")
        print(f"  Number: {str(sch_num.value)[:30]}{'...' if len(str(sch_num.value)) > 30 else ''}")
        print(f"  Length: {len(str(sch_num.value))} digits")
        print(f"  Palindromic: {analysis['is_palindromic']}")
        
        # Use number properties in coordinate system
        digit_sum = sum(int(d) for d in str(sch_num.value))
        coord = QuadrayCoordinate(power, digit_sum % 10, len(str(sch_num.value)), 0)
        xyz = coord.to_xyz()
        
        print(f"  Derived coordinate: {coord}")
        print(f"  XYZ position: ({xyz[0]:6.3f}, {xyz[1]:6.3f}, {xyz[2]:6.3f})")
        
        # Generate dual visualizations
        sch_result = plot_scheherazade_pattern(power, backend='ascii')
        coord_result = plot_quadray_coordinate(coord, backend='ascii')
        
        print(f"  Pattern visual: {sch_result['files'][0]}")
        print(f"  Coord visual: {coord_result['files'][0]}")
        
        # Memory aid for large numbers
        if len(str(sch_num.value)) > 10:
            memory_aid = create_memory_aid(sch_num)
            print(f"  Memory aid: {memory_aid}")
            
    print("\n2. Primorial-Coordinate Mapping:")
    print("-" * 35)
    
    # Map primorials to coordinate space
    primorial_coords = []
    
    for n in range(2, 8):
        try:
            p = primorial(n)
            p_str = str(p.value)
            
            # Create coordinate based on primorial properties
            # Use n, length, digit sum, and last digit
            last_digit = int(p_str[-1])
            digit_sum = sum(int(d) for d in p_str) % 100  # Mod to keep manageable
            coord = QuadrayCoordinate(n, len(p_str), digit_sum, last_digit)
            
            primorial_coords.append({
                'n': n,
                'primorial': p.value,
                'coordinate': coord,
                'xyz': coord.to_xyz()
            })
            
            print(f"{n:2d}# = {p_str[:20]}{'...' if len(p_str) > 20 else ''}")
            print(f"     Coordinate: {coord}")
            print(f"     XYZ: ({coord.to_xyz()[0]:6.3f}, {coord.to_xyz()[1]:6.3f}, {coord.to_xyz()[2]:6.3f})")
            
        except Exception as e:
            print(f"{n:2d}# = [Error: {e}]")
            break
            
    # Analyze primorial coordinate patterns
    if len(primorial_coords) >= 2:
        print(f"\nPrimorial coordinate movement:")
        for i in range(1, len(primorial_coords)):
            current = primorial_coords[i]
            previous = primorial_coords[i-1]
            
            distance = current['coordinate'].distance_to(previous['coordinate'])
            print(f"  {previous['n']}# -> {current['n']}#: distance = {distance:.3f}")


def integrated_research_workflow():
    """Demonstrate a complete research workflow using multiple modules."""
    print("\n🔬 INTEGRATED RESEARCH WORKFLOW")
    print("="*40)
    
    print("\n1. Research Question: Golden Ratio in Symergetics")
    print("-" * 50)
    
    # Research the golden ratio using multiple approaches
    phi_exact = (SymergeticsNumber(1) + rational_sqrt(SymergeticsNumber(5))) / SymergeticsNumber(2)
    phi_float = float(phi_exact.value)
    
    print(f"Golden ratio φ = {phi_exact.value} = {phi_float:.15f}")
    
    # 1. Continued fraction analysis
    cf = continued_fraction_approximation(phi_float, max_terms=10)
    print(f"Continued fraction: {cf}")
    
    cf_result = plot_continued_fraction(phi_float, max_terms=8, backend='ascii')
    print(f"CF visualization: {cf_result['files'][0]}")
    
    # 2. Check for palindromic properties in φ approximations
    phi_convergents = [
        (1, 1), (2, 1), (3, 2), (5, 3), (8, 5), (13, 8), (21, 13), (34, 21)
    ]  # Fibonacci ratios
    
    print(f"\nFibonacci-based φ approximations:")
    for i, (num, den) in enumerate(phi_convergents):
        approx = num / den
        error = abs(phi_float - approx)
        
        # Check if numerator or denominator is palindromic
        num_pal = is_palindromic(num)
        den_pal = is_palindromic(den)
        
        print(f"  F({i+7})/F({i+6}) = {num:2d}/{den:2d} = {approx:.10f} (error: {error:.2e})")
        print(f"    Palindromic: num={num_pal}, den={den_pal}")
        
        if num_pal or den_pal:
            pal_result = plot_palindromic_pattern(num if num_pal else den, backend='ascii')
            print(f"    Palindrome visual: {pal_result['files'][0]}")
            
    # 3. Geometric representation of φ
    print(f"\nGeometric representation:")
    
    # Use φ in coordinate system
    phi_coord = QuadrayCoordinate(phi_float, 1, phi_float-1, 0)
    xyz = phi_coord.to_xyz()
    
    print(f"φ coordinate: {phi_coord}")
    print(f"XYZ: ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f})")
    
    coord_result = plot_quadray_coordinate(phi_coord, backend='ascii')
    print(f"Coord visualization: {coord_result['files'][0]}")
    
    print("\n2. Cross-Module Pattern Discovery:")
    print("-" * 35)
    
    # Discover patterns by combining different modules
    interesting_numbers = []
    
    # Collect numbers from different sources
    # From Scheherazade sequence
    for power in range(1, 5):
        sch = scheherazade_power(power)
        interesting_numbers.append(('Scheherazade', f'1001^{power}', sch.value))
        
    # From primorials
    for n in [5, 7]:
        try:
            p = primorial(n)
            interesting_numbers.append(('Primorial', f'{n}#', p.value))
        except:
            pass
            
    # From polyhedron volumes (scaled to integers)
    poly = Cuboctahedron()
    vol = poly.volume()
    scaled_vol = int(float(vol) * 1000)  # Scale for integer analysis
    interesting_numbers.append(('Geometric', 'Cuboctahedron×1000', scaled_vol))
    
    print("Cross-module pattern analysis:")
    for category, description, number in interesting_numbers:
        num_str = str(number)
        
        # Multi-faceted analysis
        is_pal = is_palindromic(number)
        digit_sum = sum(int(d) for d in num_str)
        
        # Base conversions
        binary = convert_between_bases(number, 10, 2) if number < 10**6 else "[too large]"
        
        # Coordinate mapping
        coord = QuadrayCoordinate(len(num_str), digit_sum % 10, number % 100, 0)
        
        print(f"\n{category:12} - {description}:")
        print(f"  Value: {num_str[:30]}{'...' if len(num_str) > 30 else ''}")
        print(f"  Properties: palindromic={is_pal}, digits={len(num_str)}, sum={digit_sum}")
        print(f"  Binary: {binary}")
        print(f"  Coordinate: {coord}")
        
        # Memory aid for complex numbers
        if len(num_str) > 8:
            try:
                memory_aid = create_memory_aid(number)
                print(f"  Memory aid: {memory_aid[:50]}{'...' if len(memory_aid) > 50 else ''}")
            except:
                print(f"  Memory aid: [generation failed]")


def demonstrate_batch_research_pipeline():
    """Demonstrate batch processing for research workflows."""
    print("\n🔄 BATCH RESEARCH PIPELINE")
    print("="*30)
    
    print("\n1. Automated Visualization Pipeline:")
    print("-" * 40)
    
    # Create a research-focused batch of visualizations
    research_tasks = []
    
    # Golden ratio analysis series
    phi = (1 + math.sqrt(5)) / 2
    research_tasks.extend([
        {'function': 'plot_continued_fraction', 'args': [phi], 'kwargs': {'max_terms': 10}},
        {'function': 'plot_continued_fraction', 'args': [math.pi], 'kwargs': {'max_terms': 10}},
        {'function': 'plot_continued_fraction', 'args': [math.e], 'kwargs': {'max_terms': 10}},
    ])
    
    # Scheherazade number series
    for power in range(2, 5):
        research_tasks.append({
            'function': 'plot_scheherazade_pattern', 
            'args': [power]
        })
        
    # Geometric series
    polyhedra = ['tetrahedron', 'octahedron', 'cube', 'cuboctahedron']
    for poly in polyhedra:
        research_tasks.append({
            'function': 'plot_polyhedron',
            'args': [poly]
        })
        
    # Palindromic number series
    palindromes = [121, 12321, 1234321, 123454321]
    for pal in palindromes:
        research_tasks.append({
            'function': 'plot_palindromic_pattern',
            'args': [pal]
        })
        
    print(f"Processing {len(research_tasks)} research visualizations...")
    
    # Execute batch processing
    results = batch_visualize(research_tasks, backend='ascii')
    
    # Analyze results
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\nBatch processing results:")
    print(f"  Successful: {len(successful)}/{len(research_tasks)} ({len(successful)/len(research_tasks)*100:.1f}%)")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"  Failures:")
        for result in failed:
            print(f"    - {result.get('function', 'unknown')}: {result.get('error', 'unknown error')}")
            
    # Generate comprehensive report
    report_path = create_visualization_report(results, "Integrated Research Pipeline Report")
    print(f"\n📊 Comprehensive report: {report_path}")
    
    print("\n2. Research Data Analysis:")
    print("-" * 30)
    
    # Analyze the batch results for patterns
    visualization_types = {}
    total_files = 0
    
    for result in successful:
        if 'metadata' in result:
            viz_type = result['metadata'].get('type', 'unknown')
            visualization_types[viz_type] = visualization_types.get(viz_type, 0) + 1
            total_files += len(result.get('files', []))
            
    print(f"Research output analysis:")
    print(f"  Total files generated: {total_files}")
    print(f"  Visualization types:")
    for viz_type, count in sorted(visualization_types.items()):
        print(f"    {viz_type}: {count} instances")
        
    # Success metrics
    overall_success_rate = len(successful) / len(research_tasks) * 100
    print(f"  Overall pipeline success rate: {overall_success_rate:.1f}%")
    
    if overall_success_rate >= 90:
        print("  Assessment: ✅ Research pipeline highly reliable")
    elif overall_success_rate >= 75:
        print("  Assessment: ⚠️ Research pipeline moderately reliable")
    else:
        print("  Assessment: ❌ Research pipeline needs improvement")


def main():
    """Run the comprehensive integration demonstration."""
    print("🎯 SYMERGETICS INTEGRATION DEMONSTRATION")
    print("="*60)
    print()
    print("This demonstration showcases how different modules of the")
    print("Symergetics package integrate for advanced mathematical research:")
    print()
    print("• 🔗 Cross-module integration workflows")
    print("• 🔢 Combined numerical and geometric analysis")
    print("• 🔬 Complete research methodologies")
    print("• 🔄 Batch processing pipelines")
    print("• 📊 Comprehensive result analysis")
    print()
    
    # Configure for comprehensive organized output
    set_config({
        'backend': 'ascii',
        'output_dir': 'output',
        'organize_by_type': True,
        'include_timestamps': False
    })
    
    # Create organized structure with documentation
    create_output_structure_readme()
    
    try:
        # Run all integration demonstrations
        integrated_geometric_analysis()
        integrated_number_pattern_analysis()
        integrated_research_workflow()
        demonstrate_batch_research_pipeline()
        
        print(f"\n" + "="*60)
        print("🎉 SYMERGETICS INTEGRATION DEMONSTRATION COMPLETE!")
        print("="*60)
        print()
        print("🔬 Research Capabilities Demonstrated:")
        print("✓ Multi-module mathematical analysis workflows")
        print("✓ Cross-domain pattern recognition and discovery")
        print("✓ Automated batch processing for research efficiency")
        print("✓ Comprehensive visualization and reporting systems")
        print("✓ Integrated geometric and numerical methodologies")
        print()
        print("📊 Integration Benefits:")
        print("• Seamless workflow between different mathematical domains")
        print("• Automatic organization of complex research outputs")
        print("• Pattern discovery across numerical and geometric systems")
        print("• Scalable processing for large research projects")
        print("• Comprehensive documentation and reproducibility")
        print()
        print("🚀 Research Outcomes:")
        print("• Golden ratio analysis combining multiple approaches")
        print("• Scheherazade-geometric coordinate integration")
        print("• Primorial spatial mapping methodologies")
        print("• Cross-module pattern discovery techniques")
        print("• Automated research pipeline validation")
        print()
        print("📖 The Symergetics package successfully demonstrates Fuller's")
        print("   vision of integrated mathematical understanding where")
        print("   'everything is connected to everything else'!")
        
    except Exception as e:
        print(f"\n❌ Integration demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
