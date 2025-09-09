#!/usr/bin/env python3
"""
Comprehensive Visualizations Demo - Symergetics Package

This script demonstrates all the advanced visualization methods in the Symergetics package,
showcasing geometric, mathematical, and numerical pattern analysis capabilities.

Features demonstrated:
- Enhanced 3D polyhedron visualizations with wireframe and surface rendering
- Graphical abstracts combining multiple visualization perspectives
- Palindromic heatmaps showing number pattern analysis
- Scheherazade network visualizations
- Primorial spectrum analysis
- Continued fraction convergence plots
- Base conversion matrices
- Pattern analysis radar charts

Usage:
    python examples/comprehensive_visualizations_demo.py
    uv run python examples/comprehensive_visualizations_demo.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import json

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from symergetics.core.numbers import SymergeticsNumber
from symergetics.geometry.polyhedra import Tetrahedron, Octahedron, Cube, Cuboctahedron
from symergetics.visualization import (
    # Enhanced geometry visualizations
    plot_polyhedron_3d,
    plot_polyhedron_graphical_abstract,
    plot_polyhedron_wireframe,

    # Enhanced number visualizations
    plot_palindromic_heatmap,
    plot_scheherazade_network,
    plot_primorial_spectrum,

    # Enhanced mathematical visualizations
    plot_continued_fraction_convergence,
    plot_base_conversion_matrix,
    plot_pattern_analysis_radar,

    # Batch processing
    batch_visualize,
    create_visualization_report,
)


def log(message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = "🎨" if level == "VISUAL" else "📊" if level == "DATA" else "ℹ️"
    print(f"[{timestamp}] {prefix} {message}")


def demonstrate_geometry_visualizations() -> List[Dict[str, Any]]:
    """Demonstrate all geometry visualization methods."""
    log("Starting geometry visualization demonstrations", "VISUAL")

    results = []

    # Create polyhedron instances
    tetrahedron = Tetrahedron()
    octahedron = Octahedron()
    cube = Cube()
    cuboctahedron = Cuboctahedron()

    polyhedra = [
        ("tetrahedron", tetrahedron),
        ("octahedron", octahedron),
        ("cube", cube),
        ("cuboctahedron", cuboctahedron)
    ]

    # Demonstrate enhanced 3D visualizations
    for name, polyhedron in polyhedra:
        log(f"Creating 3D visualization for {name}")

        # 3D with wireframe and surface
        result = plot_polyhedron_3d(
            polyhedron,
            show_wireframe=True,
            show_surface=True,
            elevation=25,
            azimuth=45,
            backend='matplotlib'
        )
        results.append(result)
        log(f"✅ 3D visualization created for {name}")

        # Graphical abstract
        result = plot_polyhedron_graphical_abstract(
            polyhedron,
            show_volume_ratios=True,
            show_coordinates=True,
            backend='matplotlib'
        )
        results.append(result)
        log(f"✅ Graphical abstract created for {name}")

        # Wireframe only
        result = plot_polyhedron_wireframe(
            polyhedron,
            elevation=30,
            azimuth=60,
            backend='matplotlib'
        )
        results.append(result)
        log(f"✅ Wireframe visualization created for {name}")

    return results


def demonstrate_number_visualizations() -> List[Dict[str, Any]]:
    """Demonstrate all number pattern visualization methods."""
    log("Starting number pattern visualization demonstrations", "VISUAL")

    results = []

    # Palindromic heatmap
    log("Creating palindromic heatmap")
    result = plot_palindromic_heatmap(
        sequence_start=100,
        sequence_end=200,
        backend='matplotlib'
    )
    results.append(result)
    log("✅ Palindromic heatmap created")

    # Scheherazade network
    log("Creating Scheherazade network visualization")
    result = plot_scheherazade_network(
        power=4,
        backend='matplotlib'
    )
    results.append(result)
    log("✅ Scheherazade network created")

    # Primorial spectrum
    log("Creating primorial spectrum visualization")
    result = plot_primorial_spectrum(
        max_n=12,
        backend='matplotlib'
    )
    results.append(result)
    log("✅ Primorial spectrum created")

    return results


def demonstrate_mathematical_visualizations() -> List[Dict[str, Any]]:
    """Demonstrate all mathematical visualization methods."""
    log("Starting mathematical visualization demonstrations", "VISUAL")

    results = []

    # Continued fraction convergence
    log("Creating continued fraction convergence visualization")
    result = plot_continued_fraction_convergence(
        value=3.14159,
        max_terms=15,
        backend='matplotlib'
    )
    results.append(result)
    log("✅ Continued fraction convergence created")

    # Base conversion matrix
    log("Creating base conversion matrix")
    result = plot_base_conversion_matrix(
        start_base=2,
        end_base=12,
        number=1001,
        backend='matplotlib'
    )
    results.append(result)
    log("✅ Base conversion matrix created")

    # Pattern analysis radar
    log("Creating pattern analysis radar")
    sequence = "123454321"  # Palindromic sequence
    result = plot_pattern_analysis_radar(
        sequence=sequence,
        backend='matplotlib'
    )
    results.append(result)
    log("✅ Pattern analysis radar created")

    return results


def demonstrate_batch_processing() -> List[Dict[str, Any]]:
    """Demonstrate batch processing of visualizations."""
    log("Starting batch processing demonstration", "DATA")

    # Create a comprehensive batch of visualization tasks
    batch_tasks = [
        # Geometry visualizations
        {
            'function': 'plot_polyhedron_3d',
            'args': [Tetrahedron()],
            'kwargs': {'show_wireframe': True, 'backend': 'matplotlib'}
        },
        {
            'function': 'plot_polyhedron_graphical_abstract',
            'args': [Octahedron()],
            'kwargs': {'show_volume_ratios': True, 'backend': 'matplotlib'}
        },

        # Number visualizations
        {
            'function': 'plot_palindromic_heatmap',
            'args': [],
            'kwargs': {'sequence_start': 50, 'sequence_end': 150, 'backend': 'matplotlib'}
        },
        {
            'function': 'plot_primorial_spectrum',
            'args': [],
            'kwargs': {'max_n': 10, 'backend': 'matplotlib'}
        },

        # Mathematical visualizations
        {
            'function': 'plot_continued_fraction_convergence',
            'args': [],
            'kwargs': {'value': 2.71828, 'max_terms': 12, 'backend': 'matplotlib'}
        },
        {
            'function': 'plot_base_conversion_matrix',
            'args': [],
            'kwargs': {'start_base': 2, 'end_base': 10, 'number': 12345, 'backend': 'matplotlib'}
        },
    ]

    # Execute batch processing
    batch_results = batch_visualize(batch_tasks)
    log(f"✅ Batch processing completed: {len([r for r in batch_results if 'error' not in r])} successful, {len([r for r in batch_results if 'error' in r])} failed")

    return batch_results


def main():
    """Main demonstration function."""
    print("🎨 COMPREHENSIVE VISUALIZATIONS DEMO")
    print("=" * 50)
    print("Demonstrating advanced visualization capabilities of Symergetics")
    print()

    start_time = time.time()
    all_results = []

    try:
        # Phase 1: Geometry visualizations
        log("Phase 1: Geometry Visualizations", "INFO")
        geometry_results = demonstrate_geometry_visualizations()
        all_results.extend(geometry_results)
        log(f"Geometry phase completed: {len(geometry_results)} visualizations")

        # Phase 2: Number pattern visualizations
        log("Phase 2: Number Pattern Visualizations", "INFO")
        number_results = demonstrate_number_visualizations()
        all_results.extend(number_results)
        log(f"Number patterns phase completed: {len(number_results)} visualizations")

        # Phase 3: Mathematical visualizations
        log("Phase 3: Mathematical Visualizations", "INFO")
        math_results = demonstrate_mathematical_visualizations()
        all_results.extend(math_results)
        log(f"Mathematical phase completed: {len(math_results)} visualizations")

        # Phase 4: Batch processing
        log("Phase 4: Batch Processing Demonstration", "INFO")
        batch_results = demonstrate_batch_processing()
        all_results.extend(batch_results)
        log(f"Batch processing phase completed: {len(batch_results)} operations")

        # Generate comprehensive report
        log("Generating visualization report", "DATA")
        report_path = create_visualization_report(all_results, "Comprehensive Visualizations Demo")
        log(f"Report generated: {report_path}")

        # Summary statistics
        total_duration = time.time() - start_time
        successful = len([r for r in all_results if 'error' not in r])
        failed = len([r for r in all_results if 'error' in r])
        total_files = sum(len(r.get('files', [])) for r in all_results if 'error' not in r)

        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE VISUALIZATIONS SUMMARY")
        print("=" * 60)
        print(f"Total execution time: {total_duration:.1f} seconds")
        print(f"Visualizations attempted: {len(all_results)}")
        print(f"Visualizations successful: {successful}")
        print(f"Visualizations failed: {failed}")
        print(f"Success rate: {successful/len(all_results)*100:.1f}%" if all_results else "No visualizations")
        print(f"Total files generated: {total_files}")

        if failed > 0:
            print("\n❌ Failed visualizations:")
            for result in all_results:
                if 'error' in result:
                    print(f"   • {result.get('error', 'Unknown error')}")

        print("\n✅ Demo completed successfully!")
        print(f"📁 Check the output/ directory for generated visualizations")
        print(f"📋 Detailed report: {report_path}")

        return True

    except Exception as e:
        log(f"💥 Demo failed with exception: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
