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
    # Mega graphical abstract
    create_mega_graphical_abstract,

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

    # Advanced analysis visualizations
    create_comparative_analysis_visualization,
    create_pattern_discovery_visualization,
    create_statistical_analysis_dashboard,

    # Batch processing
    batch_visualize,
    create_visualization_report,
)

# Import advanced analysis and reporting methods
from symergetics.computation.analysis import (
    analyze_mathematical_patterns,
    compare_mathematical_domains,
    generate_comprehensive_report,
)
from symergetics.utils.reporting import (
    generate_statistical_summary,
    generate_comparative_report,
    export_report_to_json,
    export_report_to_csv,
    export_report_to_markdown,
)


def demonstrate_advanced_analysis() -> List[Dict[str, Any]]:
    """Demonstrate advanced analysis, reporting, and visualization methods."""
    log("Starting advanced analysis demonstrations", "DATA")

    results = []

    try:
        # Create test datasets for analysis
        palindromic_numbers = [121, 12321, 123454321, 12345678987654321]
        mixed_numbers = [123, 456, 789, 12345678901234567890]
        scheherazade_numbers = [SymergeticsNumber(1001), SymergeticsNumber(2002)]

        log("Performing mathematical pattern analysis", "DATA")

        # Analyze all number sets
        palindromic_analyses = [analyze_mathematical_patterns(num, analysis_depth=4)
                               for num in palindromic_numbers]
        mixed_analyses = [analyze_mathematical_patterns(num, analysis_depth=4)
                         for num in mixed_numbers]
        scheherazade_analyses = [analyze_mathematical_patterns(num, analysis_depth=4)
                                for num in scheherazade_numbers]

        log(f"✅ Analyzed {len(palindromic_analyses + mixed_analyses + scheherazade_analyses)} numbers")

        # Generate statistical summaries
        log("Generating statistical summaries", "DATA")

        palindromic_summary = generate_statistical_summary(
            palindromic_analyses,
            "Palindromic Numbers Analysis"
        )
        mixed_summary = generate_statistical_summary(
            mixed_analyses,
            "Mixed Numbers Analysis"
        )

        log("✅ Statistical summaries generated")

        # Create comparative analysis
        log("Creating comparative analysis", "DATA")

        comparative_report = generate_comparative_report(
            palindromic_analyses, mixed_analyses,
            "Palindromic Numbers", "Mixed Numbers"
        )

        log("✅ Comparative analysis completed")

        # Export reports in multiple formats
        log("Exporting reports to multiple formats", "DATA")

        output_dir = Path("output/mathematical/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export statistical summaries
        export_report_to_json(palindromic_summary, output_dir / "palindromic_analysis.json")
        export_report_to_markdown(palindromic_summary, output_dir / "palindromic_analysis.md")
        export_report_to_json(mixed_summary, output_dir / "mixed_analysis.json")
        export_report_to_csv(mixed_summary, output_dir / "mixed_analysis.csv")

        # Export comparative report
        export_report_to_json(comparative_report, output_dir / "comparative_analysis.json")
        export_report_to_markdown(comparative_report, output_dir / "comparative_analysis.md")

        log("✅ Reports exported to multiple formats")

        # Create advanced visualizations
        log("Creating advanced analysis visualizations", "VISUAL")

        # Comparative analysis visualization
        comparative_viz = create_comparative_analysis_visualization(
            palindromic_analyses, mixed_analyses,
            "Palindromic Numbers", "Mixed Numbers"
        )
        results.append({
            'type': 'comparative_visualization',
            'files': comparative_viz['files'],
            'metadata': comparative_viz['metadata']
        })
        log("✅ Comparative analysis visualization created")

        # Pattern discovery visualization
        all_analyses = palindromic_analyses + mixed_analyses + scheherazade_analyses
        pattern_viz = create_pattern_discovery_visualization(
            all_analyses,
            title="Comprehensive Pattern Discovery Analysis"
        )
        results.append({
            'type': 'pattern_discovery_visualization',
            'files': pattern_viz['files'],
            'metadata': pattern_viz['metadata']
        })
        log("✅ Pattern discovery visualization created")

        # Statistical dashboard
        dashboard_viz = create_statistical_analysis_dashboard(
            all_analyses,
            title="Advanced Mathematical Analysis Dashboard"
        )
        results.append({
            'type': 'statistical_dashboard',
            'files': dashboard_viz['files'],
            'metadata': dashboard_viz['metadata']
        })
        log("✅ Statistical analysis dashboard created")

        # Generate comprehensive analysis report
        log("Generating comprehensive analysis report", "DATA")

        comprehensive_report = generate_comprehensive_report(
            all_analyses[0],  # Use first analysis as example
            title="Symergetics Advanced Analysis Report",
            include_visualizations=True
        )

        export_report_to_json(comprehensive_report, output_dir / "comprehensive_report.json")
        export_report_to_markdown(comprehensive_report, output_dir / "comprehensive_report.md")

        log("✅ Comprehensive analysis report generated")

        # Record analysis results
        results.extend([
            {
                'type': 'statistical_summary',
                'domain': 'palindromic',
                'files': [
                    str(output_dir / "palindromic_analysis.json"),
                    str(output_dir / "palindromic_analysis.md")
                ]
            },
            {
                'type': 'statistical_summary',
                'domain': 'mixed',
                'files': [
                    str(output_dir / "mixed_analysis.json"),
                    str(output_dir / "mixed_analysis.csv")
                ]
            },
            {
                'type': 'comparative_report',
                'files': [
                    str(output_dir / "comparative_analysis.json"),
                    str(output_dir / "comparative_analysis.md")
                ]
            },
            {
                'type': 'comprehensive_report',
                'files': [
                    str(output_dir / "comprehensive_report.json"),
                    str(output_dir / "comprehensive_report.md")
                ]
            }
        ])

        log(f"Advanced analysis demonstration completed successfully: {len(results)} items generated", "DATA")

    except Exception as e:
        log(f"❌ Advanced analysis demonstration failed: {e}", "ERROR")
        results.append({
            'type': 'error',
            'error': str(e)
        })

    return results


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

        # Phase 4: Advanced Analysis and Reporting
        log("Phase 4: Advanced Analysis and Reporting", "INFO")
        analysis_results = demonstrate_advanced_analysis()
        all_results.extend(analysis_results)
        log(f"Advanced analysis phase completed: {len(analysis_results)} items")

        # Phase 5: Geometric Mnemonics Demonstration
        log("Phase 5: Geometric Mnemonics Analysis", "INFO")
        try:
            from geometric_mnemonics_demo import main as geometric_main
            log("Running geometric mnemonics demonstration")
            geometric_success = geometric_main()
            if geometric_success:
                log("✅ Geometric mnemonics demonstration completed successfully")
            else:
                log("⚠️ Geometric mnemonics demonstration had issues")
        except Exception as e:
            log(f"⚠️ Geometric mnemonics demonstration failed: {e}")

        # Phase 6: Batch processing
        log("Phase 6: Batch Processing Demonstration", "INFO")
        batch_results = demonstrate_batch_processing()
        all_results.extend(batch_results)
        log(f"Batch processing phase completed: {len(batch_results)} operations")

        # Phase 7: Mega Graphical Abstract
        log("Phase 7: Mega Graphical Abstract Creation", "INFO")
        try:
            mega_result = create_mega_graphical_abstract(
                title="Symergetics Package - Comprehensive Visual Overview",
                backend="matplotlib"
            )
            all_results.append(mega_result)
            log(f"✅ Mega graphical abstract created: {mega_result['files'][0]}")
            log(f"   Panels: {mega_result['metadata']['panels']}")
            log(f"   Dimensions: {mega_result['metadata']['dimensions']}")
        except Exception as e:
            log(f"⚠️ Mega graphical abstract creation failed: {e}", "WARNING")

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
