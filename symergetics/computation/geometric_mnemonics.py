#!/usr/bin/env python3
"""
Geometric Mnemonics for Integer Ratio Systems

This module explores the relationship between large integers, rational approximations
of irrational numbers, and their geometric interpretations as mnemonics of platonic
shapes in Synergetic close-packed sphere (IVM) geometry.

Key Concepts:
- Integer ratios as geometric encodings
- Platonic solid volume/edge/surface relationships
- IVM (Isotropic Vector Matrix) geometric interpretations
- Decimal approximation via geometric scaling

Author: Symergetics Team
"""

import math
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict
from fractions import Fraction
import matplotlib.pyplot as plt

from ..core.numbers import SymergeticsNumber
from ..core.constants import SymergeticsConstants
from ..geometry.polyhedra import (
    Tetrahedron, Octahedron, Cube, Cuboctahedron,
    integer_tetra_volume
)
from ..core.coordinates import QuadrayCoordinate
from ..computation.palindromes import is_palindromic


class GeometricMnemonic:
    """
    Represents a geometric interpretation of an integer as a mnemonic
    for platonic solid properties in IVM geometry.
    """

    def __init__(self, value: Union[int, str, SymergeticsNumber],
                 interpretation: str = "", geometric_form: str = ""):
        self.value = value if isinstance(value, SymergeticsNumber) else SymergeticsNumber(value)
        self.interpretation = interpretation
        self.geometric_form = geometric_form
        self.rational_approximation = None
        self.platonic_relationships = {}

    def __str__(self):
        return f"GeometricMnemonic({self.value}, {self.interpretation}, {self.geometric_form})"

    def __repr__(self):
        return self.__str__()


def analyze_geometric_mnemonics(number: Union[int, str, SymergeticsNumber],
                               analysis_depth: int = 3) -> Dict[str, Any]:
    """
    Analyze a number for geometric mnemonic relationships in IVM space.

    Args:
        number: Number to analyze
        analysis_depth: Depth of analysis (1-5)

    Returns:
        Dict containing geometric mnemonic analysis
    """
    if isinstance(number, SymergeticsNumber):
        num_value = number
    else:
        num_value = SymergeticsNumber(number)

    analysis = {
        'number': str(number) if not isinstance(number, SymergeticsNumber) else str(number.value),
        'numeric_value': float(num_value.value),
        'is_palindromic': is_palindromic(number),
        'geometric_mnemonics': [],
        'platonic_relationships': {},
        'ivm_scaling_factors': {},
        'rational_approximations': {}
    }

    # Basic platonic solid relationships
    constants = SymergeticsConstants()
    platonic_volumes = constants.VOLUME_RATIOS

    # Check for direct volume relationships
    for solid, volume in platonic_volumes.items():
        ratio = num_value / volume
        if ratio.value.is_integer() and ratio.value != 0:
            analysis['platonic_relationships'][solid] = {
                'scaling_factor': int(ratio.value),
                'relationship_type': 'volume_multiple',
                'description': f"{int(ratio.value)} × {solid} volume"
            }

    # Check for edge/face relationships
    _analyze_edge_face_relationships(num_value, analysis)

    # Check for IVM lattice relationships
    if analysis_depth >= 2:
        _analyze_ivm_lattice_relationships(num_value, analysis)

    # Analyze rational approximations
    if analysis_depth >= 3:
        _analyze_rational_approximations(num_value, analysis)

    # Advanced geometric interpretations
    if analysis_depth >= 4:
        _analyze_advanced_geometric_interpretations(num_value, analysis)

    # Create geometric mnemonic objects
    analysis['geometric_mnemonics'] = _create_geometric_mnemonics(analysis)

    return analysis


def _analyze_edge_face_relationships(num_value: SymergeticsNumber,
                                   analysis: Dict[str, Any]) -> None:
    """Analyze relationships to edges and faces of platonic solids."""

    # Edge counts for platonic solids
    edge_counts = {
        'tetrahedron': 6,
        'octahedron': 12,
        'cube': 12,
        'cuboctahedron': 24,
        'icosahedron': 30,
        'dodecahedron': 30
    }

    # Face counts
    face_counts = {
        'tetrahedron': 4,
        'octahedron': 8,
        'cube': 6,
        'cuboctahedron': 14,
        'icosahedron': 20,
        'dodecahedron': 12
    }

    # Check edge relationships
    for solid, edges in edge_counts.items():
        if num_value.value % edges == 0:
            scaling = int(num_value.value // edges)
            analysis['platonic_relationships'][f'{solid}_edges'] = {
                'scaling_factor': scaling,
                'relationship_type': 'edge_count',
                'description': f"{scaling} × {solid} edges ({edges})"
            }

    # Check face relationships
    for solid, faces in face_counts.items():
        if num_value.value % faces == 0:
            scaling = int(num_value.value // faces)
            analysis['platonic_relationships'][f'{solid}_faces'] = {
                'scaling_factor': scaling,
                'relationship_type': 'face_count',
                'description': f"{scaling} × {solid} faces ({faces})"
            }


def _analyze_ivm_lattice_relationships(num_value: SymergeticsNumber,
                                     analysis: Dict[str, Any]) -> None:
    """Analyze relationships to IVM lattice structures."""

    # IVM sphere packing relationships
    ivm_constants = {
        'tetrahedral_number': lambda n: n * (n + 1) * (n + 2) // 6,
        'octahedral_number': lambda n: n * (2*n**2 + 1) // 3,
        'cubic_number': lambda n: n**3,
        'fibonacci_sphere': lambda n: int(((1 + math.sqrt(5))/2)**n / math.sqrt(5) + 0.5)
    }

    # Check for sphere packing relationships
    for name, formula in ivm_constants.items():
        for n in range(1, 21):  # Check first 20 layers
            sphere_count = formula(n)
            if num_value.value == sphere_count:
                analysis['ivm_scaling_factors'][name] = {
                    'layer': n,
                    'sphere_count': sphere_count,
                    'description': f"{name} sphere packing for layer {n}"
                }
                break


def _analyze_rational_approximations(num_value: SymergeticsNumber,
                                   analysis: Dict[str, Any]) -> None:
    """Analyze rational approximations of mathematical constants."""

    # Key mathematical constants and their well-known rational approximations
    rational_approximations = {
        'π': [
            (22, 7),     # Famous 22/7 approximation
            (355, 113),  # More accurate 355/113
            (314, 100),  # 314/100
            (31, 10),    # 31/10
        ],
        'e': [
            (19, 7),     # 19/7 approximation
            (87, 32),    # 87/32
            (193, 71),   # 193/71
        ],
        'φ': [
            (13, 8),     # 13/8 (golden ratio)
            (21, 13),    # 21/13
            (34, 21),    # 34/21
            (55, 34),    # 55/34
            (89, 55),    # 89/55
            (144, 89),   # 144/89
            (233, 144),  # 233/144
            (377, 233),  # 377/233
            (610, 377),  # 610/377
            (577, 376),  # 577/376 (alternative approximation)
            (987, 610),  # 987/610
            (1597, 987), # 1597/987
            (2584, 1597), # 2584/1597
            (4181, 2584), # 4181/2584
            (6765, 4181), # 6765/4181
            (10946, 6765), # 10946/6765
            (17711, 10946), # 17711/10946
            (28657, 17711), # 28657/17711
            (46368, 28657), # 46368/28657
            (75025, 46368), # 75025/46368
            (121393, 75025), # 121393/75025
            (196418, 121393), # 196418/121393
            (317811, 196418), # 317811/196418
            (514229, 317811), # 514229/317811
            (832040, 514229), # 832040/514229
        ],
        '√2': [
            (7, 5),      # 7/5
            (17, 12),    # 17/12
            (41, 29),    # 41/29
        ],
        '√3': [
            (7, 4),      # 7/4
            (17, 10),    # 17/10
            (24, 14),    # 24/14
        ]
    }

    num_int = int(float(num_value.value))

    for constant_name, approximations in rational_approximations.items():
        # Calculate actual constant value
        if constant_name == 'π':
            actual_value = math.pi
        elif constant_name == 'e':
            actual_value = math.e
        elif constant_name == 'φ':
            actual_value = (1 + math.sqrt(5)) / 2
        elif constant_name == '√2':
            actual_value = math.sqrt(2)
        elif constant_name == '√3':
            actual_value = math.sqrt(3)
        else:
            continue  # Skip unknown constants

        for numerator, denominator in approximations:
            if num_int == numerator:
                error = abs(numerator/denominator - actual_value)

                analysis['rational_approximations'][f'{constant_name}_numerator_{numerator}_{denominator}'] = {
                    'constant': constant_name,
                    'approximation': f"{numerator}/{denominator}",
                    'error': error,
                    'description': f"Numerator in rational approximation of {constant_name}"
                }
            elif num_int == denominator:
                error = abs(numerator/denominator - actual_value)

                analysis['rational_approximations'][f'{constant_name}_denominator_{numerator}_{denominator}'] = {
                    'constant': constant_name,
                    'approximation': f"{numerator}/{denominator}",
                    'error': error,
                    'description': f"Denominator in rational approximation of {constant_name}"
                }


def _analyze_advanced_geometric_interpretations(num_value: SymergeticsNumber,
                                              analysis: Dict[str, Any]) -> None:
    """Analyze advanced geometric interpretations."""

    # Check for relationships to cosmic scaling factors
    constants = SymergeticsConstants()
    if hasattr(constants, 'COSMIC_SCALING_FACTORS'):
        scaling_factors = constants.COSMIC_SCALING_FACTORS
        for scale_name, factor in scaling_factors.items():
            ratio = num_value / factor
            if abs(float(ratio.value) - round(float(ratio.value))) < 0.01:
                analysis['ivm_scaling_factors'][f'cosmic_{scale_name}'] = {
                    'scaling_factor': round(float(ratio.value)),
                    'description': f"Cosmic scaling relationship: {scale_name}"
                }

    # Check for vector equilibrium relationships
    # Vector equilibrium has 12 vertices, 24 edges, 8 faces, 24 faces when considering internal structure
    vector_equilibrium_counts = [12, 24, 8, 48]  # Including internal tetrahedra

    for count in vector_equilibrium_counts:
        if num_value.value % count == 0:
            scaling = int(num_value.value // count)
            analysis['platonic_relationships']['vector_equilibrium'] = {
                'scaling_factor': scaling,
                'relationship_type': 'vector_equilibrium_element',
                'description': f"{scaling} × vector equilibrium {count}"
            }


def _create_geometric_mnemonics(analysis: Dict[str, Any]) -> List[GeometricMnemonic]:
    """Create GeometricMnemonic objects from analysis results."""

    mnemonics = []

    # Create mnemonics from platonic relationships
    for relationship_name, relationship_data in analysis['platonic_relationships'].items():
        mnemonic = GeometricMnemonic(
            value=analysis['number'],
            interpretation=relationship_data['description'],
            geometric_form=relationship_name.split('_')[0]  # Extract solid name
        )
        mnemonics.append(mnemonic)

    # Create mnemonics from IVM relationships
    for relationship_name, relationship_data in analysis['ivm_scaling_factors'].items():
        mnemonic = GeometricMnemonic(
            value=analysis['number'],
            interpretation=relationship_data['description'],
            geometric_form='ivm_lattice'
        )
        mnemonics.append(mnemonic)

    # Create mnemonics from rational approximations
    for approximation_name, approximation_data in analysis['rational_approximations'].items():
        mnemonic = GeometricMnemonic(
            value=analysis['number'],
            interpretation=approximation_data['description'],
            geometric_form='rational_approximation'
        )
        mnemonics.append(mnemonic)

    return mnemonics


def create_integer_ratio_visualization(
    numbers: List[Union[int, str, SymergeticsNumber]],
    title: str = "Integer Ratio Geometric Mnemonics",
    backend: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create visualization of integer ratio geometric relationships.

    Args:
        numbers: List of numbers to analyze
        title: Visualization title
        backend: Visualization backend

    Returns:
        Dict containing visualization metadata
    """
    backend = backend or 'matplotlib'

    if backend == 'matplotlib':
        return _create_integer_ratio_matplotlib(numbers, title)
    else:
        raise ValueError(f"Integer ratio visualization requires matplotlib backend, got: {backend}")


def _create_integer_ratio_matplotlib(
    numbers: List[Union[int, str, SymergeticsNumber]],
    title: str
) -> Dict[str, Any]:
    """Create integer ratio visualization using matplotlib."""

    if not numbers:
        raise ValueError("Cannot create visualization with empty number list")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib is required for integer ratio visualization")

    # Analyze all numbers
    analyses = [analyze_geometric_mnemonics(num, analysis_depth=4) for num in numbers]

    # Extract geometric relationships
    relationships = _extract_geometric_relationships(analyses)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Geometric form distribution
    _plot_geometric_distribution(axes[0, 0], relationships)

    # Plot 2: Scaling factor analysis
    _plot_scaling_factors(axes[0, 1], relationships)

    # Plot 3: IVM relationship network
    _plot_ivm_relationships(axes[1, 0], relationships)

    # Plot 4: Rational approximation quality
    _plot_rational_approximations(axes[1, 1], analyses)

    plt.tight_layout()

    # Save visualization
    from ..visualization import get_organized_output_path
    safe_title = title.lower().replace(' ', '_').replace('-', '_')
    output_path = get_organized_output_path('mathematical', 'geometric_mnemonics',
                                          f'geometric_mnemonics_{safe_title}.png')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'files': [str(output_path)],
        'metadata': {
            'type': 'geometric_mnemonics',
            'title': title,
            'numbers_analyzed': len(numbers),
            'relationships_found': len(relationships),
            'backend': 'matplotlib'
        }
    }


def _extract_geometric_relationships(analyses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Extract geometric relationships from analyses."""

    relationships = defaultdict(list)

    for i, analysis in enumerate(analyses):
        for rel_name, rel_data in analysis.get('platonic_relationships', {}).items():
            relationships[rel_name].append({
                'number_index': i,
                'number': analysis['number'],
                'scaling_factor': rel_data.get('scaling_factor', 1),
                'description': rel_data.get('description', '')
            })

        for rel_name, rel_data in analysis.get('ivm_scaling_factors', {}).items():
            relationships[f'ivm_{rel_name}'].append({
                'number_index': i,
                'number': analysis['number'],
                'layer': rel_data.get('layer', 0),
                'description': rel_data.get('description', '')
            })

    return dict(relationships)


def _plot_geometric_distribution(ax, relationships):
    """Plot distribution of geometric forms."""
    if not relationships:
        ax.text(0.5, 0.5, 'No geometric relationships found',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Geometric Form Distribution\n(No Data)')
        return

    # Count relationships by geometric form
    form_counts = defaultdict(int)
    for rel_name in relationships.keys():
        if '_' in rel_name:
            form = rel_name.split('_')[0]
        else:
            form = rel_name
        form_counts[form] += len(relationships[rel_name])

    forms = list(form_counts.keys())
    counts = list(form_counts.values())

    bars = ax.bar(forms, counts, color='lightblue', edgecolor='navy', alpha=0.7)
    ax.set_ylabel('Number of Relationships')
    ax.set_title('Geometric Form Distribution')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{count}', ha='center', va='bottom')


def _plot_scaling_factors(ax, relationships):
    """Plot scaling factor distribution."""
    if not relationships:
        ax.text(0.5, 0.5, 'No scaling relationships found',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Scaling Factor Analysis\n(No Data)')
        return

    scaling_factors = []
    for rel_name, rel_list in relationships.items():
        for rel in rel_list:
            if 'scaling_factor' in rel:
                scaling_factors.append(rel['scaling_factor'])

    if scaling_factors:
        ax.hist(scaling_factors, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        ax.set_xlabel('Scaling Factor')
        ax.set_ylabel('Frequency')
        ax.set_title('Scaling Factor Distribution')
        ax.grid(True, alpha=0.3)


def _plot_ivm_relationships(ax, relationships):
    """Plot IVM relationship network."""
    ivm_rels = {k: v for k, v in relationships.items() if k.startswith('ivm_')}

    if not ivm_rels:
        ax.text(0.5, 0.5, 'No IVM relationships found',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('IVM Relationships\n(No Data)')
        return

    # Create a simple network visualization
    rel_types = list(ivm_rels.keys())
    rel_counts = [len(ivm_rels[rel]) for rel in rel_types]

    # Create positions for nodes
    angles = np.linspace(0, 2*np.pi, len(rel_types), endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Plot nodes
    scatter = ax.scatter(x, y, s=[count*100 for count in rel_counts],
                        c=range(len(rel_types)), cmap='viridis', alpha=0.7)

    # Add labels
    for i, (rel_type, angle) in enumerate(zip(rel_types, angles)):
        # Clean up label
        label = rel_type.replace('ivm_', '').replace('_', ' ').title()
        ax.text(1.2 * np.cos(angle), 1.2 * np.sin(angle), label,
               ha='center', va='center', fontsize=8)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('IVM Relationship Network')

    # Add colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Relationship Type')


def _plot_rational_approximations(ax, analyses):
    """Plot rational approximation quality."""
    approximations = []

    for analysis in analyses:
        for approx_name, approx_data in analysis.get('rational_approximations', {}).items():
            approximations.append({
                'error': approx_data.get('error', 1.0),
                'constant': approx_data.get('constant', 'unknown'),
                'description': approx_data.get('description', '')
            })

    if not approximations:
        ax.text(0.5, 0.5, 'No rational approximations found',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Rational Approximation Quality\n(No Data)')
        return

    # Group by constant
    constants = defaultdict(list)
    for approx in approximations:
        constants[approx['constant']].append(approx['error'])

    # Plot box plot of errors
    data = [constants[const] for const in constants.keys()]
    labels = list(constants.keys())

    if data:
        ax.boxplot(data, labels=labels)
        ax.set_ylabel('Approximation Error')
        ax.set_title('Rational Approximation Quality by Constant')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')


def generate_geometric_mnemonic_report(
    numbers: List[Union[int, str, SymergeticsNumber]],
    title: str = "Geometric Mnemonics Analysis Report"
) -> Dict[str, Any]:
    """
    Generate comprehensive report on geometric mnemonics.

    Args:
        numbers: Numbers to analyze
        title: Report title

    Returns:
        Dict containing comprehensive report
    """
    # Analyze all numbers
    analyses = [analyze_geometric_mnemonics(num, analysis_depth=5) for num in numbers]

    report = {
        'title': title,
        'timestamp': '2024-01-01T00:00:00Z',  # Would use datetime in real implementation
        'summary': _create_mnemonic_summary(analyses),
        'detailed_analysis': {},
        'insights': _generate_mnemonic_insights(analyses),
        'recommendations': _generate_mnemonic_recommendations(analyses)
    }

    # Detailed analysis by category
    report['detailed_analysis'] = {
        'platonic_relationships': _aggregate_platonic_relationships(analyses),
        'ivm_relationships': _aggregate_ivm_relationships(analyses),
        'rational_approximations': _aggregate_rational_approximations(analyses),
        'geometric_mnemonics': [str(m) for analysis in analyses
                               for m in analysis.get('geometric_mnemonics', [])]
    }

    return report


def _create_mnemonic_summary(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create summary statistics for mnemonic analysis."""
    total_numbers = len(analyses)

    platonic_rels = sum(len(a.get('platonic_relationships', {})) for a in analyses)
    ivm_rels = sum(len(a.get('ivm_scaling_factors', {})) for a in analyses)
    rational_approxs = sum(len(a.get('rational_approximations', {})) for a in analyses)

    return {
        'total_numbers_analyzed': total_numbers,
        'total_geometric_relationships': platonic_rels + ivm_rels,
        'platonic_relationships': platonic_rels,
        'ivm_relationships': ivm_rels,
        'rational_approximations': rational_approxs,
        'average_relationships_per_number': (platonic_rels + ivm_rels + rational_approxs) / total_numbers if total_numbers > 0 else 0
    }


def _aggregate_platonic_relationships(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate platonic solid relationships."""
    aggregated = defaultdict(list)

    for analysis in analyses:
        for rel_name, rel_data in analysis.get('platonic_relationships', {}).items():
            aggregated[rel_name].append(rel_data)

    return dict(aggregated)


def _aggregate_ivm_relationships(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate IVM lattice relationships."""
    aggregated = defaultdict(list)

    for analysis in analyses:
        for rel_name, rel_data in analysis.get('ivm_scaling_factors', {}).items():
            aggregated[rel_name].append(rel_data)

    return dict(aggregated)


def _aggregate_rational_approximations(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate rational approximation relationships."""
    aggregated = defaultdict(list)

    for analysis in analyses:
        for approx_name, approx_data in analysis.get('rational_approximations', {}).items():
            aggregated[approx_name].append(approx_data)

    return dict(aggregated)


def _generate_mnemonic_insights(analyses: List[Dict[str, Any]]) -> List[str]:
    """Generate insights from mnemonic analysis."""
    insights = []

    # Analyze geometric diversity
    all_forms = set()
    for analysis in analyses:
        for rel_name in analysis.get('platonic_relationships', {}):
            if '_' in rel_name:
                all_forms.add(rel_name.split('_')[0])
            else:
                all_forms.add(rel_name)

    if len(all_forms) > 3:
        insights.append(f"High geometric diversity: {len(all_forms)} different platonic forms represented")

    # Analyze scaling patterns
    scaling_factors = []
    for analysis in analyses:
        for rel_data in analysis.get('platonic_relationships', {}).values():
            if 'scaling_factor' in rel_data:
                scaling_factors.append(rel_data['scaling_factor'])

    if scaling_factors:
        avg_scaling = sum(scaling_factors) / len(scaling_factors)
        if avg_scaling > 10:
            insights.append(".1f")
        elif avg_scaling < 2:
            insights.append(".1f")
    # Analyze IVM relationships
    ivm_count = sum(len(a.get('ivm_scaling_factors', {})) for a in analyses)
    if ivm_count > 0:
        insights.append(f"Found {ivm_count} relationships to IVM lattice structures")

    # Analyze rational approximations
    rational_count = sum(len(a.get('rational_approximations', {})) for a in analyses)
    if rational_count > 0:
        insights.append(f"Identified {rational_count} rational approximations of mathematical constants")

    return insights


def _generate_mnemonic_recommendations(analyses: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on mnemonic analysis."""
    recommendations = []

    # Check for visualization opportunities
    has_platonic = any(len(a.get('platonic_relationships', {})) > 0 for a in analyses)
    has_ivm = any(len(a.get('ivm_scaling_factors', {})) > 0 for a in analyses)
    has_rational = any(len(a.get('rational_approximations', {})) > 0 for a in analyses)

    if has_platonic:
        recommendations.append("Create visualization showing platonic solid scaling relationships")

    if has_ivm:
        recommendations.append("Generate IVM lattice structure visualizations for identified relationships")

    if has_rational:
        recommendations.append("Develop interactive visualization of rational approximation convergence")

    if has_platonic and has_ivm:
        recommendations.append("Explore connections between platonic forms and IVM sphere packing")

    recommendations.append("Consider creating geometric mnemonic encoding system for data compression")

    return recommendations
