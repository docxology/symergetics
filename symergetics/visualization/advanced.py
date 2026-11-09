#!/usr/bin/env python3
"""
Advanced Visualization Methods for Symergetics Package

This module provides advanced visualization capabilities for comparative analysis,
statistical analysis, and complex mathematical pattern visualization.
"""

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..computation.geometric_mnemonics import analyze_geometric_mnemonics
from ..core.numbers import SymergeticsNumber
from ..visualization import (
    _validate_canvas,
    get_config,
    get_organized_output_path,  # noqa: F401  (patch target in tests)
)

# Global variables for matplotlib and numpy (initialized when needed)
_plt = None
_np = None


def _ensure_matplotlib_numpy():
    """Ensure matplotlib and numpy are properly initialized."""
    global _plt, _np
    if _plt is None or _np is None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            _plt = plt
            _np = np
        except ImportError as e:
            raise ImportError(f"Required visualization libraries not available: {e}")
    return _plt, _np


def create_comparative_analysis_visualization(
    domain1_data: List[Dict[str, Any]],
    domain2_data: List[Dict[str, Any]],
    domain1_name: str = "Domain 1",
    domain2_name: str = "Domain 2",
    backend: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create comprehensive comparative analysis visualization.

    Args:
        domain1_data: Analysis data from first domain
        domain2_data: Analysis data from second domain
        domain1_name: Name of first domain
        domain2_name: Name of second domain
        backend: Visualization backend
        **kwargs: Additional visualization parameters

    Returns:
        Dict containing visualization metadata and file paths
    """
    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _create_comparative_matplotlib(domain1_data, domain2_data,
                                            domain1_name, domain2_name, **kwargs)
    else:
        raise ValueError(f"Comparative analysis visualization requires matplotlib backend, got: {backend}")


def _create_comparative_matplotlib(
    domain1_data: List[Dict[str, Any]],
    domain2_data: List[Dict[str, Any]],
    domain1_name: str,
    domain2_name: str,
    **kwargs
) -> Dict[str, Any]:
    """Create comparative analysis visualization using matplotlib."""
    # Check for empty domains
    if not domain1_data and not domain2_data:
        raise ValueError("Cannot create comparative analysis visualization with empty domains")

    # Ensure matplotlib and numpy are available
    plt, np = _ensure_matplotlib_numpy()

    # Make plt available to helper functions
    global _plt
    _plt = plt

    # Extract metrics for comparison
    domain1_metrics = _extract_domain_metrics(domain1_data)
    domain2_metrics = _extract_domain_metrics(domain2_data)

    # Create multi-panel figure
    _validate_canvas((18, 12), max(get_config()["dpi"], float(plt.rcParams["figure.dpi"])))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    try:
        fig.suptitle(f'Comparative Analysis: {domain1_name} vs {domain2_name}',
                    fontsize=16, fontweight='bold')

        # Panel 1: Complexity comparison
        _plot_complexity_comparison(axes[0, 0], domain1_metrics, domain2_metrics,
                                  domain1_name, domain2_name)

        # Panel 2: Symmetry comparison
        _plot_symmetry_comparison(axes[0, 1], domain1_metrics, domain2_metrics,
                                domain1_name, domain2_name)

        # Panel 3: Palindromic density comparison
        _plot_palindromic_comparison(axes[0, 2], domain1_metrics, domain2_metrics,
                                   domain1_name, domain2_name)

        # Panel 4: Length distribution
        _plot_length_distribution(axes[1, 0], domain1_data, domain2_data,
                                domain1_name, domain2_name)

        # Panel 5: Digit distribution comparison
        _plot_digit_distribution_comparison(axes[1, 1], domain1_data, domain2_data,
                                         domain1_name, domain2_name)

        # Panel 6: Correlation heatmap
        _plot_correlation_heatmap(axes[1, 2], domain1_data, domain2_data,
                                domain1_name, domain2_name)

        fig.tight_layout()

        # Save the visualization
        output_path = get_organized_output_path('mathematical', 'comparative_analysis',
                                              f'comparative_analysis_{domain1_name.lower()}_vs_{domain2_name.lower()}.png')

        fig.savefig(output_path, dpi=get_config()["dpi"], bbox_inches='tight')

        return {
            'files': [str(output_path)],
            'metadata': {
                'type': 'comparative_analysis',
                'domain1': domain1_name,
                'domain2': domain2_name,
                'domain1_count': len(domain1_data),
                'domain2_count': len(domain2_data),
                'panels': 6,
                'backend': 'matplotlib'
            }
        }
    finally:
        _plt.close(fig)


def _extract_domain_metrics(data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Extract numerical metrics from domain data."""
    metrics = {
        'complexity': [],
        'symmetry': [],
        'palindromic_density': [],
        'length': []
    }

    for item in data:
        if 'pattern_complexity' in item:
            comp = item['pattern_complexity'].get('complexity_score', 0)
            metrics['complexity'].append(comp)

        if 'symmetry_analysis' in item:
            sym = item['symmetry_analysis'].get('symmetry_score', 0)
            metrics['symmetry'].append(sym)

        if 'palindromic_density' in item:
            density = item['palindromic_density']
            metrics['palindromic_density'].append(density)

        if 'length' in item:
            metrics['length'].append(item['length'])

    return metrics


def _plot_complexity_comparison(ax, domain1_metrics, domain2_metrics, name1, name2):
    """Plot complexity comparison."""
    comp1 = domain1_metrics['complexity']
    comp2 = domain2_metrics['complexity']

    if comp1 and comp2:
        ax.hist(comp1, alpha=0.7, label=name1, bins=10, density=True)
        ax.hist(comp2, alpha=0.7, label=name2, bins=10, density=True)
        ax.set_xlabel('Complexity Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title('Complexity Distribution')
    else:
        ax.text(0.5, 0.5, 'No complexity data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Complexity Distribution\n(No Data)')


def _plot_symmetry_comparison(ax, domain1_metrics, domain2_metrics, name1, name2):
    """Plot symmetry comparison."""
    sym1 = domain1_metrics['symmetry']
    sym2 = domain2_metrics['symmetry']

    if sym1 and sym2:
        ax.hist(sym1, alpha=0.7, label=name1, bins=10, density=True)
        ax.hist(sym2, alpha=0.7, label=name2, bins=10, density=True)
        ax.set_xlabel('Symmetry Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title('Symmetry Distribution')
    else:
        ax.text(0.5, 0.5, 'No symmetry data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Symmetry Distribution\n(No Data)')


def _plot_palindromic_comparison(ax, domain1_metrics, domain2_metrics, name1, name2):
    """Plot palindromic density comparison."""
    pal1 = domain1_metrics['palindromic_density']
    pal2 = domain2_metrics['palindromic_density']

    if pal1 and pal2:
        data = [pal1, pal2]
        labels = [name1, name2]

        ax.boxplot(data)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_ylabel('Palindromic Density')
        ax.set_title('Palindromic Density Comparison')

        # Add mean markers
        for i, d in enumerate(data):
            if d:
                mean_val = statistics.mean(d)
            else:
                mean_val = 0
            ax.plot(i+1, mean_val, 'ro', markersize=8, label=f'Mean: {mean_val:.3f}' if i == 0 else "")

        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No palindromic data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Palindromic Density\n(No Data)')


def _plot_length_distribution(ax, domain1_data, domain2_data, name1, name2):
    """Plot length distribution comparison."""
    lengths1 = [item.get('length', 0) for item in domain1_data]
    lengths2 = [item.get('length', 0) for item in domain2_data]

    if lengths1 and lengths2:
        ax.hist(lengths1, alpha=0.7, label=name1, bins=15, density=True)
        ax.hist(lengths2, alpha=0.7, label=name2, bins=15, density=True)
        ax.set_xlabel('Length')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title('Length Distribution')
    else:
        ax.text(0.5, 0.5, 'No length data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Length Distribution\n(No Data)')


def _plot_digit_distribution_comparison(ax, domain1_data, domain2_data, name1, name2):
    """Plot digit distribution comparison."""
    # Extract digit distributions
    digits1 = {}
    digits2 = {}

    for item in domain1_data:
        dist = item.get('digit_distribution', {})
        for digit, count in dist.items():
            digits1[digit] = digits1.get(digit, 0) + count

    for item in domain2_data:
        dist = item.get('digit_distribution', {})
        for digit, count in dist.items():
            digits2[digit] = digits2.get(digit, 0) + count

    if digits1 and digits2:
        digits = sorted(set(digits1.keys()) | set(digits2.keys()))

        values1 = [digits1.get(d, 0) for d in digits]
        values2 = [digits2.get(d, 0) for d in digits]

        x = range(len(digits))
        width = 0.35

        ax.bar([i - width/2 for i in x], values1, width, label=name1, alpha=0.7)
        ax.bar([i + width/2 for i in x], values2, width, label=name2, alpha=0.7)

        ax.set_xlabel('Digit')
        ax.set_ylabel('Total Count')
        ax.set_xticks(x)
        ax.set_xticklabels(digits)
        ax.legend()
        ax.set_title('Digit Distribution Comparison')
    else:
        ax.text(0.5, 0.5, 'No digit data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Digit Distribution\n(No Data)')


def _plot_correlation_heatmap(ax, domain1_data, domain2_data, name1, name2):
    """Plot correlation heatmap between domains."""
    # Calculate correlations between different metrics
    metrics = ['length', 'palindromic_density']

    correlations = {}
    for metric in metrics:
        values1 = [item.get(metric, 0) for item in domain1_data if metric in item]
        values2 = [item.get(metric, 0) for item in domain2_data if metric in item]

        if len(values1) > 1 and len(values2) > 1 and len(values1) == len(values2):
            try:
                corr = statistics.correlation(values1, values2)
                correlations[metric] = corr
            except (statistics.StatisticsError, ValueError):
                correlations[metric] = 0
        else:
            correlations[metric] = 0

    if correlations:
        metrics_list = list(correlations.keys())
        corr_values = [correlations[m] for m in metrics_list]

        # Create heatmap-like visualization
        im = ax.imshow([[corr] for corr in corr_values], cmap='coolwarm', aspect='auto',
                      vmin=-1, vmax=1)

        ax.set_yticks(range(len(metrics_list)))
        ax.set_yticklabels(metrics_list)
        ax.set_xticks([0])
        ax.set_xticklabels([f'{name1}\nvs\n{name2}'])
        ax.set_title('Metric Correlations')

        # Add colorbar
        _plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    else:
        ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Metric Correlations\n(No Data)')


def create_pattern_discovery_visualization(
    analysis_results: List[Dict[str, Any]],
    title: str = "Pattern Discovery Analysis",
    backend: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create pattern discovery visualization showing discovered patterns across data.

    Args:
        analysis_results: List of analysis result dictionaries
        title: Visualization title
        backend: Visualization backend
        **kwargs: Additional visualization parameters

    Returns:
        Dict containing visualization metadata and file paths
    """
    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _create_pattern_discovery_matplotlib(analysis_results, title, **kwargs)
    else:
        raise ValueError(f"Pattern discovery visualization requires matplotlib backend, got: {backend}")


def _create_pattern_discovery_matplotlib(
    analysis_results: List[Dict[str, Any]],
    title: str,
    **kwargs
) -> Dict[str, Any]:
    """Create pattern discovery visualization using matplotlib."""
    plt, _ = _ensure_matplotlib_numpy()

    # Extract pattern information
    palindromic_numbers = []
    high_complexity_numbers = []
    high_symmetry_numbers = []

    for i, result in enumerate(analysis_results):
        if result.get('is_palindromic', False):
            palindromic_numbers.append((i, result.get('length', 0)))

        complexity = result.get('pattern_complexity', {}).get('complexity_score', 0)
        if complexity > 2:
            high_complexity_numbers.append((i, complexity))

        symmetry = result.get('symmetry_analysis', {}).get('symmetry_score', 0)
        if symmetry > 0.7:
            high_symmetry_numbers.append((i, symmetry))

    # Create visualization
    _validate_canvas((15, 12), max(get_config()["dpi"], float(plt.rcParams["figure.dpi"])))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    try:
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Panel 1: Pattern type distribution
        _plot_pattern_type_distribution(axes[0, 0], analysis_results)

        # Panel 2: Complexity vs Symmetry scatter
        _plot_complexity_symmetry_scatter(axes[0, 1], analysis_results)

        # Panel 3: Pattern discovery timeline
        _plot_pattern_discovery_timeline(axes[1, 0], analysis_results)

        # Panel 4: Outstanding patterns
        _plot_outstanding_patterns(axes[1, 1], palindromic_numbers, high_complexity_numbers, high_symmetry_numbers)

        fig.tight_layout()

        # Save visualization
        safe_title = title.lower().replace(' ', '_').replace('-', '_')
        output_path = get_organized_output_path('mathematical', 'pattern_discovery',
                                              f'pattern_discovery_{safe_title}.png')

        fig.savefig(output_path, dpi=get_config()["dpi"], bbox_inches='tight')

        return {
            'files': [str(output_path)],
            'metadata': {
                'type': 'pattern_discovery',
                'title': title,
                'total_analyses': len(analysis_results),
                'palindromic_count': len(palindromic_numbers),
                'high_complexity_count': len(high_complexity_numbers),
                'high_symmetry_count': len(high_symmetry_numbers),
                'backend': 'matplotlib'
            }
        }
    finally:
        _plt.close(fig)


def _plot_pattern_type_distribution(ax, analysis_results):
    """Plot distribution of different pattern types."""
    palindromic_count = sum(1 for r in analysis_results if r.get('is_palindromic', False))
    high_complexity_count = sum(1 for r in analysis_results
                               if r.get('pattern_complexity', {}).get('complexity_score', 0) > 2)
    high_symmetry_count = sum(1 for r in analysis_results
                             if r.get('symmetry_analysis', {}).get('symmetry_score', 0) > 0.7)

    pattern_types = ['Palindromic', 'High Complexity', 'High Symmetry']
    counts = [palindromic_count, high_complexity_count, high_symmetry_count]

    bars = ax.bar(pattern_types, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('Count')
    ax.set_title('Pattern Type Distribution')

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{count}', ha='center', va='bottom')


def _plot_complexity_symmetry_scatter(ax, analysis_results):
    """Plot scatter plot of complexity vs symmetry."""
    complexities = []
    symmetries = []

    for result in analysis_results:
        comp = result.get('pattern_complexity', {}).get('complexity_score', 0)
        sym = result.get('symmetry_analysis', {}).get('symmetry_score', 0)
        complexities.append(comp)
        symmetries.append(sym)

    if complexities and symmetries:
        scatter = ax.scatter(complexities, symmetries, alpha=0.6, s=50)

        # Add trend line
        if len(complexities) > 1:
            try:
                slope, intercept = _np.polyfit(complexities, symmetries, 1)
                x_trend = _np.linspace(min(complexities), max(complexities), 100)
                y_trend = slope * x_trend + intercept
                ax.plot(x_trend, y_trend, 'r--', alpha=0.8, label=f'slope={slope:.2f}')
                ax.legend()
            except (_np.linalg.LinAlgError, ValueError):
                pass

        ax.set_xlabel('Complexity Score')
        ax.set_ylabel('Symmetry Score')
        ax.set_title('Complexity vs Symmetry')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No complexity/symmetry data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Complexity vs Symmetry\n(No Data)')


def _plot_pattern_discovery_timeline(ax, analysis_results):
    """Plot pattern discovery over time/sequence."""
    palindromic_positions = []
    complexity_positions = []
    symmetry_positions = []

    for i, result in enumerate(analysis_results):
        if result.get('is_palindromic', False):
            palindromic_positions.append(i)

        if result.get('pattern_complexity', {}).get('complexity_score', 0) > 2:
            complexity_positions.append(i)

        if result.get('symmetry_analysis', {}).get('symmetry_score', 0) > 0.7:
            symmetry_positions.append(i)

    # Plot as cumulative discovery
    max_pos = len(analysis_results) - 1

    palindromic_cumulative = []
    complexity_cumulative = []
    symmetry_cumulative = []

    for i in range(max_pos + 1):
        palindromic_cumulative.append(sum(1 for p in palindromic_positions if p <= i))
        complexity_cumulative.append(sum(1 for p in complexity_positions if p <= i))
        symmetry_cumulative.append(sum(1 for p in symmetry_positions if p <= i))

    ax.plot(range(max_pos + 1), palindromic_cumulative, label='Palindromic', marker='o', markersize=3)
    ax.plot(range(max_pos + 1), complexity_cumulative, label='High Complexity', marker='s', markersize=3)
    ax.plot(range(max_pos + 1), symmetry_cumulative, label='High Symmetry', marker='^', markersize=3)

    ax.set_xlabel('Analysis Position')
    ax.set_ylabel('Cumulative Count')
    ax.set_title('Pattern Discovery Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_outstanding_patterns(ax, palindromic, high_complexity, high_symmetry):
    """Plot outstanding patterns summary."""
    pattern_counts = {
        'Palindromic': len(palindromic),
        'High Complexity': len(high_complexity),
        'High Symmetry': len(high_symmetry)
    }

    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())

    bars = ax.bar(patterns, counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('Count')
    ax.set_title('Outstanding Patterns')

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{count}', ha='center', va='bottom')

    # Add percentage if we have total count
    if hasattr(ax, '_parent') and hasattr(ax._parent, 'get_children'):
        # This is a bit hacky, but we can get the total from the parent figure
        total = sum(counts)
        if total > 0:
            for i, (bar, count) in enumerate(zip(bars, counts)):
                percentage = count / total * 100
                ax.text(bar.get_x() + bar.get_width()/2., count/2,
                       f'{percentage:.1f}%', ha='center', va='center',
                       fontweight='bold', color='white')


def create_statistical_analysis_dashboard(
    analysis_results: List[Dict[str, Any]],
    title: str = "Statistical Analysis Dashboard",
    backend: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create statistical analysis dashboard visualization.

    Args:
        analysis_results: List of analysis result dictionaries
        title: Dashboard title
        backend: Visualization backend
        **kwargs: Additional visualization parameters

    Returns:
        Dict containing visualization metadata and file paths
    """
    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _create_statistical_dashboard_matplotlib(analysis_results, title, **kwargs)
    else:
        raise ValueError(f"Statistical dashboard requires matplotlib backend, got: {backend}")


def _create_statistical_dashboard_matplotlib(
    analysis_results: List[Dict[str, Any]],
    title: str,
    **kwargs
) -> Dict[str, Any]:
    """Create statistical dashboard using matplotlib."""
    _ensure_matplotlib_numpy()

    # Extract statistical data
    lengths = [r.get('length', 0) for r in analysis_results]
    complexities = [r.get('pattern_complexity', {}).get('complexity_score', 0) for r in analysis_results]
    symmetries = [r.get('symmetry_analysis', {}).get('symmetry_score', 0) for r in analysis_results]
    palindromic_densities = [r.get('palindromic_density', 0) for r in analysis_results]

    # Create dashboard layout
    _validate_canvas((20, 16), max(get_config()["dpi"], float(_plt.rcParams["figure.dpi"])))
    fig = _plt.figure(figsize=(20, 16), constrained_layout=True)
    try:
        fig.suptitle(title, fontsize=20, fontweight='bold')

        # Create subplots with GridSpec for better layout
        gs = fig.add_gridspec(4, 4)

        # Main statistics panel
        ax_stats = fig.add_subplot(gs[0, :2])
        _plot_main_statistics(ax_stats, analysis_results)

        # Distribution panel
        ax_dist = fig.add_subplot(gs[0, 2:])
        _plot_distribution_summary(ax_dist, lengths, complexities, symmetries)

        # Correlation matrix
        ax_corr = fig.add_subplot(gs[1, :2])
        _plot_correlation_matrix(ax_corr, complexities, symmetries, palindromic_densities, lengths)

        # Time series (if applicable)
        ax_time = fig.add_subplot(gs[1, 2:])
        _plot_metric_evolution(ax_time, complexities, symmetries, palindromic_densities)

        # Detailed histograms
        ax_hist1 = fig.add_subplot(gs[2, :2])
        _plot_detailed_histogram(ax_hist1, complexities, 'Complexity Scores', 'blue')

        ax_hist2 = fig.add_subplot(gs[2, 2:])
        _plot_detailed_histogram(ax_hist2, symmetries, 'Symmetry Scores', 'green')

        # Box plot comparison
        ax_box = fig.add_subplot(gs[3, :2])
        _plot_box_plot_comparison(ax_box, complexities, symmetries, palindromic_densities)

        # Summary statistics table
        ax_table = fig.add_subplot(gs[3, 2:])
        _plot_summary_table(ax_table, complexities, symmetries, palindromic_densities, lengths)

        # Save dashboard
        safe_title = title.lower().replace(' ', '_').replace('-', '_')
        output_path = get_organized_output_path('mathematical', 'statistical_dashboard',
                                              f'statistical_dashboard_{safe_title}.png')

        fig.savefig(output_path, dpi=get_config()["dpi"], bbox_inches='tight')

        return {
            'files': [str(output_path)],
            'metadata': {
                'type': 'statistical_dashboard',
                'title': title,
                'total_analyses': len(analysis_results),
                'panels': 8,
                'backend': 'matplotlib'
            }
        }
    finally:
        _plt.close(fig)


def _plot_main_statistics(ax, analysis_results):
    """Plot main statistics overview."""
    total_count = len(analysis_results)
    palindromic_count = sum(1 for r in analysis_results if r.get('is_palindromic', False))
    palindromic_ratio = palindromic_count / total_count if total_count > 0 else 0

    # Create summary text
    summary_text = f"""
    Mathematical Analysis Summary

    Total Analyses: {total_count}
    Palindromic Numbers: {palindromic_count} ({palindromic_ratio:.1%})

    Average Length: {statistics.mean([r.get('length', 0) for r in analysis_results]) if analysis_results else 0:.1f}
    Average Complexity: {statistics.mean([r.get('pattern_complexity', {}).get('complexity_score', 0) for r in analysis_results]) if analysis_results else 0:.2f}
    Average Symmetry: {statistics.mean([r.get('symmetry_analysis', {}).get('symmetry_score', 0) for r in analysis_results]) if analysis_results else 0:.2f}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Analysis Summary', fontweight='bold')


def _plot_distribution_summary(ax, lengths, complexities, symmetries):
    """Plot distribution summary."""
    # Create box plots for each metric
    data = [lengths, complexities, symmetries]
    labels = ['Length', 'Complexity', 'Symmetry']

    ax.boxplot(data)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel('Value')
    ax.set_title('Metric Distributions')
    ax.grid(True, alpha=0.3)


def _plot_correlation_matrix(ax, complexities, symmetries, densities, lengths):
    """Plot correlation matrix."""
    # Handle empty data
    if not complexities and not symmetries and not densities and not lengths:
        ax.text(0.5, 0.5, 'No data for correlation analysis', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Correlation Matrix\n(No Data)')
        return
    
    # Correlation is undefined for constant columns or fewer than two samples.
    data = _np.array([complexities, symmetries, densities, lengths], dtype=float)
    corr_matrix = _np.full((4, 4), _np.nan)
    if data.shape[1] > 1:
        varying = _np.flatnonzero(_np.any(data != data[:, :1], axis=1))
        if varying.size:
            corr_matrix[_np.ix_(varying, varying)] = _np.corrcoef(data[varying])

    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Add labels
    labels = ['Complexity', 'Symmetry', 'Density', 'Length']
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr_matrix[i, j]
            label = f'{value:.2f}' if _np.isfinite(value) else 'N/A'
            ax.text(j, i, label, ha='center', va='center', color='black')

    ax.set_title('Correlation Matrix')
    _plt.colorbar(im, ax=ax, shrink=0.8)


def _plot_metric_evolution(ax, complexities, symmetries, densities):
    """Plot metric evolution over time/sequence."""
    x = range(len(complexities))

    ax.plot(x, complexities, label='Complexity', marker='o', markersize=2, alpha=0.7)
    ax.plot(x, symmetries, label='Symmetry', marker='s', markersize=2, alpha=0.7)
    ax.plot(x, densities, label='Palindromic Density', marker='^', markersize=2, alpha=0.7)

    ax.set_xlabel('Analysis Index')
    ax.set_ylabel('Metric Value')
    ax.set_title('Metric Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_detailed_histogram(ax, data, title, color):
    """Plot detailed histogram with statistics."""
    if data:
        ax.hist(data, bins=15, alpha=0.7, color=color, edgecolor='black')
        mean_val = statistics.mean(data)
        median_val = statistics.median(data)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                   label=f'Median: {median_val:.2f}')

        ax.legend()
        ax.grid(True, alpha=0.3)

    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)


def _plot_box_plot_comparison(ax, complexities, symmetries, densities):
    """Plot box plot comparison of metrics."""
    data = [complexities, symmetries, densities]
    labels = ['Complexity', 'Symmetry', 'Palindromic\nDensity']

    ax.boxplot(data, patch_artist=True,
              boxprops=dict(facecolor='lightblue', color='blue'),
              medianprops=dict(color='red', linewidth=2))
    ax.set_xticks(range(1, len(labels) + 1), labels)

    ax.set_ylabel('Value')
    ax.set_title('Metric Comparison')
    ax.grid(True, alpha=0.3)


def _plot_summary_table(ax, complexities, symmetries, densities, lengths):
    """Plot summary statistics; sample standard deviation needs two values."""
    # Calculate statistics
    stats_data = []
    for name, data in [('Complexity', complexities), ('Symmetry', symmetries),
                      ('Density', densities), ('Length', lengths)]:
        if data:
            stats_data.append([
                name,
                f'{statistics.mean(data):.2f}',
                f'{statistics.median(data):.2f}',
                f'{min(data):.2f}',
                f'{max(data):.2f}',
                f'{statistics.stdev(data):.2f}' if len(data) > 1 else 'N/A',
            ])
        else:
            stats_data.append([name, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

    # Create table
    table = ax.table(cellText=stats_data,
                    colLabels=['Metric', 'Mean', 'Median', 'Min', 'Max', 'StdDev'],
                    loc='center',
                    cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    ax.axis('off')
    ax.set_title('Summary Statistics', fontweight='bold')


def create_geometric_mnemonics_visualization(
    numbers: List[Union[int, str, SymergeticsNumber]],
    title: str = "Geometric Mnemonics Analysis",
    backend: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create visualization of geometric mnemonics relationships.

    Args:
        numbers: List of numbers to analyze for geometric relationships
        title: Visualization title
        backend: Visualization backend
        **kwargs: Additional visualization parameters

    Returns:
        Dict containing visualization metadata and file paths
    """
    backend = backend or get_config()["backend"]

    if backend == "matplotlib":
        return _create_geometric_mnemonics_matplotlib(numbers, title, **kwargs)
    else:
        raise ValueError(f"Geometric mnemonics visualization requires matplotlib backend, got: {backend}")


def _create_geometric_mnemonics_matplotlib(
    numbers: List[Union[int, str, SymergeticsNumber]],
    title: str,
    **kwargs
) -> Dict[str, Any]:
    """Create geometric mnemonics visualization using matplotlib."""

    # Ensure matplotlib and numpy are available
    plt, np = _ensure_matplotlib_numpy()

    # Make plt and np available to helper functions
    global _plt, _np
    _plt = plt
    _np = np

    # Analyze all numbers
    analyses = [analyze_geometric_mnemonics(num, analysis_depth=4) for num in numbers]

    # Extract geometric relationships
    relationships = _extract_geometric_relationships(analyses)

    # Create comprehensive visualization
    _validate_canvas((18, 15), max(get_config()["dpi"], float(plt.rcParams["figure.dpi"])))
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    try:
        fig.suptitle(title, fontsize=18, fontweight='bold')

        # Panel 1: Overview statistics
        _plot_mnemonics_overview(axes[0, 0], analyses)

        # Panel 2: Geometric form distribution
        _plot_geometric_distribution(axes[0, 1], relationships)

        # Panel 3: Scaling factor analysis
        _plot_scaling_factors(axes[1, 0], relationships)

        # Panel 4: IVM relationship network
        _plot_ivm_relationships(axes[1, 1], relationships)

        # Panel 5: Rational approximation quality
        _plot_rational_approximations(axes[2, 0], analyses)

        # Panel 6: Platonic solid relationships
        _plot_platonic_relationships(axes[2, 1], relationships)

        fig.tight_layout()

        # Save visualization
        safe_title = title.lower().replace(' ', '_').replace('-', '_')
        output_path = get_organized_output_path('mathematical', 'geometric_mnemonics',
                                              f'geometric_mnemonics_{safe_title}.png')

        fig.savefig(output_path, dpi=get_config()["dpi"], bbox_inches='tight')

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
    finally:
        _plt.close(fig)


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


def _plot_mnemonics_overview(ax, analyses):
    """Plot overview statistics of geometric mnemonics."""

    # Calculate statistics
    total_numbers = len(analyses)
    platonic_rels = sum(len(a.get('platonic_relationships', {})) for a in analyses)
    ivm_rels = sum(len(a.get('ivm_scaling_factors', {})) for a in analyses)
    rational_approxs = sum(len(a.get('rational_approximations', {})) for a in analyses)

    # Create summary text
    summary_text = f"""
    Geometric Mnemonics Overview

    Total Numbers: {total_numbers}
    Platonic Relationships: {platonic_rels}
    IVM Relationships: {ivm_rels}
    Rational Approximations: {rational_approxs}

    Avg Relationships/Number: {(platonic_rels + ivm_rels + rational_approxs) / total_numbers:.2f}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Analysis Overview', fontweight='bold')


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
    angles = _np.linspace(0, 2*_np.pi, len(rel_types), endpoint=False)
    x = _np.cos(angles)
    y = _np.sin(angles)

    # Plot nodes
    scatter = ax.scatter(x, y, s=[count*100 for count in rel_counts],
                        c=range(len(rel_types)), cmap='viridis', alpha=0.7)

    # Add labels
    for i, (rel_type, angle) in enumerate(zip(rel_types, angles)):
        # Clean up label
        label = rel_type.replace('ivm_', '').replace('_', ' ').title()
        ax.text(1.2 * _np.cos(angle), 1.2 * _np.sin(angle), label,
               ha='center', va='center', fontsize=8)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('IVM Relationship Network')

    # Add colorbar
    _plt.colorbar(scatter, ax=ax, shrink=0.8, label='Relationship Type')


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
        ax.boxplot(data)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_ylabel('Approximation Error')
        ax.set_title('Rational Approximation Quality by Constant')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')


def _plot_platonic_relationships(ax, relationships):
    """Plot platonic solid relationship details."""
    platonic_rels = {k: v for k, v in relationships.items() if not k.startswith('ivm_')}

    if not platonic_rels:
        ax.text(0.5, 0.5, 'No platonic relationships found',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Platonic Relationships\n(No Data)')
        return

    # Extract scaling factors by solid type
    solid_scaling = defaultdict(list)
    for rel_name, rel_list in platonic_rels.items():
        solid_type = rel_name.split('_')[0] if '_' in rel_name else rel_name
        for rel in rel_list:
            if 'scaling_factor' in rel:
                solid_scaling[solid_type].append(rel['scaling_factor'])

    # Create violin plot or box plot
    solid_types = list(solid_scaling.keys())
    scaling_data = [solid_scaling[solid] for solid in solid_types]

    if scaling_data:
        # Create box plot
        ax.boxplot(scaling_data)
        ax.set_xticks(range(1, len(solid_types) + 1), solid_types)
        ax.set_ylabel('Scaling Factor')
        ax.set_title('Platonic Solid Scaling Factors')
        ax.tick_params(axis='x', rotation=45)


def create_mega_graphical_abstract(
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    backend: Optional[str] = None,
    *,
    section: str = "overview",
) -> Dict[str, Any]:
    """Render a focused scientific overview or a computed topic group.

    ``section`` is keyword-only: ``overview`` (six representative topics),
    ``geometry``, ``arithmetic``, ``patterns``, or ``all`` (all 17 topics).
    An omitted title follows the selected section; explicit titles are retained.
    The first three positional arguments and canonical default PNG path remain
    supported. Section selection was introduced in 1.4.0.

    Returns file paths, actual panel identities/layout metadata, and JSON-safe
    ``data`` keyed by panel identity. Exact rational results are full strings;
    floating references and geometric units are identified alongside their data.
    Required computation/save errors propagate and the owned figure is closed.
    The footer uses the UTC date of ``SOURCE_DATE_EPOCH`` when set, otherwise
    the current local date. This alone does not guarantee byte-identical exports.
    """
    groups = {
        'geometry': (
            'geometric_ratios', 'quadray_coordinates', 'sphere_packing',
            'ivm_lattice', 'vector_equilibrium', 'frequency_ratios',
        ),
        'arithmetic': (
            'symbolic_operations', 'decimal_to_symbolic', 'continued_fractions',
            'decimal_to_sphere_packing', 'natural_language', 'geometry_precision',
        ),
        'patterns': (
            'mathematical_patterns', 'palindrome_analysis', 'primorial_distribution',
            'pattern_discovery_summary', 'synergetics_sphere_frequency',
        ),
    }
    sections = {
        'overview': (
            'symbolic_operations', 'quadray_coordinates', 'sphere_packing',
            'frequency_ratios', 'decimal_to_symbolic', 'mathematical_patterns',
        ),
        **groups,
        'all': tuple(name for group in groups.values() for name in group),
    }
    if not isinstance(section, str) or section not in sections:
        raise ValueError(f"section must be one of {', '.join(sections)}")
    titles = {
        'overview': 'Symergetics — Exact Numbers, Physical Geometry, Finite Patterns',
        'geometry': 'Symergetics Geometry — Quadrays, FCC Contacts and Physical Scale',
        'arithmetic': 'Symergetics Arithmetic — Exact Results and Approximation Boundaries',
        'patterns': 'Symergetics Patterns — Integer Recurrences and Finite Enumeration',
        'all': 'Symergetics — All 17 Arithmetic, Geometry and Pattern Topics',
    }
    renderers = {
        'symbolic_operations': _create_symbolic_operations_panel,
        'geometric_ratios': _create_geometric_ratios_panel,
        'mathematical_patterns': _create_mathematical_patterns_panel,
        'decimal_to_sphere_packing': _create_decimal_to_sphere_packing_panel,
        'frequency_ratios': _create_frequency_ratios_panel,
        'natural_language': _create_natural_language_panel,
        'sphere_packing': _create_sphere_packing_panel,
        'ivm_lattice': _create_ivm_lattice_panel,
        'vector_equilibrium': _create_vector_equilibrium_panel,
        'decimal_to_symbolic': _create_decimal_to_symbolic_panel,
        'continued_fractions': _create_continued_fractions_panel,
        'quadray_coordinates': _create_quadray_coordinate_panel,
        'palindrome_analysis': _create_palindrome_analysis_panel,
        'primorial_distribution': _create_primorial_distribution_panel,
        'pattern_discovery_summary': _create_pattern_discovery_summary_panel,
        'synergetics_sphere_frequency': _create_synergetics_sphere_frequency_panel,
        'geometry_precision': _create_geometry_precision_panel,
    }
    plt, _ = _ensure_matplotlib_numpy()
    backend = backend or get_config()["backend"]
    if backend != "matplotlib":
        raise ValueError("Mega graphical abstract requires matplotlib backend")
    panel_ids = sections[section]
    columns = 3 if section == 'all' else 2
    rows = (len(panel_ids) + columns - 1) // columns
    size = (8 * columns, 4.5 * rows + 1)
    dpi = 300
    _validate_canvas(size, dpi)
    title = titles[section] if title is None else title
    fig = plt.figure(figsize=size, dpi=150)
    try:
        fig.suptitle(title, fontsize=21, fontweight='bold', y=0.98)
        gs = fig.add_gridspec(
            rows, columns, hspace=0.18, wspace=0.12,
            left=0.035, right=0.975, bottom=0.06, top=0.93,
        )
        data = {}
        panel_titles = {}
        for index, name in enumerate(panel_ids):
            ax = fig.add_subplot(gs[index // columns, index % columns])
            ax.set(xlim=(0, 1), ylim=(0, 1))
            data[name] = renderers[name](ax)
            panel_titles[name] = ax.texts[0].get_text()
            for text in ax.texts:
                text.set_fontsize(max(9, text.get_fontsize()))
            for child in ax.child_axes:
                child.tick_params(labelsize=8)
                for text in child.texts:
                    text.set_fontsize(max(8, text.get_fontsize()))
        _add_contextualizing_text(fig)
        if output_path is None:
            from . import get_organized_output_path
            output_path = get_organized_output_path(
                'mathematical', 'mega_graphical_abstract', 'mega_graphical_abstract.png',
            )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        return {
            'files': [str(output_path)],
            'metadata': {
                'type': 'mega_graphical_abstract', 'title': title,
                'section': section, 'panels': len(panel_ids),
                'panel_ids': list(panel_ids), 'panel_titles': panel_titles,
                'figure_size_inches': list(size), 'dpi': dpi,
                'dimensions': f'{size[0]:g}x{size[1]:g} inches at {dpi} DPI',
                'description': 'Computed rational arithmetic, scale-declared geometry and finite sequence examples',
            },
            'data': data,
        }
    finally:
        plt.close(fig)




def _create_symbolic_operations_panel(ax):
    """Show exact scalar results from the actual rational arithmetic API."""
    from fractions import Fraction
    from ..core.numbers import SymergeticsNumber as N

    rows = [
        ('1/3 + 1/6', str((N(Fraction(1, 3)) + N(Fraction(1, 6))).value)),
        ('(7/5) × (5/21)', str((N(Fraction(7, 5)) * N(Fraction(5, 21))).value)),
        ('sqrt(16/81)', str((N(Fraction(16, 81)) ** Fraction(1, 2)).value)),
        ('(2¹²⁰ + 1) − 2¹²⁰', str(((N(2**120) + 1) - N(2**120)).value)),
    ]
    ax.text(0.5, 0.94, 'EXACT RATIONAL ARITHMETIC', ha='center',
            fontsize=15, fontweight='bold', transform=ax.transAxes)
    table = ax.table(cellText=rows, colLabels=['Operation', 'Exact result'],
                     cellLoc='center', bbox=[0.05, 0.18, 0.9, 0.62])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax.text(0.5, 0.05, 'Integer/Fraction operations; no floating intermediate.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'operations': [{'expression': expression, 'exact_result': result}
                           for expression, result in rows]}




def _create_geometric_ratios_panel(ax):
    """Compare canonical models at one physical scale (translation only)."""
    import numpy as np
    from ..geometry.polyhedra import Tetrahedron, Cube, Octahedron
    from .geometry import _draw_polyhedron, _fit_3d
    ax.text(0.5, 0.92, 'CANONICAL GEOMETRY — ONE PHYSICAL SCALE',
            ha='center', fontsize=13, fontweight='bold')
    view = ax.inset_axes([0.02, 0.23, 0.72, 0.6], projection='3d')
    all_vertices = []
    models = [Tetrahedron(), Cube(), Octahedron()]
    descriptions = []
    model_data = []
    for i, model in enumerate(models):
        vertices = np.array(model.to_xyz_vertices(), dtype=float)
        vertices += np.array([(i - 1) * 6, 0, 0])
        _draw_polyhedron(view, model, vertices, alpha=0.25)
        all_vertices.extend(vertices)
        center = vertices.mean(axis=0)
        volume, area = model.volume(), model.surface_area()
        view.text(*center, f'{type(model).__name__}\n{volume} IVM', fontsize=8)
        descriptions.append(f'{type(model).__name__}:\nV={volume} IVM\nA={area:.4g} XYZ²')
        model_data.append({
            'model': type(model).__name__, 'volume_ivm_exact': str(volume),
            'area_xyz_squared': area, 'display_vertices_xyz': vertices.tolist(),
            'translation_xyz': [(i - 1) * 6, 0, 0],
        })
    _fit_3d(view, all_vertices)
    view.tick_params(labelsize=6)
    ax.text(0.76, 0.7, '\n\n'.join(descriptions), va='top', fontsize=9)
    ax.text(0.5, 0.08, 'Scale-1 XYZ; solids translated, never individually normalized.\n'
            '1 IVM volume = 8/3 XYZ³. Areas use the declared triangle fan.',
            ha='center', fontsize=8)
    ax.axis('off')
    return {'xyz_scale': 1, 'xyz_cubed_per_ivm_exact': '8/3', 'models': model_data}


def _create_sphere_packing_panel(ax):
    """The twelve equal-distance canonical FCC contacts, with actual spheres."""
    import numpy as np
    from ..core.coordinates import IVM_NEIGHBORS
    from .geometry import _fit_3d
    ax.text(0.5, 0.92, 'FCC: TWELVE EQUAL-DISTANCE CONTACTS',
            ha='center', fontsize=12, fontweight='bold')
    view = ax.inset_axes([0.02, 0.22, 0.7, 0.6], projection='3d')
    radius = 1.0
    neighbors = np.array([q.to_xyz() for q in IVM_NEIGHBORS])
    neighbors *= (2 * radius) / np.linalg.norm(neighbors[0])
    centers = np.vstack([np.zeros(3), neighbors])
    view.scatter(*centers.T, color='#8c564b', s=25)
    u, v = np.mgrid[0:2*np.pi:17j, 0:np.pi:9j]
    sphere = np.array([radius * np.cos(u) * np.sin(v),
                       radius * np.sin(u) * np.sin(v), radius * np.cos(v)])
    for center in centers:
        view.plot_surface(*(sphere + center[:, None, None]),
                          color='#8c564b', alpha=0.15, linewidth=0)
    for center in neighbors:
        view.plot(*np.array([np.zeros(3), center]).T, color='#e377c2', linewidth=1)
    _fit_3d(view, np.vstack([centers - radius, centers + radius]))
    view.tick_params(labelsize=6)
    ax.text(0.75, 0.73, 'Sphere radius r=1\n\n12 center contacts\nCenter→neighbor = 2r\n\nFCC packing fraction\nπ/(3√2) ≈ 0.74048\n(infinite packing)',
            fontsize=8, va='top')
    ax.text(0.5, 0.08, 'Centers from the local 12 IVM neighbors, uniformly scaled to contact.',
            ha='center', fontsize=8)
    ax.axis('off')
    return {'radius_xyz': radius, 'centers_xyz': centers.tolist(),
            'contact_distances_xyz': np.linalg.norm(neighbors, axis=1).tolist(),
            'infinite_fcc_packing_fraction': float(np.pi / (3 * np.sqrt(2)))}


def _create_frequency_ratios_panel(ax):
    """Measure integer dilations of one cube, not an invented shape hierarchy."""
    import numpy as np
    from ..core.coordinates import QuadrayCoordinate
    from ..geometry.polyhedra import Cube

    reference = Cube()
    first, second = reference.edges()[0]
    edge = first.distance_to(second)
    area, volume = reference.surface_area(), reference.volume()
    frequencies = np.arange(1, 7)
    ratios = []
    for frequency in frequencies:
        model = Cube()
        model.vertices = [
            QuadrayCoordinate(*(int(frequency) * value for value in point.as_tuple()))
            for point in reference.vertices
        ]
        a, b = model.edges()[0]
        ratios.append((a.distance_to(b) / edge, model.surface_area() / area,
                       float(model.volume() / volume)))
    chart = ax.inset_axes([0.14, 0.39, 0.8, 0.38])
    for values, label in zip(np.asarray(ratios).T, ('Edge / E₁', 'Area / A₁', 'Volume / V₁')):
        chart.loglog(frequencies, values, '-o', label=label)
    chart.set(xlabel='Uniform integer dilation f', ylabel='Dimensionless ratio')
    chart.legend(fontsize=7)
    chart.grid(True, alpha=0.3)
    ax.text(0.5, 0.94, 'MEASURED UNIFORM DILATION', ha='center',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.06, 'One cube: edge ×f; surface area ×f²; volume ×f³.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'dilations': frequencies.tolist(), 'ratios_edge_area_volume': ratios,
            'reference_edge_xyz': edge, 'reference_area_xyz_squared': area,
            'reference_volume_ivm_exact': str(volume)}




def _create_decimal_to_symbolic_panel(ax):
    """Plot measured errors of actual bounded rational approximations."""
    import math
    from ..utils.conversion import best_rational_approximation

    references = [math.pi, math.e, math.sqrt(2), (1 + math.sqrt(5)) / 2]
    names = ['π', 'e', '√2', 'φ']
    approximations = [
        best_rational_approximation(value, max_denominator=1000).value
        for value in references
    ]
    errors = [abs(float(rational) - value)
              for rational, value in zip(approximations, references)]
    chart = ax.inset_axes([0.14, 0.4, 0.8, 0.34])
    chart.bar(range(len(names)), errors, color='#2ca02c')
    chart.set_yscale('log')
    chart.set_xticks(range(len(names)), [
        f'{name}\n{rational}' for name, rational in zip(names, approximations)
    ])
    chart.set_ylabel('Absolute error')
    chart.tick_params(labelsize=8)
    ax.text(0.5, 0.94, 'BOUNDED RATIONAL APPROXIMATION', ha='center',
            fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.07, 'Denominator ≤ 1000; errors against binary-float references.\n'
            'Approximation is not symbolic equality to an irrational constant.',
            ha='center', fontsize=8, transform=ax.transAxes)
    ax.axis('off')
    return {'names': names, 'binary_float_references': references,
            'max_denominator': 1000, 'rational_approximations_exact': [
                str(value) for value in approximations], 'absolute_errors': errors}




def _create_vector_equilibrium_panel(ax):
    """Render the local cuboctahedron's genuine edges and physical units."""
    import numpy as np
    from ..geometry.polyhedra import Cuboctahedron
    from .geometry import _draw_polyhedron, _edge_indices, _fit_3d
    model = Cuboctahedron()
    vertices = np.array(model.to_xyz_vertices())
    ax.text(0.5, 0.92, 'VECTOR EQUILIBRIUM — CUBOCTAHEDRON',
            ha='center', fontsize=12, fontweight='bold')
    view = ax.inset_axes([0.02, 0.22, 0.7, 0.6], projection='3d')
    _draw_polyhedron(view, model, vertices, alpha=0.2)
    _fit_3d(view, vertices)
    view.tick_params(labelsize=6)
    edges = _edge_indices(model)
    edge_length = np.linalg.norm(vertices[edges[0][0]] - vertices[edges[0][1]])
    ax.text(0.75, 0.74,
            f'{len(vertices)} vertices\n{len(edges)} edges\n{len(model.face_indices())} faces\n'
            f'6 squares + 8 triangles\n\nV = {model.volume()} IVM\n'
            f'V = {float(model.volume()) * 8 / 3:.5g} XYZ³\n'
            f'Edge = {edge_length:.5g} XYZ\nA = {model.surface_area():.5g} XYZ²',
            fontsize=8, va='top')
    ax.text(0.5, 0.08, 'Native scale-1 model; no separate circumradius normalization.',
            ha='center', fontsize=8)
    ax.axis('off')
    return {'xyz_scale': 1, 'vertices_xyz': vertices.tolist(),
            'edge_indices': [list(pair) for pair in edges],
            'face_indices': [list(face) for face in model.face_indices()],
            'volume_ivm_exact': str(model.volume()),
            'volume_xyz_cubed_exact': str(model.volume() * 8 / 3),
            'edge_length_xyz': float(edge_length),
            'area_xyz_squared': model.surface_area()}


def _create_decimal_to_sphere_packing_panel(ax):
    """An exact count-volume ratio, distinct from cubic frequency scaling."""
    import math
    from ..geometry.polyhedra import Tetrahedron
    from ..utils.conversion import best_rational_approximation

    rational = best_rational_approximation(math.pi, max_denominator=1000).value
    unit = Tetrahedron().volume()
    counts = [rational.numerator, rational.denominator]
    volumes = [count * unit for count in counts]
    chart = ax.inset_axes([0.14, 0.4, 0.8, 0.34])
    chart.bar([0, 1], [float(value) for value in volumes], color=['#1f77b4', '#ff7f0e'])
    chart.set_xticks([0, 1], [f'{count} unit tetrahedra' for count in counts])
    chart.set_ylabel('Total IVM volume')
    chart.tick_params(labelsize=8)
    ax.text(0.5, 0.94, 'COUNT RATIOS, NOT FREQUENCY POWERS', ha='center',
            fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.08, f'Volume ratio = {volumes[0] / volumes[1]} ≈ π.\n'
            'These are counts of congruent units, not scaled tetrahedra.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'counts': counts, 'unit_tetrahedron_volume_ivm_exact': str(unit),
            'total_volumes_ivm_exact': [str(value) for value in volumes],
            'volume_ratio_exact': str(volumes[0] / volumes[1]),
            'binary_float_reference': math.pi}




def _create_natural_language_panel(ax):
    """Distinguish rational count statements from irrational limits."""
    ax.text(0.5, 0.94, 'GEOMETRIC LANGUAGE WITH DECLARED MEANING', ha='center',
            fontsize=12, fontweight='bold', transform=ax.transAxes)
    explanations = [
        'Counts: volume of n congruent unit tetrahedra divided by\n'
        'volume of m congruent unit tetrahedra is exactly n/m (m > 0).',
        'Dilation: an n/m edge-scale ratio gives area ratio (n/m)²\n'
        'and volume ratio (n/m)³ for the same shape.',
        'Every rational has an integer-pair representation.\n'
        'Irrational constants require approximation or a limiting process.',
    ]
    for y, text in zip((0.72, 0.44, 0.16), explanations):
        ax.text(0.5, y, text, ha='center', va='center', fontsize=10,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9))
    ax.axis('off')
    return {'statements': explanations}




def _create_synergetics_sphere_frequency_panel(ax):
    """Tetrahedral cluster layer counts, not FCC shells or volume frequencies."""
    layers = [1, 2, 4, 8, 12, 20, 30, 42, 60, 84]
    # Same exact tetrahedral-number formula used by geometric_mnemonics.
    counts = [n * (n + 1) * (n + 2) // 6 for n in layers]
    ax.text(0.5, 0.95, 'TETRAHEDRAL SPHERE CLUSTERS', ha='center',
            fontsize=17, fontweight='bold')
    ax.text(0.5, 0.87, 'T(n) = n(n+1)(n+2)/6; n is the number of triangular layers',
            ha='center', fontsize=12)
    chart = ax.inset_axes([0.08, 0.43, 0.85, 0.32])
    chart.semilogy(layers, counts, '-o', color='#8c564b', label='Tetrahedral cluster sites')
    for index in (0, 4, 7, 9):
        n, total = layers[index], counts[index]
        chart.annotate(f'n={n}\nT={total}', (n, total), xytext=(4, 4),
                       textcoords='offset points', fontsize=8)
    chart.margins(y=0.25)
    chart.set_xlabel('Tetrahedral cluster layer index n', fontsize=9)
    chart.set_ylabel('Sphere count T(n)', fontsize=9)
    chart.tick_params(labelsize=8)
    chart.grid(True, alpha=0.3)
    ax.text(0.5, 0.16, 'Counts computed from triangular-layer sums.',
            ha='center', fontsize=10)
    ax.text(0.5, 0.08, 'Finite tetrahedral clusters — not FCC graph shells or subdivision volumes.',
            ha='center', fontsize=9)
    ax.axis('off')
    return {'layer_indices': layers, 'sphere_counts': counts,
            'count_formula': 'n(n+1)(n+2)/6'}


def _create_mathematical_patterns_panel(ax):
    """Actual consecutive Fibonacci ratios, not a synthetic convergence curve."""
    import math
    from fractions import Fraction
    fibonacci = [1, 1]
    for _ in range(13):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    ratios = [b / a for a, b in zip(fibonacci, fibonacci[1:])]
    chart = ax.inset_axes([0.14, 0.39, 0.8, 0.38])
    chart.plot(range(1, len(ratios) + 1), ratios, '-o', label='F(n+1) / F(n)')
    chart.axhline((1 + math.sqrt(5)) / 2, linestyle='--', color='#d62728',
                  label='Binary-float φ reference')
    chart.set(xlabel='Consecutive Fibonacci index n', ylabel='Ratio')
    chart.legend(fontsize=7)
    chart.grid(True, alpha=0.3)
    ax.text(0.5, 0.94, 'FIBONACCI RATIOS FROM INTEGER RECURRENCE', ha='center',
            fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.06, 'F(n+2) = F(n+1) + F(n), starting at 1, 1.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'fibonacci_integers': fibonacci, 'consecutive_ratios_exact': [
                str(Fraction(b, a)) for a, b in zip(fibonacci, fibonacci[1:])],
            'plotted_ratios': ratios, 'binary_float_phi_reference': (1 + math.sqrt(5)) / 2}




def _create_quadray_coordinate_panel(ax):
    """Actual scale-1 quadray embedding and common-offset gauge equivalence."""
    import numpy as np
    from ..core.coordinates import QuadrayCoordinate
    from .geometry import _fit_3d
    ax.text(0.5, 0.92, 'QUADRAYS — FOUR LABELS, THREE PHYSICAL DIMENSIONS',
            ha='center', fontsize=11, fontweight='bold')
    view = ax.inset_axes([0.02, 0.23, 0.6, 0.58], projection='3d')
    basis = [QuadrayCoordinate(*(int(i == j) for j in range(4))) for i in range(4)]
    vertices = np.array([q.to_xyz() for q in basis])
    for name, vertex, color in zip('ABCD', vertices, ['red', 'green', 'blue', 'orange']):
        view.plot(*np.array([np.zeros(3), vertex]).T, color=color)
        view.scatter(*vertex, color=color, s=30)
        view.text(*vertex, name, fontsize=8)
    examples = [QuadrayCoordinate(1, 1, 0, 0), QuadrayCoordinate(1, 1, 1, 0),
                QuadrayCoordinate(1, 1, 1, 1, normalize=False)]
    for q in examples:
        xyz = q.to_xyz()
        view.scatter(*xyz, color='purple', s=25)
        view.text(*xyz, str(q.as_tuple()), fontsize=6)
    _fit_3d(view, [*vertices, (0, 0, 0), *(q.to_xyz() for q in examples)])
    view.tick_params(labelsize=6)
    equations = ('x = a − b − c + d\n'
                 'y = a + b − c − d\n'
                 'z = a − b + c − d\n\n'
                 '(a,b,c,d) ∼ (a+k,b+k,c+k,d+k)\n'
                 '(1,1,1,1) is the physical origin.\n\n'
                 'No sum-to-one constraint;\nnot normalized barycentric weights.')
    ax.text(0.64, 0.72, equations, fontsize=7, va='top')
    ax.text(0.5, 0.08, 'All plotted examples use QuadrayCoordinate.to_xyz() at scale 1.',
            ha='center', fontsize=8)
    ax.axis('off')
    return {'xyz_scale': 1, 'basis_quadrays': [list(q.as_tuple()) for q in basis],
            'basis_xyz': vertices.tolist(),
            'examples': [{'quadray': list(q.as_tuple()), 'xyz': list(q.to_xyz())}
                         for q in examples],
            'common_offset_kernel': [1, 1, 1, 1]}




def _create_continued_fractions_panel(ax):
    """Compute convergents from the actual finite-reference expansion."""
    import math
    from fractions import Fraction
    from ..utils.conversion import continued_fraction_approximation

    terms = continued_fraction_approximation(math.pi, max_terms=8)
    previous_h, current_h, previous_k, current_k = 0, 1, 1, 0
    convergents = []
    exact_convergents = []
    for term in terms:
        previous_h, current_h = current_h, term * current_h + previous_h
        previous_k, current_k = current_k, term * current_k + previous_k
        exact_convergents.append(str(Fraction(current_h, current_k)))
        convergents.append(current_h / current_k)
    chart = ax.inset_axes([0.14, 0.39, 0.8, 0.38])
    chart.plot(range(1, len(convergents) + 1), convergents, '-o', label='Convergent')
    chart.axhline(math.pi, color='#d62728', linestyle='--', label='Binary-float π reference')
    chart.set(xlabel='Number of terms', ylabel='Rational value')
    chart.legend(fontsize=7)
    chart.grid(True, alpha=0.3)
    ax.text(0.5, 0.94, 'CONTINUED FRACTIONS', ha='center',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.07, 'Finite float-reference expansion, not a proof of all π coefficients.',
            ha='center', fontsize=8, transform=ax.transAxes)
    ax.axis('off')
    return {'binary_float_pi_reference': math.pi, 'terms': terms,
            'convergents_exact': exact_convergents, 'plotted_convergents': convergents}




def _create_palindrome_analysis_panel(ax):
    """Digit profiles centered between the two physical end positions."""
    import numpy as np
    from ..computation.palindromes import is_palindromic

    chart = ax.inset_axes([0.14, 0.39, 0.8, 0.38])
    profiles = []
    for number in ('121', '12321', '123454321', '1001'):
        if not is_palindromic(number):
            raise ValueError(f'Expected a palindrome: {number}')
        digits = [int(digit) for digit in number]
        positions = np.arange(len(digits)) - (len(digits) - 1) / 2
        chart.plot(positions, digits, '-o', label=number)
        profiles.append({'number': number, 'digits': digits,
                         'midpoint_positions': positions.tolist()})
    chart.axvline(0, color='#d62728', linestyle='--', alpha=0.7)
    chart.set(xlabel='Digit position relative to midpoint', ylabel='Digit value')
    chart.legend(fontsize=7)
    chart.grid(True, alpha=0.3)
    ax.text(0.5, 0.94, 'PALINDROME REFLECTION SYMMETRY', ha='center',
            fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.06, 'Odd and even digit counts reflect about the same x=0 axis.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'base': 10, 'profiles': profiles}




def _create_primorial_distribution_panel(ax):
    """Use the actual prime-bound primorial API rather than a second sieve."""
    from ..computation.primorials import primorial

    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    values = [int(primorial(prime).value) for prime in primes]
    chart = ax.inset_axes([0.14, 0.39, 0.8, 0.38])
    chart.semilogy(primes, values, '-o', color='#9467bd')
    chart.set(xlabel='Prime bound p', ylabel='p# = product of primes ≤ p')
    chart.grid(True, alpha=0.3)
    ax.text(0.5, 0.94, 'PRIMORIAL GROWTH', ha='center',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.06, 'Actual integer products; ln(p#) ~ p asymptotically.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'prime_bounds': list(primes), 'primorials_exact': values}




def _create_ivm_lattice_panel(ax):
    """Render the actual radius-two FCC ball, not a textual lattice placeholder."""
    import numpy as np
    from ..core.coordinates import enumerate_ivm_sites
    from .geometry import _fit_3d

    sites = enumerate_ivm_sites(2)
    points = np.asarray([site.to_xyz() for site in sites])
    view = ax.inset_axes([0.08, 0.18, 0.84, 0.64], projection='3d')
    view.scatter(*points.T, c=np.linalg.norm(points, axis=1), cmap='viridis', s=20)
    _fit_3d(view, points)
    view.tick_params(labelsize=6)
    ax.text(0.5, 0.94, 'CANONICAL FCC GRAPH BALL', ha='center',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.05, f'{len(sites)} physical sites through graph radius 2; scale-one XYZ.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'graph_radius': 2, 'xyz_scale': 1, 'physical_site_count': len(sites),
            'site_quadrays': [list(site.as_tuple()) for site in sites],
            'sites_xyz': points.tolist()}




def _create_pattern_discovery_summary_panel(ax):
    """Enumerate a declared finite population rather than imply an open search."""
    from ..computation.palindromes import is_palindromic

    ranges = (range(1, 10), range(10, 100), range(100, 1000))
    counts = [sum(is_palindromic(str(number)) for number in values) for values in ranges]
    chart = ax.inset_axes([0.14, 0.35, 0.8, 0.42])
    chart.bar([1, 2, 3], counts, color='#bcbd22')
    chart.set_xticks([1, 2, 3], ['1 digit', '2 digits', '3 digits'])
    chart.set_ylabel('Palindrome count')
    ax.text(0.5, 0.94, 'FINITE PALINDROME ENUMERATION', ha='center',
            fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.06, 'All positive base-10 integers 1–999; no leading zeros.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'base': 10, 'positive_integer_ranges_inclusive': [
                [values.start, values.stop - 1] for values in ranges],
            'palindrome_counts': counts, 'leading_zeros': False}




def _create_geometry_precision_panel(ax):
    """Computed geometry examples; never embed invented test/coverage metadata."""
    from ..core.coordinates import QuadrayCoordinate, urner_embedding
    from ..geometry.polyhedra import Cube, integer_tetra_volume

    q = QuadrayCoordinate(2**60 + 1, 2**60, 0, 0)
    alias = QuadrayCoordinate(*(value + 2**80 for value in q.as_tuple()), normalize=False)
    origin = QuadrayCoordinate(0, 0, 0, 0)
    basis = [QuadrayCoordinate(*(int(i == j) for j in range(4))) for i in range(3)]
    rows = [
        ('Huge-gauge squared separation', str(q.squared_distance_exact(alias))),
        ('Resolved cancellation dot', str(q.dot_exact(QuadrayCoordinate(1, 0, 1, 0)))),
        ('Primitive tetrahedron, IVM', str(integer_tetra_volume(origin, *basis))),
        ('Cube area, XYZ scale 1e−162', str(Cube().surface_area(urner_embedding(1e-162)))),
    ]
    table = ax.table(cellText=rows, colLabels=['Actual API example', 'Computed result'],
                     cellLoc='center', bbox=[0.04, 0.2, 0.92, 0.58])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    ax.text(0.5, 0.94, 'EXACT AND FLOATING BOUNDARIES', ha='center',
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.06, 'Exact rational metrics and volumes; area rounds once to float.\n'
            'Examples are not a test-suite or universal-correctness certificate.',
            ha='center', fontsize=9, transform=ax.transAxes)
    ax.axis('off')
    return {'computed_results': [{'example': example, 'result': value}
                                 for example, value in rows],
            'metric_units': 'scale-1 XYZ squared',
            'primitive_volume_unit': 'IVM', 'area_unit': 'XYZ squared',
            'area_embedding_scale': 1e-162,
            'quadray': list(q.as_tuple()), 'gauge_alias': list(alias.as_tuple())}




def _add_contextualizing_text(fig):
    """Use the UTC SOURCE_DATE_EPOCH date, or the current local date if unset."""
    from datetime import datetime, timezone
    import os

    epoch = os.environ.get('SOURCE_DATE_EPOCH')
    generated = (datetime.fromtimestamp(int(epoch), timezone.utc)
                 if epoch is not None else datetime.now())

    fig.text(0.05, 0.025,
             'Computed rational arithmetic, declared geometric units and finite sequence examples.\n'
             'Continuous irrational quantities remain approximations; mesh validity has explicit preconditions.',
             fontsize=10, ha='left', va='bottom')
    fig.text(0.97, 0.025, f'Generated: {generated:%Y-%m-%d}',
             fontsize=10, ha='right', va='bottom')
