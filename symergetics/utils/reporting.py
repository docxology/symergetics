#!/usr/bin/env python3
"""
Advanced Reporting Methods for Symergetics Package

This module provides comprehensive reporting capabilities for mathematical
analysis results, including statistical summaries, comparative analysis,
and export functionality in multiple formats.
"""

import json
import csv
import statistics
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ReportMetrics:
    """Container for report-level metrics."""
    total_items: int
    successful_analyses: int
    failed_analyses: int
    average_complexity: float
    average_symmetry: float
    palindromic_ratio: float
    generation_time: float


@dataclass
class AnalysisSummary:
    """Summary of analysis results."""
    domain: str
    item_count: int
    palindromic_count: int
    average_complexity: float
    average_symmetry: float
    unique_patterns: int
    timestamp: str


def generate_statistical_summary(
    analysis_results: List[Dict[str, Any]],
    title: str = "Statistical Analysis Summary"
) -> Dict[str, Any]:
    """
    Generate statistical summary of analysis results.

    Args:
        analysis_results: List of analysis result dictionaries
        title: Title for the summary

    Returns:
        Dict containing statistical summary
    """
    if not analysis_results:
        return {
            'title': title,
            'timestamp': datetime.now().isoformat(),
            'summary': 'No analysis results provided',
            'total_analyses': 0,
            'metrics': {}
        }

    # Extract numerical metrics
    complexities = []
    symmetries = []
    densities = []
    lengths = []
    palindromic_count = 0

    for result in analysis_results:
        if 'pattern_complexity' in result:
            comp = result['pattern_complexity'].get('complexity_score', 0)
            if comp is not None:
                complexities.append(comp)

        if 'symmetry_analysis' in result:
            sym = result['symmetry_analysis'].get('symmetry_score', 0)
            if sym is not None:
                symmetries.append(sym)

        if 'palindromic_density' in result:
            density = result['palindromic_density']
            if density is not None:
                densities.append(density)

        if 'length' in result:
            length = result['length']
            if length is not None:
                lengths.append(length)

        if result.get('is_palindromic', False):
            palindromic_count += 1

    # Calculate statistics
    summary = {
        'title': title,
        'timestamp': datetime.now().isoformat(),
        'total_analyses': len(analysis_results),
        'palindromic_count': palindromic_count,
        'palindromic_ratio': palindromic_count / len(analysis_results) if analysis_results else 0,
        'metrics': {}
    }

    # Add statistical metrics
    if complexities:
        summary['metrics']['complexity'] = {
            'mean': statistics.mean(complexities),
            'median': statistics.median(complexities),
            'stdev': statistics.stdev(complexities) if len(complexities) > 1 else 0,
            'min': min(complexities),
            'max': max(complexities)
        }

    if symmetries:
        summary['metrics']['symmetry'] = {
            'mean': statistics.mean(symmetries),
            'median': statistics.median(symmetries),
            'stdev': statistics.stdev(symmetries) if len(symmetries) > 1 else 0,
            'min': min(symmetries),
            'max': max(symmetries)
        }

    if densities:
        summary['metrics']['palindromic_density'] = {
            'mean': statistics.mean(densities),
            'median': statistics.median(densities),
            'stdev': statistics.stdev(densities) if len(densities) > 1 else 0,
            'min': min(densities),
            'max': max(densities)
        }

    if lengths:
        summary['metrics']['length'] = {
            'mean': statistics.mean(lengths),
            'median': statistics.median(lengths),
            'stdev': statistics.stdev(lengths) if len(lengths) > 1 else 0,
            'min': min(lengths),
            'max': max(lengths)
        }

    # Add insights
    summary['insights'] = _generate_statistical_insights(summary)

    return summary


def _generate_statistical_insights(summary: Dict[str, Any]) -> List[str]:
    """Generate insights from statistical summary."""
    insights = []

    palindromic_ratio = summary.get('palindromic_ratio', 0)
    if palindromic_ratio > 0.5:
        insights.append(f"High palindromic ratio ({palindromic_ratio:.1%}) indicates strong pattern presence")
    elif palindromic_ratio < 0.2:
        insights.append(f"Low palindromic ratio ({palindromic_ratio:.1%}) suggests diverse pattern distribution")

    if 'metrics' in summary:
        metrics = summary['metrics']

        # Complexity insights
        if 'complexity' in metrics:
            comp_mean = metrics['complexity']['mean']
            if comp_mean > 2:
                insights.append(f"High average complexity ({comp_mean:.2f}) indicates intricate patterns")
            elif comp_mean < 1:
                insights.append(f"Low average complexity ({comp_mean:.2f}) suggests simple structures")

        # Symmetry insights
        if 'symmetry' in metrics:
            sym_mean = metrics['symmetry']['mean']
            if sym_mean > 0.7:
                insights.append(f"High average symmetry ({sym_mean:.2f}) indicates regular structures")
            elif sym_mean < 0.3:
                insights.append(f"Low average symmetry ({sym_mean:.2f}) suggests irregular patterns")

        # Length insights
        if 'length' in metrics:
            len_mean = metrics['length']['mean']
            len_stdev = metrics['length']['stdev']
            if len_stdev > len_mean * 0.5:
                insights.append("High length variation suggests diverse mathematical domains")

    return insights


def generate_comparative_report(
    domain1_results: List[Dict[str, Any]],
    domain2_results: List[Dict[str, Any]],
    domain1_name: str = "Domain 1",
    domain2_name: str = "Domain 2",
    title: str = "Comparative Analysis Report"
) -> Dict[str, Any]:
    """
    Generate comparative report between two domains.

    Args:
        domain1_results: Analysis results from first domain
        domain2_results: Analysis results from second domain
        domain1_name: Name of first domain
        domain2_name: Name of second domain
        title: Report title

    Returns:
        Dict containing comparative analysis
    """
    report = {
        'title': title,
        'timestamp': datetime.now().isoformat(),
        'domains': {
            'domain1': domain1_name,
            'domain2': domain2_name
        },
        'summary': {},
        'comparisons': {},
        'insights': []
    }

    # Generate summaries for each domain
    domain1_summary = generate_statistical_summary(domain1_results, f"{domain1_name} Summary")
    domain2_summary = generate_statistical_summary(domain2_results, f"{domain2_name} Summary")

    report['domain_summaries'] = {
        domain1_name: domain1_summary,
        domain2_name: domain2_summary
    }

    # Calculate comparisons
    comparisons = {}

    # Compare palindromic ratios
    pal1 = domain1_summary.get('palindromic_ratio', 0)
    pal2 = domain2_summary.get('palindromic_ratio', 0)
    comparisons['palindromic_ratio'] = {
        domain1_name: pal1,
        domain2_name: pal2,
        'difference': pal2 - pal1,
        'ratio': pal2 / pal1 if pal1 > 0 else float('inf')
    }

    # Compare complexity
    comp1 = domain1_summary.get('metrics', {}).get('complexity', {}).get('mean', 0)
    comp2 = domain2_summary.get('metrics', {}).get('complexity', {}).get('mean', 0)
    comparisons['complexity'] = {
        domain1_name: comp1,
        domain2_name: comp2,
        'difference': comp2 - comp1,
        'similarity': 1 - abs(comp1 - comp2) / max(comp1, comp2) if max(comp1, comp2) > 0 else 1
    }

    # Compare symmetry
    sym1 = domain1_summary.get('metrics', {}).get('symmetry', {}).get('mean', 0)
    sym2 = domain2_summary.get('metrics', {}).get('symmetry', {}).get('mean', 0)
    comparisons['symmetry'] = {
        domain1_name: sym1,
        domain2_name: sym2,
        'difference': sym2 - sym1,
        'similarity': 1 - abs(sym1 - sym2) / max(sym1, sym2) if max(sym1, sym2) > 0 else 1
    }

    report['comparisons'] = comparisons

    # Generate comparative insights
    report['insights'] = _generate_comparative_insights(comparisons, domain1_name, domain2_name)

    return report


def _generate_comparative_insights(
    comparisons: Dict[str, Any],
    domain1_name: str,
    domain2_name: str
) -> List[str]:
    """Generate insights from comparative analysis."""
    insights = []

    # Palindromic comparison
    if 'palindromic_ratio' in comparisons:
        pal_comp = comparisons['palindromic_ratio']
        diff = pal_comp['difference']
        if abs(diff) > 0.2:
            higher = domain2_name if diff > 0 else domain1_name
            lower = domain1_name if diff > 0 else domain2_name
            insights.append(f"{higher} shows significantly higher palindromic patterns than {lower}")
        else:
            insights.append(f"Similar palindromic pattern distribution between domains")

    # Complexity comparison
    if 'complexity' in comparisons:
        comp_comp = comparisons['complexity']
        similarity = comp_comp['similarity']
        if similarity > 0.8:
            insights.append("Domains show similar complexity patterns")
        elif similarity < 0.5:
            insights.append("Domains exhibit different complexity characteristics")

    # Symmetry comparison
    if 'symmetry' in comparisons:
        sym_comp = comparisons['symmetry']
        similarity = sym_comp['similarity']
        if similarity > 0.8:
            insights.append("Domains demonstrate similar symmetry properties")
        elif similarity < 0.5:
            insights.append("Domains show different symmetry characteristics")

    return insights


def export_report_to_json(
    report_data: Dict[str, Any],
    output_path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Export report to JSON format.

    Args:
        report_data: Report data dictionary
        output_path: Path to output file
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=indent, ensure_ascii=False)

    print(f"Report exported to: {output_path}")


def export_report_to_csv(
    report_data: Dict[str, Any],
    output_path: Union[str, Path],
    include_nested: bool = False
) -> None:
    """
    Export report to CSV format (flattened structure).

    Args:
        report_data: Report data dictionary
        output_path: Path to output file
        include_nested: Whether to include nested dictionary data
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten the report data for CSV export
    flattened_data = _flatten_dict(report_data)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Key', 'Value'])

        for key, value in flattened_data.items():
            writer.writerow([key, str(value)])

    print(f"Report exported to: {output_path}")


def _flatten_dict(d: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Flatten nested dictionary for CSV export."""
    flattened = {}

    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, new_key))
        elif isinstance(value, list):
            # Convert lists to comma-separated strings
            flattened[new_key] = ', '.join(str(item) for item in value)
        else:
            flattened[new_key] = value

    return flattened


def export_report_to_markdown(
    report_data: Dict[str, Any],
    output_path: Union[str, Path]
) -> None:
    """
    Export report to Markdown format.

    Args:
        report_data: Report data dictionary
        output_path: Path to output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown_content = _convert_report_to_markdown(report_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"Report exported to: {output_path}")


def _convert_report_to_markdown(report_data: Dict[str, Any]) -> str:
    """Convert report data to Markdown format."""
    lines = []

    # Title
    title = report_data.get('title', 'Analysis Report')
    lines.append(f"# {title}")
    lines.append("")

    # Timestamp
    timestamp = report_data.get('timestamp', 'Unknown')
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")

    # Summary section
    if 'summary' in report_data:
        lines.append("## Summary")
        lines.append("")

        summary = report_data['summary']
        if isinstance(summary, dict):
            for key, value in summary.items():
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        else:
            lines.append(str(summary))
        lines.append("")

    # Metrics section
    if 'metrics' in report_data:
        lines.append("## Metrics")
        lines.append("")

        metrics = report_data['metrics']
        for category, category_metrics in metrics.items():
            lines.append(f"### {category.replace('_', ' ').title()}")
            if isinstance(category_metrics, dict):
                for metric, value in category_metrics.items():
                    if isinstance(value, float):
                        lines.append(f"- **{metric}:** {value:.4f}")
                    else:
                        lines.append(f"- **{metric}:** {value}")
            lines.append("")

    # Insights section
    if 'insights' in report_data and report_data['insights']:
        lines.append("## Key Insights")
        lines.append("")

        for insight in report_data['insights']:
            lines.append(f"- {insight}")
        lines.append("")

    # Comparisons section (for comparative reports)
    if 'comparisons' in report_data:
        lines.append("## Comparisons")
        lines.append("")

        comparisons = report_data['comparisons']
        for comparison_type, comparison_data in comparisons.items():
            lines.append(f"### {comparison_type.replace('_', ' ').title()}")
            if isinstance(comparison_data, dict):
                for key, value in comparison_data.items():
                    if isinstance(value, float):
                        lines.append(f"- **{key}:** {value:.4f}")
                    else:
                        lines.append(f"- **{key}:** {value}")
            lines.append("")

    return "\n".join(lines)


def generate_performance_report(
    analysis_results: List[Dict[str, Any]],
    execution_times: Optional[List[float]] = None,
    title: str = "Performance Analysis Report"
) -> Dict[str, Any]:
    """
    Generate performance analysis report.

    Args:
        analysis_results: List of analysis result dictionaries
        execution_times: Optional list of execution times for each analysis
        title: Report title

    Returns:
        Dict containing performance analysis
    """
    report = {
        'title': title,
        'timestamp': datetime.now().isoformat(),
        'performance_metrics': {},
        'efficiency_analysis': {},
        'recommendations': []
    }

    total_analyses = len(analysis_results)
    report['performance_metrics']['total_analyses'] = total_analyses

    # Execution time analysis
    if execution_times:
        report['performance_metrics']['execution_times'] = {
            'total_time': sum(execution_times),
            'average_time': statistics.mean(execution_times),
            'median_time': statistics.median(execution_times),
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'time_stddev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        }

        # Calculate throughput
        total_time = sum(execution_times)
        if total_time > 0:
            report['performance_metrics']['throughput'] = total_analyses / total_time

    # Analysis complexity vs time correlation
    if execution_times:
        complexities = []
        for result in analysis_results:
            comp = result.get('pattern_complexity', {}).get('complexity_score', 0)
            complexities.append(comp)

        if complexities and len(complexities) == len(execution_times):
            # Calculate correlation between complexity and execution time
            try:
                correlation = statistics.correlation(complexities, execution_times)
                report['efficiency_analysis']['complexity_time_correlation'] = correlation

                if correlation > 0.7:
                    report['efficiency_analysis']['insight'] = "High correlation between complexity and execution time"
                elif correlation < 0.3:
                    report['efficiency_analysis']['insight'] = "Low correlation between complexity and execution time"
                else:
                    report['efficiency_analysis']['insight'] = "Moderate correlation between complexity and execution time"
            except:
                pass

    # Memory efficiency analysis (estimated)
    avg_length = statistics.mean([r.get('length', 0) for r in analysis_results]) if analysis_results else 0
    report['performance_metrics']['average_input_size'] = avg_length

    # Generate recommendations
    recommendations = []

    if execution_times:
        avg_time = statistics.mean(execution_times)
        if avg_time > 1.0:
            recommendations.append("Consider optimizing analysis for large inputs")
        elif avg_time < 0.01:
            recommendations.append("Analysis is very fast - good performance")

    throughput = report['performance_metrics'].get('throughput', 0)
    if throughput > 100:
        recommendations.append("High throughput achieved - excellent performance")
    elif throughput < 10:
        recommendations.append("Consider batch processing optimization")

    report['recommendations'] = recommendations

    return report
