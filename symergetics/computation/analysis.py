#!/usr/bin/env python3
"""
Advanced Analysis Methods for Symergetics Package

This module provides comprehensive analysis methods for mathematical patterns,
statistical analysis, pattern discovery, and cross-domain correlations.
These methods are designed to work with the core Symergetics mathematical
framework and provide deeper insights into mathematical structures.
"""

import math
import statistics
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict, Counter
from dataclasses import dataclass

from ..core.numbers import SymergeticsNumber
from ..computation.primorials import primorial, scheherazade_power, is_prime
from ..computation.palindromes import is_palindromic, calculate_palindromic_density


@dataclass
class PatternMetrics:
    """Container for pattern analysis metrics."""
    complexity_score: float
    symmetry_index: float
    density_score: float
    entropy_measure: float
    fractal_dimension: Optional[float] = None
    self_similarity_score: float = 0.0


@dataclass
class ComparativeAnalysis:
    """Container for comparative analysis results."""
    correlation_coefficient: float
    similarity_score: float
    domain_overlap: float
    structural_similarity: float
    cross_domain_patterns: List[str]


def analyze_mathematical_patterns(
    number: Union[int, str, SymergeticsNumber],
    analysis_depth: int = 3
) -> Dict[str, Any]:
    """
    Perform comprehensive mathematical pattern analysis.

    Args:
        number: Number to analyze
        analysis_depth: Depth of analysis (1-5, higher = more detailed)

    Returns:
        Dict containing pattern analysis results
    """
    if isinstance(number, SymergeticsNumber):
        num_str = str(number.value)
        numeric_value = float(number.value)
    else:
        num_str = str(number)
        # Handle empty strings and non-numeric strings
        try:
            numeric_value = float(number) if number else 0.0
        except (ValueError, TypeError):
            numeric_value = 0.0

    results = {
        'number': str(number),
        'numeric_value': numeric_value,
        'string_representation': num_str,
        'length': len(num_str)
    }

    # Basic pattern analysis
    if num_str:  # Only analyze non-empty strings
        try:
            results['is_palindromic'] = is_palindromic(number)
            results['palindromic_density'] = calculate_palindromic_density(number)
        except (TypeError, ValueError):
            results['is_palindromic'] = False
            results['palindromic_density'] = 0.0
    else:
        results['is_palindromic'] = True  # Empty string is technically palindromic
        results['palindromic_density'] = 0.0

    # Digit distribution analysis
    digit_counts = Counter(num_str)
    results['digit_distribution'] = dict(digit_counts)
    results['unique_digits'] = len(digit_counts)
    results['most_common_digit'] = digit_counts.most_common(1)[0][0] if digit_counts else None

    # Statistical analysis
    if num_str and all(c.isdigit() for c in num_str):
        digits = [int(c) for c in num_str]
        if digits:  # Ensure we have at least one digit
            results['digit_statistics'] = {
                'mean': statistics.mean(digits),
                'median': statistics.median(digits),
                'stdev': statistics.stdev(digits) if len(digits) > 1 else 0,
                'variance': statistics.variance(digits) if len(digits) > 1 else 0,
                'range': max(digits) - min(digits),
                'mode': statistics.mode(digits)
            }

        # Entropy calculation
        total_digits = len(digits)
        if total_digits > 0:
            entropy = 0.0
            for count in digit_counts.values():
                if count > 0:
                    prob = count / total_digits
                    entropy -= prob * math.log2(prob)
            results['digit_entropy'] = entropy
            results['entropy_ratio'] = entropy / math.log2(10)  # Max possible entropy
        else:
            results['digit_entropy'] = 0.0
            results['entropy_ratio'] = 0.0

    # Pattern complexity analysis
    if analysis_depth >= 2:
        results['pattern_complexity'] = _analyze_pattern_complexity(num_str)

    # Symmetry analysis
    if analysis_depth >= 3:
        results['symmetry_analysis'] = _analyze_symmetry_patterns(num_str)

    # Advanced analysis for deeper levels
    if analysis_depth >= 4:
        results['fractal_analysis'] = _analyze_fractal_properties(num_str)

    if analysis_depth >= 5:
        results['cross_domain_correlations'] = _analyze_cross_domain_patterns(number)

    return results


def _analyze_pattern_complexity(number_str: str) -> Dict[str, Any]:
    """Analyze the complexity of patterns in a number string."""
    complexity_metrics = {}

    # Repetition analysis
    repetitions = []
    for length in range(1, min(len(number_str) // 2 + 1, 10)):
        for i in range(len(number_str) - length + 1):
            pattern = number_str[i:i+length]
            count = number_str.count(pattern)
            if count > 1:
                repetitions.append({
                    'pattern': pattern,
                    'length': length,
                    'count': count,
                    'positions': [j for j in range(len(number_str) - length + 1)
                                if number_str[j:j+length] == pattern]
                })

    complexity_metrics['repetitions'] = repetitions

    # Sequence analysis
    if len(number_str) >= 3:
        is_arithmetic = _is_arithmetic_sequence(number_str)
        is_geometric = _is_geometric_sequence(number_str)

        complexity_metrics['sequence_patterns'] = {
            'arithmetic': is_arithmetic,
            'geometric': is_geometric,
            'fibonacci_like': _is_fibonacci_like(number_str)
        }

    # Complexity score calculation
    base_complexity = len(number_str) / 10  # Length factor
    pattern_complexity = len(repetitions) / 5  # Pattern diversity factor
    entropy_factor = len(set(number_str)) / 10  # Uniqueness factor

    complexity_metrics['complexity_score'] = base_complexity + pattern_complexity + entropy_factor

    return complexity_metrics


def _analyze_symmetry_patterns(number_str: str) -> Dict[str, Any]:
    """Analyze symmetry patterns in a number string."""
    symmetry_metrics = {}

    # Mirror symmetry
    reversed_str = number_str[::-1]
    symmetry_metrics['mirror_symmetry'] = number_str == reversed_str

    # Partial symmetries
    half_len = len(number_str) // 2
    first_half = number_str[:half_len]
    second_half = number_str[-half_len:]
    symmetry_metrics['partial_symmetry'] = first_half == second_half[::-1]

    # Central symmetry
    if len(number_str) % 2 == 1:
        center = len(number_str) // 2
        left_part = number_str[:center]
        right_part = number_str[center+1:]
        symmetry_metrics['central_symmetry'] = left_part == right_part[::-1]

    # Rotational symmetry (for even-length strings)
    if len(number_str) % 2 == 0:
        mid = len(number_str) // 2
        first_half = number_str[:mid]
        second_half = number_str[mid:]
        symmetry_metrics['rotational_symmetry'] = first_half == second_half[::-1]

    # Symmetry score
    symmetry_count = sum(symmetry_metrics.values())
    symmetry_metrics['symmetry_score'] = symmetry_count / len(symmetry_metrics)

    return symmetry_metrics


def _analyze_fractal_properties(number_str: str) -> Dict[str, Any]:
    """Analyze fractal-like properties in number patterns."""
    fractal_metrics = {}

    # Self-similarity analysis
    self_similarity_scores = []
    for scale in range(2, min(len(number_str) // 2, 10)):
        if len(number_str) % scale == 0:
            segments = [number_str[i:i+scale] for i in range(0, len(number_str), scale)]
            if len(segments) > 1:
                # Check if all segments are similar
                similarity = sum(1 for i in range(1, len(segments))
                               if segments[i] == segments[0]) / len(segments)
                self_similarity_scores.append(similarity)

    fractal_metrics['self_similarity_scores'] = self_similarity_scores
    fractal_metrics['average_self_similarity'] = (
        statistics.mean(self_similarity_scores) if self_similarity_scores else 0
    )

    # Approximate fractal dimension using box-counting method
    if len(number_str) >= 8:
        fractal_metrics['fractal_dimension'] = _estimate_fractal_dimension(number_str)

    return fractal_metrics


def _analyze_cross_domain_patterns(number: Union[int, str, SymergeticsNumber]) -> Dict[str, Any]:
    """Analyze patterns that appear across different mathematical domains."""
    cross_domain_patterns = {}

    if isinstance(number, SymergeticsNumber):
        numeric_value = float(number.value)
    else:
        numeric_value = float(number)

    # Check for connections to prime numbers
    if isinstance(number, int) or (isinstance(number, str) and number.isdigit()):
        num_int = int(number) if isinstance(number, str) else number
        cross_domain_patterns['prime_related'] = _analyze_prime_relationships(num_int)

    # Check for geometric relationships
    cross_domain_patterns['geometric_relationships'] = _analyze_geometric_relationships(numeric_value)

    # Check for physical/mathematical constants relationships
    cross_domain_patterns['constant_relationships'] = _analyze_constant_relationships(numeric_value)

    return cross_domain_patterns


def _is_arithmetic_sequence(number_str: str) -> bool:
    """Check if digits form an arithmetic sequence."""
    if not all(c.isdigit() for c in number_str):
        return False

    digits = [int(c) for c in number_str]
    if len(digits) < 3:
        return False

    # Check for common difference
    diff = digits[1] - digits[0]
    return all(digits[i] - digits[i-1] == diff for i in range(2, len(digits)))


def _is_geometric_sequence(number_str: str) -> bool:
    """Check if digits form a geometric sequence."""
    if not all(c.isdigit() for c in number_str) or '0' in number_str:
        return False

    digits = [int(c) for c in number_str]
    if len(digits) < 3:
        return False

    # Check for common ratio
    ratio = digits[1] / digits[0]
    return all(abs(digits[i] / digits[i-1] - ratio) < 0.01 for i in range(2, len(digits)))


def _is_fibonacci_like(number_str: str) -> bool:
    """Check if digits resemble Fibonacci sequence pattern."""
    if not all(c.isdigit() for c in number_str):
        return False

    digits = [int(c) for c in number_str]
    if len(digits) < 3:
        return False

    # Check Fibonacci-like pattern
    for i in range(2, len(digits)):
        if digits[i] != (digits[i-1] + digits[i-2]):
            return False
    return True


def _estimate_fractal_dimension(number_str: str) -> float:
    """Estimate fractal dimension using simple box-counting method."""
    # Simplified fractal dimension estimation
    n = len(number_str)
    if n < 8:
        return 1.0

    # Count unique substrings of different lengths
    dimensions = []
    for size in range(1, min(6, n//2)):
        substrings = set()
        for i in range(n - size + 1):
            substrings.add(number_str[i:i+size])
        if substrings:
            dimensions.append(len(substrings))

    if len(dimensions) >= 2:
        # Simple estimation based on substring diversity
        diversity_ratio = dimensions[-1] / dimensions[0] if dimensions[0] > 0 else 1
        return 1 + math.log(diversity_ratio) / math.log(len(dimensions))

    return 1.0


def _prime_factors(number: int) -> List[int]:
    """Find prime factors of a number."""
    if number <= 1:
        return []
    
    factors = []
    d = 2
    while d * d <= number:
        while number % d == 0:
            factors.append(d)
            number //= d
        d += 1
    if number > 1:
        factors.append(number)
    return factors


def _analyze_prime_relationships(number: int) -> Dict[str, Any]:
    """Analyze relationships between number and prime numbers."""
    prime_relationships = {}

    if number <= 1:
        return prime_relationships

    # Check if number itself is prime
    prime_relationships['is_prime'] = is_prime(number)

    # Find prime factors
    prime_relationships['prime_factors'] = _prime_factors(number)

    # Check for primorial relationships
    primorial_relationships = []
    for n in range(2, 20):
        try:
            p = primorial(n)
            if number % int(p.value) == 0:
                primorial_relationships.append(f"divisible by {n}#")
        except:
            break
    prime_relationships['primorial_relationships'] = primorial_relationships

    return prime_relationships


def _analyze_geometric_relationships(value: float) -> Dict[str, Any]:
    """Analyze geometric relationships of the number."""
    geometric_relationships = {}

    # Check for relationships to π
    pi_ratio = value / math.pi
    geometric_relationships['pi_ratio'] = pi_ratio
    geometric_relationships['pi_relationship'] = abs(pi_ratio - round(pi_ratio)) < 0.01

    # Check for relationships to e
    e_ratio = value / math.e
    geometric_relationships['e_ratio'] = e_ratio
    geometric_relationships['e_relationship'] = abs(e_ratio - round(e_ratio)) < 0.01

    # Check for relationships to φ (golden ratio)
    phi = (1 + math.sqrt(5)) / 2
    phi_ratio = value / phi
    geometric_relationships['phi_ratio'] = phi_ratio
    geometric_relationships['phi_relationship'] = abs(phi_ratio - round(phi_ratio)) < 0.01

    return geometric_relationships


def _analyze_constant_relationships(value: float) -> Dict[str, Any]:
    """Analyze relationships to mathematical constants."""
    constant_relationships = {}

    constants = {
        'π': math.pi,
        'e': math.e,
        'φ': (1 + math.sqrt(5)) / 2,
        '√2': math.sqrt(2),
        '√3': math.sqrt(3),
        'γ': 0.5772156649015329,  # Euler-Mascheroni constant
    }

    relationships = {}
    for name, const_value in constants.items():
        ratio = value / const_value
        closeness = abs(ratio - round(ratio))
        if closeness < 0.001:  # Very close relationship
            relationships[name] = {
                'ratio': ratio,
                'rounded_ratio': round(ratio),
                'closeness': closeness
            }

    constant_relationships['relationships'] = relationships
    constant_relationships['strongest_relationship'] = (
        min(relationships.items(), key=lambda x: x[1]['closeness'])[0]
        if relationships else None
    )

    return constant_relationships


def compare_mathematical_domains(
    domain1_data: Dict[str, Any],
    domain2_data: Dict[str, Any],
    domain1_name: str = "Domain1",
    domain2_name: str = "Domain2"
) -> ComparativeAnalysis:
    """
    Compare patterns between two mathematical domains.

    Args:
        domain1_data: Analysis data from first domain
        domain2_data: Analysis data from second domain
        domain1_name: Name of first domain
        domain2_name: Name of second domain

    Returns:
        ComparativeAnalysis object with comparison results
    """
    # Calculate correlation coefficient based on aggregated metrics
    correlations = []

    # Aggregate metrics across all items in each domain
    domain1_aggregated = _aggregate_domain_metrics(domain1_data)
    domain2_aggregated = _aggregate_domain_metrics(domain2_data)

    # Compare complexity scores
    comp1 = domain1_aggregated.get('avg_complexity', 0)
    comp2 = domain2_aggregated.get('avg_complexity', 0)
    if comp1 > 0 and comp2 > 0:
        correlations.append(abs(comp1 - comp2) < 0.5)  # Similar complexity

    # Compare symmetry scores
    sym1 = domain1_aggregated.get('avg_symmetry', 0)
    sym2 = domain2_aggregated.get('avg_symmetry', 0)
    if sym1 > 0 and sym2 > 0:
        correlations.append(abs(sym1 - sym2) < 0.2)  # Similar symmetry

    # Compare palindromic ratios
    pal_ratio1 = domain1_aggregated.get('palindromic_ratio', 0)
    pal_ratio2 = domain2_aggregated.get('palindromic_ratio', 0)
    correlations.append(abs(pal_ratio1 - pal_ratio2) < 0.3)  # Similar palindromic patterns

    # For empty domains, return 0 correlation since there's no meaningful comparison
    if not domain1_data and not domain2_data:
        correlation_coefficient = 0
    else:
        correlation_coefficient = sum(correlations) / len(correlations) if correlations else 0

    # Calculate similarity score
    similarity_score = _calculate_domain_similarity(domain1_aggregated, domain2_aggregated)

    # Calculate domain overlap
    domain_overlap = _calculate_domain_overlap(domain1_aggregated, domain2_aggregated)

    # Identify cross-domain patterns
    cross_domain_patterns = _identify_cross_domain_patterns(domain1_aggregated, domain2_aggregated)

    # Calculate structural similarity
    structural_similarity = _calculate_structural_similarity(domain1_aggregated, domain2_aggregated)

    return ComparativeAnalysis(
        correlation_coefficient=correlation_coefficient,
        similarity_score=similarity_score,
        domain_overlap=domain_overlap,
        structural_similarity=structural_similarity,
        cross_domain_patterns=cross_domain_patterns
    )


def _aggregate_domain_metrics(domain_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all items in a domain."""
    if not domain_data:
        return {}

    aggregated = {
        'total_items': len(domain_data),
        'palindromic_count': 0,
        'total_complexity': 0,
        'total_symmetry': 0,
        'total_length': 0,
        'complexity_count': 0,
        'symmetry_count': 0,
        'length_count': 0,
        'all_digit_distributions': []
    }

    for item in domain_data:
        # Count palindromic numbers
        if item.get('is_palindromic', False):
            aggregated['palindromic_count'] += 1

        # Aggregate complexity scores
        if 'pattern_complexity' in item:
            comp = item['pattern_complexity'].get('complexity_score', 0)
            if comp > 0:
                aggregated['total_complexity'] += comp
                aggregated['complexity_count'] += 1

        # Aggregate symmetry scores
        if 'symmetry_analysis' in item:
            sym = item['symmetry_analysis'].get('symmetry_score', 0)
            if sym >= 0:
                aggregated['total_symmetry'] += sym
                aggregated['symmetry_count'] += 1

        # Aggregate lengths
        if 'length' in item:
            aggregated['total_length'] += item['length']
            aggregated['length_count'] += 1

        # Collect digit distributions
        if 'digit_distribution' in item:
            aggregated['all_digit_distributions'].append(item['digit_distribution'])

    # Calculate averages
    aggregated['avg_complexity'] = (
        aggregated['total_complexity'] / aggregated['complexity_count']
        if aggregated['complexity_count'] > 0 else 0
    )
    aggregated['avg_symmetry'] = (
        aggregated['total_symmetry'] / aggregated['symmetry_count']
        if aggregated['symmetry_count'] > 0 else 0
    )
    aggregated['avg_length'] = (
        aggregated['total_length'] / aggregated['length_count']
        if aggregated['length_count'] > 0 else 0
    )
    aggregated['palindromic_ratio'] = (
        aggregated['palindromic_count'] / aggregated['total_items']
        if aggregated['total_items'] > 0 else 0
    )

    return aggregated


def _calculate_domain_similarity(data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
    """Calculate overall similarity between two domains."""
    similarity_factors = []

    # Average length similarity
    len1 = data1.get('avg_length', 0)
    len2 = data2.get('avg_length', 0)
    if len1 > 0 and len2 > 0:
        length_similarity = 1 - abs(len1 - len2) / max(len1, len2)
        similarity_factors.append(length_similarity)

    # Palindromic ratio similarity
    pal_ratio1 = data1.get('palindromic_ratio', 0)
    pal_ratio2 = data2.get('palindromic_ratio', 0)
    palindromic_similarity = 1 - abs(pal_ratio1 - pal_ratio2)
    similarity_factors.append(palindromic_similarity)

    # Complexity similarity
    comp1 = data1.get('avg_complexity', 0)
    comp2 = data2.get('avg_complexity', 0)
    if comp1 > 0 and comp2 > 0:
        complexity_similarity = 1 - abs(comp1 - comp2) / max(comp1, comp2)
        similarity_factors.append(complexity_similarity)

    # Symmetry similarity
    sym1 = data1.get('avg_symmetry', 0)
    sym2 = data2.get('avg_symmetry', 0)
    if sym1 > 0 and sym2 > 0:
        symmetry_similarity = 1 - abs(sym1 - sym2)
        similarity_factors.append(symmetry_similarity)

    return statistics.mean(similarity_factors) if similarity_factors else 0.0


def _calculate_domain_overlap(data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
    """Calculate the overlap between two domains."""
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    intersection = keys1 & keys2
    union = keys1 | keys2

    return len(intersection) / len(union) if union else 0.0


def _identify_cross_domain_patterns(data1: Dict[str, Any], data2: Dict[str, Any]) -> List[str]:
    """Identify patterns that appear in both domains."""
    patterns = []

    # Check for shared digit distributions
    dist1 = data1.get('digit_distribution', {})
    dist2 = data2.get('digit_distribution', {})
    common_digits = set(dist1.keys()) & set(dist2.keys())
    if common_digits:
        patterns.append(f"Shared digit patterns: {sorted(common_digits)}")

    # Check for similar complexity scores
    comp1 = data1.get('pattern_complexity', {}).get('complexity_score', 0)
    comp2 = data2.get('pattern_complexity', {}).get('complexity_score', 0)
    if abs(comp1 - comp2) < 0.1:
        patterns.append(f"Similar complexity: {comp1:.2f} vs {comp2:.2f}")

    # Check for similar symmetry scores
    sym1 = data1.get('symmetry_analysis', {}).get('symmetry_score', 0)
    sym2 = data2.get('symmetry_analysis', {}).get('symmetry_score', 0)
    if abs(sym1 - sym2) < 0.1:
        patterns.append(f"Similar symmetry: {sym1:.2f} vs {sym2:.2f}")

    return patterns


def _calculate_structural_similarity(data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
    """Calculate structural similarity between domains."""
    # Compare the structure of the analysis data
    structure1 = _extract_structure(data1)
    structure2 = _extract_structure(data2)

    common_structure = structure1 & structure2
    total_structure = structure1 | structure2

    return len(common_structure) / len(total_structure) if total_structure else 0.0


def _extract_structure(data: Dict[str, Any]) -> set:
    """Extract structural information from analysis data."""
    structure = set()

    for key, value in data.items():
        if isinstance(value, dict):
            structure.add(f"{key}_dict")
            for subkey in value.keys():
                structure.add(f"{key}.{subkey}")
        elif isinstance(value, list):
            structure.add(f"{key}_list")
        elif isinstance(value, (int, float)):
            structure.add(f"{key}_numeric")
        elif isinstance(value, str):
            structure.add(f"{key}_string")
        elif isinstance(value, bool):
            structure.add(f"{key}_boolean")

    return structure


def generate_comprehensive_report(
    analysis_data: Dict[str, Any],
    title: str = "Mathematical Analysis Report",
    include_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive analysis report.

    Args:
        analysis_data: Results from analyze_mathematical_patterns
        title: Report title
        include_visualizations: Whether to include visualization recommendations

    Returns:
        Dict containing formatted report data
    """
    # Handle None or invalid input
    if analysis_data is None:
        analysis_data = {}
    
    report = {
        'title': title,
        'timestamp': '2024-01-01T00:00:00Z',  # Would use datetime in real implementation
        'summary': {},
        'detailed_analysis': {},
        'insights': [],
        'recommendations': []
    }

    # Generate summary statistics
    report['summary'] = {
        'total_numbers_analyzed': 1,
        'palindromic_numbers': 1 if analysis_data.get('is_palindromic', False) else 0,
        'average_palindromic_density': analysis_data.get('palindromic_density', 0),
        'average_complexity': analysis_data.get('pattern_complexity', {}).get('complexity_score', 0),
        'average_symmetry': analysis_data.get('symmetry_analysis', {}).get('symmetry_score', 0)
    }

    # Detailed analysis sections
    report['detailed_analysis'] = {
        'basic_properties': {
            'number': analysis_data.get('number', ''),
            'length': analysis_data.get('length', 0),
            'is_palindromic': analysis_data.get('is_palindromic', False),
            'palindromic_density': analysis_data.get('palindromic_density', 0)
        },
        'pattern_analysis': analysis_data.get('pattern_complexity', {}),
        'symmetry_analysis': analysis_data.get('symmetry_analysis', {}),
        'statistical_properties': analysis_data.get('digit_statistics', {})
    }

    # Generate insights
    insights = []

    if analysis_data.get('is_palindromic', False):
        insights.append("Number exhibits perfect palindromic symmetry")

    density = analysis_data.get('palindromic_density', 0)
    if density > 0.5:
        insights.append(f"High palindromic density ({density:.1%}) indicates strong pattern structure")
    elif density < 0.2:
        insights.append(f"Low palindromic density ({density:.1%}) suggests more random distribution")

    complexity = analysis_data.get('pattern_complexity', {}).get('complexity_score', 0)
    if complexity > 2:
        insights.append(f"High complexity score ({complexity:.2f}) indicates intricate pattern structure")
    elif complexity < 1:
        insights.append(f"Low complexity score ({complexity:.2f}) suggests simple pattern structure")

    symmetry = analysis_data.get('symmetry_analysis', {}).get('symmetry_score', 0)
    if symmetry > 0.7:
        insights.append(f"High symmetry score ({symmetry:.2f}) indicates strong structural regularity")
    elif symmetry < 0.3:
        insights.append(f"Low symmetry score ({symmetry:.2f}) suggests asymmetric structure")

    report['insights'] = insights

    # Generate recommendations
    recommendations = []

    if include_visualizations:
        recommendations.append("Generate palindromic pattern visualization")
        recommendations.append("Create symmetry analysis heatmap")
        recommendations.append("Produce complexity metric time series")

        if analysis_data.get('is_palindromic', False):
            recommendations.append("Compare with other palindromic numbers in the same domain")

        if complexity > 2:
            recommendations.append("Investigate fractal properties and self-similarity")

        if symmetry > 0.7:
            recommendations.append("Analyze group theoretical properties")

    report['recommendations'] = recommendations

    return report
