#!/usr/bin/env python3
"""
Tests for advanced analysis methods in symergetics.computation.analysis
"""

import pytest
import math
from symergetics.computation.analysis import (
    analyze_mathematical_patterns,
    compare_mathematical_domains,
    generate_comprehensive_report,
    PatternMetrics,
    ComparativeAnalysis
)
from symergetics.core.numbers import SymergeticsNumber


class TestMathematicalPatternAnalysis:
    """Test mathematical pattern analysis functions."""

    def test_analyze_palindromic_number(self):
        """Test analysis of a palindromic number."""
        result = analyze_mathematical_patterns(121)

        assert result['number'] == '121'
        assert result['is_palindromic'] == True
        assert result['length'] == 3
        assert 'palindromic_density' in result
        assert 'pattern_complexity' in result
        assert 'symmetry_analysis' in result

    def test_analyze_non_palindromic_number(self):
        """Test analysis of a non-palindromic number."""
        result = analyze_mathematical_patterns(123)

        assert result['number'] == '123'
        assert result['is_palindromic'] == False
        assert result['length'] == 3

    def test_analyze_symergetics_number(self):
        """Test analysis of a SymergeticsNumber."""
        num = SymergeticsNumber(22, 7)
        result = analyze_mathematical_patterns(num)

        assert '22/7' in result['number'] or '22/7' in str(result['number'])
        assert 'pattern_complexity' in result

    def test_analyze_with_different_depths(self):
        """Test analysis with different depth levels."""
        # Depth 1
        result1 = analyze_mathematical_patterns(121, analysis_depth=1)
        assert 'pattern_complexity' not in result1

        # Depth 3
        result3 = analyze_mathematical_patterns(121, analysis_depth=3)
        assert 'pattern_complexity' in result3
        assert 'symmetry_analysis' in result3

    def test_analyze_large_number(self):
        """Test analysis of a large number."""
        large_num = 12345678901234567890
        result = analyze_mathematical_patterns(large_num)

        assert result['length'] > 10
        assert 'digit_statistics' in result


class TestDomainComparison:
    """Test domain comparison functions."""

    def test_compare_identical_domains(self):
        """Test comparison of identical domains."""
        domain1 = [
            analyze_mathematical_patterns(121),
            analyze_mathematical_patterns(12321)
        ]
        domain2 = [
            analyze_mathematical_patterns(121),
            analyze_mathematical_patterns(12321)
        ]

        comparison = compare_mathematical_domains(domain1, domain2, "Test1", "Test2")

        assert isinstance(comparison, ComparativeAnalysis)
        assert comparison.correlation_coefficient >= 0
        assert 'similarity_score' in comparison.__dict__

    def test_compare_different_domains(self):
        """Test comparison of different domains."""
        domain1 = [analyze_mathematical_patterns(121)]  # Palindromic
        domain2 = [analyze_mathematical_patterns(123)]  # Non-palindromic

        comparison = compare_mathematical_domains(domain1, domain2, "Palindromic", "Non-palindromic")

        assert isinstance(comparison, ComparativeAnalysis)
        assert len(comparison.cross_domain_patterns) >= 0

    def test_compare_empty_domains(self):
        """Test comparison with empty domains."""
        comparison = compare_mathematical_domains([], [], "Empty1", "Empty2")

        assert isinstance(comparison, ComparativeAnalysis)
        assert comparison.correlation_coefficient == 0


class TestComprehensiveReporting:
    """Test comprehensive reporting functions."""

    def test_generate_report_basic(self):
        """Test basic report generation."""
        analysis_data = analyze_mathematical_patterns(121)
        report = generate_comprehensive_report(analysis_data)

        assert 'title' in report
        assert 'timestamp' in report
        assert 'summary' in report
        assert 'detailed_analysis' in report
        assert 'insights' in report
        assert 'recommendations' in report

    def test_generate_report_with_visualizations(self):
        """Test report generation with visualization recommendations."""
        analysis_data = analyze_mathematical_patterns(121)
        report = generate_comprehensive_report(analysis_data, include_visualizations=True)

        assert 'recommendations' in report
        assert len(report['recommendations']) > 0

    def test_generate_report_palindromic(self):
        """Test report generation for palindromic number."""
        analysis_data = analyze_mathematical_patterns(12321)
        report = generate_comprehensive_report(analysis_data)

        insights_str = ' '.join(report['insights'])
        assert 'palindromic' in insights_str.lower()

    def test_generate_report_custom_title(self):
        """Test report generation with custom title."""
        analysis_data = analyze_mathematical_patterns(121)
        custom_title = "Custom Analysis Report"
        report = generate_comprehensive_report(analysis_data, title=custom_title)

        assert report['title'] == custom_title


class TestPatternMetrics:
    """Test PatternMetrics dataclass."""

    def test_pattern_metrics_creation(self):
        """Test PatternMetrics dataclass creation."""
        metrics = PatternMetrics(
            complexity_score=2.5,
            symmetry_index=0.8,
            density_score=0.6,
            entropy_measure=3.2,
            fractal_dimension=1.8
        )

        assert metrics.complexity_score == 2.5
        assert metrics.symmetry_index == 0.8
        assert metrics.density_score == 0.6
        assert metrics.entropy_measure == 3.2
        assert metrics.fractal_dimension == 1.8
        assert metrics.self_similarity_score == 0.0

    def test_pattern_metrics_defaults(self):
        """Test PatternMetrics with default values."""
        metrics = PatternMetrics(
            complexity_score=1.0,
            symmetry_index=0.5,
            density_score=0.3,
            entropy_measure=2.1
        )

        assert metrics.fractal_dimension is None
        assert metrics.self_similarity_score == 0.0


class TestComparativeAnalysis:
    """Test ComparativeAnalysis dataclass."""

    def test_comparative_analysis_creation(self):
        """Test ComparativeAnalysis dataclass creation."""
        analysis = ComparativeAnalysis(
            correlation_coefficient=0.85,
            similarity_score=0.72,
            domain_overlap=0.6,
            structural_similarity=0.88,
            cross_domain_patterns=['Shared digit patterns', 'Similar complexity']
        )

        assert analysis.correlation_coefficient == 0.85
        assert analysis.similarity_score == 0.72
        assert analysis.domain_overlap == 0.6
        assert analysis.structural_similarity == 0.88
        assert len(analysis.cross_domain_patterns) == 2

    def test_comparative_analysis_empty_patterns(self):
        """Test ComparativeAnalysis with empty cross-domain patterns."""
        analysis = ComparativeAnalysis(
            correlation_coefficient=0.5,
            similarity_score=0.4,
            domain_overlap=0.3,
            structural_similarity=0.5,
            cross_domain_patterns=[]
        )

        assert len(analysis.cross_domain_patterns) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_analyze_empty_string(self):
        """Test analysis of empty string."""
        result = analyze_mathematical_patterns("")

        assert result['length'] == 0
        assert result['is_palindromic'] == True  # Empty string is technically palindromic

    def test_analyze_single_digit(self):
        """Test analysis of single digit."""
        result = analyze_mathematical_patterns(5)

        assert result['length'] == 1
        assert result['is_palindromic'] == True

    def test_analyze_zero(self):
        """Test analysis of zero."""
        result = analyze_mathematical_patterns(0)

        assert result['number'] == '0'
        assert result['length'] == 1

    def test_compare_single_item_domains(self):
        """Test comparison with single items in domains."""
        domain1 = [analyze_mathematical_patterns(121)]
        domain2 = [analyze_mathematical_patterns(123)]

        comparison = compare_mathematical_domains(domain1, domain2)
        assert isinstance(comparison, ComparativeAnalysis)

    def test_generate_report_minimal_data(self):
        """Test report generation with minimal data."""
        minimal_data = {
            'number': '123',
            'length': 3,
            'is_palindromic': False,
            'palindromic_density': 0.0
        }

        report = generate_comprehensive_report(minimal_data)
        assert 'title' in report
        assert 'summary' in report


class TestIntegrationScenarios:
    """Test integration with other Symergetics modules."""

    def test_analyze_with_symergetics_constants(self):
        """Test analysis integrated with Symergetics constants."""
        from symergetics.core.constants import SymergeticsConstants

        constants = SymergeticsConstants()
        volume = constants.VOLUME_RATIOS['tetrahedron']

        result = analyze_mathematical_patterns(volume)
        assert 'pattern_complexity' in result

    def test_cross_module_pattern_analysis(self):
        """Test pattern analysis across different modules."""
        # Create data from different sources
        palindromic_data = analyze_mathematical_patterns(121)
        non_palindromic_data = analyze_mathematical_patterns(123)
        scheherazade_data = analyze_mathematical_patterns(SymergeticsNumber(1001))

        # Compare them
        comparison = compare_mathematical_domains(
            [palindromic_data],
            [non_palindromic_data, scheherazade_data],
            "Palindromic",
            "Mixed"
        )

        assert isinstance(comparison, ComparativeAnalysis)
        assert 'correlation_coefficient' in comparison.__dict__

    def test_large_scale_analysis(self):
        """Test analysis with larger dataset."""
        # Generate a series of numbers to analyze
        test_numbers = [121, 12321, 123454321, 12345678987654321][:3]  # Limit for speed
        analysis_results = [analyze_mathematical_patterns(num) for num in test_numbers]

        # Generate comparative report
        report = generate_comprehensive_report(analysis_results[0], "Large Scale Analysis")

        assert 'insights' in report
        assert len(report['insights']) >= 0
