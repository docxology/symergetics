#!/usr/bin/env python3
"""
Comprehensive tests for symergetics.computation.analysis module.

This module provides extensive test coverage for analysis functions
to achieve 90%+ test coverage, focusing on edge cases, error handling, and
comprehensive functionality testing.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import json
from typing import List, Dict, Any

# Test data and fixtures
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestAnalyzeMathematicalPatternsComprehensive:
    """Test comprehensive mathematical pattern analysis functionality."""

    def test_analyze_mathematical_patterns_basic(self):
        """Test basic mathematical pattern analysis."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12321
        result = analyze_mathematical_patterns(number)
        
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result
        assert 'is_palindromic' in result
        assert 'palindromic_density' in result

    def test_analyze_mathematical_patterns_empty_string(self):
        """Test mathematical pattern analysis with empty string."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = ""
        result = analyze_mathematical_patterns(number)
        
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result
        assert result['is_palindromic'] == True  # Empty string is palindromic
        assert result['palindromic_density'] == 0.0

    def test_analyze_mathematical_patterns_single_digit(self):
        """Test mathematical pattern analysis with single digit."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 5
        result = analyze_mathematical_patterns(number)
        
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result
        assert result['is_palindromic'] == True  # Single digit is palindromic

    def test_analyze_mathematical_patterns_negative_number(self):
        """Test mathematical pattern analysis with negative number."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = -12321
        result = analyze_mathematical_patterns(number)
        
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result

    def test_analyze_mathematical_patterns_float_number(self):
        """Test mathematical pattern analysis with float number."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 123.456
        result = analyze_mathematical_patterns(number)
        
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result

    def test_analyze_mathematical_patterns_symergetics_number(self):
        """Test mathematical pattern analysis with SymergeticsNumber."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        from symergetics.core.numbers import SymergeticsNumber
        
        number = SymergeticsNumber(12321)
        result = analyze_mathematical_patterns(number)
        
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result

    def test_analyze_mathematical_patterns_large_number(self):
        """Test mathematical pattern analysis with large number."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12345678987654321
        result = analyze_mathematical_patterns(number)
        
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result

    def test_analyze_mathematical_patterns_analysis_depth_1(self):
        """Test mathematical pattern analysis with depth 1."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12321
        result = analyze_mathematical_patterns(number, analysis_depth=1)
        
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result
        assert 'is_palindromic' in result
        assert 'palindromic_density' in result

    def test_analyze_mathematical_patterns_analysis_depth_2(self):
        """Test mathematical pattern analysis with depth 2."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12321
        result = analyze_mathematical_patterns(number, analysis_depth=2)
        
        assert 'number' in result
        assert 'pattern_complexity' in result
        assert 'complexity_score' in result['pattern_complexity']

    def test_analyze_mathematical_patterns_analysis_depth_3(self):
        """Test mathematical pattern analysis with depth 3."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12321
        result = analyze_mathematical_patterns(number, analysis_depth=3)
        
        assert 'number' in result
        assert 'symmetry_analysis' in result
        assert 'symmetry_score' in result['symmetry_analysis']

    def test_analyze_mathematical_patterns_analysis_depth_4(self):
        """Test mathematical pattern analysis with depth 4."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12321
        result = analyze_mathematical_patterns(number, analysis_depth=4)
        
        assert 'number' in result
        assert 'fractal_analysis' in result

    def test_analyze_mathematical_patterns_analysis_depth_5(self):
        """Test mathematical pattern analysis with depth 5."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12321
        result = analyze_mathematical_patterns(number, analysis_depth=5)
        
        assert 'number' in result
        assert 'cross_domain_correlations' in result

    def test_analyze_mathematical_patterns_invalid_input(self):
        """Test mathematical pattern analysis with invalid input."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        # Test with None input
        result = analyze_mathematical_patterns(None)
        assert 'number' in result
        assert result['numeric_value'] == 0.0

    def test_analyze_mathematical_patterns_non_numeric_string(self):
        """Test mathematical pattern analysis with non-numeric string."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = "abc"
        result = analyze_mathematical_patterns(number)
        
        assert 'number' in result
        assert result['numeric_value'] == 0.0
        assert result['string_representation'] == "abc"


class TestCompareMathematicalDomainsComprehensive:
    """Test comprehensive mathematical domain comparison functionality."""

    def test_compare_mathematical_domains_basic(self):
        """Test basic mathematical domain comparison."""
        from symergetics.computation.analysis import compare_mathematical_domains
        
        domain1_data = [{'is_palindromic': True, 'length': 5}]
        domain2_data = [{'is_palindromic': True, 'length': 5}]
        
        result = compare_mathematical_domains(domain1_data, domain2_data)
        
        assert hasattr(result, 'correlation_coefficient')
        assert hasattr(result, 'similarity_score')
        assert hasattr(result, 'domain_overlap')
        assert hasattr(result, 'structural_similarity')
        assert hasattr(result, 'cross_domain_patterns')

    def test_compare_mathematical_domains_empty_domains(self):
        """Test mathematical domain comparison with empty domains."""
        from symergetics.computation.analysis import compare_mathematical_domains
        
        domain1_data = []
        domain2_data = []
        
        result = compare_mathematical_domains(domain1_data, domain2_data)
        
        assert hasattr(result, 'correlation_coefficient')
        assert hasattr(result, 'similarity_score')
        assert hasattr(result, 'domain_overlap')
        assert hasattr(result, 'structural_similarity')
        assert hasattr(result, 'cross_domain_patterns')

    def test_compare_mathematical_domains_different_sizes(self):
        """Test mathematical domain comparison with different domain sizes."""
        from symergetics.computation.analysis import compare_mathematical_domains
        
        domain1_data = [{'is_palindromic': True, 'length': 5}]
        domain2_data = [
            {'is_palindromic': True, 'length': 5},
            {'is_palindromic': False, 'length': 3},
            {'is_palindromic': True, 'length': 7}
        ]
        
        result = compare_mathematical_domains(domain1_data, domain2_data)
        
        assert hasattr(result, 'correlation_coefficient')
        assert hasattr(result, 'similarity_score')
        assert hasattr(result, 'domain_overlap')
        assert hasattr(result, 'structural_similarity')
        assert hasattr(result, 'cross_domain_patterns')

    def test_compare_mathematical_domains_custom_names(self):
        """Test mathematical domain comparison with custom names."""
        from symergetics.computation.analysis import compare_mathematical_domains
        
        domain1_data = [{'is_palindromic': True, 'length': 5}]
        domain2_data = [{'is_palindromic': True, 'length': 5}]
        
        result = compare_mathematical_domains(
            domain1_data, 
            domain2_data,
            domain1_name="Palindromes",
            domain2_name="Symmetric Numbers"
        )
        
        assert hasattr(result, 'correlation_coefficient')
        assert hasattr(result, 'similarity_score')
        assert hasattr(result, 'domain_overlap')
        assert hasattr(result, 'structural_similarity')
        assert hasattr(result, 'cross_domain_patterns')

    def test_compare_mathematical_domains_complex_data(self):
        """Test mathematical domain comparison with complex data."""
        from symergetics.computation.analysis import compare_mathematical_domains
        
        domain1_data = [
            {
                'is_palindromic': True,
                'length': 5,
                'pattern_complexity': {'complexity_score': 2.5},
                'symmetry_analysis': {'symmetry_score': 0.8},
                'digit_distribution': {'1': 2, '2': 1, '3': 1, '4': 1}
            }
        ]
        domain2_data = [
            {
                'is_palindromic': False,
                'length': 4,
                'pattern_complexity': {'complexity_score': 1.8},
                'symmetry_analysis': {'symmetry_score': 0.3},
                'digit_distribution': {'1': 1, '2': 1, '3': 1, '4': 1}
            }
        ]
        
        result = compare_mathematical_domains(domain1_data, domain2_data)
        
        assert hasattr(result, 'correlation_coefficient')
        assert hasattr(result, 'similarity_score')
        assert hasattr(result, 'domain_overlap')
        assert hasattr(result, 'structural_similarity')
        assert hasattr(result, 'cross_domain_patterns')


class TestGenerateComprehensiveReportComprehensive:
    """Test comprehensive report generation functionality."""

    def test_generate_comprehensive_report_basic(self):
        """Test basic comprehensive report generation."""
        from symergetics.computation.analysis import generate_comprehensive_report
        
        analysis_data = {
            'number': '12321',
            'length': 5,
            'is_palindromic': True,
            'palindromic_density': 1.0,
            'pattern_complexity': {'complexity_score': 2.5},
            'symmetry_analysis': {'symmetry_score': 0.8}
        }
        
        result = generate_comprehensive_report(analysis_data)
        
        assert 'title' in result
        assert 'timestamp' in result
        assert 'summary' in result
        assert 'detailed_analysis' in result
        assert 'insights' in result
        assert 'recommendations' in result

    def test_generate_comprehensive_report_custom_title(self):
        """Test comprehensive report generation with custom title."""
        from symergetics.computation.analysis import generate_comprehensive_report
        
        analysis_data = {
            'number': '12321',
            'length': 5,
            'is_palindromic': True,
            'palindromic_density': 1.0
        }
        
        result = generate_comprehensive_report(
            analysis_data,
            title="Custom Analysis Report"
        )
        
        assert result['title'] == "Custom Analysis Report"
        assert 'summary' in result
        assert 'detailed_analysis' in result
        assert 'insights' in result
        assert 'recommendations' in result

    def test_generate_comprehensive_report_without_visualizations(self):
        """Test comprehensive report generation without visualizations."""
        from symergetics.computation.analysis import generate_comprehensive_report
        
        analysis_data = {
            'number': '12321',
            'length': 5,
            'is_palindromic': True,
            'palindromic_density': 1.0
        }
        
        result = generate_comprehensive_report(
            analysis_data,
            include_visualizations=False
        )
        
        assert 'title' in result
        assert 'summary' in result
        assert 'detailed_analysis' in result
        assert 'insights' in result
        assert 'recommendations' in result

    def test_generate_comprehensive_report_complex_data(self):
        """Test comprehensive report generation with complex data."""
        from symergetics.computation.analysis import generate_comprehensive_report
        
        analysis_data = {
            'number': '12345678987654321',
            'length': 17,
            'is_palindromic': True,
            'palindromic_density': 1.0,
            'pattern_complexity': {'complexity_score': 3.5},
            'symmetry_analysis': {'symmetry_score': 0.9},
            'digit_statistics': {
                'mean': 5.0,
                'median': 5.0,
                'stdev': 2.5,
                'variance': 6.25,
                'range': 8,
                'mode': 1
            }
        }
        
        result = generate_comprehensive_report(analysis_data)
        
        assert 'title' in result
        assert 'summary' in result
        assert 'detailed_analysis' in result
        assert 'insights' in result
        assert 'recommendations' in result

    def test_generate_comprehensive_report_minimal_data(self):
        """Test comprehensive report generation with minimal data."""
        from symergetics.computation.analysis import generate_comprehensive_report
        
        analysis_data = {
            'number': '123',
            'length': 3,
            'is_palindromic': False,
            'palindromic_density': 0.0
        }
        
        result = generate_comprehensive_report(analysis_data)
        
        assert 'title' in result
        assert 'summary' in result
        assert 'detailed_analysis' in result
        assert 'insights' in result
        assert 'recommendations' in result


class TestAnalysisErrorHandling:
    """Test comprehensive error handling in analysis functions."""

    def test_analyze_mathematical_patterns_invalid_analysis_depth(self):
        """Test mathematical pattern analysis with invalid analysis depth."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12321
        
        # Test with negative analysis depth
        result = analyze_mathematical_patterns(number, analysis_depth=-1)
        assert 'number' in result
        
        # Test with zero analysis depth
        result = analyze_mathematical_patterns(number, analysis_depth=0)
        assert 'number' in result
        
        # Test with very large analysis depth
        result = analyze_mathematical_patterns(number, analysis_depth=100)
        assert 'number' in result

    def test_compare_mathematical_domains_invalid_input(self):
        """Test mathematical domain comparison with invalid input."""
        from symergetics.computation.analysis import compare_mathematical_domains
        
        # Test with None input
        result = compare_mathematical_domains(None, None)
        assert hasattr(result, 'correlation_coefficient')
        
        # Test with mixed input types
        result = compare_mathematical_domains([], {})
        assert hasattr(result, 'correlation_coefficient')

    def test_generate_comprehensive_report_invalid_input(self):
        """Test comprehensive report generation with invalid input."""
        from symergetics.computation.analysis import generate_comprehensive_report
        
        # Test with None input
        result = generate_comprehensive_report(None)
        assert 'title' in result
        
        # Test with empty dict
        result = generate_comprehensive_report({})
        assert 'title' in result


class TestAnalysisIntegration:
    """Test integration between different analysis functions."""

    def test_analysis_workflow_integration(self):
        """Test complete analysis workflow integration."""
        from symergetics.computation.analysis import (
            analyze_mathematical_patterns,
            compare_mathematical_domains,
            generate_comprehensive_report
        )
        
        # Analyze two numbers
        number1 = 12321
        number2 = 12345
        
        analysis1 = analyze_mathematical_patterns(number1)
        analysis2 = analyze_mathematical_patterns(number2)
        
        # Compare domains
        domain1_data = [analysis1]
        domain2_data = [analysis2]
        
        comparison = compare_mathematical_domains(domain1_data, domain2_data)
        
        # Generate report
        report = generate_comprehensive_report(analysis1)
        
        # Check that all results are valid
        assert 'number' in analysis1
        assert 'number' in analysis2
        assert hasattr(comparison, 'correlation_coefficient')
        assert 'title' in report

    def test_analysis_metadata_consistency(self):
        """Test that analysis metadata is consistent across functions."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12321
        result = analyze_mathematical_patterns(number)
        
        # Check that result has expected structure
        assert 'number' in result
        assert 'numeric_value' in result
        assert 'string_representation' in result
        assert 'length' in result
        assert 'is_palindromic' in result
        assert 'palindromic_density' in result

    def test_analysis_cross_validation(self):
        """Test cross-validation between analysis functions."""
        from symergetics.computation.analysis import (
            analyze_mathematical_patterns,
            compare_mathematical_domains
        )
        
        # Test with same data in different formats
        number1 = 12321
        number2 = 12321
        
        analysis1 = analyze_mathematical_patterns(number1)
        analysis2 = analyze_mathematical_patterns(number2)
        
        # Compare identical analyses
        comparison = compare_mathematical_domains([analysis1], [analysis2])
        
        # Should have high correlation for identical data
        assert hasattr(comparison, 'correlation_coefficient')
        assert hasattr(comparison, 'similarity_score')


class TestAnalysisPerformance:
    """Test performance aspects of analysis functions."""

    def test_analysis_execution_time(self):
        """Test that analysis functions execute within reasonable time."""
        import time
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 12345678987654321
        
        start_time = time.time()
        result = analyze_mathematical_patterns(number)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 2.0  # Should complete within 2 seconds
        assert 'number' in result

    def test_large_number_performance(self):
        """Test performance with large numbers."""
        import time
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        number = 10**100  # Very large number
        
        start_time = time.time()
        result = analyze_mathematical_patterns(number)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert 'number' in result

    def test_domain_comparison_performance(self):
        """Test performance of domain comparison."""
        import time
        from symergetics.computation.analysis import compare_mathematical_domains
        
        domain1_data = [{'is_palindromic': True, 'length': 5} for _ in range(100)]
        domain2_data = [{'is_palindromic': False, 'length': 3} for _ in range(100)]
        
        start_time = time.time()
        result = compare_mathematical_domains(domain1_data, domain2_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 3.0  # Should complete within 3 seconds
        assert hasattr(result, 'correlation_coefficient')

    def test_report_generation_performance(self):
        """Test performance of report generation."""
        import time
        from symergetics.computation.analysis import generate_comprehensive_report
        
        analysis_data = {
            'number': '12345678987654321',
            'length': 17,
            'is_palindromic': True,
            'palindromic_density': 1.0,
            'pattern_complexity': {'complexity_score': 3.5},
            'symmetry_analysis': {'symmetry_score': 0.9}
        }
        
        start_time = time.time()
        result = generate_comprehensive_report(analysis_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 1.0  # Should complete within 1 second
        assert 'title' in result


class TestAnalysisEdgeCases:
    """Test edge cases and boundary conditions for analysis functions."""

    def test_analyze_mathematical_patterns_boundary_values(self):
        """Test mathematical pattern analysis with boundary values."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        # Test with very small numbers
        small_number = 0.001
        result = analyze_mathematical_patterns(small_number)
        assert 'number' in result
        
        # Test with very large numbers
        large_number = 10**100
        result = analyze_mathematical_patterns(large_number)
        assert 'number' in result

    def test_analyze_mathematical_patterns_extreme_sequences(self):
        """Test mathematical pattern analysis with extreme sequences."""
        from symergetics.computation.analysis import analyze_mathematical_patterns
        
        # Test with constant sequence
        constant = "1111111111"
        result = analyze_mathematical_patterns(constant)
        assert 'number' in result
        
        # Test with alternating sequence
        alternating = "1010101010"
        result = analyze_mathematical_patterns(alternating)
        assert 'number' in result

    def test_compare_mathematical_domains_extreme_data(self):
        """Test mathematical domain comparison with extreme data."""
        from symergetics.computation.analysis import compare_mathematical_domains
        
        # Test with very large datasets
        large_domain1 = [{'is_palindromic': True, 'length': 5} for _ in range(1000)]
        large_domain2 = [{'is_palindromic': False, 'length': 3} for _ in range(1000)]
        
        result = compare_mathematical_domains(large_domain1, large_domain2)
        assert hasattr(result, 'correlation_coefficient')

    def test_generate_comprehensive_report_extreme_data(self):
        """Test comprehensive report generation with extreme data."""
        from symergetics.computation.analysis import generate_comprehensive_report
        
        # Test with very complex analysis data
        complex_data = {
            'number': '1' * 1000,  # Very long number
            'length': 1000,
            'is_palindromic': True,
            'palindromic_density': 1.0,
            'pattern_complexity': {'complexity_score': 10.0},
            'symmetry_analysis': {'symmetry_score': 1.0}
        }
        
        result = generate_comprehensive_report(complex_data)
        assert 'title' in result


if __name__ == "__main__":
    pytest.main([__file__])

