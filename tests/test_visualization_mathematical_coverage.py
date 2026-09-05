#!/usr/bin/env python3
"""
Visualization Mathematical Coverage Tests

This module contains tests to improve code coverage for visualization.mathematical module,
focusing on the missing lines identified in the coverage report.
"""

import math
import os
import tempfile

import pytest

from symergetics.visualization import reset_config, set_config


def assert_files_real(result):
    """Assert the plot produced real, non-empty files."""
    assert result.get('files'), 'expected rendered files'
    for file_path in result['files']:
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0


# Import visualization modules
from symergetics.visualization.mathematical import (
    plot_base_conversion,
    plot_base_conversion_matrix,
    plot_continued_fraction,
    plot_continued_fraction_convergence,
    plot_pattern_analysis,
    plot_pattern_analysis_radar,
)


class TestPlotContinuedFractionConvergenceCoverage:
    """Test coverage for plot_continued_fraction_convergence function."""

    def test_plot_continued_fraction_convergence_matplotlib(self):
        """Test plot_continued_fraction_convergence with a real matplotlib render."""
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                result = plot_continued_fraction_convergence(
                    math.pi, max_terms=15, backend='matplotlib')
            finally:
                reset_config()

            assert isinstance(result, dict)
            assert_files_real(result)
        assert result['metadata']['backend'] == 'matplotlib'

    def test_plot_continued_fraction_convergence_plotly(self):
        """Test plot_continued_fraction_convergence with a real plotly render."""
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                result = plot_continued_fraction_convergence(
                    math.e, max_terms=10, backend='plotly')
            finally:
                reset_config()

            assert isinstance(result, dict)
            assert_files_real(result)
        assert result['metadata']['backend'] == 'plotly'


    def test_plot_continued_fraction_convergence_ascii(self):
        """Test plot_continued_fraction_convergence with ascii (should fail)."""
        with pytest.raises(ValueError, match="Convergence visualization requires matplotlib or plotly backend"):
            plot_continued_fraction_convergence(1.414213562, max_terms=8, backend='ascii')

    def test_plot_continued_fraction_convergence_integer(self):
        """Test plot_continued_fraction_convergence with integer value."""
        result = plot_continued_fraction_convergence(3.0, max_terms=5, backend='matplotlib')

        # Should handle integer values
        assert isinstance(result, dict)

    def test_plot_continued_fraction_convergence_few_terms(self):
        """Test plot_continued_fraction_convergence with few terms."""
        # Golden ratio: (1 + sqrt(5)) / 2
        phi = (1 + math.sqrt(5)) / 2
        result = plot_continued_fraction_convergence(phi, max_terms=3, backend='matplotlib')

        # Should handle small number of terms
        assert isinstance(result, dict)


class TestPlotBaseConversionMatrixCoverage:
    """Test coverage for plot_base_conversion_matrix function."""

    def test_plot_base_conversion_matrix_matplotlib(self):
        """Test plot_base_conversion_matrix with a real matplotlib render."""
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                result = plot_base_conversion_matrix(
                    start_base=3, end_base=8, number=123, backend='matplotlib')
            finally:
                reset_config()

            assert isinstance(result, dict)
            assert_files_real(result)
        assert result['metadata']['backend'] == 'matplotlib'

    def test_plot_base_conversion_matrix_plotly(self):
        """Test plot_base_conversion_matrix with a real plotly render."""
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                result = plot_base_conversion_matrix(
                    start_base=3, end_base=8, number=123, backend='plotly')
            finally:
                reset_config()

            assert isinstance(result, dict)
            assert_files_real(result)
        assert result['metadata']['backend'] == 'plotly'

    def test_plot_base_conversion_matrix_ascii(self):
        """Test plot_base_conversion_matrix with ascii (should fail)."""
        with pytest.raises(ValueError, match="Matrix visualization requires matplotlib or plotly backend"):
            plot_base_conversion_matrix(start_base=2, end_base=6, number=42, backend='ascii')

    def test_plot_base_conversion_matrix_wide_range(self):
        """Test plot_base_conversion_matrix with wide base range."""
        result = plot_base_conversion_matrix(start_base=2, end_base=16, number=255, backend='matplotlib')

        # Should handle wider range
        assert isinstance(result, dict)

    def test_plot_base_conversion_matrix_invalid_range(self):
        """Test plot_base_conversion_matrix with invalid base range."""
        # This function doesn't validate range, it just works with whatever is given
        result = plot_base_conversion_matrix(start_base=10, end_base=5, number=100, backend='matplotlib')
        assert isinstance(result, dict)


class TestPlotPatternAnalysisRadarCoverage:
    """Test coverage for plot_pattern_analysis_radar function."""

    def test_plot_pattern_analysis_radar_matplotlib(self):
        """Test plot_pattern_analysis_radar with a real matplotlib render."""
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                result = plot_pattern_analysis_radar("111222333", backend='matplotlib')
            finally:
                reset_config()

            assert isinstance(result, dict)
            assert_files_real(result)
        assert result['metadata']['backend'] == 'matplotlib'

    def test_plot_pattern_analysis_radar_plotly(self):
        """Test plot_pattern_analysis_radar with a real plotly render."""
        with tempfile.TemporaryDirectory() as tmp:
            set_config({'output_dir': tmp})
            try:
                result = plot_pattern_analysis_radar("111222333", backend='plotly')
            finally:
                reset_config()

            assert isinstance(result, dict)
            assert_files_real(result)
        assert result['metadata']['backend'] == 'plotly'

    def test_plot_pattern_analysis_radar_ascii(self):
        """Test plot_pattern_analysis_radar with ascii (should fail)."""
        sequence = "121212"
        with pytest.raises(ValueError, match="Radar visualization requires matplotlib or plotly backend"):
            plot_pattern_analysis_radar(sequence, backend='ascii')

    def test_plot_pattern_analysis_radar_empty_sequence(self):
        """Test plot_pattern_analysis_radar with empty sequence."""
        with pytest.raises(ValueError):
            plot_pattern_analysis_radar("")

    def test_plot_pattern_analysis_radar_single_character(self):
        """Test plot_pattern_analysis_radar with single character."""
        result = plot_pattern_analysis_radar("5", backend='matplotlib')

        # Should handle single character
        assert isinstance(result, dict)

    def test_plot_pattern_analysis_radar_long_sequence(self):
        """Test plot_pattern_analysis_radar with long sequence."""
        sequence = "123456789" * 10  # 90 characters
        result = plot_pattern_analysis_radar(sequence, backend='matplotlib')

        # Should handle long sequences
        assert isinstance(result, dict)


class TestOriginalMathematicalVisualizationsCoverage:
    """Test coverage for original mathematical visualization functions."""

    def test_plot_continued_fraction_custom_value(self):
        """Test plot_continued_fraction with custom value."""
        result = plot_continued_fraction(math.sqrt(2), max_terms=12)

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert_files_real(result)

    def test_plot_base_conversion_large_number(self):
        """Test plot_base_conversion with large number."""
        result = plot_base_conversion(1001, max_base=16)

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert_files_real(result)

    def test_plot_pattern_analysis_complex_sequence(self):
        """Test plot_pattern_analysis with complex sequence."""
        result = plot_pattern_analysis("1234554321")

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert_files_real(result)

