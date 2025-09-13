#!/usr/bin/env python3
"""
Visualization Mathematical Coverage Tests

This module contains tests to improve code coverage for visualization.mathematical module,
focusing on the missing lines identified in the coverage report.
"""

import pytest
import math
from unittest.mock import patch, MagicMock

# Import visualization modules
from symergetics.visualization.mathematical import (
    plot_continued_fraction, plot_base_conversion, plot_pattern_analysis,
    plot_continued_fraction_convergence, plot_base_conversion_matrix, plot_pattern_analysis_radar
)


class TestPlotContinuedFractionConvergenceCoverage:
    """Test coverage for plot_continued_fraction_convergence function."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.tight_layout')
    def test_plot_continued_fraction_convergence_matplotlib(self, mock_layout, mock_subplot, mock_savefig, mock_figure):
        """Test plot_continued_fraction_convergence with matplotlib."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_continued_fraction_convergence(math.pi, max_terms=15, backend='matplotlib')

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result
        assert 'metadata' in result

    def test_plot_continued_fraction_convergence_plotly(self):
        """Test plot_continued_fraction_convergence with plotly (should fail)."""
        with pytest.raises(ValueError, match="Convergence visualization requires matplotlib backend"):
            plot_continued_fraction_convergence(math.e, max_terms=10, backend='plotly')

    def test_plot_continued_fraction_convergence_ascii(self):
        """Test plot_continued_fraction_convergence with ascii (should fail)."""
        with pytest.raises(ValueError, match="Convergence visualization requires matplotlib backend"):
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

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.table')
    def test_plot_base_conversion_matrix_matplotlib(self, mock_table, mock_savefig, mock_figure):
        """Test plot_base_conversion_matrix with matplotlib."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_base_conversion_matrix(start_base=2, end_base=10, number=1001, backend='matplotlib')

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

    def test_plot_base_conversion_matrix_plotly(self):
        """Test plot_base_conversion_matrix with plotly (should fail)."""
        with pytest.raises(ValueError, match="Matrix visualization requires matplotlib backend"):
            plot_base_conversion_matrix(start_base=3, end_base=8, number=123, backend='plotly')

    def test_plot_base_conversion_matrix_ascii(self):
        """Test plot_base_conversion_matrix with ascii (should fail)."""
        with pytest.raises(ValueError, match="Matrix visualization requires matplotlib backend"):
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

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplot')
    def test_plot_pattern_analysis_radar_matplotlib(self, mock_subplot, mock_savefig, mock_figure):
        """Test plot_pattern_analysis_radar with matplotlib."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        sequence = "123454321"
        result = plot_pattern_analysis_radar(sequence, backend='matplotlib')

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

    def test_plot_pattern_analysis_radar_plotly(self):
        """Test plot_pattern_analysis_radar with plotly (should fail)."""
        sequence = "111222333"
        with pytest.raises(ValueError, match="Radar visualization requires matplotlib backend"):
            plot_pattern_analysis_radar(sequence, backend='plotly')

    def test_plot_pattern_analysis_radar_ascii(self):
        """Test plot_pattern_analysis_radar with ascii (should fail)."""
        sequence = "121212"
        with pytest.raises(ValueError, match="Radar visualization requires matplotlib backend"):
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
        assert 'files' in result

    def test_plot_base_conversion_large_number(self):
        """Test plot_base_conversion with large number."""
        result = plot_base_conversion(1001, max_base=16)

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

    def test_plot_pattern_analysis_complex_sequence(self):
        """Test plot_pattern_analysis with complex sequence."""
        result = plot_pattern_analysis("1234554321")

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

