#!/usr/bin/env python3
"""
Visualization Numbers Coverage Tests

This module contains tests to improve code coverage for visualization.numbers module,
focusing on the missing lines identified in the coverage report.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import visualization modules
from symergetics.visualization.numbers import (
    plot_palindromic_pattern, plot_scheherazade_pattern, plot_primorial_distribution,
    plot_palindromic_heatmap, plot_scheherazade_network, plot_primorial_spectrum
)


class TestPlotPalindromicHeatmapCoverage:
    """Test coverage for plot_palindromic_heatmap function."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.heatmap')
    def test_plot_palindromic_heatmap_matplotlib(self, mock_heatmap, mock_savefig, mock_figure):
        """Test plot_palindromic_heatmap with matplotlib backend."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_palindromic_heatmap(100, 120, backend='matplotlib')

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result
        assert 'metadata' in result

    @patch('plotly.graph_objects.Figure')
    @patch('plotly.offline.plot')
    def test_plot_palindromic_heatmap_plotly(self, mock_plot, mock_figure):
        """Test plot_palindromic_heatmap with plotly backend."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_palindromic_heatmap(100, 120, backend='plotly')

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

    def test_plot_palindromic_heatmap_small_range(self):
        """Test plot_palindromic_heatmap with small number range."""
        result = plot_palindromic_heatmap(100, 105, backend='ascii')

        # Should return a dict with ASCII art
        assert isinstance(result, dict)
        assert 'ascii_art' in result

    def test_plot_palindromic_heatmap_invalid_range(self):
        """Test plot_palindromic_heatmap with invalid range."""
        with pytest.raises(ValueError):
            plot_palindromic_heatmap(120, 100)  # end < start


class TestPlotScheherazadeNetworkCoverage:
    """Test coverage for plot_scheherazade_network function."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('networkx.draw')
    @patch('networkx.spring_layout')
    def test_plot_scheherazade_network_matplotlib(self, mock_layout, mock_draw, mock_savefig, mock_figure):
        """Test plot_scheherazade_network with matplotlib backend."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_scheherazade_network(6, backend='matplotlib')

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

    def test_plot_scheherazade_network_plotly(self):
        """Test plot_scheherazade_network with plotly (should fail)."""
        with pytest.raises(ValueError, match="Network visualization requires matplotlib backend"):
            plot_scheherazade_network(6, backend='plotly')

    def test_plot_scheherazade_network_ascii(self):
        """Test plot_scheherazade_network with ascii (should fail)."""
        with pytest.raises(ValueError, match="Network visualization requires matplotlib backend"):
            plot_scheherazade_network(4, backend='ascii')

    def test_plot_scheherazade_network_power_zero(self):
        """Test plot_scheherazade_network with power 0."""
        result = plot_scheherazade_network(0)

        # Should handle power 0 gracefully
        assert isinstance(result, dict)


class TestPlotPrimorialSpectrumCoverage:
    """Test coverage for plot_primorial_spectrum function."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplot')
    def test_plot_primorial_spectrum_matplotlib(self, mock_subplot, mock_savefig, mock_figure):
        """Test plot_primorial_spectrum with matplotlib backend."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_primorial_spectrum(max_n=10, backend='matplotlib')

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result
        assert 'metadata' in result

    def test_plot_primorial_spectrum_plotly(self):
        """Test plot_primorial_spectrum with plotly (should fail)."""
        with pytest.raises(ValueError, match="Spectrum visualization requires matplotlib backend"):
            plot_primorial_spectrum(max_n=8, backend='plotly')

    def test_plot_primorial_spectrum_ascii(self):
        """Test plot_primorial_spectrum with ascii (should fail)."""
        with pytest.raises(ValueError, match="Spectrum visualization requires matplotlib backend"):
            plot_primorial_spectrum(max_n=5, backend='ascii')

    def test_plot_primorial_spectrum_large_max_n(self):
        """Test plot_primorial_spectrum with larger max_n."""
        result = plot_primorial_spectrum(max_n=15)

        # Should handle larger values
        assert isinstance(result, dict)

    def test_plot_primorial_spectrum_invalid_max_n(self):
        """Test plot_primorial_spectrum with invalid max_n."""
        with pytest.raises(ValueError):
            plot_primorial_spectrum(max_n=0)


class TestOriginalNumberVisualizationsCoverage:
    """Test coverage for original number visualization functions."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_palindromic_pattern_large_number(self, mock_savefig, mock_figure):
        """Test plot_palindromic_pattern with large number."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_palindromic_pattern(123454321)

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_scheherazade_pattern_higher_power(self, mock_savefig, mock_figure):
        """Test plot_scheherazade_pattern with higher power."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_scheherazade_pattern(8)

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_primorial_distribution_custom_range(self, mock_savefig, mock_figure):
        """Test plot_primorial_distribution with custom range."""
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig

        result = plot_primorial_distribution(max_n=12)

        # Should return a dict with file information
        assert isinstance(result, dict)
        assert 'files' in result

