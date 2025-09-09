#!/usr/bin/env python3
"""
Visualization Init Coverage Tests

This module contains tests to improve code coverage for visualization.__init__.py module,
focusing on the missing lines identified in the coverage report.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import the visualization module
from symergetics.visualization import (
    plot_polyhedron, plot_quadray_coordinate, plot_ivm_lattice,
    plot_palindromic_pattern, plot_scheherazade_pattern, plot_primorial_distribution,
    plot_continued_fraction, plot_base_conversion, plot_pattern_analysis,
    plot_ssrcd_analysis,
    # New functions
    plot_polyhedron_3d, plot_polyhedron_graphical_abstract, plot_polyhedron_wireframe,
    plot_palindromic_heatmap, plot_scheherazade_network, plot_primorial_spectrum,
    plot_continued_fraction_convergence, plot_base_conversion_matrix, plot_pattern_analysis_radar
)


class TestVisualizationInitFunctionMap:
    """Test the function mapping and batch processing in __init__.py."""

    @patch('symergetics.visualization.plot_polyhedron')
    @patch('symergetics.visualization.plot_quadray_coordinate')
    def test_batch_visualize_with_original_functions(self, mock_plot_quadray, mock_plot_polyhedron):
        """Test batch_visualize with original visualization functions."""
        from symergetics.visualization import batch_visualize

        # Mock return values
        mock_plot_polyhedron.return_value = {'files': ['poly.png'], 'metadata': {}}
        mock_plot_quadray.return_value = {'files': ['coord.png'], 'metadata': {}}

        tasks = [
            {'function': 'plot_polyhedron', 'args': ['tetrahedron']},
            {'function': 'plot_quadray_coordinate', 'args': [[1, 2, 3, 4]]}
        ]

        result = batch_visualize(tasks)

        # Should call the functions and return results
        assert len(result) == 2
        assert result[0] == {'files': ['poly.png'], 'metadata': {}}
        assert result[1] == {'files': ['coord.png'], 'metadata': {}}

    @patch('symergetics.visualization.plot_polyhedron_3d')
    @patch('symergetics.visualization.plot_palindromic_heatmap')
    def test_batch_visualize_with_new_functions(self, mock_plot_heatmap, mock_plot_3d):
        """Test batch_visualize with new visualization functions."""
        from symergetics.visualization import batch_visualize

        # Mock return values
        mock_plot_3d.return_value = {'files': ['3d.png'], 'metadata': {}}
        mock_plot_heatmap.return_value = {'files': ['heatmap.png'], 'metadata': {}}

        tasks = [
            {'function': 'plot_polyhedron_3d', 'args': ['tetrahedron']},
            {'function': 'plot_palindromic_heatmap', 'args': [100, 120]}
        ]

        result = batch_visualize(tasks)

        # Should call the functions and return results
        assert len(result) == 2
        assert result[0] == {'files': ['3d.png'], 'metadata': {}}
        assert result[1] == {'files': ['heatmap.png'], 'metadata': {}}

    def test_batch_visualize_invalid_function(self):
        """Test batch_visualize with invalid function name."""
        from symergetics.visualization import batch_visualize

        tasks = [
            {'function': 'invalid_function', 'args': []}
        ]

        result = batch_visualize(tasks)

        # Should return error dict for invalid function
        assert len(result) == 1
        assert 'error' in result[0]
        assert 'Unknown visualization function' in result[0]['error']

    def test_batch_visualize_empty_tasks(self):
        """Test batch_visualize with empty tasks list."""
        from symergetics.visualization import batch_visualize

        result = batch_visualize([])

        # Should return empty list
        assert result == []


class TestVisualizationInitUtilityFunctions:
    """Test utility functions in visualization __init__.py."""

    def test_get_organized_output_path(self):
        """Test get_organized_output_path function."""
        from symergetics.visualization import get_organized_output_path
        from pathlib import Path

        # Test with parameters
        result = get_organized_output_path('test', 'category', 'file.png')

        # Should return a Path object
        assert isinstance(result, Path)

    def test_ensure_output_dir(self):
        """Test ensure_output_dir function."""
        from symergetics.visualization import ensure_output_dir
        from pathlib import Path

        result = ensure_output_dir()

        # Should return a Path object
        assert isinstance(result, Path)

    def test_list_output_structure(self):
        """Test list_output_structure function."""
        from symergetics.visualization import list_output_structure

        result = list_output_structure()

        # Should return a dict
        assert isinstance(result, dict)

    def test_batch_visualize_with_kwargs(self):
        """Test batch_visualize with kwargs in tasks."""
        from symergetics.visualization import batch_visualize

        tasks = [
            {'function': 'plot_polyhedron', 'args': ['tetrahedron'], 'kwargs': {'wireframe': True}}
        ]

        with patch('symergetics.visualization.plot_polyhedron') as mock_plot:
            mock_plot.return_value = {'files': ['test.png'], 'metadata': {}}
            result = batch_visualize(tasks)

            # Should pass kwargs to the function
            mock_plot.assert_called_once()
            call_args = mock_plot.call_args
            assert call_args.kwargs['wireframe'] == True


class TestVisualizationInitErrorHandling:
    """Test error handling in visualization __init__.py."""

    def test_batch_visualize_with_function_error(self):
        """Test batch_visualize when a function raises an error."""
        from symergetics.visualization import batch_visualize

        with patch('symergetics.visualization.plot_polyhedron') as mock_plot:
            mock_plot.side_effect = ValueError("Test error")

            tasks = [
                {'function': 'plot_polyhedron', 'args': ['tetrahedron']}
            ]

            result = batch_visualize(tasks)

            # Should handle error gracefully
            assert len(result) == 1
            assert 'error' in result[0]

    def test_batch_visualize_partial_success(self):
        """Test batch_visualize with some successes and some failures."""
        from symergetics.visualization import batch_visualize

        with patch('symergetics.visualization.plot_polyhedron') as mock_plot, \
             patch('symergetics.visualization.plot_quadray_coordinate') as mock_coord:

            mock_plot.return_value = {'files': ['poly.png'], 'metadata': {}}
            mock_coord.side_effect = RuntimeError("Coord error")

            tasks = [
                {'function': 'plot_polyhedron', 'args': ['tetrahedron']},
                {'function': 'plot_quadray_coordinate', 'args': [[1, 2, 3, 4]]}
            ]

            result = batch_visualize(tasks)

            # Should have one success and one error
            assert len(result) == 2
            assert result[0] == {'files': ['poly.png'], 'metadata': {}}
            assert 'error' in result[1]


class TestVisualizationInitConfiguration:
    """Test configuration functions in visualization __init__.py."""

    def test_get_config_default(self):
        """Test get_config with default values."""
        from symergetics.visualization import get_config

        config = get_config()

        # Should return a dictionary with default values
        assert isinstance(config, dict)
        assert 'backend' in config
        assert 'output_dir' in config

    def test_set_config(self):
        """Test set_config function."""
        from symergetics.visualization import set_config, get_config

        # Set custom config
        set_config({'backend': 'plotly', 'custom_param': 'value'})

        # Get config and verify
        config = get_config()
        assert config.get('backend') == 'plotly'
        assert config.get('custom_param') == 'value'

    def test_reset_config(self):
        """Test reset_config function."""
        from symergetics.visualization import set_config, reset_config, get_config

        # Set custom config
        set_config({'backend': 'plotly'})

        # Reset config
        reset_config()

        # Should be back to defaults
        config = get_config()
        assert config.get('backend') != 'plotly'  # Should not be plotly anymore
