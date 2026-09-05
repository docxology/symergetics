#!/usr/bin/env python3
"""
Visualization Init Coverage Tests

Covers the public surface of symergetics.visualization.__init__ with real
function calls (no mocks), writing into temporary output directories.
"""

import os
import tempfile
from pathlib import Path

import pytest

from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.visualization import (
    batch_visualize,
    ensure_output_dir,
    get_config,
    get_organized_output_path,
    list_output_structure,
    reset_config,
    set_config,
)


@pytest.fixture(autouse=True)
def temp_output_config():
    """Run every test against a temporary output directory, then reset config."""
    with tempfile.TemporaryDirectory() as tmp:
        set_config({'output_dir': tmp})
        yield tmp
        reset_config()


def _assert_rendered(result):
    """Assert a plot result references real, non-empty files."""
    assert 'files' in result and result['files']
    for file_path in result['files']:
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0
    assert 'metadata' in result


class TestVisualizationInitFunctionMap:
    """batch_visualize must dispatch to the real visualization functions."""

    def test_batch_visualize_with_original_functions(self, temp_output_config):
        """Test batch_visualize with original visualization functions."""
        tasks = [
            {'function': 'plot_polyhedron', 'args': ['tetrahedron']},
            {'function': 'plot_quadray_coordinate',
             'args': [QuadrayCoordinate(1, 2, 3, 4)]},
        ]

        result = batch_visualize(tasks)

        assert len(result) == 2
        _assert_rendered(result[0])
        _assert_rendered(result[1])
        assert result[0]['metadata']['type'] == 'polyhedron_3d'

    def test_batch_visualize_with_new_functions(self, temp_output_config):
        """Test batch_visualize dispatches the enhanced functions too."""
        tasks = [
            {'function': 'plot_polyhedron_3d', 'args': ['tetrahedron']},
            {'function': 'plot_palindromic_pattern', 'args': [12321]},
        ]

        result = batch_visualize(tasks)

        assert len(result) == 2
        _assert_rendered(result[0])
        _assert_rendered(result[1])

    def test_batch_visualize_invalid_function(self):
        """Test batch_visualize with invalid function name."""
        tasks = [
            {'function': 'invalid_function', 'args': []}
        ]

        result = batch_visualize(tasks)

        assert len(result) == 1
        assert 'error' in result[0]
        assert 'Unknown visualization function' in result[0]['error']

    def test_batch_visualize_empty_tasks(self):
        """Test batch_visualize with empty tasks list."""
        result = batch_visualize([])
        assert result == []


class TestVisualizationInitUtilityFunctions:
    """Test utility functions in visualization __init__.py."""

    def test_get_organized_output_path(self, temp_output_config):
        """Test get_organized_output_path returns a real path under output_dir."""
        result = get_organized_output_path('test', 'category', 'file.png')
        assert isinstance(result, Path)
        assert temp_output_config in str(result)

    def test_ensure_output_dir(self, temp_output_config):
        """Test ensure_output_dir creates the directory."""
        result = ensure_output_dir()
        assert isinstance(result, Path)
        assert result.exists()

    def test_list_output_structure(self, temp_output_config):
        """Test list_output_structure returns structure info."""
        result = list_output_structure()
        assert isinstance(result, dict)

    def test_batch_visualize_with_kwargs(self, temp_output_config):
        """Test batch_visualize forwards kwargs to the real functions."""
        tasks = [
            {'function': 'plot_polyhedron',
             'args': ['tetrahedron'],
             'kwargs': {'show_vertices': False}},
        ]

        result = batch_visualize(tasks)

        assert len(result) == 1
        _assert_rendered(result[0])


class TestVisualizationInitErrorHandling:
    """Test error handling in visualization __init__.py with real failures."""

    def test_batch_visualize_with_function_error(self):
        """A failing plot task yields an error entry, not a crash."""
        # 'invalid' is not a number, so plot_base_conversion raises ValueError
        tasks = [
            {'function': 'plot_base_conversion', 'args': ['invalid']}
        ]

        result = batch_visualize(tasks)

        assert len(result) == 1
        assert 'error' in result[0]

    def test_batch_visualize_partial_success(self, temp_output_config):
        """One valid and one failing task produce one result and one error."""
        tasks = [
            {'function': 'plot_polyhedron', 'args': ['tetrahedron']},
            {'function': 'plot_base_conversion', 'args': ['invalid']},
        ]

        result = batch_visualize(tasks)

        assert len(result) == 2
        _assert_rendered(result[0])
        assert 'error' in result[1]


class TestVisualizationInitConfiguration:
    """Test configuration functions in visualization __init__.py."""

    def test_config_round_trip(self, temp_output_config):
        """set_config is visible through get_config and reset_config restores."""
        set_config({'backend': 'plotly', 'dpi': 150})
        config = get_config()
        assert config['backend'] == 'plotly'
        assert config['dpi'] == 150

        reset_config()
        config = get_config()
        assert config['backend'] == 'matplotlib'
        assert config['output_dir'] == temp_output_config or config['output_dir'] == 'output'
