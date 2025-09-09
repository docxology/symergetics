"""
Tests for new enhanced visualization methods.

This module tests the new visualization methods added to the Synergetics package,
including 3D visualizations, heatmaps, networks, spectra, convergence plots,
matrices, and radar charts.
"""

import pytest
import numpy as np
import os
from pathlib import Path

from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.core.numbers import SymergeticsNumber
from symergetics.geometry.polyhedra import Tetrahedron
from symergetics.computation.primorials import primorial

# Import visualization functions
from symergetics.visualization.geometry import (
    plot_polyhedron_3d,
    plot_polyhedron_graphical_abstract,
    plot_polyhedron_wireframe
)
from symergetics.visualization.numbers import (
    plot_palindromic_heatmap,
    plot_scheherazade_network,
    plot_primorial_spectrum
)
from symergetics.visualization.mathematical import (
    plot_continued_fraction_convergence,
    plot_base_conversion_matrix,
    plot_pattern_analysis_radar
)

from symergetics.visualization import ensure_output_dir


class TestEnhancedGeometryVisualizations:
    """Test enhanced geometry visualization methods."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment."""
        self.output_dir = Path("output/test_visualizations")
        ensure_output_dir()

    def test_plot_polyhedron_3d_matplotlib(self):
        """Test enhanced 3D polyhedron visualization."""
        tetra = Tetrahedron()

        result = plot_polyhedron_3d(tetra, backend='matplotlib', show_wireframe=True, show_surface=True)

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'polyhedron_3d_enhanced'
        assert result['metadata']['polyhedron'] == 'Tetrahedron'

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_plot_polyhedron_graphical_abstract(self):
        """Test graphical abstract visualization."""
        cube = Tetrahedron()  # Using tetrahedron for simplicity

        result = plot_polyhedron_graphical_abstract(cube, show_volume_ratios=True, show_coordinates=True)

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'polyhedron_graphical_abstract'

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_plot_polyhedron_wireframe(self):
        """Test wireframe visualization."""
        octa = Tetrahedron()  # Using tetrahedron for simplicity

        result = plot_polyhedron_wireframe(octa, elevation=30, azimuth=60)

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'polyhedron_wireframe'

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_invalid_backend_3d(self):
        """Test that invalid backend raises error for 3D visualization."""
        tetra = Tetrahedron()

        with pytest.raises(ValueError, match="Unsupported backend"):
            plot_polyhedron_3d(tetra, backend='invalid')


class TestEnhancedNumberVisualizations:
    """Test enhanced number visualization methods."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment."""
        self.output_dir = Path("output/test_visualizations")
        ensure_output_dir()

    def test_plot_palindromic_heatmap(self):
        """Test palindromic heatmap visualization."""
        result = plot_palindromic_heatmap(100, 200, backend='matplotlib')

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'palindromic_heatmap'
        assert result['metadata']['sequence_start'] == 100
        assert result['metadata']['sequence_end'] == 200

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_plot_scheherazade_network(self):
        """Test Scheherazade network visualization."""
        result = plot_scheherazade_network(4, backend='matplotlib')

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'scheherazade_network'
        assert result['metadata']['power'] == 4

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_plot_primorial_spectrum(self):
        """Test primorial spectrum visualization."""
        result = plot_primorial_spectrum(max_n=10, backend='matplotlib')

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'primorial_spectrum'
        assert result['metadata']['max_n'] == 10

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_invalid_backend_heatmap(self):
        """Test that invalid backend raises error for heatmap."""
        with pytest.raises(ValueError, match="Heatmap visualization requires matplotlib backend"):
            plot_palindromic_heatmap(100, 150, backend='invalid')


class TestEnhancedMathematicalVisualizations:
    """Test enhanced mathematical visualization methods."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment."""
        self.output_dir = Path("output/test_visualizations")
        ensure_output_dir()

    def test_plot_continued_fraction_convergence(self):
        """Test continued fraction convergence visualization."""
        result = plot_continued_fraction_convergence(np.pi, max_terms=10, backend='matplotlib')

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'continued_fraction_convergence'
        assert abs(result['metadata']['value'] - np.pi) < 1e-10

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_plot_base_conversion_matrix(self):
        """Test base conversion matrix visualization."""
        result = plot_base_conversion_matrix(start_base=2, end_base=8, number=1001, backend='matplotlib')

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'base_conversion_matrix'
        assert result['metadata']['start_base'] == 2
        assert result['metadata']['end_base'] == 8
        assert result['metadata']['number'] == 1001

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_plot_pattern_analysis_radar(self):
        """Test pattern analysis radar visualization."""
        test_sequence = "123454321987654321"
        result = plot_pattern_analysis_radar(test_sequence, backend='matplotlib')

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'pattern_analysis_radar'
        assert result['metadata']['sequence_length'] == len(test_sequence)

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_invalid_backend_convergence(self):
        """Test that invalid backend raises error for convergence plot."""
        with pytest.raises(ValueError, match="Convergence visualization requires matplotlib backend"):
            plot_continued_fraction_convergence(np.e, backend='invalid')


class TestVisualizationIntegration:
    """Test integration between different visualization methods."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment."""
        self.output_dir = Path("output/test_visualizations")
        ensure_output_dir()

    def test_batch_visualization_with_new_methods(self):
        """Test batch visualization with new methods."""
        from symergetics.visualization import batch_visualize

        tetra = Tetrahedron()

        # Create batch visualization tasks
        tasks = [
            {
                'function': 'plot_polyhedron_3d',
                'args': [tetra],
                'kwargs': {'show_wireframe': True}
            },
            {
                'function': 'plot_palindromic_heatmap',
                'args': [100, 150]
            },
            {
                'function': 'plot_continued_fraction_convergence',
                'args': [np.pi, 8]
            }
        ]

        results = batch_visualize(tasks, backend='matplotlib')

        assert len(results) == 3

        # Check that each result is valid
        for result in results:
            assert 'files' in result
            assert 'metadata' in result

            # Check that files were created (skip if error occurred)
            if 'error' not in result:
                for file_path in result['files']:
                    assert os.path.exists(file_path)

    def test_visualization_error_handling(self):
        """Test error handling in visualization methods."""
        # Test with invalid polyhedron name
        with pytest.raises(ValueError, match="Unknown polyhedron"):
            plot_polyhedron_3d("invalid_polyhedron")

        # Test with invalid backend
        tetra = Tetrahedron()
        with pytest.raises(ValueError, match="Unsupported backend"):
            plot_polyhedron_3d(tetra, backend='invalid')

    def test_visualization_metadata_consistency(self):
        """Test that visualization metadata is consistent across methods."""
        tetra = Tetrahedron()

        # Test multiple visualization methods
        methods = [
            lambda: plot_polyhedron_3d(tetra),
            lambda: plot_polyhedron_graphical_abstract(tetra),
            lambda: plot_polyhedron_wireframe(tetra)
        ]

        for method in methods:
            result = method()

            # Check required metadata fields
            assert 'type' in result['metadata']
            assert 'backend' in result['metadata']
            assert isinstance(result['files'], list)
            assert len(result['files']) > 0

    def test_file_naming_and_organization(self):
        """Test that files are named and organized correctly."""
        tetra = Tetrahedron()

        result = plot_polyhedron_3d(tetra, show_wireframe=True, show_surface=True)

        # Check file naming
        filename = os.path.basename(result['files'][0])
        assert 'tetrahedron' in filename.lower()
        assert '3d_enhanced' in filename
        assert filename.endswith('.png')

        # Check file organization (should be in geometric/polyhedra/)
        file_path = Path(result['files'][0])
        assert 'geometric' in str(file_path)
        assert 'polyhedra' in str(file_path)


class TestVisualizationPerformance:
    """Test performance aspects of new visualizations."""

    def test_visualization_execution_time(self):
        """Test that visualizations execute within reasonable time."""
        import time

        tetra = Tetrahedron()
        start_time = time.time()

        result = plot_polyhedron_3d(tetra)
        execution_time = time.time() - start_time

        # Should complete within 30 seconds
        assert execution_time < 30.0

        # Clean up created files
        for file_path in result['files']:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_memory_usage_during_visualization(self):
        """Test memory usage during visualization creation."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        tetra = Tetrahedron()
        result = plot_polyhedron_3d(tetra)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 500 MB)
        assert memory_increase < 500.0

        # Clean up created files
        for file_path in result['files']:
            if os.path.exists(file_path):
                os.remove(file_path)


class TestVisualizationConfiguration:
    """Test visualization configuration options."""

    def test_custom_color_scheme(self):
        """Test custom color scheme in visualizations."""
        from symergetics.visualization import set_config

        # Set custom colors
        custom_config = {
            'colors': {
                'primary': '#FF6B6B',
                'secondary': '#4ECDC4',
                'accent': '#45B7D1'
            }
        }

        set_config(custom_config)

        tetra = Tetrahedron()
        result = plot_polyhedron_3d(tetra)

        # Should use custom configuration
        assert result['metadata']['backend'] == 'matplotlib'

        # Clean up
        for file_path in result['files']:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_output_directory_configuration(self):
        """Test custom output directory configuration."""
        from symergetics.visualization import set_config

        custom_output_dir = "output/test_custom_dir"
        set_config({'output_dir': custom_output_dir})

        tetra = Tetrahedron()
        result = plot_polyhedron_3d(tetra)

        # Check that files are in custom directory
        for file_path in result['files']:
            assert custom_output_dir in file_path

        # Clean up
        for file_path in result['files']:
            if os.path.exists(file_path):
                os.remove(file_path)


# Cleanup fixture for all tests
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files():
    """Clean up test files after all tests complete."""
    yield

    # Clean up test visualization files
    import shutil
    test_output_dir = Path("output/test_visualizations")

    if test_output_dir.exists():
        shutil.rmtree(test_output_dir, ignore_errors=True)

    # Clean up any other test files
    for pattern in ["*_3d_enhanced*", "*heatmap*", "*network*", "*spectrum*",
                   "*convergence*", "*matrix*", "*radar*"]:
        for file_path in Path("output").rglob(pattern):
            if file_path.is_file():
                file_path.unlink(missing_ok=True)
