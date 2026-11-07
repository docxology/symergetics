"""
Tests for new enhanced visualization methods.

This module tests the new visualization methods added to the Synergetics package,
including 3D visualizations, heatmaps, networks, spectra, convergence plots,
matrices, and radar charts.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from symergetics.computation.primorials import primorial
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.core.numbers import SymergeticsNumber
from symergetics.geometry.polyhedra import Tetrahedron

# Import visualization functions
from symergetics.visualization.geometry import (
    plot_polyhedron_3d,
    plot_polyhedron_graphical_abstract,
    plot_polyhedron_wireframe,
)
from symergetics.visualization.mathematical import (
    plot_base_conversion_matrix,
    plot_continued_fraction_convergence,
    plot_pattern_analysis_radar,
)
from symergetics.visualization.numbers import (
    plot_palindromic_heatmap,
    plot_primorial_spectrum,
    plot_scheherazade_network,
)


class TestEnhancedGeometryVisualizations:
    """Test enhanced geometry visualization methods."""

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

        result = plot_polyhedron_graphical_abstract(cube, show_volume_ratios=True, show_coordinates=True, backend='matplotlib')

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'polyhedron_graphical_abstract'

        # Check that files were created
        for file_path in result['files']:
            assert os.path.exists(file_path)

    def test_plot_polyhedron_wireframe(self):
        """Test wireframe visualization."""
        octa = Tetrahedron()  # Using tetrahedron for simplicity

        result = plot_polyhedron_wireframe(octa, elevation=30, azimuth=60, backend='matplotlib')

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
        with pytest.raises(ValueError, match="Heatmap visualization requires matplotlib or plotly backend"):
            plot_palindromic_heatmap(100, 150, backend='invalid')


class TestEnhancedMathematicalVisualizations:
    """Test enhanced mathematical visualization methods."""

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
        with pytest.raises(ValueError, match="Convergence visualization requires matplotlib or plotly backend"):
            plot_continued_fraction_convergence(np.e, backend='invalid')


class TestVisualizationIntegration:
    """Test integration between different visualization methods."""

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
            lambda: plot_polyhedron_graphical_abstract(tetra, backend='matplotlib'),
            lambda: plot_polyhedron_wireframe(tetra, backend='matplotlib')
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

        result = plot_polyhedron_3d(tetra, show_wireframe=True, show_surface=True, backend='matplotlib')

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

    def test_memory_usage_during_visualization(self):
        """Test memory usage during visualization creation."""
        try:
            import os

            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # Skip memory testing if psutil not available
            pytest.skip("psutil not available for memory testing")

        tetra = Tetrahedron()
        result = plot_polyhedron_3d(tetra)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 500 MB)
        assert memory_increase < 500.0


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
        result = plot_polyhedron_3d(tetra, backend='matplotlib')

        # Should use custom configuration
        assert result['metadata']['backend'] == 'matplotlib'

    def test_output_directory_configuration(self, tmp_path):
        """Test custom output directory configuration (self-cleaning via tmp_path)."""
        from symergetics.visualization import reset_config, set_config

        custom_output_dir = tmp_path / "test_custom_dir"
        set_config({'output_dir': str(custom_output_dir)})
        try:
            tetra = Tetrahedron()
            result = plot_polyhedron_3d(tetra)

            # Check that files are in custom directory
            for file_path in result['files']:
                assert Path(file_path).resolve().is_relative_to(custom_output_dir.resolve())
                assert Path(file_path).is_file()
        finally:
            reset_config()



def test_child_pytest_preserves_preexisting_outputs(tmp_path):
    """A complete pytest teardown must leave unowned output artifacts intact."""
    source_tests = Path(__file__).resolve().parent
    child_root = tmp_path / "child-session"
    child_tests = child_root / "tests"
    child_tests.mkdir(parents=True)
    shutil.copy2(source_tests / "conftest.py", child_tests / "conftest.py")
    shutil.copy2(source_tests / "test_visualization_methods.py", child_tests / "test_visualization_methods.py")
    (child_root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    sentinels = [
        child_root / "output" / name for name in (
            "test_visualizations/keep.txt",
            "geometric/polyhedra/tetrahedron_3d_enhanced.png",
            "numbers/palindromes/user_heatmap.png",
            "numbers/scheherazade/user_network.png",
            "numbers/primorials/user_spectrum.png",
            "mathematical/user_convergence.png",
            "mathematical/user_matrix.png",
            "mathematical/user_radar.png",
        )
    ]
    for sentinel in sentinels:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(b"pre-existing user artifact")
    original_outputs = set((child_root / "output").rglob("*"))
    (child_tests / "test_isolation_probe.py").write_text(textwrap.dedent('''
        from pathlib import Path
        from symergetics.visualization import get_config, reset_config, set_config
        from symergetics.visualization.geometry import plot_polyhedron

        def assert_local_files(result):
            assert result["files"]
            for filename in result["files"]:
                path = Path(filename).resolve()
                assert path.is_relative_to(Path.cwd())
                assert path.is_file()

        def test_reset_does_not_escape_temporary_cwd(tmp_path):
            set_config({"output_dir": str(tmp_path / "custom"), "backend": "ascii"})
            assert_local_files(plot_polyhedron("tetrahedron"))
            reset_config()
            assert_local_files(plot_polyhedron("tetrahedron", backend="ascii"))
            set_config({"backend": "plotly", "colors": {"primary": "#abcdef"}})

        def test_neighbor_starts_with_defaults(tmp_path):
            assert Path.cwd() == tmp_path
            assert get_config()["backend"] == "matplotlib"
            assert get_config()["colors"]["primary"] == "#1f77b4"
            assert get_config()["output_dir"] == "output"
            assert not (Path.cwd() / "output").exists()
    '''), encoding="utf-8")
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(source_tests.parent), env.get("PYTHONPATH"),
    ]))
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "-c", str(child_root / "pytest.ini"),
            str(child_tests / "test_visualization_methods.py") + "::TestEnhancedGeometryVisualizations::test_invalid_backend_3d",
            str(child_tests / "test_isolation_probe.py"),
        ],
        cwd=child_root, env=env, text=True, capture_output=True,
        timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(sentinel.read_bytes() == b"pre-existing user artifact" for sentinel in sentinels)
    assert set((child_root / "output").rglob("*")) == original_outputs
