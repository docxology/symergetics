"""
Tests for the visualization module.

Tests cover geometric, mathematical, and number visualizations
in the Synergetics framework.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from symergetics.visualization import (
    set_config, get_config, reset_config,
    plot_polyhedron, plot_quadray_coordinate, plot_ivm_lattice,
    plot_palindromic_pattern, plot_scheherazade_pattern,
    plot_continued_fraction, plot_base_conversion,
    plot_pattern_analysis,
    batch_visualize, export_visualization, create_visualization_report
)
from symergetics.core.coordinates import QuadrayCoordinate
from symergetics.core.numbers import SymergeticsNumber


class TestVisualizationConfig:
    """Test visualization configuration functions."""

    def test_get_config(self):
        """Test getting configuration."""
        config = get_config()
        assert isinstance(config, dict)
        assert 'backend' in config
        assert 'output_dir' in config

    def test_set_config(self):
        """Test setting configuration."""
        original_config = get_config()
        new_config = {'backend': 'ascii', 'dpi': 200}

        set_config(new_config)
        updated_config = get_config()

        assert updated_config['backend'] == 'ascii'
        assert updated_config['dpi'] == 200

        # Reset for other tests
        set_config(original_config)

    def test_reset_config(self):
        """Test resetting configuration."""
        original_config = get_config()
        set_config({'backend': 'plotly'})

        reset_config()
        assert get_config()['backend'] == original_config['backend']


class TestGeometricVisualizations:
    """Test geometric visualization functions."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        from symergetics.visualization import reset_config
        reset_config()

    def test_plot_polyhedron_matplotlib(self, temp_output_dir):
        """Test polyhedron plotting with matplotlib backend."""
        set_config({
            'output_dir': str(temp_output_dir), 
            'backend': 'matplotlib',
            'organize_by_type': True
        })

        result = plot_polyhedron('tetrahedron', backend='matplotlib')

        assert 'files' in result
        assert 'metadata' in result
        assert result['metadata']['type'] == 'polyhedron_3d'
        assert result['metadata']['polyhedron'] == 'Tetrahedron'

        # Check that files were created in organized structure
        if result['files']:
            for file_path in result['files']:
                file_obj = Path(file_path)
                assert file_obj.exists()
                # Should be in organized structure: temp_dir/geometric/polyhedra/
                assert 'geometric' in file_obj.parts
                assert 'polyhedra' in file_obj.parts

    def test_plot_polyhedron_ascii(self):
        """Test polyhedron plotting with ASCII backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'ascii'})

            result = plot_polyhedron('octahedron', backend='ascii')

            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['backend'] == 'ascii'
            assert len(result['files']) == 1

    def test_plot_quadray_coordinate(self):
        """Test Quadray coordinate visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            coord = QuadrayCoordinate(1, 2, 0, 1)
            result = plot_quadray_coordinate(coord, show_lattice=False, backend='matplotlib')

            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'quadray_coordinate'

    def test_plot_ivm_lattice(self):
        """Test IVM lattice visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            result = plot_ivm_lattice(size=3, backend='matplotlib')

            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'ivm_lattice'
            assert result['metadata']['lattice_size'] == 3


class TestNumberVisualizations:
    """Test number visualization functions."""

    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        from symergetics.visualization import reset_config
        reset_config()

    def test_plot_palindromic_pattern(self):
        """Test palindromic pattern visualization with organized output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({
                'output_dir': temp_dir, 
                'backend': 'matplotlib',
                'organize_by_type': True
            })

            result = plot_palindromic_pattern(121, backend='matplotlib')  # Palindrome

            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'palindrome_pattern'
            assert result['metadata']['is_palindromic'] == True
            
            # Check organized structure: temp_dir/numbers/palindromes/
            if result['files']:
                file_obj = Path(result['files'][0])
                assert file_obj.exists()
                assert 'numbers' in file_obj.parts
                assert 'palindromes' in file_obj.parts

    def test_plot_scheherazade_pattern(self):
        """Test Scheherazade number pattern visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            result = plot_scheherazade_pattern(2)  # 1001^2

            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'scheherazade_pattern'
            assert result['metadata']['power'] == 2

    def test_plot_scheherazade_pattern_ascii(self):
        """Test Scheherazade pattern with ASCII backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'ascii'})

            result = plot_scheherazade_pattern(1, backend='ascii')  # 1001^1

            assert 'files' in result
            assert result['metadata']['backend'] == 'ascii'


class TestMathematicalVisualizations:
    """Test mathematical visualization functions."""

    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        from symergetics.visualization import reset_config
        reset_config()

    def test_plot_continued_fraction(self):
        """Test continued fraction visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            result = plot_continued_fraction(np.pi, max_terms=5)

            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'continued_fraction'
            assert abs(result['metadata']['value'] - np.pi) < 1e-10

    def test_plot_base_conversion(self):
        """Test base conversion visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            result = plot_base_conversion(42, 10, 2)  # 42 in binary

            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'base_conversion'
            assert result['metadata']['number'] == 42
            assert result['metadata']['from_base'] == 10
            assert result['metadata']['to_base'] == 2

    def test_plot_pattern_analysis(self):
        """Test pattern analysis visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            result = plot_pattern_analysis(12321, 'palindrome')  # Palindrome

            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'pattern_analysis'
            assert result['metadata']['pattern_type'] == 'palindrome'


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        from symergetics.visualization import reset_config
        reset_config()

    def test_organized_output_structure(self):
        """Test the organized output structure functionality."""
        from symergetics.visualization import (
            get_organized_output_path, create_output_structure_readme,
            list_output_structure, set_config
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({
                'output_dir': temp_dir,
                'organize_by_type': True
            })
            
            # Test path generation
            path = get_organized_output_path('geometric', 'polyhedra', 'test.png')
            assert 'geometric' in str(path)
            assert 'polyhedra' in str(path)
            assert 'test.png' in str(path)
            
            # Create structure and README
            create_output_structure_readme()
            
            # Check that README files were created
            base_readme = Path(temp_dir) / 'README.md'
            assert base_readme.exists()
            
            # Check structure info
            structure_info = list_output_structure()
            assert structure_info['exists'] == True
            assert structure_info['organized'] == True
            assert 'geometric' in structure_info['categories']

    def test_batch_visualize(self):
        """Test batch visualization function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'ascii'})

            tasks = [
                {'function': 'plot_polyhedron', 'args': ['tetrahedron']},
                {'function': 'plot_palindromic_pattern', 'args': [121]},
            ]

            results = batch_visualize(tasks)

            assert len(results) == 2
            for result in results:
                assert 'files' in result or 'error' in result

    def test_batch_visualize_with_errors(self):
        """Test batch visualization with error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            tasks = [
                {'function': 'plot_polyhedron', 'args': ['tetrahedron'], 'kwargs': {'backend': 'matplotlib'}},
                {'function': 'nonexistent_function', 'args': []},
            ]

            results = batch_visualize(tasks)

            assert len(results) == 2
            # First should succeed, second should have error
            assert 'files' in results[0]
            assert 'error' in results[1]

    def test_export_visualization(self):
        """Test visualization export function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            test_data = {
                'type': 'test',
                'metadata': {'test': True},
                'files': ['test.png']
            }

            # Test JSON export
            filepath = export_visualization(test_data, format='json')
            assert Path(filepath).exists()
            assert filepath.endswith('.json')

            # Test text export
            filepath_txt = export_visualization(test_data, format='txt')
            assert Path(filepath_txt).exists()
            assert filepath_txt.endswith('.txt')

    def test_create_visualization_report(self):
        """Test visualization report creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            # Create mock results
            results = [
                {'files': ['test1.png'], 'metadata': {'type': 'test1'}},
                {'error': 'Test error', 'task': {'function': 'test2'}},
                {'files': ['test3.png'], 'metadata': {'type': 'test3'}},
            ]

            report_path = create_visualization_report(results, "Test Report")

            assert Path(report_path).exists()
            assert report_path.endswith('.txt')

            # Check report content
            with open(report_path, 'r') as f:
                content = f.read()
                assert "Test Report" in content
                assert "Total visualizations: 3" in content
                assert "Successful: 2" in content
                assert "Failed: 1" in content

    def test_invalid_export_format(self):
        """Test error handling for invalid export format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_visualization({}, format="invalid")


class TestVisualizationIntegration:
    """Test integration of visualization functions."""

    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        from symergetics.visualization import reset_config
        reset_config()

    def test_full_workflow(self):
        """Test a complete visualization workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'ascii'})

            # Step 1: Create visualizations
            results = []

            # Geometric visualization
            result1 = plot_polyhedron('cube', backend='ascii')
            results.append(result1)

            # Number visualization
            result2 = plot_palindromic_pattern(SymergeticsNumber(12321), backend='ascii')
            results.append(result2)

            # Mathematical visualization
            result3 = plot_continued_fraction(np.e, max_terms=8, backend='ascii')
            results.append(result3)

            # Step 2: Export results
            export_path = export_visualization({
                'workflow': 'test',
                'results': results
            })

            # Step 3: Create report
            report_path = create_visualization_report(results, "Integration Test Report")

            # Verify all outputs exist
            assert Path(export_path).exists()
            assert Path(report_path).exists()

            for result in results:
                if 'files' in result:
                    for file_path in result['files']:
                        assert Path(file_path).exists()


class TestBackendCompatibility:
    """Test compatibility with different backends."""

    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        from symergetics.visualization import reset_config
        reset_config()

    def test_backend_switching(self):
        """Test switching between different backends."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test matplotlib backend
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            result_matplotlib = plot_polyhedron('tetrahedron', backend='matplotlib')
            assert result_matplotlib['metadata']['backend'] == 'matplotlib'

            # Test ASCII backend
            set_config({'backend': 'ascii'})
            result_ascii = plot_polyhedron('tetrahedron', backend='ascii')
            assert result_ascii['metadata']['backend'] == 'ascii'

    def test_unsupported_backend(self):
        """Test error handling for unsupported backends."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'unsupported'})

            with pytest.raises(ValueError, match="Unsupported backend"):
                plot_polyhedron('tetrahedron', backend='unsupported')


class TestErrorHandling:
    """Test error handling in visualization functions."""

    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        from symergetics.visualization import reset_config
        reset_config()

    def test_invalid_polyhedron(self):
        """Test error handling for invalid polyhedron names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            with pytest.raises(ValueError, match="Unknown polyhedron"):
                plot_polyhedron('invalid_shape')

    def test_invalid_pattern_type(self):
        """Test error handling for invalid pattern types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir})

            with pytest.raises(ValueError, match="Unsupported backend"):
                plot_pattern_analysis(123, 'invalid_pattern', backend='invalid')

    def test_empty_visualization_tasks(self):
        """Test batch visualization with empty task list."""
        results = batch_visualize([])
        assert results == []


# Cleanup after all tests
@pytest.fixture(scope="session", autouse=True)
def cleanup_visualization_config():
    """Reset visualization configuration after all tests."""
    yield
    reset_config()
