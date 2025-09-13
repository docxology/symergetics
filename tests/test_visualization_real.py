"""
Comprehensive real tests for visualization modules.

Tests actual visualization functions using ASCII backend to avoid GUI dependencies.
All tests use real Symergetics methods and functions.
"""

import pytest
import tempfile
import os
from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.constants import SymergeticsConstants
from symergetics.computation.palindromes import is_palindromic
from symergetics.computation.primorials import primorial, scheherazade_power
from symergetics.utils.mnemonics import mnemonic_encode

# Import visualization functions
from symergetics.visualization.numbers import (
    plot_palindromic_pattern,
    plot_scheherazade_pattern,
    plot_primorial_distribution,
    plot_mnemonic_visualization
)
from symergetics.visualization.geometry import (
    plot_polyhedron,
    plot_quadray_coordinate,
    plot_ivm_lattice
)
from symergetics.visualization.mathematical import (
    plot_continued_fraction,
    plot_base_conversion,
    plot_pattern_analysis,
    plot_ssrcd_analysis
)
from symergetics.visualization.advanced import (
    create_comparative_analysis_visualization
)


class TestVisualizationNumbers:
    """Test number visualization functions."""
    
    def test_plot_palindromic_pattern_basic(self):
        """Test basic palindromic pattern plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_palindromic_pattern(
                12321, 
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result
            assert 'metadata' in result
    
    def test_plot_palindromic_pattern_symergetics_number(self):
        """Test palindromic pattern with SymergeticsNumber."""
        with tempfile.TemporaryDirectory() as temp_dir:
            number = SymergeticsNumber(12321)
            result = plot_palindromic_pattern(
                number,
                backend="ascii", 
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_scheherazade_pattern(self):
        """Test Scheherazade pattern plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_scheherazade_pattern(
                3,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_primorial_distribution(self):
        """Test primorial distribution plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_primorial_distribution(
                max_n=10,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_mnemonic_visualization(self):
        """Test mnemonic visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            number = SymergeticsNumber(12345)
            result = plot_mnemonic_visualization(
                number,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result


class TestVisualizationGeometry:
    """Test geometry visualization functions."""
    
    def test_plot_polyhedron_tetrahedron(self):
        """Test tetrahedron plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_polyhedron(
                "tetrahedron",
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_polyhedron_octahedron(self):
        """Test octahedron plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_polyhedron(
                "octahedron",
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_polyhedron_cube(self):
        """Test cube plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_polyhedron(
                "cube",
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_quadray_coordinate(self):
        """Test quadray coordinate plotting."""
        from symergetics.core.coordinates import QuadrayCoordinate
        coord = QuadrayCoordinate(1, 2, 3, 4)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_quadray_coordinate(
                coord,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_ivm_lattice(self):
        """Test IVM lattice plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_ivm_lattice(
                size=3,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result


class TestVisualizationMathematical:
    """Test mathematical visualization functions."""
    
    def test_plot_continued_fraction(self):
        """Test continued fraction plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_continued_fraction(
                value=3.14159,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_base_conversion(self):
        """Test base conversion plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_base_conversion(
                number=12345,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_pattern_analysis(self):
        """Test pattern analysis plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_pattern_analysis(
                number=12321,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_plot_ssrcd_analysis(self):
        """Test SSRCD analysis plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_ssrcd_analysis(
                power=3,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result


class TestVisualizationAdvanced:
    """Test advanced visualization functions."""
    
    def test_create_comparative_analysis_visualization(self):
        """Test comparative analysis visualization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data for comparison
            domain1_data = [{"number": 12321, "is_palindromic": True}]
            domain2_data = [{"number": 12345, "is_palindromic": False}]
            
            result = create_comparative_analysis_visualization(
                domain1_data=domain1_data,
                domain2_data=domain2_data,
                domain1_name="Palindromes",
                domain2_name="Non-palindromes",
                backend="matplotlib",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result


class TestVisualizationIntegration:
    """Test integration between visualization and core modules."""
    
    def test_visualization_with_constants(self):
        """Test visualization using Symergetics constants."""
        with tempfile.TemporaryDirectory() as temp_dir:
            constants = SymergeticsConstants()
            
            # Test with volume ratios
            tetra_vol = constants.get_volume_ratio('tetrahedron')
            result = plot_palindromic_pattern(
                tetra_vol,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_visualization_with_coordinates(self):
        """Test visualization with coordinate data."""
        from symergetics.core.coordinates import QuadrayCoordinate
        
        with tempfile.TemporaryDirectory() as temp_dir:
            coord = QuadrayCoordinate(1, 2, 3, 4)
            
            # Test coordinate visualization
            result = plot_quadray_coordinate(
                coord,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_visualization_error_handling(self):
        """Test visualization error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with invalid input
            result = plot_palindromic_pattern(
                "invalid",
                backend="ascii",
                output_dir=temp_dir
            )
            # Should handle gracefully
            assert isinstance(result, dict)


class TestVisualizationPerformance:
    """Test visualization performance and edge cases."""
    
    def test_large_number_visualization(self):
        """Test visualization with large numbers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            large_number = SymergeticsNumber(10**20)
            result = plot_palindromic_pattern(
                large_number,
                backend="ascii",
                output_dir=temp_dir
            )
            assert isinstance(result, dict)
    
    def test_multiple_output_formats(self):
        """Test multiple output format generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = plot_scheherazade_pattern(
                2,
                backend="ascii",
                output_dir=temp_dir,
                formats=["txt", "json"]
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
    
    def test_custom_configuration(self):
        """Test visualization with custom configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_config = {
                "figure_size": (12, 10),
                "colors": {"primary": "#ff0000"},
                "dpi": 150
            }
            
            result = plot_primorial_distribution(
                max_n=5,
                backend="ascii",
                output_dir=temp_dir,
                config=custom_config
            )
            assert isinstance(result, dict)
            assert 'files' in result or 'metadata' in result
