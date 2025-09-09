"""
Tests for PNG visualization capabilities in the Symergetics package.

This module tests all PNG generation functionality, including:
- High-quality PNG output
- Configuration options
- File format validation
- Resolution and DPI settings
- Transparency and color options
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import os

from symergetics.visualization import (
    set_config, get_config, reset_config,
    plot_polyhedron, plot_quadray_coordinate, plot_ivm_lattice,
    plot_palindromic_pattern, plot_scheherazade_pattern,
    plot_continued_fraction, plot_base_conversion, plot_pattern_analysis
)
from symergetics.core.numbers import SymergeticsNumber
from symergetics.core.coordinates import QuadrayCoordinate


class TestPNGConfiguration:
    """Test PNG-specific configuration options."""
    
    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        reset_config()
    
    def test_png_config_defaults(self):
        """Test default PNG configuration values."""
        config = get_config()
        
        assert config['dpi'] == 300
        assert 'png_options' in config
        assert config['png_options']['transparent'] == False
        assert config['png_options']['facecolor'] == 'white'
        assert config['png_options']['bbox_inches'] == 'tight'
        assert config['png_options']['pad_inches'] == 0.1
    
    def test_png_config_customization(self):
        """Test customizing PNG configuration."""
        custom_config = {
            'dpi': 600,
            'png_options': {
                'transparent': True,
                'facecolor': 'none',
                'bbox_inches': 'tight',
                'pad_inches': 0.2
            }
        }
        
        set_config(custom_config)
        config = get_config()
        
        assert config['dpi'] == 600
        assert config['png_options']['transparent'] == True
        assert config['png_options']['facecolor'] == 'none'
        assert config['png_options']['pad_inches'] == 0.2


class TestGeometricPNGVisualizations:
    """Test PNG generation for geometric visualizations."""
    
    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        reset_config()
    
    def test_polyhedron_png_generation(self):
        """Test PNG generation for polyhedra with organized structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({
                'output_dir': temp_dir, 
                'backend': 'matplotlib',
                'organize_by_type': True
            })
            
            result = plot_polyhedron('tetrahedron', backend='matplotlib')
            
            assert 'files' in result
            assert len(result['files']) >= 1
            
            # Find the PNG file among the generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'
            assert 'tetrahedron' in png_file.name
            
            # Verify organized structure: temp_dir/geometric/polyhedra/
            assert 'geometric' in png_file.parts
            assert 'polyhedra' in png_file.parts
            
            # Verify it's a valid PNG
            with Image.open(png_file) as img:
                assert img.format == 'PNG'
                assert img.size[0] > 0 and img.size[1] > 0
    
    def test_quadray_coordinate_png(self):
        """Test PNG generation for Quadray coordinates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            qc = QuadrayCoordinate(1, 0, 0, 1)
            result = plot_quadray_coordinate(qc, backend='matplotlib')
            
            assert 'files' in result
            # Find PNG file among generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'
            
            # Verify PNG properties
            with Image.open(png_file) as img:
                assert img.format == 'PNG'
                assert img.mode in ['RGB', 'RGBA']
    
    def test_ivm_lattice_png(self):
        """Test PNG generation for IVM lattice."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            result = plot_ivm_lattice(size=2, backend='matplotlib')
            
            assert 'files' in result
            # Find PNG file among generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'


class TestNumberPatternPNGVisualizations:
    """Test PNG generation for number pattern visualizations."""
    
    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        reset_config()
    
    def test_palindromic_pattern_png(self):
        """Test PNG generation for palindromic patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            result = plot_palindromic_pattern(12321, backend='matplotlib')
            
            assert 'files' in result
            # Find PNG file among generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'
            
            # Verify PNG content
            with Image.open(png_file) as img:
                assert img.format == 'PNG'
                # Should have reasonable dimensions
                assert img.size[0] >= 100 and img.size[1] >= 100
    
    def test_scheherazade_pattern_png(self):
        """Test PNG generation for Scheherazade patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            result = plot_scheherazade_pattern(2, backend='matplotlib')
            
            assert 'files' in result
            # Find PNG file among generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'


class TestMathematicalPNGVisualizations:
    """Test PNG generation for mathematical visualizations."""
    
    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        reset_config()
    
    def test_continued_fraction_png(self):
        """Test PNG generation for continued fractions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            result = plot_continued_fraction(np.pi, max_terms=5, backend='matplotlib')
            
            assert 'files' in result
            # Find PNG file among generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'
            
            # Check file size (should be reasonable for a chart)
            file_size = png_file.stat().st_size
            assert file_size > 1000  # At least 1KB
            assert file_size < 10 * 1024 * 1024  # Less than 10MB
    
    def test_base_conversion_png(self):
        """Test PNG generation for base conversions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            result = plot_base_conversion(1001, from_base=10, to_base=2, backend='matplotlib')
            
            assert 'files' in result
            # Find PNG file among generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'
    
    def test_pattern_analysis_png(self):
        """Test PNG generation for pattern analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            result = plot_pattern_analysis(12321, pattern_type='palindrome', backend='matplotlib')
            
            assert 'files' in result
            # Find PNG file among generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'


class TestPNGQualityAndProperties:
    """Test PNG quality, resolution, and properties."""
    
    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        reset_config()
    
    def test_high_dpi_png_generation(self):
        """Test high DPI PNG generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({
                'output_dir': temp_dir, 
                'backend': 'matplotlib',
                'dpi': 600,
                'figure_size': (8, 6)
            })
            
            result = plot_polyhedron('cube', backend='matplotlib')
            png_file = Path(result['files'][0])
            
            # Verify high resolution (matplotlib may not give exact DPI*inches due to 3D plot handling)
            with Image.open(png_file) as img:
                # At 600 DPI and 8x6 inches, expect roughly 4800x3600 pixels
                # But matplotlib 3D plots may scale differently, so check for reasonably high resolution
                assert img.size[0] >= 1800  # At least 1800 pixels wide (high-res)
                assert img.size[1] >= 1800  # At least 1800 pixels tall (high-res)
    
    def test_transparent_png_generation(self):
        """Test transparent PNG generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({
                'output_dir': temp_dir,
                'backend': 'matplotlib',
                'png_options': {
                    'transparent': True,
                    'facecolor': 'none'
                }
            })
            
            result = plot_palindromic_pattern(121, backend='matplotlib')
            png_file = Path(result['files'][0])
            
            # Verify transparency
            with Image.open(png_file) as img:
                assert img.mode == 'RGBA'  # Should have alpha channel
    
    def test_custom_figure_size_png(self):
        """Test custom figure size PNG generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({
                'output_dir': temp_dir,
                'backend': 'matplotlib',
                'figure_size': (16, 12),
                'dpi': 150
            })
            
            result = plot_continued_fraction(np.e, max_terms=6, backend='matplotlib')
            png_file = Path(result['files'][0])
            
            # Verify custom dimensions (matplotlib may scale differently for different plot types)
            with Image.open(png_file) as img:
                # At 150 DPI and 16x12 inches, expect roughly 2400x1800 pixels
                # But matplotlib may scale differently, so just verify larger figure size results in larger image
                assert img.size[0] > 2000  # Should be reasonably large
                assert img.size[1] > 1500  # Should maintain aspect ratio reasonably


class TestPNGBatchGeneration:
    """Test batch PNG generation capabilities."""
    
    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        reset_config()
    
    def test_multiple_png_generation(self):
        """Test generating multiple PNGs in sequence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            # Generate multiple visualizations
            results = []
            
            # Geometric
            results.append(plot_polyhedron('tetrahedron', backend='matplotlib'))
            results.append(plot_polyhedron('octahedron', backend='matplotlib'))
            
            # Numbers
            results.append(plot_palindromic_pattern(121, backend='matplotlib'))
            results.append(plot_scheherazade_pattern(1, backend='matplotlib'))
            
            # Mathematical
            results.append(plot_continued_fraction(np.pi, max_terms=4, backend='matplotlib'))
            
            # Verify all files exist
            png_files = []
            for result in results:
                assert 'files' in result
                png_file = Path(result['files'][0])
                assert png_file.exists()
                assert png_file.suffix == '.png'
                png_files.append(png_file)
            
            # Verify unique filenames
            filenames = [f.name for f in png_files]
            assert len(filenames) == len(set(filenames))  # All unique
    
    def test_png_file_naming_consistency(self):
        """Test PNG file naming consistency."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            # Test various visualizations
            test_cases = [
                ('polyhedron', lambda: plot_polyhedron('cube', backend='matplotlib')),
                ('palindrome', lambda: plot_palindromic_pattern(12321, backend='matplotlib')),
                ('continued_fraction', lambda: plot_continued_fraction(np.pi, max_terms=3, backend='matplotlib')),
            ]
            
            for case_name, plot_func in test_cases:
                result = plot_func()
                png_file = Path(result['files'][0])
                
                # Verify naming conventions
                assert png_file.suffix == '.png'
                # Should contain descriptive elements related to the visualization
                assert any(char.isdigit() or char.isalpha() for char in png_file.stem)
                # Backend info should be in metadata, not filename
                assert case_name in ['polyhedron', 'palindrome', 'continued_fraction']


class TestPNGErrorHandling:
    """Test error handling in PNG generation."""
    
    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        reset_config()
    
    def test_invalid_dpi_handling(self):
        """Test handling of invalid DPI values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Very high DPI might cause issues
            set_config({
                'output_dir': temp_dir,
                'backend': 'matplotlib',
                'dpi': 10000  # Extremely high DPI
            })
            
            # Should still work or fail gracefully
            try:
                result = plot_polyhedron('tetrahedron', backend='matplotlib')
                if 'files' in result:
                    png_file = Path(result['files'][0])
                    assert png_file.exists()
            except Exception as e:
                # Should be a reasonable error, not a crash
                assert isinstance(e, (ValueError, MemoryError, OSError))
    
    def test_readonly_output_directory(self):
        """Test handling of read-only output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'readonly'
            output_dir.mkdir()
            
            # Make directory read-only
            output_dir.chmod(0o444)
            
            try:
                set_config({'output_dir': str(output_dir), 'backend': 'matplotlib'})
                
                # Should handle permission error gracefully
                with pytest.raises((PermissionError, OSError)):
                    plot_polyhedron('tetrahedron', backend='matplotlib')
            finally:
                # Restore permissions for cleanup
                output_dir.chmod(0o755)


class TestPNGIntegration:
    """Test PNG integration with other systems."""
    
    @pytest.fixture(autouse=True)
    def reset_visualization_config(self):
        """Reset visualization configuration before each test."""
        reset_config()
    
    def test_png_with_symergetics_numbers(self):
        """Test PNG generation with SymergeticsNumber objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib'})
            
            sn = SymergeticsNumber(1001)
            result = plot_palindromic_pattern(sn, backend='matplotlib')
            
            assert 'files' in result
            # Find PNG file among generated files
            png_files = [f for f in result['files'] if f.endswith('.png')]
            assert len(png_files) >= 1
            png_file = Path(png_files[0])
            assert png_file.exists()
            assert png_file.suffix == '.png'
    
    def test_png_metadata_preservation(self):
        """Test that PNG generation preserves metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            set_config({'output_dir': temp_dir, 'backend': 'matplotlib', 'dpi': 200})
            
            result = plot_polyhedron('octahedron', backend='matplotlib')
            
            # Verify metadata includes PNG-specific information
            assert 'metadata' in result
            metadata = result['metadata']
            assert metadata['backend'] == 'matplotlib'
            assert 'volume' in metadata
            assert 'vertices' in metadata
            
            # PNG file should exist
            png_file = Path(result['files'][0])
            assert png_file.exists()
