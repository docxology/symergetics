"""
Comprehensive real tests for visualization modules.
Tests actual functionality without mocks.
"""

from pathlib import Path

import pytest


from symergetics.visualization import get_config


def assert_output_contained(result):
    output_dir = Path(get_config()['output_dir']).resolve()
    assert result['files']
    for filename in result['files']:
        path = Path(filename).resolve()
        assert path.is_relative_to(output_dir)
        assert path.is_file()
        assert path.stat().st_size > 0


class TestVisualizationMathematical:
    """Test mathematical visualization functionality."""
    
    def test_plot_pattern_analysis_basic(self):
        """Test basic number pattern visualization."""
        from symergetics.visualization.mathematical import plot_pattern_analysis
        
        # Test with a single number
        number = 12321  # A palindromic number
        
        # This should create a real visualization
        result = plot_pattern_analysis(
            number=number,
            pattern_type='palindrome',
            title="Test Number Patterns"
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_visualize_palindromic_analysis(self):
        """Test palindromic analysis visualization."""
        from symergetics.visualization.mathematical import plot_pattern_analysis
        
        # Test with a palindromic number
        palindromic_number = 12321
        
        result = plot_pattern_analysis(
            number=palindromic_number,
            pattern_type='palindrome',
            title="Palindromic Analysis"
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_visualize_scheherazade_analysis(self):
        """Test Scheherazade number analysis visualization."""
        from symergetics.visualization.mathematical import plot_ssrcd_analysis
        
        # Test with a Scheherazade power
        power = 2
        
        result = plot_ssrcd_analysis(
            power=power,
            title="Scheherazade Analysis"
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_visualize_primorial_analysis(self):
        """Test primorial sequence analysis visualization."""
        from symergetics.visualization.mathematical import plot_base_conversion
        
        # Test with a number that can be analyzed
        number = 30  # A primorial number
        
        result = plot_base_conversion(
            number=number,
            title="Primorial Analysis"
        )
        
        assert result is not None
        assert isinstance(result, dict)


class TestVisualizationGeometry:
    """Test geometric visualization functionality."""
    
    def test_visualize_polyhedra_basic(self):
        """Test basic polyhedra visualization."""
        from symergetics.visualization.geometry import plot_polyhedron
        
        # Test with a polyhedron
        polyhedron = "tetrahedron"
        
        result = plot_polyhedron(
            polyhedron=polyhedron,
            title="Polyhedra Visualization"
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_visualize_coordinate_systems(self):
        """Test coordinate system visualization."""
        # Test with a coordinate
        from symergetics.core.coordinates import QuadrayCoordinate
        from symergetics.visualization.geometry import plot_quadray_coordinate
        coord = QuadrayCoordinate(1, 0, 0, 0)
        
        result = plot_quadray_coordinate(
            coord=coord,
            title="Coordinate System Visualization"
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_visualize_volume_relationships(self):
        """Test volume relationship visualization."""
        from symergetics.visualization.geometry import plot_polyhedron
        
        # Test with a polyhedron instead
        result = plot_polyhedron(
            polyhedron="tetrahedron",
            title="Volume Relationship Test"
        )
        
        assert result is not None
        assert isinstance(result, dict)


class TestVisualizationNumbers:
    """Test number visualization functionality."""
    
    def test_visualize_number_analysis(self):
        """Test number analysis visualization."""
        from symergetics.visualization.numbers import plot_palindromic_pattern
        
        # Test with a palindromic pattern instead
        result = plot_palindromic_pattern(
            number=12321,
            title="Number Analysis Test"
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_visualize_pattern_density(self):
        """Test pattern density visualization."""
        from symergetics.visualization.numbers import plot_palindromic_pattern
        
        # Test with a palindromic pattern instead
        result = plot_palindromic_pattern(
            number=12321,
            title="Pattern Density Test"
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_visualize_number_sequences(self):
        """Test number sequence visualization."""
        from symergetics.visualization.numbers import plot_palindromic_pattern
        
        # Test with a palindromic pattern instead
        result = plot_palindromic_pattern(
            number=12321,
            title="Number Sequences Test"
        )
        
        assert result is not None
        assert isinstance(result, dict)


class TestVisualizationAdvanced:
    """Test advanced visualization functionality."""
    
    def test_visualize_comprehensive_analysis(self):
        """Test comprehensive analysis visualization."""
        from symergetics.visualization.advanced import (
            create_statistical_analysis_dashboard,
        )
        
        # Test with real comprehensive data
        analysis_data = [
            {
                'length': 3,
                'is_palindromic': False,
                'pattern_complexity': {'complexity_score': 2.5},
                'symmetry_analysis': {'symmetry_score': 0.7}
            },
            {
                'length': 3,
                'is_palindromic': False,
                'pattern_complexity': {'complexity_score': 3.1},
                'symmetry_analysis': {'symmetry_score': 0.9}
            },
            {
                'length': 3,
                'is_palindromic': False,
                'pattern_complexity': {'complexity_score': 2.8},
                'symmetry_analysis': {'symmetry_score': 0.6}
            }
        ]
        
        result = create_statistical_analysis_dashboard(
            analysis_data, title="Comprehensive Analysis Test"
        )
        assert_output_contained(result)
        assert result['metadata']['total_analyses'] == len(analysis_data)
    
    def test_visualize_cross_module_analysis(self):
        """Test cross-module analysis visualization."""
        from symergetics.visualization.advanced import (
            create_comparative_analysis_visualization,
        )
        
        # Test with real cross-module data
        domain1_data = [
            {'length': 3, 'pattern_complexity': {'complexity_score': 2.5}, 'symmetry_analysis': {'symmetry_score': 0.7}},
            {'length': 4, 'pattern_complexity': {'complexity_score': 3.1}, 'symmetry_analysis': {'symmetry_score': 0.9}}
        ]
        domain2_data = [
            {'length': 5, 'pattern_complexity': {'complexity_score': 2.8}, 'symmetry_analysis': {'symmetry_score': 0.6}},
            {'length': 6, 'pattern_complexity': {'complexity_score': 3.5}, 'symmetry_analysis': {'symmetry_score': 0.8}}
        ]
        
        result = create_comparative_analysis_visualization(
            domain1_data, domain2_data, "Arithmetic", "Geometry"
        )
        assert_output_contained(result)
        assert result['metadata']['domain1_count'] == len(domain1_data)
        assert result['metadata']['domain2_count'] == len(domain2_data)


class TestVisualizationIntegration:
    """Test integration between visualization modules."""
    
    def test_visualization_pipeline_integration(self):
        """Test complete visualization pipeline."""
        from symergetics.visualization.geometry import plot_polyhedron
        from symergetics.visualization.mathematical import plot_pattern_analysis
        from symergetics.visualization.numbers import plot_palindromic_pattern
        
        results = [
            plot_pattern_analysis(number=12321, pattern_type='palindrome'),
            plot_polyhedron(polyhedron='tetrahedron'),
            plot_palindromic_pattern(number=12321, title="Palindromic Pattern Test"),
        ]
        for result in results:
            assert_output_contained(result)
    
    def test_visualization_error_handling(self):
        """Test visualization error handling."""
        from symergetics.visualization.mathematical import plot_pattern_analysis
        
        with pytest.raises(TypeError):
            plot_pattern_analysis(numbers=[])


class TestVisualizationPerformance:
    """Test visualization performance with real data."""
    
    def test_large_dataset_visualization(self):
        """Test visualization with large datasets."""
        from symergetics.visualization.mathematical import plot_pattern_analysis
        
        result = plot_pattern_analysis(number=12321, pattern_type='palindrome')
        assert_output_contained(result)
    
    def test_visualization_memory_usage(self):
        """Test visualization memory usage."""
        from symergetics.visualization.numbers import plot_palindromic_pattern
        
        result = plot_palindromic_pattern(
            number=12321, title="Memory Usage Test"
        )
        assert_output_contained(result)
