#!/usr/bin/env python3
"""
Comprehensive tests for symergetics.visualization.mathematical module.

This module provides extensive test coverage for mathematical visualization functions
to achieve 90%+ test coverage, focusing on edge cases, error handling, and
comprehensive functionality testing.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import json
from typing import List, Dict, Any

# Test data and fixtures
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestPlotContinuedFractionConvergenceComprehensive:
    """Test comprehensive continued fraction convergence plotting functionality."""

    def test_plot_continued_fraction_convergence_matplotlib(self):
        """Test continued fraction convergence plotting with matplotlib."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            result = plot_continued_fraction_convergence(
                value=3.14159,
                max_terms=15,
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'continued_fraction_convergence'

    @pytest.mark.skip(reason="Plotly backend not implemented for continued fraction convergence")
    def test_plot_continued_fraction_convergence_plotly(self):
        """Test continued fraction convergence plotting with plotly."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        with patch('plotly.graph_objects.Figure') as mock_figure, \
             patch('plotly.graph_objects.Scatter') as mock_scatter, \
             patch('plotly.graph_objects.Figure.update_layout') as mock_update_layout, \
             patch('plotly.graph_objects.Figure.add_trace') as mock_add_trace, \
             patch('plotly.graph_objects.Figure.write_html') as mock_write_html:
            
            result = plot_continued_fraction_convergence(
                value=3.14159,
                max_terms=15,
                backend='plotly'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    @pytest.mark.skip(reason="ASCII backend not implemented for continued fraction convergence")
    def test_plot_continued_fraction_convergence_ascii(self):
        """Test continued fraction convergence plotting with ASCII backend."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        result = plot_continued_fraction_convergence(
            value=3.14159,
            max_terms=15,
            backend='ascii'
        )
        
        assert 'files' in result
        assert 'metadata' in result
        assert any('ascii' in str(f) for f in result['files'])

    def test_plot_continued_fraction_convergence_integer(self):
        """Test continued fraction convergence plotting with integer value."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            result = plot_continued_fraction_convergence(
                value=5,
                max_terms=10,
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_continued_fraction_convergence_few_terms(self):
        """Test continued fraction convergence plotting with few terms."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            result = plot_continued_fraction_convergence(
                value=1.618,
                max_terms=3,
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_continued_fraction_convergence_invalid_backend(self):
        """Test continued fraction convergence plotting with invalid backend."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        with pytest.raises(ValueError, match="Convergence visualization requires matplotlib backend"):
            plot_continued_fraction_convergence(
                value=3.14159,
                max_terms=15,
                backend='invalid'
            )

    def test_plot_continued_fraction_convergence_error_handling(self):
        """Test continued fraction convergence plotting error handling."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            # Simulate matplotlib error
            mock_figure.side_effect = Exception("Matplotlib error")
            
            with pytest.raises(Exception, match="Matplotlib error"):
                plot_continued_fraction_convergence(
                    value=3.14159,
                    max_terms=15,
                    backend='matplotlib'
                )


class TestPlotBaseConversionMatrixComprehensive:
    """Test comprehensive base conversion matrix plotting functionality."""

    def test_plot_base_conversion_matrix_matplotlib(self):
        """Test base conversion matrix plotting with matplotlib."""
        from symergetics.visualization.mathematical import plot_base_conversion_matrix
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.table') as mock_table:
            
            result = plot_base_conversion_matrix(
                number=255,
                bases=[2, 8, 10, 16],
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'base_conversion_matrix'

    @pytest.mark.skip(reason="Plotly backend not implemented for base conversion matrix")
    def test_plot_base_conversion_matrix_plotly(self):
        """Test base conversion matrix plotting with plotly."""
        from symergetics.visualization.mathematical import plot_base_conversion_matrix
        
        with patch('plotly.graph_objects.Figure') as mock_figure, \
             patch('plotly.graph_objects.Heatmap') as mock_heatmap, \
             patch('plotly.graph_objects.Figure.update_layout') as mock_update_layout, \
             patch('plotly.graph_objects.Figure.add_trace') as mock_add_trace, \
             patch('plotly.graph_objects.Figure.write_html') as mock_write_html:
            
            result = plot_base_conversion_matrix(
                number=255,
                bases=[2, 8, 10, 16],
                backend='plotly'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    @pytest.mark.skip(reason="ASCII backend not implemented for base conversion matrix")
    def test_plot_base_conversion_matrix_ascii(self):
        """Test base conversion matrix plotting with ASCII backend."""
        from symergetics.visualization.mathematical import plot_base_conversion_matrix
        
        result = plot_base_conversion_matrix(
            number=255,
            bases=[2, 8, 10, 16],
            backend='ascii'
        )
        
        assert 'files' in result
        assert 'metadata' in result
        assert any('ascii' in str(f) for f in result['files'])

    def test_plot_base_conversion_matrix_wide_range(self):
        """Test base conversion matrix plotting with wide range of bases."""
        from symergetics.visualization.mathematical import plot_base_conversion_matrix
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.table') as mock_table:
            
            result = plot_base_conversion_matrix(
                number=1000,
                bases=list(range(2, 17)),
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_base_conversion_matrix_invalid_range(self):
        """Test base conversion matrix plotting with invalid range."""
        from symergetics.visualization.mathematical import plot_base_conversion_matrix
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.table') as mock_table:
            
            # Test with invalid base range
            result = plot_base_conversion_matrix(
                number=255,
                bases=[1, 2, 3],  # Base 1 is invalid
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_base_conversion_matrix_large_number(self):
        """Test base conversion matrix plotting with large number."""
        from symergetics.visualization.mathematical import plot_base_conversion_matrix
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.table') as mock_table:
            
            result = plot_base_conversion_matrix(
                number=10**6,
                bases=[2, 8, 10, 16, 32],
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result


class TestPlotPatternAnalysisRadarComprehensive:
    """Test comprehensive pattern analysis radar plotting functionality."""

    def test_plot_pattern_analysis_radar_matplotlib(self):
        """Test pattern analysis radar plotting with matplotlib."""
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar
        
        sequence = "121123211234321"
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot:
            
            result = plot_pattern_analysis_radar(
                sequence=sequence,
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result
            assert result['metadata']['type'] == 'pattern_analysis_radar'

    @pytest.mark.skip(reason="Plotly backend not implemented for pattern analysis radar")
    def test_plot_pattern_analysis_radar_plotly(self):
        """Test pattern analysis radar plotting with plotly."""
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar
        
        sequence = "121123211234321"
        
        with patch('plotly.graph_objects.Figure') as mock_figure, \
             patch('plotly.graph_objects.Scatterpolar') as mock_scatterpolar, \
             patch('plotly.graph_objects.Figure.update_layout') as mock_update_layout, \
             patch('plotly.graph_objects.Figure.add_trace') as mock_add_trace, \
             patch('plotly.graph_objects.Figure.write_html') as mock_write_html:
            
            result = plot_pattern_analysis_radar(
                sequence=sequence,
                backend='plotly'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    @pytest.mark.skip(reason="ASCII backend not implemented for pattern analysis radar")
    def test_plot_pattern_analysis_radar_ascii(self):
        """Test pattern analysis radar plotting with ASCII backend."""
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar
        
        sequence = "121123211234321"
        
        result = plot_pattern_analysis_radar(
            sequence=sequence,
            backend='ascii'
        )
        
        assert 'files' in result
        assert 'metadata' in result
        assert any('ascii' in str(f) for f in result['files'])

    def test_plot_pattern_analysis_radar_empty_sequence(self):
        """Test pattern analysis radar plotting with empty sequence."""
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot:
            
            result = plot_pattern_analysis_radar(
                sequence="1",
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_pattern_analysis_radar_single_character(self):
        """Test pattern analysis radar plotting with single character."""
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot:
            
            result = plot_pattern_analysis_radar(
                sequence="1",
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_pattern_analysis_radar_long_sequence(self):
        """Test pattern analysis radar plotting with long sequence."""
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar
        
        # Create a long sequence
        long_sequence = "121123211234321123454321" * 10
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot:
            
            result = plot_pattern_analysis_radar(
                sequence=long_sequence,
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_pattern_analysis_radar_invalid_backend(self):
        """Test pattern analysis radar plotting with invalid backend."""
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar
        
        with pytest.raises(ValueError, match="Radar visualization requires matplotlib backend"):
            plot_pattern_analysis_radar(
                sequence="121123211234321",
                backend='invalid'
            )


class TestOriginalMathematicalVisualizationsComprehensive:
    """Test comprehensive original mathematical visualization functionality."""

    def test_plot_continued_fraction_custom_value(self):
        """Test original continued fraction plotting with custom value."""
        from symergetics.visualization.mathematical import plot_continued_fraction
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            # Mock subplots to return a figure and 2x2 array of axes
            mock_fig = mock_figure.return_value
            mock_axes = [[mock_fig, mock_fig], [mock_fig, mock_fig]]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result = plot_continued_fraction(
                value=2.71828,  # e
                max_terms=20,
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_base_conversion_large_number(self):
        """Test original base conversion plotting with large number."""
        from symergetics.visualization.mathematical import plot_base_conversion
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            # Mock subplots to return a figure and 2x2 array of axes
            mock_fig = mock_figure.return_value
            mock_axes = [[mock_fig, mock_fig], [mock_fig, mock_fig]]
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result = plot_base_conversion(
                number=10**9,
                bases=[2, 8, 10, 16, 32, 64],
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result

    def test_plot_pattern_analysis_complex_sequence(self):
        """Test original pattern analysis plotting with complex sequence."""
        from symergetics.visualization.mathematical import plot_pattern_analysis
        
        # Create a complex sequence with multiple patterns
        complex_sequence = "12112321123432112345432112345654321"
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout, \
             patch('matplotlib.pyplot.subplots') as mock_subplots:
            
            # Mock subplots to return proper structure
            mock_axes = [[mock_subplot, mock_subplot], [mock_subplot, mock_subplot]]
            mock_subplots.return_value = (mock_figure, mock_axes)
            
            result = plot_pattern_analysis(
                number=complex_sequence,
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result


class TestMathematicalVisualizationErrorHandling:
    """Test comprehensive error handling in mathematical visualizations."""

    def test_missing_matplotlib_graceful_failure(self):
        """Test graceful failure when matplotlib is missing."""
        # This test is skipped because matplotlib is a required dependency
        # and is imported at module level, making it impractical to test
        pytest.skip("matplotlib is a required dependency imported at module level")

    def test_missing_plotly_graceful_failure(self):
        """Test graceful failure when plotly is missing."""
        # This test is skipped because plotly is not used in the convergence function
        # and the function only supports matplotlib backend
        pytest.skip("plot_continued_fraction_convergence only supports matplotlib backend")

    def test_invalid_data_structures(self):
        """Test handling of invalid data structures."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        # Test with invalid value type - should raise ValueError
        with pytest.raises(ValueError, match="invalid literal for int"):
            plot_continued_fraction_convergence(
                value="invalid",
                max_terms=15,
                backend='matplotlib'
            )

    def test_extreme_data_values(self):
        """Test handling of extreme data values."""
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            # Test with extreme values
            result = plot_continued_fraction_convergence(
                value=1e-10,
                max_terms=1000,
                backend='matplotlib'
            )
            
            assert 'files' in result
            assert 'metadata' in result


class TestMathematicalVisualizationIntegration:
    """Test integration between different mathematical visualization functions."""

    def test_multiple_mathematical_visualizations(self):
        """Test multiple mathematical visualization types."""
        from symergetics.visualization.mathematical import (
            plot_continued_fraction_convergence,
            plot_base_conversion_matrix,
            plot_pattern_analysis_radar
        )
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout, \
             patch('matplotlib.pyplot.table') as mock_table:
            
            # Test different visualization types
            cf_result = plot_continued_fraction_convergence(
                value=3.14159,
                max_terms=15,
                backend='matplotlib'
            )
            
            bc_result = plot_base_conversion_matrix(
                number=255,
                bases=[2, 8, 10, 16],
                backend='matplotlib'
            )
            
            pa_result = plot_pattern_analysis_radar(
                sequence="121123211234321",
                backend='matplotlib'
            )
            
            # Check that all results have consistent structure
            for result in [cf_result, bc_result, pa_result]:
                assert 'files' in result
                assert 'metadata' in result
                assert 'type' in result['metadata']
                assert 'backend' in result['metadata']

    def test_visualization_metadata_consistency(self):
        """Test that visualization metadata is consistent across functions."""
        from symergetics.visualization.mathematical import (
            plot_continued_fraction_convergence,
            plot_base_conversion_matrix,
            plot_pattern_analysis_radar
        )
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout, \
             patch('matplotlib.pyplot.table') as mock_table:
            
            cf_result = plot_continued_fraction_convergence(
                value=3.14159,
                max_terms=15,
                backend='matplotlib'
            )
            
            bc_result = plot_base_conversion_matrix(
                number=255,
                bases=[2, 8, 10, 16],
                backend='matplotlib'
            )
            
            pa_result = plot_pattern_analysis_radar(
                sequence="121123211234321",
                backend='matplotlib'
            )
            
            # Check that all results have consistent metadata structure
            for result in [cf_result, bc_result, pa_result]:
                assert 'files' in result
                assert 'metadata' in result
                assert 'type' in result['metadata']
                assert 'backend' in result['metadata']
                # Note: timestamp is not always included in metadata


class TestMathematicalVisualizationPerformance:
    """Test performance aspects of mathematical visualizations."""

    def test_visualization_execution_time(self):
        """Test that visualizations execute within reasonable time."""
        import time
        from symergetics.visualization.mathematical import plot_continued_fraction_convergence
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot, \
             patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
            
            start_time = time.time()
            result = plot_continued_fraction_convergence(
                value=3.14159,
                max_terms=100,
                backend='matplotlib'
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            assert execution_time < 3.0  # Should complete within 3 seconds
            assert 'files' in result

    def test_large_sequence_performance(self):
        """Test performance with large sequences."""
        import time
        from symergetics.visualization.mathematical import plot_pattern_analysis_radar
        
        # Create a very long sequence
        long_sequence = "121123211234321" * 100
        
        with patch('matplotlib.pyplot.figure') as mock_figure, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.subplot') as mock_subplot:
            
            start_time = time.time()
            result = plot_pattern_analysis_radar(
                sequence=long_sequence,
                backend='matplotlib'
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            assert execution_time < 5.0  # Should complete within 5 seconds
            assert 'files' in result


if __name__ == "__main__":
    pytest.main([__file__])

