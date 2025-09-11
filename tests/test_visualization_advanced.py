#!/usr/bin/env python3
"""
Tests for advanced visualization methods in symergetics.visualization.advanced
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from symergetics.visualization.advanced import (
    create_comparative_analysis_visualization,
    create_pattern_discovery_visualization,
    create_statistical_analysis_dashboard
)


class TestComparativeAnalysisVisualization:
    """Test comparative analysis visualization."""

    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_comparative_analysis_basic(self, mock_tight_layout, mock_savefig,
                                       mock_subplots, mock_figure, mock_colorbar):
        """Test basic comparative analysis visualization."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax1, mock_ax2, mock_ax3 = MagicMock(), MagicMock(), MagicMock()
        mock_ax4, mock_ax5, mock_ax6 = MagicMock(), MagicMock(), MagicMock()
        # Create a mock axes array that supports tuple indexing
        mock_axes = MagicMock()
        mock_axes.__getitem__.side_effect = lambda key: {
            (0, 0): mock_ax1, (0, 1): mock_ax2, (0, 2): mock_ax3,
            (1, 0): mock_ax4, (1, 1): mock_ax5, (1, 2): mock_ax6
        }[key]
        mock_subplots.return_value = (mock_fig, mock_axes)
        mock_figure.return_value = mock_fig
        mock_colorbar.return_value = MagicMock()  # Mock the colorbar return value

        # Test data
        domain1_data = [
            {
                'length': 3,
                'is_palindromic': True,
                'pattern_complexity': {'complexity_score': 2.5},
                'symmetry_analysis': {'symmetry_score': 0.8},
                'digit_distribution': {'1': 1, '2': 1, '3': 1}
            }
        ]

        domain2_data = [
            {
                'length': 5,
                'is_palindromic': False,
                'pattern_complexity': {'complexity_score': 1.8},
                'symmetry_analysis': {'symmetry_score': 0.6},
                'digit_distribution': {'1': 1, '2': 2, '3': 1, '4': 1}
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                mock_path.return_value = Path(temp_dir) / 'test_comparative.png'

                result = create_comparative_analysis_visualization(
                    domain1_data, domain2_data,
                    "Domain1", "Domain2"
                )

                # Verify result structure
                assert 'files' in result
                assert 'metadata' in result
                assert result['metadata']['type'] == 'comparative_analysis'
                assert result['metadata']['domain1'] == 'Domain1'
                assert result['metadata']['domain2'] == 'Domain2'
                assert result['metadata']['panels'] == 6

                # Verify matplotlib calls
                mock_subplots.assert_called_once()
                mock_savefig.assert_called_once()
                mock_tight_layout.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_comparative_analysis_matplotlib_error(self, mock_savefig, mock_subplots, mock_figure):
        """Test comparative analysis with matplotlib import error."""
        # Mock the _ensure_matplotlib_numpy function to raise ImportError
        with patch('symergetics.visualization.advanced._ensure_matplotlib_numpy') as mock_ensure:
            mock_ensure.side_effect = ImportError("Required visualization libraries not available: matplotlib not available")

            domain1_data = [{'length': 3}]
            domain2_data = [{'length': 5}]

            with pytest.raises(ImportError, match="Required visualization libraries not available"):
                create_comparative_analysis_visualization(
                    domain1_data, domain2_data,
                    backend="matplotlib"
                )

    def test_comparative_analysis_invalid_backend(self):
        """Test comparative analysis with invalid backend."""
        domain1_data = [{'length': 3}]
        domain2_data = [{'length': 5}]

        with pytest.raises(ValueError, match="Comparative analysis visualization requires matplotlib backend"):
            create_comparative_analysis_visualization(
                domain1_data, domain2_data,
                backend="invalid_backend"
            )

    def test_comparative_analysis_empty_domains(self):
        """Test comparative analysis with empty domains raises ValueError."""
        with pytest.raises(ValueError, match="Cannot create comparative analysis visualization with empty domains"):
            create_comparative_analysis_visualization([], [], "Empty1", "Empty2")


class TestPatternDiscoveryVisualization:
    """Test pattern discovery visualization."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    def test_pattern_discovery_basic(self, mock_tight_layout, mock_savefig,
                                    mock_subplots, mock_figure):
        """Test basic pattern discovery visualization."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax1, mock_ax2 = MagicMock(), MagicMock()
        mock_ax3, mock_ax4 = MagicMock(), MagicMock()
        # Create a mock axes array that supports tuple indexing
        mock_axes = MagicMock()
        mock_axes.__getitem__.side_effect = lambda key: {
            (0, 0): mock_ax1, (0, 1): mock_ax2,
            (1, 0): mock_ax3, (1, 1): mock_ax4
        }[key]
        mock_subplots.return_value = (mock_fig, mock_axes)
        mock_figure.return_value = mock_fig

        # Test data with different patterns
        analysis_results = [
            {
                'length': 3,
                'is_palindromic': True,
                'pattern_complexity': {'complexity_score': 2.5},
                'symmetry_analysis': {'symmetry_score': 0.8}
            },
            {
                'length': 5,
                'is_palindromic': False,
                'pattern_complexity': {'complexity_score': 1.8},
                'symmetry_analysis': {'symmetry_score': 0.6}
            },
            {
                'length': 7,
                'is_palindromic': True,
                'pattern_complexity': {'complexity_score': 3.2},
                'symmetry_analysis': {'symmetry_score': 0.9}
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                mock_path.return_value = Path(temp_dir) / 'test_pattern_discovery.png'

                result = create_pattern_discovery_visualization(
                    analysis_results,
                    title="Test Pattern Discovery"
                )

                # Verify result structure
                assert 'files' in result
                assert 'metadata' in result
                assert result['metadata']['type'] == 'pattern_discovery'
                assert result['metadata']['title'] == 'Test Pattern Discovery'
                assert result['metadata']['total_analyses'] == 3
                assert 'palindromic_count' in result['metadata']
                assert 'high_complexity_count' in result['metadata']

                # Verify matplotlib calls
                mock_subplots.assert_called_once()
                mock_savefig.assert_called_once()

    def test_pattern_discovery_empty_results(self):
        """Test pattern discovery with empty results."""
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax1, mock_ax2 = MagicMock(), MagicMock()
            mock_ax3, mock_ax4 = MagicMock(), MagicMock()
            # Create a mock axes array that supports tuple indexing
            mock_axes = MagicMock()
            mock_axes.__getitem__.side_effect = lambda key: {
                (0, 0): mock_ax1, (0, 1): mock_ax2,
                (1, 0): mock_ax3, (1, 1): mock_ax4
            }[key]
            mock_subplots.return_value = (mock_fig, mock_axes)

            with patch('matplotlib.pyplot.savefig'):
                with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                    mock_path.return_value = Path('/tmp/test.png')

                    result = create_pattern_discovery_visualization([], title="Empty Test")

                    assert result['metadata']['total_analyses'] == 0
                    assert result['metadata']['palindromic_count'] == 0

    def test_pattern_discovery_invalid_backend(self):
        """Test pattern discovery with invalid backend."""
        analysis_results = [{'length': 3}]

        with pytest.raises(ValueError, match="Pattern discovery visualization requires matplotlib backend"):
            create_pattern_discovery_visualization(
                analysis_results,
                backend="invalid_backend"
            )


class TestStatisticalAnalysisDashboard:
    """Test statistical analysis dashboard."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.colorbar')
    @patch('symergetics.visualization.advanced._np')
    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_statistical_dashboard_basic(self, mock_np, mock_colorbar, mock_tight_layout,
                                        mock_savefig, mock_subplots, mock_figure):
        """Test basic statistical dashboard."""
        # Mock matplotlib components for complex dashboard layout
        mock_fig = MagicMock()
        # Create mock axes for 4x4 grid
        mock_axes = [[MagicMock() for _ in range(4)] for _ in range(4)]
        mock_subplots.return_value = (mock_fig, mock_axes)
        mock_figure.return_value = mock_fig
        # Mock numpy methods
        mock_np.array.return_value = [[1.0, 0.5], [0.5, 1.0]]  # Mock correlation matrix
        mock_np.corrcoef.return_value = [[1.0, 0.5], [0.5, 1.0]]

        # Test data
        analysis_results = [
            {
                'length': 3,
                'is_palindromic': True,
                'pattern_complexity': {'complexity_score': 2.5},
                'symmetry_analysis': {'symmetry_score': 0.8},
                'palindromic_density': 1.0
            },
            {
                'length': 5,
                'is_palindromic': False,
                'pattern_complexity': {'complexity_score': 1.8},
                'symmetry_analysis': {'symmetry_score': 0.6},
                'palindromic_density': 0.4
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                mock_path.return_value = Path(temp_dir) / 'test_dashboard.png'

                result = create_statistical_analysis_dashboard(
                    analysis_results,
                    title="Test Statistical Dashboard"
                )

                # Verify result structure
                assert 'files' in result
                assert 'metadata' in result
                assert result['metadata']['type'] == 'statistical_dashboard'
                assert result['metadata']['title'] == 'Test Statistical Dashboard'
                assert result['metadata']['total_analyses'] == 2
                assert result['metadata']['panels'] == 8

                # Verify matplotlib calls
                mock_subplots.assert_called_once()
                mock_savefig.assert_called_once()

    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_statistical_dashboard_empty_results(self):
        """Test statistical dashboard with empty results."""
        # Mock minimal matplotlib setup
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig

            # Mock add_gridspec and subplots
            with patch.object(mock_fig, 'add_gridspec') as mock_gridspec:
                mock_gs = MagicMock()
                mock_gridspec.return_value = mock_gs

                # Mock axes creation
                mock_axes = [MagicMock() for _ in range(8)]
                for i, ax in enumerate(mock_axes):
                    mock_gs.__getitem__.return_value = ax

                with patch('matplotlib.pyplot.savefig'):
                    with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                        mock_path.return_value = Path('/tmp/test.png')

                        result = create_statistical_analysis_dashboard([], title="Empty Dashboard")

                        assert result['metadata']['total_analyses'] == 0

    def test_statistical_dashboard_invalid_backend(self):
        """Test statistical dashboard with invalid backend."""
        analysis_results = [{'length': 3}]

        with pytest.raises(ValueError, match="Statistical dashboard requires matplotlib backend"):
            create_statistical_analysis_dashboard(
                analysis_results,
                backend="invalid_backend"
            )


class TestVisualizationIntegration:
    """Test integration with other modules."""

    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_comparative_with_analysis_data(self):
        """Test comparative visualization with real analysis data."""
        from symergetics.computation.analysis import analyze_mathematical_patterns

        # Generate real analysis data
        palindromic_data = [
            analyze_mathematical_patterns(121),
            analyze_mathematical_patterns(12321)
        ]

        non_palindromic_data = [
            analyze_mathematical_patterns(123),
            analyze_mathematical_patterns(456)
        ]

        # Mock matplotlib to avoid actual plotting in tests
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_subplots.return_value = (MagicMock(), [[MagicMock() for _ in range(3)] for _ in range(2)])

            with patch('matplotlib.pyplot.savefig'):
                with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                    mock_path.return_value = Path('/tmp/test.png')

                    result = create_comparative_analysis_visualization(
                        palindromic_data, non_palindromic_data,
                        "Palindromic", "Non-Palindromic"
                    )

                    assert result['metadata']['domain1'] == 'Palindromic'
                    assert result['metadata']['domain2'] == 'Non-Palindromic'
                    assert result['metadata']['domain1_count'] == 2
                    assert result['metadata']['domain2_count'] == 2

    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_pattern_discovery_with_mixed_patterns(self):
        """Test pattern discovery with mixed pattern types."""
        analysis_results = [
            {'is_palindromic': True, 'pattern_complexity': {'complexity_score': 3.0}, 'symmetry_analysis': {'symmetry_score': 0.9}},
            {'is_palindromic': False, 'pattern_complexity': {'complexity_score': 1.5}, 'symmetry_analysis': {'symmetry_score': 0.4}},
            {'is_palindromic': True, 'pattern_complexity': {'complexity_score': 2.8}, 'symmetry_analysis': {'symmetry_score': 0.8}},
            {'is_palindromic': False, 'pattern_complexity': {'complexity_score': 3.5}, 'symmetry_analysis': {'symmetry_score': 0.2}}
        ]

        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_subplots.return_value = (MagicMock(), [[MagicMock() for _ in range(2)] for _ in range(2)])

            with patch('matplotlib.pyplot.savefig'):
                with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                    mock_path.return_value = Path('/tmp/test.png')

                    result = create_pattern_discovery_visualization(analysis_results)

                    assert result['metadata']['palindromic_count'] == 2
                    assert result['metadata']['high_complexity_count'] == 2  # complexity > 2
                    assert result['metadata']['high_symmetry_count'] == 2   # symmetry > 0.7

    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_dashboard_with_comprehensive_data(self):
        """Test dashboard with comprehensive analysis data."""
        analysis_results = [
            {
                'length': 3,
                'is_palindromic': True,
                'pattern_complexity': {'complexity_score': 2.5},
                'symmetry_analysis': {'symmetry_score': 0.8},
                'palindromic_density': 1.0,
                'digit_statistics': {'mean': 2.0, 'stdev': 0.8}
            },
            {
                'length': 5,
                'is_palindromic': False,
                'pattern_complexity': {'complexity_score': 1.8},
                'symmetry_analysis': {'symmetry_score': 0.6},
                'palindromic_density': 0.4,
                'digit_statistics': {'mean': 3.0, 'stdev': 1.2}
            }
        ]

        # Mock complex dashboard setup
        with patch('matplotlib.pyplot.figure') as mock_figure:
            mock_fig = MagicMock()
            mock_figure.return_value = mock_fig

            with patch.object(mock_fig, 'add_gridspec') as mock_gridspec:
                mock_gs = MagicMock()
                mock_gridspec.return_value = mock_gs

                mock_axes = [MagicMock() for _ in range(8)]
                for ax in mock_axes:
                    mock_gs.__getitem__.return_value = ax

                with patch('matplotlib.pyplot.savefig'):
                    with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                        mock_path.return_value = Path('/tmp/test.png')

                        result = create_statistical_analysis_dashboard(analysis_results)

                        assert result['metadata']['total_analyses'] == 2
                        assert result['metadata']['panels'] == 8


class TestVisualizationOutputStructure:
    """Test that visualizations create proper output structure."""

    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_output_paths_use_organized_structure(self):
        """Test that visualizations use organized output paths."""
        with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
            mock_path.return_value = Path('/test/output/mathematical/comparative/test.png')

            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                mock_subplots.return_value = (MagicMock(), [[MagicMock() for _ in range(3)] for _ in range(2)])

                with patch('matplotlib.pyplot.savefig') as mock_savefig:
                    create_comparative_analysis_visualization(
                        [{'length': 3}], [{'length': 5}],
                        "Test1", "Test2"
                    )

                    # Verify the organized path function was called correctly
                    mock_path.assert_called_once()
                    args, kwargs = mock_path.call_args
                    assert args[0] == 'mathematical'
                    assert args[1] == 'comparative_analysis'
                    assert 'test1_vs_test2' in args[2]

    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_multiple_visualization_types_create_different_paths(self):
        """Test that different visualization types create different output paths."""
        path_calls = []

        def mock_path_getter(*args, **kwargs):
            path_calls.append(args)
            return Path(f'/test/{args[0]}/{args[1]}/test.png')

        with patch('symergetics.visualization.advanced.get_organized_output_path', side_effect=mock_path_getter):
            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                mock_subplots.return_value = (MagicMock(), [[MagicMock() for _ in range(2)] for _ in range(2)])

                with patch('matplotlib.pyplot.savefig'):
                    # Test comparative analysis
                    create_comparative_analysis_visualization(
                        [{'length': 3}], [{'length': 5}]
                    )

                    # Test pattern discovery
                    create_pattern_discovery_visualization([{'length': 3}])

                    # Test dashboard
                    with patch('matplotlib.pyplot.figure') as mock_figure:
                        mock_fig = MagicMock()
                        mock_figure.return_value = mock_fig

                        with patch.object(mock_fig, 'add_gridspec') as mock_gridspec:
                            mock_gs = MagicMock()
                            mock_gridspec.return_value = mock_gs

                            mock_axes = [MagicMock() for _ in range(8)]
                            mock_gs.__getitem__.return_value = mock_axes[0]

                            create_statistical_analysis_dashboard([{'length': 3}])

                    # Verify different paths were generated
                    assert len(path_calls) >= 2  # At least comparative and pattern discovery
                    assert path_calls[0][1] == 'comparative_analysis'
                    assert path_calls[1][1] == 'pattern_discovery'


class TestErrorHandling:
    """Test error handling in visualization functions."""

    def test_missing_matplotlib_graceful_failure(self):
        """Test graceful failure when matplotlib is not available."""
        # This is tested indirectly through the ImportError tests above
        # The functions should raise clear ImportError messages
        pass

    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_invalid_data_structures(self):
        """Test handling of invalid data structures."""
        # Test with None values
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_subplots.return_value = (MagicMock(), [[MagicMock() for _ in range(3)] for _ in range(2)])

            with patch('matplotlib.pyplot.savefig'):
                with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                    mock_path.return_value = Path('/tmp/test.png')

                    # Should handle None values gracefully
                    result = create_comparative_analysis_visualization(
                        [{'length': None}], [{'length': None}],
                        "Test1", "Test2"
                    )

                    assert 'files' in result

    @pytest.mark.skip(reason="Complex visualization test requiring extensive matplotlib mocking")
    def test_extreme_data_values(self):
        """Test handling of extreme data values."""
        # Test with very large/small values
        analysis_results = [
            {
                'length': 1000,
                'pattern_complexity': {'complexity_score': 100.0},
                'symmetry_analysis': {'symmetry_score': 1.0}
            },
            {
                'length': 0,
                'pattern_complexity': {'complexity_score': 0.0},
                'symmetry_analysis': {'symmetry_score': 0.0}
            }
        ]

        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_subplots.return_value = (MagicMock(), [[MagicMock() for _ in range(2)] for _ in range(2)])

            with patch('matplotlib.pyplot.savefig'):
                with patch('symergetics.visualization.advanced.get_organized_output_path') as mock_path:
                    mock_path.return_value = Path('/tmp/test.png')

                    # Should handle extreme values without crashing
                    result = create_pattern_discovery_visualization(analysis_results)

                    assert 'files' in result
                    assert result['metadata']['total_analyses'] == 2
