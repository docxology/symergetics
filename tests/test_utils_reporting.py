#!/usr/bin/env python3
"""
Tests for reporting methods in symergetics.utils.reporting
"""

import pytest
import json
import csv
import os
import tempfile
from pathlib import Path
from symergetics.utils.reporting import (
    generate_statistical_summary,
    generate_comparative_report,
    export_report_to_json,
    export_report_to_csv,
    export_report_to_markdown,
    generate_performance_report,
    ReportMetrics,
    AnalysisSummary
)


class TestStatisticalSummary:
    """Test statistical summary generation."""

    def test_generate_basic_summary(self):
        """Test basic statistical summary generation."""
        analysis_results = [
            {
                'length': 3,
                'is_palindromic': True,
                'palindromic_density': 1.0,
                'pattern_complexity': {'complexity_score': 2.5},
                'symmetry_analysis': {'symmetry_score': 0.8}
            },
            {
                'length': 5,
                'is_palindromic': False,
                'palindromic_density': 0.4,
                'pattern_complexity': {'complexity_score': 1.8},
                'symmetry_analysis': {'symmetry_score': 0.6}
            }
        ]

        summary = generate_statistical_summary(analysis_results)

        assert 'title' in summary
        assert 'timestamp' in summary
        assert summary['total_analyses'] == 2
        assert summary['palindromic_count'] == 1
        assert summary['palindromic_ratio'] == 0.5
        assert 'metrics' in summary
        assert 'insights' in summary

    def test_generate_empty_summary(self):
        """Test summary generation with empty results."""
        summary = generate_statistical_summary([])

        assert summary['summary'] == 'No analysis results provided'
        assert summary['total_analyses'] == 0

    def test_generate_summary_with_missing_data(self):
        """Test summary with missing data fields."""
        analysis_results = [
            {'length': 3, 'is_palindromic': True},
            {'is_palindromic': False}
        ]

        summary = generate_statistical_summary(analysis_results)

        assert summary['total_analyses'] == 2
        assert 'metrics' in summary

    def test_generate_summary_insights(self):
        """Test insights generation in summary."""
        # High palindromic ratio
        high_palindromic = [
            {'is_palindromic': True, 'length': 3},
            {'is_palindromic': True, 'length': 3}
        ]

        summary = generate_statistical_summary(high_palindromic)
        insights_str = ' '.join(summary['insights'])

        assert 'High palindromic ratio' in insights_str or 'high' in insights_str.lower()

    def test_generate_summary_custom_title(self):
        """Test summary with custom title."""
        analysis_results = [{'is_palindromic': True, 'length': 3}]
        custom_title = "Custom Statistical Summary"

        summary = generate_statistical_summary(analysis_results, custom_title)

        assert summary['title'] == custom_title


class TestComparativeReporting:
    """Test comparative report generation."""

    def test_generate_comparative_report(self):
        """Test basic comparative report generation."""
        domain1_results = [
            {
                'length': 3,
                'is_palindromic': True,
                'palindromic_density': 1.0,
                'pattern_complexity': {'complexity_score': 2.5},
                'symmetry_analysis': {'symmetry_score': 0.8}
            }
        ]

        domain2_results = [
            {
                'length': 5,
                'is_palindromic': False,
                'palindromic_density': 0.4,
                'pattern_complexity': {'complexity_score': 1.8},
                'symmetry_analysis': {'symmetry_score': 0.6}
            }
        ]

        report = generate_comparative_report(domain1_results, domain2_results)

        assert 'title' in report
        assert 'timestamp' in report
        assert 'domains' in report
        assert 'comparisons' in report
        assert 'insights' in report
        assert 'domain_summaries' in report

    def test_generate_comparative_report_empty_domains(self):
        """Test comparative report with empty domains."""
        report = generate_comparative_report([], [], "Empty1", "Empty2")

        assert 'comparisons' in report
        assert 'insights' in report

    def test_generate_comparative_report_custom_names(self):
        """Test comparative report with custom domain names."""
        domain1_results = [{'is_palindromic': True, 'length': 3}]
        domain2_results = [{'is_palindromic': False, 'length': 3}]

        report = generate_comparative_report(
            domain1_results, domain2_results,
            "Palindromic Domain", "Non-Palindromic Domain"
        )

        assert report['domains']['domain1'] == "Palindromic Domain"
        assert report['domains']['domain2'] == "Non-Palindromic Domain"

    def test_generate_comparative_report_different_sizes(self):
        """Test comparative report with different domain sizes."""
        domain1_results = [{'is_palindromic': True, 'length': 3}]
        domain2_results = [
            {'is_palindromic': False, 'length': 3},
            {'is_palindromic': True, 'length': 5}
        ]

        report = generate_comparative_report(domain1_results, domain2_results)

        assert 'comparisons' in report


class TestReportExport:
    """Test report export functionality."""

    def test_export_json_basic(self):
        """Test basic JSON export."""
        report_data = {
            'title': 'Test Report',
            'timestamp': '2024-01-01T00:00:00Z',
            'summary': {'total': 5}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            export_report_to_json(report_data, temp_path)

            # Verify file was created and contains valid JSON
            assert os.path.exists(temp_path)

            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data['title'] == 'Test Report'
            assert loaded_data['summary']['total'] == 5

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_csv_basic(self):
        """Test basic CSV export."""
        report_data = {
            'title': 'Test Report',
            'summary': {'total': 5, 'average': 2.5},
            'nested': {'data': {'value': 10}}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            export_report_to_csv(report_data, temp_path)

            # Verify file was created
            assert os.path.exists(temp_path)

            # Verify CSV content
            with open(temp_path, 'r') as f:
                content = f.read()

            assert 'Key' in content
            assert 'Value' in content
            assert 'title' in content
            assert 'Test Report' in content

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_markdown_basic(self):
        """Test basic Markdown export."""
        report_data = {
            'title': 'Test Report',
            'timestamp': '2024-01-01T00:00:00Z',
            'summary': {'total': 5},
            'insights': ['Test insight 1', 'Test insight 2']
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = f.name

        try:
            export_report_to_markdown(report_data, temp_path)

            # Verify file was created
            assert os.path.exists(temp_path)

            # Verify Markdown content
            with open(temp_path, 'r') as f:
                content = f.read()

            assert '# Test Report' in content
            assert '**Generated:**' in content
            assert '## Key Insights' in content
            assert 'Test insight 1' in content

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_with_pathlib_path(self):
        """Test export with pathlib.Path objects."""
        report_data = {'title': 'Test Report'}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            path_obj = Path(temp_path)
            export_report_to_json(report_data, path_obj)

            assert path_obj.exists()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_export_creates_directories(self):
        """Test that export creates necessary directories."""
        report_data = {'title': 'Test Report'}

        # Create a path with non-existent directories
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / 'nested' / 'dir' / 'report.json'

            export_report_to_json(report_data, nested_path)

            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_export_json_with_nested_data(self):
        """Test JSON export with deeply nested data."""
        report_data = {
            'title': 'Nested Test',
            'data': {
                'level1': {
                    'level2': {
                        'value': 42,
                        'list': [1, 2, 3]
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            export_report_to_json(report_data, temp_path, indent=2)

            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data['data']['level1']['level2']['value'] == 42
            assert loaded_data['data']['level1']['level2']['list'] == [1, 2, 3]

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestPerformanceReporting:
    """Test performance report generation."""

    def test_generate_performance_report_basic(self):
        """Test basic performance report generation."""
        analysis_results = [
            {'length': 3, 'is_palindromic': True},
            {'length': 5, 'is_palindromic': False}
        ]
        execution_times = [0.1, 0.2]

        report = generate_performance_report(analysis_results, execution_times)

        assert 'title' in report
        assert 'timestamp' in report
        assert 'performance_metrics' in report
        assert report['performance_metrics']['total_analyses'] == 2
        assert 'execution_times' in report['performance_metrics']
        assert abs(report['performance_metrics']['execution_times']['average_time'] - 0.15) < 1e-10

    def test_generate_performance_report_no_times(self):
        """Test performance report without execution times."""
        analysis_results = [{'length': 3}, {'length': 5}]

        report = generate_performance_report(analysis_results)

        assert 'performance_metrics' in report
        assert 'total_analyses' in report['performance_metrics']
        assert 'execution_times' not in report['performance_metrics']

    def test_generate_performance_report_with_correlation(self):
        """Test performance report with complexity/time correlation."""
        analysis_results = [
            {'pattern_complexity': {'complexity_score': 1.0}, 'length': 3},
            {'pattern_complexity': {'complexity_score': 2.0}, 'length': 5},
            {'pattern_complexity': {'complexity_score': 3.0}, 'length': 7}
        ]
        execution_times = [0.1, 0.15, 0.25]  # Should show positive correlation

        report = generate_performance_report(analysis_results, execution_times)

        assert 'efficiency_analysis' in report
        assert 'complexity_time_correlation' in report['efficiency_analysis']

    def test_generate_performance_report_recommendations(self):
        """Test performance report recommendations."""
        analysis_results = [{'length': 3}]
        execution_times = [2.0]  # Slow execution

        report = generate_performance_report(analysis_results, execution_times)

        assert 'recommendations' in report
        assert len(report['recommendations']) > 0

    def test_generate_performance_report_throughput(self):
        """Test throughput calculation in performance report."""
        analysis_results = [{'length': 3}, {'length': 5}]
        execution_times = [0.1, 0.1]  # Fast execution

        report = generate_performance_report(analysis_results, execution_times)

        assert 'throughput' in report['performance_metrics']
        expected_throughput = 2 / 0.2  # 2 analyses in 0.2 total time
        assert report['performance_metrics']['throughput'] == expected_throughput


class TestDataClasses:
    """Test data classes for reporting."""

    def test_report_metrics_creation(self):
        """Test ReportMetrics dataclass creation."""
        metrics = ReportMetrics(
            total_items=100,
            successful_analyses=95,
            failed_analyses=5,
            average_complexity=2.3,
            average_symmetry=0.75,
            palindromic_ratio=0.6,
            generation_time=45.2
        )

        assert metrics.total_items == 100
        assert metrics.successful_analyses == 95
        assert metrics.failed_analyses == 5
        assert metrics.average_complexity == 2.3
        assert metrics.average_symmetry == 0.75
        assert metrics.palindromic_ratio == 0.6
        assert metrics.generation_time == 45.2

    def test_analysis_summary_creation(self):
        """Test AnalysisSummary dataclass creation."""
        summary = AnalysisSummary(
            domain="Test Domain",
            item_count=50,
            palindromic_count=30,
            average_complexity=2.1,
            average_symmetry=0.8,
            unique_patterns=25,
            timestamp="2024-01-01T00:00:00Z"
        )

        assert summary.domain == "Test Domain"
        assert summary.item_count == 50
        assert summary.palindromic_count == 30
        assert summary.average_complexity == 2.1
        assert summary.average_symmetry == 0.8
        assert summary.unique_patterns == 25
        assert summary.timestamp == "2024-01-01T00:00:00Z"


class TestIntegrationWithAnalysis:
    """Test integration with analysis modules."""

    def test_full_reporting_pipeline(self):
        """Test complete reporting pipeline."""
        from symergetics.computation.analysis import analyze_mathematical_patterns

        # Generate analysis data
        analysis_results = [
            analyze_mathematical_patterns(121),
            analyze_mathematical_patterns(123),
            analyze_mathematical_patterns(12321)
        ]

        # Generate statistical summary
        summary = generate_statistical_summary(analysis_results, "Integration Test")

        assert summary['total_analyses'] == 3
        assert 'metrics' in summary

        # Export to different formats
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / 'report.json'
            csv_path = Path(temp_dir) / 'report.csv'
            md_path = Path(temp_dir) / 'report.md'

            export_report_to_json(summary, json_path)
            export_report_to_csv(summary, csv_path)
            export_report_to_markdown(summary, md_path)

            assert json_path.exists()
            assert csv_path.exists()
            assert md_path.exists()

    def test_comparative_with_real_data(self):
        """Test comparative reporting with real analysis data."""
        from symergetics.computation.analysis import analyze_mathematical_patterns

        # Create two different domains
        palindromic_domain = [
            analyze_mathematical_patterns(121),
            analyze_mathematical_patterns(12321)
        ]

        mixed_domain = [
            analyze_mathematical_patterns(123),
            analyze_mathematical_patterns(456)
        ]

        # Generate comparative report
        report = generate_comparative_report(
            palindromic_domain, mixed_domain,
            "Palindromic", "Mixed"
        )

        assert 'comparisons' in report
        assert 'palindromic_ratio' in report['comparisons']

        # Palindromic domain should have higher palindromic ratio
        pal_ratio_comp = report['comparisons']['palindromic_ratio']
        assert pal_ratio_comp['Palindromic'] > pal_ratio_comp['Mixed']


class TestEdgeCases:
    """Test edge cases in reporting."""

    def test_empty_analysis_results(self):
        """Test reporting with completely empty analysis results."""
        empty_results = []

        summary = generate_statistical_summary(empty_results)
        assert summary['total_analyses'] == 0

        report = generate_performance_report(empty_results)
        assert report['performance_metrics']['total_analyses'] == 0

    def test_malformed_analysis_data(self):
        """Test reporting with malformed analysis data."""
        malformed_results = [
            {},  # Completely empty
            {'length': None},  # None values
            {'is_palindromic': 'not_boolean'}  # Wrong types
        ]

        # Should not crash
        summary = generate_statistical_summary(malformed_results)
        assert 'total_analyses' in summary

    def test_export_to_invalid_path(self):
        """Test export to invalid path."""
        report_data = {'title': 'Test'}

        # Should handle gracefully (create directories as needed)
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = Path(temp_dir) / 'nonexistent' / 'deep' / 'path' / 'report.json'

            # This should work due to directory creation in export functions
            export_report_to_json(report_data, invalid_path)
            assert invalid_path.exists()

    def test_very_large_report_data(self):
        """Test with very large report data."""
        large_data = {
            'title': 'Large Report',
            'data': [{'value': i, 'description': f'Item {i}'} for i in range(1000)]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            export_report_to_json(large_data, temp_path)

            # Verify it was written
            with open(temp_path, 'r') as f:
                loaded = json.load(f)

            assert len(loaded['data']) == 1000

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
