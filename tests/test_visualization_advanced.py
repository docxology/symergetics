"""Exercise advanced visualizations with actual isolated rendering and files."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import matplotlib.pyplot as plt
import numpy as np

from symergetics.visualization import advanced

from symergetics.visualization import get_config, set_config
from symergetics.visualization.advanced import (
    create_comparative_analysis_visualization,
    create_pattern_discovery_visualization,
    create_geometric_mnemonics_visualization,
    create_statistical_analysis_dashboard,
)


@pytest.fixture
def analysis_results():
    return [
        {
            'length': 3, 'is_palindromic': True,
            'pattern_complexity': {'complexity_score': 2.5},
            'symmetry_analysis': {'symmetry_score': 0.8},
            'palindromic_density': 1.0,
            'digit_distribution': {'1': 1, '2': 1, '3': 1},
        },
        {
            'length': 5, 'is_palindromic': False,
            'pattern_complexity': {'complexity_score': 1.8},
            'symmetry_analysis': {'symmetry_score': 0.6},
            'palindromic_density': 0.4,
            'digit_distribution': {'1': 1, '2': 2, '3': 1, '4': 1},
        },
        {
            'length': 7, 'is_palindromic': True,
            'pattern_complexity': {'complexity_score': 3.2},
            'symmetry_analysis': {'symmetry_score': 0.9},
            'palindromic_density': 0.8,
        },
        {
            'length': 4, 'is_palindromic': False,
            'pattern_complexity': {'complexity_score': 1.5},
            'symmetry_analysis': {'symmetry_score': 0.3},
            'palindromic_density': 0.2,
        },
    ]


def assert_rendered(result):
    output_dir = Path(get_config()['output_dir']).resolve()
    assert result['files']
    for filename in result['files']:
        path = Path(filename).resolve()
        assert path.is_relative_to(output_dir)
        assert path.is_file()
        assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def run_python(code):
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [repo_root, env.get('PYTHONPATH')]))
    return subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True,
        timeout=120, check=False, env=env,
    )


class TestComparativeAnalysisVisualization:
    def test_comparative_analysis_basic(self, analysis_results):
        result = create_comparative_analysis_visualization(
            analysis_results[:2], analysis_results[2:], 'Domain1', 'Domain2'
        )
        assert_rendered(result)
        assert result['metadata']['type'] == 'comparative_analysis'
        assert result['metadata']['domain1_count'] == 2
        assert result['metadata']['domain2_count'] == 2

    def test_comparative_analysis_invalid_backend(self):
        with pytest.raises(ValueError):
            create_comparative_analysis_visualization([{'length': 3}], [{'length': 5}], backend='invalid')

    def test_comparative_analysis_empty_domains(self):
        with pytest.raises(ValueError):
            create_comparative_analysis_visualization([], [], 'Empty1', 'Empty2')


class TestPatternDiscoveryVisualization:
    def test_pattern_discovery_basic(self, analysis_results):
        result = create_pattern_discovery_visualization(
            analysis_results[:3], title='Test Pattern Discovery'
        )
        assert_rendered(result)
        assert result['metadata']['total_analyses'] == 3
        assert result['metadata']['palindromic_count'] == 2
        assert result['metadata']['high_complexity_count'] == 2

    def test_pattern_discovery_empty_results(self):
        result = create_pattern_discovery_visualization([], title='Empty Test')
        assert_rendered(result)
        assert result['metadata']['total_analyses'] == 0
        assert result['metadata']['palindromic_count'] == 0

    def test_pattern_discovery_invalid_backend(self):
        with pytest.raises(ValueError):
            create_pattern_discovery_visualization([{'length': 3}], backend='invalid')


class TestStatisticalAnalysisDashboard:
    def test_statistical_dashboard_basic(self, analysis_results):
        result = create_statistical_analysis_dashboard(
            analysis_results, title='Test Statistical Dashboard'
        )
        assert_rendered(result)
        assert result['metadata']['type'] == 'statistical_dashboard'
        assert result['metadata']['total_analyses'] == 4

    def test_dashboard_renders_as_first_visualization_in_fresh_process(self):
        result = run_python('''
from pathlib import Path
from symergetics.visualization import set_config
from symergetics.visualization.advanced import create_statistical_analysis_dashboard
set_config({"dpi": 40})
result = create_statistical_analysis_dashboard([
    {"length": 3, "pattern_complexity": {"complexity_score": 2.5},
     "symmetry_analysis": {"symmetry_score": 0.8}, "palindromic_density": 0.5},
    {"length": 5, "pattern_complexity": {"complexity_score": 1.8},
     "symmetry_analysis": {"symmetry_score": 0.6}, "palindromic_density": 0.4},
])
assert result["files"]
for filename in result["files"]:
    path = Path(filename).resolve()
    assert path.is_relative_to(Path.cwd())
    assert path.read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))
''')
        assert result.returncode == 0, result.stdout + result.stderr

    def test_statistical_dashboard_empty_results(self):
        result = create_statistical_analysis_dashboard([], title='Empty Dashboard')
        assert_rendered(result)
        assert result['metadata']['total_analyses'] == 0

    def test_statistical_dashboard_invalid_backend(self):
        with pytest.raises(ValueError):
            create_statistical_analysis_dashboard([{'length': 3}], backend='invalid')


class TestVisualizationIntegration:
    def test_comparative_with_analysis_data(self):
        from symergetics.computation.analysis import analyze_mathematical_patterns

        palindromes = [analyze_mathematical_patterns(n) for n in (121, 12321)]
        non_palindromes = [analyze_mathematical_patterns(n) for n in (123, 456)]
        result = create_comparative_analysis_visualization(
            palindromes, non_palindromes, 'Palindromic', 'Non-Palindromic'
        )
        assert_rendered(result)
        assert result['metadata']['domain1_count'] == 2
        assert result['metadata']['domain2_count'] == 2

    def test_pattern_discovery_with_mixed_patterns(self):
        data = [
            {'is_palindromic': True, 'pattern_complexity': {'complexity_score': 3.0}, 'symmetry_analysis': {'symmetry_score': 0.9}},
            {'is_palindromic': False, 'pattern_complexity': {'complexity_score': 1.5}, 'symmetry_analysis': {'symmetry_score': 0.4}},
            {'is_palindromic': True, 'pattern_complexity': {'complexity_score': 2.8}, 'symmetry_analysis': {'symmetry_score': 0.8}},
            {'is_palindromic': False, 'pattern_complexity': {'complexity_score': 3.5}, 'symmetry_analysis': {'symmetry_score': 0.2}},
        ]
        result = create_pattern_discovery_visualization(data)
        assert_rendered(result)
        assert result['metadata']['palindromic_count'] == 2
        assert result['metadata']['high_complexity_count'] == 3
        assert result['metadata']['high_symmetry_count'] == 2

    def test_dashboard_with_comprehensive_data(self, analysis_results):
        data = analysis_results[:2]
        data[0]['digit_statistics'] = {'mean': 2.0, 'stdev': 0.8}
        data[1]['digit_statistics'] = {'mean': 3.0, 'stdev': 1.2}
        result = create_statistical_analysis_dashboard(data)
        assert_rendered(result)
        assert result['metadata']['total_analyses'] == 2


class TestVisualizationOutputStructure:
    def test_output_paths_use_organized_structure(self, tmp_path, analysis_results):
        output_dir = tmp_path / 'configured-output'
        set_config({'output_dir': str(output_dir)})
        result = create_comparative_analysis_visualization(
            analysis_results[:2], analysis_results[2:], 'Test1', 'Test2'
        )
        assert_rendered(result)
        assert Path(result['files'][0]).parent == output_dir / 'mathematical' / 'comparative_analysis'

    def test_multiple_visualization_types_create_different_paths(self, analysis_results):
        results = [
            create_comparative_analysis_visualization(analysis_results[:2], analysis_results[2:]),
            create_pattern_discovery_visualization(analysis_results),
            create_statistical_analysis_dashboard(analysis_results),
        ]
        for result in results:
            assert_rendered(result)
        assert len({Path(result['files'][0]).resolve() for result in results}) == 3


class TestErrorHandling:
    @pytest.mark.parametrize('call', [
        'create_comparative_analysis_visualization([{"length": 3}], [{"length": 5}])',
        'create_pattern_discovery_visualization([{"length": 3}])',
        'create_statistical_analysis_dashboard([{"length": 3}])',
    ])
    def test_missing_matplotlib_graceful_failure(self, call):
        code = '''
import importlib.abc
import sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'matplotlib' or name.startswith('matplotlib.'):
            raise ImportError('matplotlib is unavailable')
        return None
sys.meta_path.insert(0, Blocker())
from symergetics.visualization.advanced import (
    create_comparative_analysis_visualization,
    create_pattern_discovery_visualization,
    create_statistical_analysis_dashboard,
)
try:
    CALL
except ImportError:
    pass
else:
    raise AssertionError('Rendering without matplotlib must fail')
'''.replace('CALL', call)
        result = run_python(code)
        assert result.returncode == 0, result.stdout + result.stderr

    def test_extreme_data_values(self):
        data = [
            {'length': 1000, 'pattern_complexity': {'complexity_score': 100.0}, 'symmetry_analysis': {'symmetry_score': 1.0}},
            {'length': 0, 'pattern_complexity': {'complexity_score': 0.0}, 'symmetry_analysis': {'symmetry_score': 0.0}},
        ]
        result = create_pattern_discovery_visualization(data)
        assert_rendered(result)
        assert result['metadata']['total_analyses'] == 2


def test_pattern_discovery_initializes_dependencies_in_cold_process():
    result = run_python('''
from pathlib import Path
import matplotlib.pyplot as plt
from symergetics.visualization import set_config
from symergetics.visualization.advanced import create_pattern_discovery_visualization
set_config({"dpi": 40})
caller = plt.figure()
before = plt.get_fignums()
result = create_pattern_discovery_visualization([
    {"length": 3, "pattern_complexity": {"complexity_score": 1},
     "symmetry_analysis": {"symmetry_score": 0.25}},
    {"length": 5, "pattern_complexity": {"complexity_score": 2},
     "symmetry_analysis": {"symmetry_score": 0.5}},
])
assert plt.get_fignums() == before
assert result["metadata"]["total_analyses"] == 2
assert Path(result["files"][0]).read_bytes().startswith(bytes.fromhex("89504e470d0a1a0a"))
plt.close(caller)
''')
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('renderer', [
    create_comparative_analysis_visualization,
    create_pattern_discovery_visualization,
    create_statistical_analysis_dashboard,
    create_geometric_mnemonics_visualization,
])
@pytest.mark.parametrize('blocked', [False, True])
def test_legacy_renderers_own_only_their_figure(
    renderer, blocked, tmp_path, analysis_results,
):
    if blocked:
        destination = tmp_path / 'blocked'
        destination.write_text('not a directory')
        set_config({'output_dir': str(destination)})
    set_config({'dpi': 40})
    if renderer is create_comparative_analysis_visualization:
        args = (analysis_results[:2], analysis_results[2:])
    elif renderer is create_geometric_mnemonics_visualization:
        args = ([1, 2],)
    else:
        args = (analysis_results,)
    caller = plt.figure()
    before = plt.get_fignums()
    try:
        if blocked:
            with pytest.raises(NotADirectoryError):
                renderer(*args)
        else:
            assert_rendered(renderer(*args))
        assert plt.get_fignums() == before
        assert plt.fignum_exists(caller.number)
    finally:
        plt.close(caller)


def test_dashboard_analysis_failure_releases_figure():
    caller = plt.figure()
    before = plt.get_fignums()
    try:
        with pytest.raises(ValueError):
            create_statistical_analysis_dashboard([
                {'length': 3, 'pattern_complexity': {'complexity_score': float('inf')}},
            ])
        assert plt.get_fignums() == before
    finally:
        plt.close(caller)


@pytest.mark.parametrize('constant', [False, True])
def test_dashboard_correlations_are_numeric_or_undefined(constant):
    advanced._ensure_matplotlib_numpy()
    figure, ax = plt.subplots()
    try:
        density = [1, 1, 1] if constant else [3, 2, 1]
        advanced._plot_correlation_matrix(
            ax, [1, 2, 3], [2, 4, 6], density, [6, 4, 2],
        )
        actual = ax.images[0].get_array()
        expected = np.array([
            [1, 1, -1, -1], [1, 1, -1, -1],
            [-1, -1, 1, 1], [-1, -1, 1, 1],
        ], dtype=float)
        if constant:
            expected[2, :] = np.nan
            expected[:, 2] = np.nan
        np.testing.assert_allclose(actual.filled(np.nan), expected, equal_nan=True)
        for text in ax.texts:
            column, row = text.get_position()
            value = expected[row, column]
            if np.isnan(value):
                assert text.get_text() == 'N/A'
            else:
                assert float(text.get_text()) == pytest.approx(value)
    finally:
        plt.close(figure)


def test_histogram_labels_and_table_follow_actual_statistics():
    advanced._ensure_matplotlib_numpy()
    figure, axes = plt.subplots(1, 2)
    try:
        advanced._plot_detailed_histogram(axes[0], [1, 2, 6], 'Data', 'blue')
        for line, expected in zip(axes[0].lines, [3, 2]):
            np.testing.assert_array_equal(line.get_xdata(), [expected, expected])
            assert float(line.get_label().split(':')[1]) == expected
        advanced._plot_summary_table(
            axes[1], [1, 2, 6], [2], [1, 1, 1], [],
        )
        cells = axes[1].tables[0].get_celld()
        assert [cells[1, col].get_text().get_text() for col in range(1, 6)] == [
            '3.00', '2.00', '1.00', '6.00', '2.65',
        ]
        assert cells[2, 5].get_text().get_text() == 'N/A'
        assert cells[3, 5].get_text().get_text() == '0.00'
        assert all(cells[4, col].get_text().get_text() == 'N/A' for col in range(1, 6))
    finally:
        plt.close(figure)


def test_analysis_overviews_report_computed_values(analysis_results):
    figure, axes = plt.subplots(1, 2)
    try:
        advanced._plot_main_statistics(axes[0], analysis_results)
        values = dict(
            line.strip().split(': ', 1)
            for line in axes[0].texts[0].get_text().splitlines() if ': ' in line
        )
        assert float(values['Average Length']) == 4.8
        assert float(values['Average Complexity']) == 2.25
        assert float(values['Average Symmetry']) == 0.65
        advanced._plot_mnemonics_overview(axes[1], [
            {'platonic_relationships': {'cube': {}}, 'ivm_scaling_factors': {'2': {}}},
            {'rational_approximations': {'pi': {}}},
        ])
        values = dict(
            line.strip().split(': ', 1)
            for line in axes[1].texts[0].get_text().splitlines() if ': ' in line
        )
        assert float(values['Avg Relationships/Number']) == 1.5
    finally:
        plt.close(figure)
