"""Exercise advanced visualizations with actual isolated rendering and files."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from symergetics.visualization import get_config, set_config
from symergetics.visualization.advanced import (
    create_comparative_analysis_visualization,
    create_pattern_discovery_visualization,
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
