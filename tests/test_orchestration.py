"""Consumer-visible local orchestration regressions using real child processes."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
GROUPS = (
    'test_core_numbers.py',
    'test_core_coordinates.py',
    'test_geometry_polyhedra.py',
    'test_computation_palindromes.py',
    'test_computation_primorials.py',
)


@pytest.fixture
def local_project(tmp_path):
    """An isolated project with actual passing arithmetic tests and a standalone demo."""
    shutil.copy2(REPOSITORY / 'run.py', tmp_path / 'run.py')
    (tmp_path / 'examples').mkdir()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'symergetics').symlink_to(REPOSITORY / 'symergetics', target_is_directory=True)
    for filename in GROUPS:
        (tmp_path / 'tests' / filename).write_text(
            'from fractions import Fraction\n'
            'def test_exact_arithmetic():\n'
            '    assert Fraction(1, 3) * 3 == 1\n'
        )
    (tmp_path / 'examples' / 'arithmetic_demo.py').write_text(
        'from fractions import Fraction\n'
        'import json, sys\n'
        'from pathlib import Path\n'
        'result = sum(Fraction(1, 2 ** n) for n in range(1, 9))\n'
        'Path("arithmetic.json").write_text(json.dumps({\n'
        '    "result": str(result), "interpreter": sys.executable, "prefix": sys.prefix}))\n'
    )
    return tmp_path


def run_local(root, *args):
    env = os.environ.copy()
    env.update(PYTHONPATH='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    # Force the real interpreter-bound pip fallback, not a machine's uv executable.
    # All subprocesses are launched with absolute paths and need no PATH lookup.
    env['PATH'] = str(root / 'no-path-tools')
    completed = subprocess.run(
        [sys.executable, str(root / 'run.py'), *args], cwd=root,
        text=True, capture_output=True, env=env, timeout=90,
    )
    report = json.loads((root / 'output' / 'orchestration_results.json').read_text())
    return completed, report


def assert_arithmetic_demo_ran(root):
    artifact = json.loads((root / 'arithmetic.json').read_text())
    assert artifact['result'] == '255/256'
    return artifact


def test_setup_failure_does_not_prevent_tests_or_demos(local_project):
    # This is an actually un-installable local project; no network or fake pip.
    (local_project / 'pyproject.toml').write_text(
        '[build-system]\nrequires = []\nbuild-backend = \"unavailable_local_backend\"\n'
        '[project]\nname = \"orchestration-fixture\"\nversion = \"0\"\n'
    )
    completed, report = run_local(local_project)
    assert completed.returncode == 1
    assert report['success'] is False
    assert report['stages']['setup']['status'] == 'failed'
    assert report['stages']['setup']['commands'][0]['success'] is False
    setup_commands = report['stages']['setup']['commands']
    assert setup_commands[0]['command'] == [
        sys.executable, '-m', 'pip', 'install', '-e', '.[scientific,test]'
    ]
    assert all(command['command'][0] == sys.executable for command in setup_commands)
    assert report['stages']['tests']['status'] == 'success'
    assert report['stages']['examples']['status'] == 'success'
    assert_arithmetic_demo_ran(local_project)

def test_test_failure_does_not_prevent_remaining_groups_or_demos(local_project):
    (local_project / 'tests' / GROUPS[0]).write_text(
        'from fractions import Fraction\n'
        'def test_wrong_identity():\n'
        '    assert Fraction(1, 3) * 3 == 2\n'
    )
    completed, report = run_local(local_project, '--skip-setup')
    assert completed.returncode == 1
    assert report['success'] is False
    stage = report['stages']['tests']
    assert stage['status'] == 'failed'
    assert len(stage['commands']) == 6
    assert stage['commands'][0]['returncode'] == 1
    assert stage['commands'][-1]['returncode'] == 0
    assert report['stages']['examples']['status'] == 'success'
    assert_arithmetic_demo_ran(local_project)


def test_demo_failure_does_not_prevent_remaining_demos(local_project):
    (local_project / 'examples' / 'aaa_broken_demo.py').write_text('print(1 / 0)\n')
    completed, report = run_local(local_project, '--examples-only')
    assert completed.returncode == 1
    assert report['success'] is False
    assert report['stages']['setup']['status'] == 'skipped'
    assert report['stages']['tests']['status'] == 'skipped'
    assert report['stages']['examples']['status'] == 'failed'
    assert [item['returncode'] for item in report['examples']] == [1, 0]
    assert report['summary']['failed'] == 1
    assert_arithmetic_demo_ran(local_project)


def test_zero_demos_produces_successful_empty_report(local_project):
    (local_project / 'examples' / 'arithmetic_demo.py').unlink()
    completed, report = run_local(local_project, '--examples-only')
    assert completed.returncode == 0
    assert report['success'] is True
    assert report['examples'] == []
    assert report['summary']['total_examples'] == 0
    assert report['stages']['examples']['commands'] == []


def test_default_discovery_runs_standalone_demos_not_suite_utility(local_project):
    shutil.copy2(REPOSITORY / 'examples' / 'suite_interoperability.py',
                 local_project / 'examples' / 'suite_interoperability.py')
    # Pre-existing output must be counted as inventory, not evidence of generation.
    (local_project / 'output').mkdir()
    (local_project / 'output' / 'old.bin').write_bytes(b'old artifact')
    completed, report = run_local(local_project, '--examples-only')
    assert completed.returncode == 0
    assert [item['script'] for item in report['examples']] == ['arithmetic_demo.py']
    assert report['output_inventory']['files'] == [
        {'path': 'old.bin', 'size_bytes': len(b'old artifact')}
    ]
    assert_arithmetic_demo_ran(local_project)


def test_selected_venv_is_retained_for_tests_and_demos(local_project):
    environment = local_project / 'selected-environment'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(environment)],
                   check=True, capture_output=True, text=True, timeout=60)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    probe = subprocess.run(
        [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        check=True, text=True, capture_output=True, timeout=15,
    )
    # Reuse installed test dependencies without changing the selected venv identity.
    package_paths = [path for path in sys.path
                     if Path(path).name in ('site-packages', 'dist-packages')]
    (Path(probe.stdout.strip()) / 'test-dependencies.pth').write_text(
        '\n'.join(package_paths) + '\n'
    )
    completed, report = run_local(local_project, '--skip-setup', '--python', str(python))
    assert completed.returncode == 0
    assert report['interpreter'] == str(python)
    assert report['success'] is True
    artifact = assert_arithmetic_demo_ran(local_project)
    assert artifact['interpreter'] == str(python)
    assert artifact['prefix'] == str(environment)
    for stage in report['stages'].values():
        for command in stage['commands']:
            assert command['command'][0] == str(python)


@pytest.fixture
def numpy_only_environment(tmp_path):
    """A real venv with NumPy but without optional plotting dependencies."""
    environment = tmp_path / 'numpy-only'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(environment)],
                   check=True, capture_output=True, text=True, timeout=60)
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    probe = subprocess.run(
        [str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
        check=True, text=True, capture_output=True, timeout=15,
    )
    site = Path(probe.stdout.strip())
    numpy_directory = Path(numpy.__file__).parent
    (site / 'numpy').symlink_to(numpy_directory, target_is_directory=True)
    # Some wheels keep native libraries beside the package rather than inside it.
    for sibling in numpy_directory.parent.glob('numpy*.libs'):
        (site / sibling.name).symlink_to(sibling, target_is_directory=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = ''
    probe = subprocess.run(
        [str(python), '-c', 'import importlib.util, numpy; '
         'assert importlib.util.find_spec("matplotlib") is None; '
         'assert numpy.arange(3).sum() == 3'],
        check=True, text=True, capture_output=True, env=env, timeout=15,
    )
    return python, env


@pytest.mark.parametrize('demo', [
    'png_visualization_demo.py',
    'comprehensive_visualizations_demo.py',
    'geometric_mnemonics_demo.py',
])
def test_missing_plotting_dependency_makes_required_generation_fail(
        tmp_path, numpy_only_environment, demo):
    python, env = numpy_only_environment
    root = tmp_path / 'project'
    (root / 'examples').mkdir(parents=True)
    (root / 'symergetics').symlink_to(REPOSITORY / 'symergetics', target_is_directory=True)
    for filename in (demo, 'geometric_mnemonics_demo.py'):
        shutil.copy2(REPOSITORY / 'examples' / filename, root / 'examples' / filename)
    completed = subprocess.run(
        [str(python), str(root / 'examples' / demo)], cwd=root,
        text=True, capture_output=True, env=env, timeout=90,
    )
    assert completed.returncode != 0
    # Imports and actual analysis execute before plotting fails: this is not a
    # missing NumPy/package/import fixture masquerading as a generation failure.
    if demo == 'geometric_mnemonics_demo.py':
        assert list((root / 'output').rglob('*.json'))
