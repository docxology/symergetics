#!/usr/bin/env python3
"""Set up, test and demonstrate Symergetics using one selected Python interpreter.

The default interpreter is the one running this script. Use --python to select
another existing environment; setup installs the declared scientific/test extras
into that environment. Requested stages continue after failures, and any failed
stage makes the final exit status nonzero. See output/orchestration_results.json
for stage, subprocess and existing-output inventory details.
"""

import argparse
import json
import logging
import logging.handlers
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


class SymergeticsOrchestrator:
    """Run independent requested stages and retain their actual outcomes."""

    command_timeout = 1800
    example_timeout = 300

    def __init__(self, python: Optional[str] = None, project_root: Optional[Path] = None):
        self.project_root = Path(project_root or Path(__file__).parent).absolute()
        self.examples_dir = self.project_root / "examples"
        self.output_dir = self.project_root / "output"
        selected = os.path.expanduser(python or sys.executable)
        # Do not resolve symlinks: a venv's Python must retain its venv identity.
        self.python = os.path.abspath(shutil.which(selected) or selected)
        self.start_time = time.time()
        self.stages: Dict[str, Dict[str, Any]] = {}
        self._commands: List[Dict[str, Any]] = []
        self.setup_logging()

    def setup_logging(self):
        """Retain readable and machine-readable rotating command logs."""
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"symergetics_orchestrator.{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%H:%M:%S'))
        log_file = logging.handlers.RotatingFileHandler(
            logs_dir / "orchestration.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        log_file.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'message': record.getMessage(),
                })

        detailed = logging.handlers.RotatingFileHandler(
            logs_dir / "orchestration_detailed.log", maxBytes=5 * 1024 * 1024, backupCount=3
        )
        detailed.setFormatter(JSONFormatter())
        for handler in (console, log_file, detailed):
            self.logger.addHandler(handler)
        self.log(f"Selected Python: {self.python}")

    def log(self, message: str, level: str = "INFO", category: Optional[str] = None):
        if category:
            message = f"[{category}] {message}"
        self.logger.log(getattr(logging, level, logging.INFO), message)

    def _execute(self, cmd: List[str], description: str, timeout: float,
                 env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute a real subprocess and record return codes, timeouts and errors."""
        started = time.monotonic()
        result: Dict[str, Any] = {
            'description': description, 'command': cmd, 'interpreter': self.python,
            'timeout_seconds': timeout, 'returncode': None, 'success': False,
            'stdout': '', 'stderr': '', 'error': None,
        }
        command_env = os.environ.copy()
        if env:
            command_env.update(env)
        self.log(f"Starting: {description}")
        try:
            completed = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True,
                timeout=timeout, env=command_env,
            )
            result.update(returncode=completed.returncode, stdout=completed.stdout,
                          stderr=completed.stderr, success=completed.returncode == 0)
            if not result['success']:
                result['error'] = completed.stderr or f"Exit status {completed.returncode}"
        except subprocess.TimeoutExpired as exc:
            result['error'] = f"Timeout after {timeout:g} seconds"
            result['stdout'] = (exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
            result['stderr'] = (exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
        except OSError as exc:
            result['error'] = str(exc)
        result['duration'] = time.monotonic() - started
        self._commands.append(result)
        if result['success']:
            self.log(f"Completed: {description} ({result['duration']:.2f}s)")
        else:
            self.log(f"Failed: {description}: {result['error']}", 'ERROR')
        self.logger.debug(f"Command result: {json.dumps(result)}")
        return result

    def run_command(self, cmd: List[str], description: str,
                    env: Optional[Dict[str, str]] = None) -> bool:
        return self._execute(cmd, description, self.command_timeout, env)['success']

    def setup_environment(self) -> bool:
        """Install declared extras into exactly the selected environment."""
        extras = '.[scientific,test]'
        uv = shutil.which('uv')
        if uv:
            command = [uv, 'pip', 'install', '--python', self.python, '-e', extras]
        else:
            command = [self.python, '-m', 'pip', 'install', '-e', extras]
        installed = self.run_command(command, 'Installing Symergetics scientific/test extras')
        # Verification is attempted even after installation failure, for diagnostics.
        verified = self._verify_installation()
        return installed and verified

    def _verify_installation(self) -> bool:
        imports = [
            'import symergetics',
            'from symergetics.core.numbers import SymergeticsNumber',
            'from symergetics.core.coordinates import QuadrayCoordinate',
            'from symergetics.geometry.polyhedra import Tetrahedron',
        ]
        results = [self.run_command([self.python, '-c', code], code) for code in imports]
        return all(results)

    def _get_test_runner(self) -> List[str]:
        return [self.python]

    def run_tests(self) -> bool:
        """Run the full suite and retain the existing focused diagnostic groups."""
        groups = [
            ('tests/', ['--tb=short', '--durations=10']),
            ('tests/test_core_numbers.py', []),
            ('tests/test_core_coordinates.py', []),
            ('tests/test_geometry_polyhedra.py', []),
            ('tests/test_computation_palindromes.py', []),
            ('tests/test_computation_primorials.py', []),
        ]
        results = []
        for path, options in groups:
            results.append(self.run_command(
                [self.python, '-m', 'pytest', path, '-v', '-o', 'addopts=', *options],
                f'Running tests: {path}',
            ))
        return all(results)

    def run_example(self, example_script: Path) -> Dict[str, Any]:
        result = self._execute([self.python, str(example_script)],
                               f'Running demo: {example_script.name}', self.example_timeout)
        return dict(result, script=example_script.name, runner=self.python)

    def run_all_examples(self) -> List[Dict[str, Any]]:
        """Discover standalone demos, not suite-level integration utilities."""
        scripts = sorted(self.examples_dir.glob('*_demo.py'))
        self.log(f"Found {len(scripts)} standalone demos")
        return [self.run_example(script) for script in scripts]

    def run_stage(self, name: str, requested: bool, action) -> Any:
        """Record skipped/success/failed independently, without short-circuiting."""
        if not requested:
            self.stages[name] = {'status': 'skipped', 'success': None, 'duration': 0,
                                 'commands': [], 'error': None}
            return None
        started = time.monotonic()
        command_start = len(self._commands)
        value = None
        error = None
        interrupted = False
        try:
            value = action()
            success = all(item['success'] for item in value) if name == 'examples' else bool(value)
        except (Exception, KeyboardInterrupt) as exc:
            success = False
            interrupted = isinstance(exc, KeyboardInterrupt)
            error = 'Interrupted by user' if interrupted else str(exc)
            self.logger.exception(f"Stage {name} failed")
        self.stages[name] = {
            'status': 'success' if success else 'failed', 'success': success,
            'duration': time.monotonic() - started,
            'commands': self._commands[command_start:], 'error': error,
        }
        if interrupted:
            raise KeyboardInterrupt
        return value

    @property
    def success(self) -> bool:
        return all(stage['status'] != 'failed' for stage in self.stages.values())

    def generate_summary_report(self, example_results: List[Dict[str, Any]]):
        """Report attempted stages and inventory, never claim old files were generated."""
        duration = time.time() - self.start_time
        successful = sum(result['success'] for result in example_results)
        results_file = self.output_dir / 'orchestration_results.json'
        inventory = []
        if self.output_dir.exists():
            for path in sorted(self.output_dir.rglob('*')):
                if path.is_file() and path != results_file:
                    inventory.append({'path': str(path.relative_to(self.output_dir)),
                                      'size_bytes': path.stat().st_size})
        size = sum(item['size_bytes'] for item in inventory)
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'interpreter': self.python, 'success': self.success,
            'total_duration': duration, 'stages': self.stages,
            'examples': example_results,
            'summary': {
                'total_examples': len(example_results), 'successful': successful,
                'failed': len(example_results) - successful,
                'output_files': len(inventory), 'output_size_mb': size / 1024 / 1024,
            },
            'output_inventory': {
                'scope': 'existing files at report time, excluding this report; may predate this run',
                'files': inventory,
            },
            'log_files': {
                'orchestration_log': str(self.project_root / 'logs' / 'orchestration.log'),
                'detailed_json_log': str(self.project_root / 'logs' / 'orchestration_detailed.log'),
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_file.write_text(json.dumps(report, indent=2) + '\n')
        print('\nSYMERGETICS ORCHESTRATION SUMMARY')
        for name, stage in self.stages.items():
            print(f"  {name}: {stage['status']} ({stage['duration']:.2f}s)")
        print(f"  Demos: {successful}/{len(example_results)} successful")
        if example_results:
            print(f"  Average demo duration: {sum(item['duration'] for item in example_results) / len(example_results):.2f}s")
        print(f"  Existing output inventory: {len(inventory)} files, {size} bytes (may predate this run)")
        print(f"  Detailed results: {results_file}")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-tests', action='store_true', help='Skip the test suite')
    parser.add_argument('--skip-setup', action='store_true', help='Skip dependency installation')
    parser.add_argument('--examples-only', action='store_true', help='Skip setup and tests')
    parser.add_argument('--python', help='Existing Python interpreter used by every stage (default: this interpreter)')
    args = parser.parse_args()
    orchestrator = SymergeticsOrchestrator(python=args.python)
    examples = []
    try:
        orchestrator.run_stage('setup', not args.examples_only and not args.skip_setup,
                               orchestrator.setup_environment)
        orchestrator.run_stage('tests', not args.examples_only and not args.skip_tests,
                               orchestrator.run_tests)
        examples = orchestrator.run_stage('examples', True, orchestrator.run_all_examples) or []
    except KeyboardInterrupt:
        orchestrator.stages['interruption'] = {
            'status': 'failed', 'success': False, 'duration': 0,
            'commands': [], 'error': 'Interrupted by user',
        }
    try:
        orchestrator.generate_summary_report(examples)
    except Exception:
        orchestrator.logger.exception('Failed to write orchestration report')
        return 1
    orchestrator.log('Orchestration completed successfully' if orchestrator.success
                     else 'Orchestration completed with errors',
                     'INFO' if orchestrator.success else 'ERROR')
    return 0 if orchestrator.success else 1


if __name__ == '__main__':
    sys.exit(main())
