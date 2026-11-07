"""Exercise orchestration with real subprocesses and isolated filesystem effects."""
import sys
from pathlib import Path

from verify_suite import Check, execute


def test_successful_steps_observe_preceding_filesystem_state(tmp_path):
    code, receipt = execute([
        Check("create", (sys.executable, "-c",
              "from pathlib import Path; Path('state').write_text('ready')"), tmp_path),
        Check("consume", (sys.executable, "-c",
              "from pathlib import Path; p=Path('state'); "
              "assert p.read_text() == 'ready'; p.write_text('consumed')"), tmp_path),
    ])
    assert code == 0
    assert (tmp_path / "state").read_text() == "consumed"
    assert receipt["status"] == "passed"
    assert receipt["not_run"] == []
    assert [step["returncode"] for step in receipt["checks"]] == [0, 0]


def test_failure_stops_execution_and_reports_unexecuted_work(tmp_path):
    code, receipt = execute([
        Check("failure", (sys.executable, "-c", "raise SystemExit(19)"), tmp_path),
        Check("must not execute", (sys.executable, "-c",
              "from pathlib import Path; Path('unexpected').touch()"), tmp_path),
    ])
    assert code == 19
    assert not (tmp_path / "unexpected").exists()
    assert receipt["status"] == "failed"
    assert receipt["checks"][0]["returncode"] == 19
    assert receipt["not_run"] == ["must not execute"]


def test_missing_executable_is_a_failed_check_not_a_success(tmp_path):
    missing = str(tmp_path / "nonexistent-executable")
    code, receipt = execute([Check("missing runtime", (missing,), tmp_path)])
    assert code != 0
    assert receipt["status"] == "failed"
    assert missing in receipt["checks"][0]["error"]


def test_cli_rejects_missing_checkout_before_running_checks(tmp_path):
    import subprocess

    script = Path(__file__).resolve().parents[1] / "verify_suite.py"
    result = subprocess.run(
        [sys.executable, str(script), "--quadmath-root", str(tmp_path / "missing")],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert str(tmp_path / "missing") in result.stderr
