"""Exercise orchestration with real subprocesses and isolated filesystem effects."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from verify_suite import Check, execute, repository_state


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

    script = Path(__file__).resolve().parents[1] / "verify_suite.py"
    result = subprocess.run(
        [sys.executable, str(script), "--quadmath-root", str(tmp_path / "missing")],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert str(tmp_path / "missing") in result.stderr


def test_portable_receipt_does_not_change_real_execution_paths(tmp_path):
    private = tmp_path / "private workspace"
    private.mkdir()
    artifact = private / "result.txt"
    code, receipt = execute([
        Check("write artifact", (sys.executable, "-I", "-c",
              "import sys; from pathlib import Path; "
              "Path(sys.argv[1]).write_text('verified')", str(artifact)), private),
        Check("missing executable", (str(private / "missing"),), private),
    ], {"checkout": private, "python": Path(sys.executable)})
    assert code != 0
    assert artifact.read_text() == "verified"
    encoded = json.dumps(receipt)
    assert str(private) not in encoded
    assert sys.executable not in encoded
    assert receipt["status"] == "failed"
    assert receipt["checks"][-1]["returncode"] != 0


def _commit_checkout(root):
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(root), "-c", "user.name=Suite test",
        "-c", "user.email=suite-test@example.invalid", "commit", "--quiet",
        "-m", "Snapshot",
    ], check=True)


def test_provenance_distinguishes_generated_output_and_source_renames(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    (root / "source.py").write_text("value = 1\n")
    (root / "output").mkdir()
    generated = root / "output/result.txt"
    generated.write_text("first")
    _commit_checkout(root)
    original = repository_state("symergetics", root, "git")
    assert not original["worktree_dirty"]
    assert not original["source_dirty"]

    generated.write_text("second")
    generated_only = repository_state("symergetics", root, "git")
    assert generated_only["worktree_dirty"]
    assert not generated_only["source_dirty"]
    assert generated_only["revision"] == original["revision"]

    # A source rename into an output directory is still a source change.
    subprocess.run(["git", "-C", str(root), "mv", "source.py", "output/source.py"],
                   check=True)
    renamed = repository_state("symergetics", root, "git")
    assert renamed["source_dirty"]
    _commit_checkout(root)
    committed = repository_state("symergetics", root, "git")
    assert committed["revision"] != original["revision"]
    assert not committed["worktree_dirty"]
    assert not committed["source_dirty"]


def test_provenance_rejects_an_enclosing_repository_revision(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    _commit_checkout(tmp_path)
    nested = tmp_path / "not-an-independent-checkout"
    nested.mkdir()
    with pytest.raises(ValueError):
        repository_state("symergetics", nested, "git")
