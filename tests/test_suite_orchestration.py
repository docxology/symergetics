"""Exercise orchestration with real subprocesses and isolated filesystem effects."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from verify_suite import Check, execute, repository_state, verify


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


@pytest.mark.parametrize("names", [[], ["duplicate", "duplicate"], [""]])
def test_invalid_plan_cannot_launch_any_child(tmp_path, names):
    checks = [Check(name, (sys.executable, "-c",
                    "from pathlib import Path; Path('unexpected').touch()"), tmp_path)
              for name in names]
    with pytest.raises(ValueError):
        execute(iter(checks))
    assert not (tmp_path / "unexpected").exists()


def _source_checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    (root / "source.py").write_text("value = 1\n")
    (root / "output").mkdir()
    _commit_checkout(root)
    return root


def test_source_bound_ordered_execution_allows_disposable_output(tmp_path):
    root = _source_checkout(tmp_path)
    revision = repository_state("symergetics", root, "git")["revision"]
    code, receipt = verify([
        Check("create", (sys.executable, "-c",
              "from pathlib import Path; Path('output/value').write_text('7')"), root),
        Check("consume", (sys.executable, "-c",
              "from pathlib import Path; p=Path('output/value'); "
              "p.write_text(str(int(p.read_text()) * 6))"), root),
    ], {"symergetics": root}, "git", require_clean=True,
       requested_refs={"symergetics": "HEAD"},
       expected_revisions={"symergetics": revision})
    assert code == 0
    assert (root / "output/value").read_text() == "42"
    binding = receipt["repositories"]["symergetics"]
    assert binding["requested_ref_revision"] == revision
    assert binding["before"]["input_sha256"] == binding["after"]["input_sha256"]
    assert binding["after"]["worktree_dirty"]
    assert not binding["after"]["source_dirty"]


@pytest.mark.parametrize("mutation", [
    "Path('source.py').write_text('value = 2\\n')",
    "Path('untracked-input.py').write_text('value = 2\\n')",
    "Path('source.py').write_text('value = 2\\n'); "
    "subprocess.run(['git','add','.'],check=True); "
    "subprocess.run(['git','-c','user.name=Suite test',"
    "'-c','user.email=suite-test@example.invalid','commit','--quiet','-m','drift'],check=True)",
])
def test_execution_cannot_certify_changed_source_or_head(tmp_path, mutation):
    root = _source_checkout(tmp_path)
    code, receipt = verify([
        Check("drift", (sys.executable, "-c",
              "from pathlib import Path; import subprocess; " + mutation), root),
    ], {"symergetics": root}, "git")
    assert code != 0
    assert receipt["status"] == "failed"
    assert receipt["checks"][0]["returncode"] == 0
    binding = receipt["repositories"]["symergetics"]
    assert binding["before"]["input_sha256"] != binding["after"]["input_sha256"]
    assert receipt["integrity_errors"]


def test_local_dirty_sources_are_reported_but_release_mode_refuses_launch(tmp_path):
    root = _source_checkout(tmp_path)
    (root / "source.py").write_text("value = 2\n")
    check = Check("consume", (sys.executable, "-c",
                  "from pathlib import Path; assert '2' in Path('source.py').read_text(); "
                  "Path('output/ran').touch()"), root)
    code, receipt = verify([check], {"symergetics": root}, "git")
    assert code == 0
    assert receipt["repositories"]["symergetics"]["before"]["source_dirty"]
    (root / "output/ran").unlink()
    code, receipt = verify([check], {"symergetics": root}, "git", require_clean=True)
    assert code != 0
    assert receipt["checks"] == []
    assert not (root / "output/ran").exists()


@pytest.mark.parametrize("name,relative", [
    ("symergetics", "output/figure.png"),
    ("QuadMath", "docs/manuscript/figures/input.png"),
    ("QuadMath", "quadmath/output/figure.png"),
])
def test_tracked_provenance_figures_are_inputs_even_under_output_prefixes(
        tmp_path, name, relative):
    root = _source_checkout(tmp_path)
    figure = root / relative
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"authored provenance")
    _commit_checkout(root)
    before = repository_state(name, root, "git")
    figure.write_bytes(b"different provenance")
    after = repository_state(name, root, "git")
    assert after["source_dirty"]
    assert before["input_sha256"] != after["input_sha256"]


def test_expected_revision_mismatch_prevents_launch(tmp_path):
    root = _source_checkout(tmp_path)
    code, receipt = verify([
        Check("must not run", (sys.executable, "-c",
              "from pathlib import Path; Path('output/ran').touch()"), root),
    ], {"symergetics": root}, "git",
       expected_revisions={"symergetics": "0" * 40})
    assert code != 0
    assert receipt["checks"] == []
    assert not (root / "output/ran").exists()


def test_clean_mode_detects_tracked_bytes_hidden_from_git_status(tmp_path):
    root = _source_checkout(tmp_path)
    subprocess.run(["git", "-C", str(root), "update-index",
                    "--assume-unchanged", "source.py"], check=True)
    (root / "source.py").write_text("value = 2\n")
    code, receipt = verify([
        Check("must not run", (sys.executable, "-c",
              "from pathlib import Path; Path('output/ran').touch()"), root),
    ], {"symergetics": root}, "git", require_clean=True)
    assert code != 0
    assert receipt["repositories"]["symergetics"]["before"]["source_dirty"]
    assert not (root / "output/ran").exists()


def test_untracked_receipt_is_excluded_but_tracked_receipt_remains_an_input(tmp_path):
    root = _source_checkout(tmp_path)
    report = root / "receipt.json"
    before = repository_state("symergetics", root, "git", [report])
    report.write_text('{"status":"historical"}')
    untracked = repository_state("symergetics", root, "git", [report])
    assert untracked["input_sha256"] == before["input_sha256"]
    assert not untracked["source_dirty"]
    _commit_checkout(root)
    tracked = repository_state("symergetics", root, "git", [report])
    report.write_text('{"status":"different"}')
    changed = repository_state("symergetics", root, "git", [report])
    assert changed["source_dirty"]
    assert changed["input_sha256"] != tracked["input_sha256"]


def test_requested_ref_must_resolve_to_checked_out_revision(tmp_path):
    root = _source_checkout(tmp_path)
    old = repository_state("symergetics", root, "git")["revision"]
    (root / "source.py").write_text("value = 2\n")
    _commit_checkout(root)
    code, receipt = verify([
        Check("must not run", (sys.executable, "-c",
              "from pathlib import Path; Path('output/ran').touch()"), root),
    ], {"symergetics": root}, "git", requested_refs={"symergetics": old})
    assert code != 0
    assert receipt["repositories"]["symergetics"]["requested_ref_revision"] == old
    assert receipt["checks"] == []
    assert not (root / "output/ran").exists()


@pytest.mark.parametrize("materialization", ["ignored_executable_bit", "crlf", "filter"])
def test_clean_git_materializations_can_execute_without_refreshing_index(
        tmp_path, materialization):
    root = _source_checkout(tmp_path)
    source = root / "source.py"
    subprocess.run(["git", "-C", str(root), "config", "core.autocrlf", "false"],
                   check=True)
    if materialization == "ignored_executable_bit":
        source.chmod(0o755)
        subprocess.run(["git", "-C", str(root), "config", "core.filemode", "true"],
                       check=True)
        _commit_checkout(root)
        subprocess.run(["git", "-C", str(root), "config", "core.filemode", "false"],
                       check=True)
        source.chmod(0o644)
    elif materialization == "crlf":
        subprocess.run(["git", "-C", str(root), "config", "core.autocrlf", "true"],
                       check=True)
        source.unlink()
        subprocess.run(["git", "-C", str(root), "checkout", "--", "source.py"],
                       check=True)
        assert source.read_bytes() == b"value = 1\r\n"
    else:
        # The filter is intentionally not a newline substitution. The filename
        # crosses both Git's NUL-delimited listing and quoted stdin-path boundary.
        source = source.rename(root / '-source\n"\\\t\r\u00e9.py')
        (root / ".gitattributes").write_text("* filter=canonical\n")
        subprocess.run(["git", "-C", str(root), "config",
                        "filter.canonical.clean", "tr a-z A-Z"], check=True)
        subprocess.run(["git", "-C", str(root), "config",
                        "filter.canonical.required", "true"], check=True)
        _commit_checkout(root)
    assert subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "status",
         "--porcelain=v1", "-z"], check=True, capture_output=True,
    ).stdout == b""
    expected = source.read_bytes()
    index = (root / ".git/index").read_bytes()
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    code, receipt = verify([
        Check("consume materialized input", (sys.executable, "-I", "-c",
              "from pathlib import Path; import sys; "
              "Path('output/consumed').write_bytes(Path(sys.argv[1]).read_bytes())",
              str(source)), root),
    ], {"symergetics": root}, "git", {"checkout": root},
       require_clean=True, requested_refs={"symergetics": "HEAD"},
       expected_revisions={"symergetics": revision})
    assert code == 0
    assert (root / "output/consumed").read_bytes() == expected
    assert (root / ".git/index").read_bytes() == index
    binding = receipt["repositories"]["symergetics"]
    assert not binding["before"]["source_dirty"]
    assert not binding["after"]["source_dirty"]
    assert binding["requested_ref_revision"] == binding["expected_revision"] == revision
    assert binding["before"]["input_sha256"] == binding["after"]["input_sha256"]
    assert str(root) not in json.dumps(receipt)


def test_clean_mode_rejects_executable_drift_hidden_from_git_status(tmp_path):
    root = _source_checkout(tmp_path)
    source = root / "source.py"
    source.chmod(0o644)
    subprocess.run(["git", "-C", str(root), "config", "core.filemode", "true"],
                   check=True)
    subprocess.run(["git", "-C", str(root), "update-index",
                    "--assume-unchanged", "source.py"], check=True)
    source.chmod(0o755)
    assert subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "status",
         "--porcelain=v1", "-z"], check=True, capture_output=True,
    ).stdout == b""
    code, receipt = verify([
        Check("must not run", (sys.executable, "-I", "-c",
              "from pathlib import Path; Path('output/ran').touch()"), root),
    ], {"symergetics": root}, "git", require_clean=True)
    assert code != 0
    assert receipt["repositories"]["symergetics"]["before"]["source_paths"] == ["source.py"]
    assert not (root / "output/ran").exists()


@pytest.mark.parametrize("mutation", ["normalized_bytes", "ignored_executable_bit"])
def test_git_clean_changes_during_execution_still_break_raw_binding(tmp_path, mutation):
    root = _source_checkout(tmp_path)
    subprocess.run(["git", "-C", str(root), "config", "core.autocrlf", "true"],
                   check=True)
    subprocess.run(["git", "-C", str(root), "config", "core.filemode", "false"],
                   check=True)
    source = root / "source.py"
    source.chmod(0o644)
    change = ("Path('source.py').write_bytes(b'value = 1\\r\\n')"
              if mutation == "normalized_bytes" else "Path('source.py').chmod(0o755)")
    code, receipt = verify([
        Check("change raw input", (sys.executable, "-I", "-c",
              "from pathlib import Path; " + change), root),
    ], {"symergetics": root}, "git", require_clean=True)
    assert code != 0
    assert receipt["checks"][0]["returncode"] == 0
    binding = receipt["repositories"]["symergetics"]
    assert not binding["before"]["source_dirty"]
    assert not binding["after"]["source_dirty"]
    assert binding["before"]["input_sha256"] != binding["after"]["input_sha256"]
    assert receipt["source_integrity"] == "failed"


def test_symlink_fingerprint_never_follows_target_but_binds_link_text(tmp_path):
    root = _source_checkout(tmp_path)
    target = tmp_path / "private"
    target.write_bytes(b"private original")
    link = root / "source.py"
    link.unlink()
    link.symlink_to(target)
    _commit_checkout(root)
    before = repository_state("symergetics", root, "git")
    target.write_bytes(b"private changed")
    after_target = repository_state("symergetics", root, "git")
    assert not after_target["source_dirty"]
    assert before["input_sha256"] == after_target["input_sha256"]
    subprocess.run(["git", "-C", str(root), "update-index",
                    "--assume-unchanged", "source.py"], check=True)
    link.unlink()
    link.symlink_to(tmp_path / "absent")
    after_link = repository_state("symergetics", root, "git")
    assert after_link["source_paths"] == ["source.py"]
    assert before["input_sha256"] != after_link["input_sha256"]


def test_sha256_git_objects_use_same_clean_filter_boundary(tmp_path):
    root = tmp_path / "sha256-checkout"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", "--object-format=sha256", str(root)],
                   check=True)
    (root / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    (root / ".gitattributes").write_text("* text eol=lf\n")
    source = root / "source.py"
    source.write_bytes(b"value = 1\r\n")
    _commit_checkout(root)
    before = repository_state("symergetics", root, "git")
    assert not before["source_dirty"]
    subprocess.run(["git", "-C", str(root), "update-index",
                    "--assume-unchanged", "source.py"], check=True)
    source.write_bytes(b"value = 2\r\n")
    after = repository_state("symergetics", root, "git")
    assert after["source_paths"] == ["source.py"]
    assert before["input_sha256"] != after["input_sha256"]


def test_git_executable_classification_uses_owner_bit_but_raw_binding_keeps_all_bits(
        tmp_path):
    root = _source_checkout(tmp_path)
    subprocess.run(["git", "-C", str(root), "config", "core.filemode", "true"],
                   check=True)
    source = root / "source.py"
    source.chmod(0o644)
    before = repository_state("symergetics", root, "git")
    source.chmod(0o654)
    assert subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), "status",
         "--porcelain=v1", "-z"], check=True, capture_output=True,
    ).stdout == b""
    after = repository_state("symergetics", root, "git")
    assert not after["source_dirty"]
    assert before["input_sha256"] != after["input_sha256"]


def test_matching_worktree_and_index_cannot_hide_staged_source_changes(tmp_path):
    root = _source_checkout(tmp_path)
    (root / "source.py").write_bytes(b"value = 2\n")
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    code, receipt = verify([
        Check("must not run", (sys.executable, "-I", "-c",
              "from pathlib import Path; Path('output/ran').touch()"), root),
    ], {"symergetics": root}, "git", require_clean=True)
    assert code != 0
    assert receipt["repositories"]["symergetics"]["before"]["source_paths"] == ["source.py"]
    assert not (root / "output/ran").exists()
