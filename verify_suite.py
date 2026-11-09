#!/usr/bin/env python3
"""Verify all three Synergetics repositories without replacing their environments.

Requires bash, git, uv, Node.js, CMake and a C++17 compiler. Locks are honored;
Python environments and the native build live in an automatically removed
scratch directory. Use --browser for the installed Chromium gate and --report
for a portable, revision-specific execution receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Check:
    name: str
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str] = field(default_factory=dict)


# Only untracked disposable products are excluded. Tracked figures/data can be
# manuscript inputs and are always source, regardless of their directory.
UNTRACKED_OUTPUT_PREFIXES = {
    "QuadMath": ("quadmath/output/",),
    "symergetics": ("output/", "paper/output/",),
    "QuadCraft": (),
}


def repository_state(name: str, root: Path, git: str,
                     excluded_reports: Iterable[Path] = ()) -> dict:
    """Fingerprint tracked inputs plus nonignored, non-output untracked inputs.

    Cleanliness uses Git's clean filters, object format and core.filemode policy,
    including edits hidden by assume-unchanged. The integrity fingerprint instead
    binds raw relative names, file kinds/executable bits, contents and staged
    index entries: even a Git-clean materialization change breaks that binding.
    Symlinks bind link text, never private target bytes. Ignored untracked builds,
    output prefixes and selected untracked receipts are excluded. Git inspection
    neither refreshes the on-disk index nor writes objects.
    """
    def inspect(*arguments: str) -> str:
        return os.fsdecode(subprocess.run(
            (git, "--no-optional-locks", *arguments), cwd=root, check=True,
            capture_output=True, stdin=subprocess.DEVNULL,
        ).stdout)

    toplevel = Path(inspect("rev-parse", "--show-toplevel").rstrip("\n")).resolve()
    if toplevel != root.resolve():
        raise ValueError(f"{name} is not a repository root")
    revision = inspect("rev-parse", "--verify", "HEAD").strip()
    filemode = inspect("config", "--type=bool", "--default=true",
                       "--get", "core.filemode").strip() == "true"
    status = inspect("status", "--porcelain=v1", "-z", "--untracked-files=all")
    tracked_changes = set()
    entries = iter(status.split("\0"))
    for entry in entries:
        if not entry:
            continue
        # Trust status for index-vs-HEAD changes only. Worktree status can be
        # stat-dirty while filtered bytes match, or hide assume-unchanged edits.
        if entry[0] not in (" ", "?"):
            tracked_changes.add(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            next(entries)
    tracked = set(filter(None, inspect("ls-files", "-z").split("\0")))
    untracked = set(filter(None, inspect(
        "ls-files", "--others", "--exclude-standard", "-z").split("\0")))
    reports = {path.resolve().relative_to(root.resolve()).as_posix()
               for path in excluded_reports if path.resolve().is_relative_to(root.resolve())}
    prefixes = UNTRACKED_OUTPUT_PREFIXES[name]

    def is_input(path: str) -> bool:
        return path in tracked or (path not in reports and not path.startswith(prefixes))

    inputs = tracked | {path for path in untracked if is_input(path)}
    index = inspect("ls-files", "--stage", "-z")
    indexed_blobs = {}
    for entry in filter(None, index.split("\0")):
        metadata, path = entry.split("\t", 1)
        mode, object_id, stage = metadata.split()
        indexed_blobs[path] = (mode, object_id)
    content_changes = set()
    regular_files = []
    digest = hashlib.sha256()
    digest.update(os.fsencode(index))
    for path in sorted(inputs):
        file = root / path
        digest.update(os.fsencode(path) + b"\0")
        indexed = indexed_blobs.get(path)
        if file.is_symlink():
            target = os.fsencode(os.readlink(file))
            payload = b"link:" + target
            mode = "120000"
            if indexed is not None:
                # Hash the link text as an unfiltered Git blob, not its target.
                object_id = subprocess.run(
                    (git, "hash-object", "--stdin", "--no-filters"), cwd=root,
                    input=target, check=True, capture_output=True,
                ).stdout.decode("ascii").strip()
                if object_id != indexed[1]:
                    content_changes.add(path)
        elif file.is_file():
            stat = file.stat()
            mode = "100755" if stat.st_mode & 0o100 else "100644"
            digest.update(f"file:{stat.st_mode & 0o111}:{stat.st_size}:".encode("ascii"))
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if indexed is not None:
                regular_files.append(path)
            payload = b""
        elif not file.exists():
            payload, mode = b"missing", None
        else:
            # Submodule directories need their own integrity policy; do not
            # certify their mutable worktrees by hashing only the gitlink.
            raise ValueError(f"{name} has an unsupported non-file input: {path}")
        digest.update(payload + b"\0")
        if indexed is not None:
            indexed_mode = indexed[0]
            same_kind = mode in ("100644", "100755") and indexed_mode in (
                "100644", "100755")
            if mode != indexed_mode and (filemode or not same_kind):
                content_changes.add(path)
    if regular_files:
        # --stdin-paths batches arbitrary path counts without argv-size limits.
        # Git expects C-quoted lines (not NUL records); quote literal newlines,
        # carriage returns, tabs, quotes and backslashes without Unicode recoding.
        quoted = []
        for path in regular_files:
            encoded = (os.fsencode(path).replace(b"\\", b"\\\\")
                       .replace(b'"', b'\\"').replace(b"\n", b"\\n")
                       .replace(b"\r", b"\\r").replace(b"\t", b"\\t"))
            quoted.append(b'"' + encoded + b'"\n')
        hashes = subprocess.run(
            (git, "hash-object", "--stdin-paths"), cwd=root,
            input=b"".join(quoted), check=True, capture_output=True,
        ).stdout.decode("ascii").splitlines()
        if len(hashes) != len(regular_files):
            raise ValueError(f"{name}: Git did not hash every tracked regular input")
        content_changes.update(
            path for path, object_id in zip(regular_files, hashes)
            if object_id != indexed_blobs[path][1])
    if name == "QuadCraft":
        declaration = re.search(
            r"\bproject\s*\(\s*QuadCraft\s+VERSION\s+([0-9]+(?:\.[0-9]+)+)",
            (root / "CMakeLists.txt").read_text(encoding="utf-8"),
            re.IGNORECASE,
        )
        if declaration is None:
            raise ValueError("QuadCraft has no declared project version")
        version = declaration.group(1)
    else:
        with (root / "pyproject.toml").open("rb") as manifest:
            version = tomllib.load(manifest)["project"]["version"]
    source_paths = sorted(content_changes | tracked_changes | {
        path for path in untracked if is_input(path)})
    return {
        "revision": revision,
        "version": version,
        "worktree_dirty": bool(status) or bool(content_changes),
        "source_dirty": bool(source_paths),
        "source_paths": source_paths,
        "input_sha256": digest.hexdigest(),
        "input_count": len(inputs),
        "untracked_output_prefixes": list(prefixes),
        "untracked_report_exclusions": sorted(reports - tracked),
    }


def execute(checks: Iterable[Check],
            locations: Mapping[str, Path] | None = None) -> tuple[int, dict]:
    """Run an arbitrary nonempty, uniquely named ordered plan, stopping on failure."""
    aliases = sorted((locations or {}).items(), key=lambda item: len(str(item[1])),
                     reverse=True)

    def portable(value: str) -> str:
        for label, path in aliases:
            value = value.replace(str(path), "${" + label + "}")
        return value

    pending = list(checks)
    if not pending:
        raise ValueError("verification plan must contain at least one check")
    names = [check.name for check in pending]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("verification check names must be nonempty and unique")
    results = []
    exit_code = 0
    for check in pending:
        print(f"\n=== {check.name} ===", flush=True)
        started = time.monotonic()
        error = None
        try:
            result = subprocess.run(
                check.command, cwd=check.cwd,
                env={**os.environ, **check.environment}, check=False,
            )
            code = result.returncode
        except OSError as exc:
            code, error = 1, str(exc)
            print(f"{check.name}: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            code, error = 130, "interrupted"
        entry = {
            "name": check.name, "command": [portable(part) for part in check.command],
            "cwd": portable(str(check.cwd)), "returncode": code,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        if error is not None:
            entry["error"] = portable(error)
        results.append(entry)
        if code:
            exit_code = code if code > 0 else 128 - code
            break
    return exit_code, {
        "status": "failed" if exit_code else "passed",
        "checks": results,
        "not_run": [check.name for check in pending[len(results):]],
    }


def verify(checks: Iterable[Check], roots: Mapping[str, Path], git: str,
           locations: Mapping[str, Path] | None = None, *,
           require_clean: bool = False,
           requested_refs: Mapping[str, str] | None = None,
           expected_revisions: Mapping[str, str] | None = None,
           excluded_reports: Iterable[Path] = ()) -> tuple[int, dict]:
    """Bind execution to before/after inputs and optional requested/exact refs.

    Ordinary dirty checkouts remain usable, but input/HEAD/index drift always
    fails. Release mode additionally requires clean authored inputs at both
    boundaries. These editable-source receipts do not certify distributions.
    """
    pending = list(checks)
    requested_refs, expected_revisions = requested_refs or {}, expected_revisions or {}
    repositories, errors = {}, []
    for name, root in roots.items():
        before = repository_state(name, root, git, excluded_reports)
        requested, expected = requested_refs.get(name), expected_revisions.get(name)
        resolved = None
        if requested:
            resolution = subprocess.run(
                (git, "rev-parse", "--verify", "--end-of-options", requested + "^{commit}"),
                cwd=root, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            )
            if resolution.returncode:
                errors.append(f"{name}: requested ref cannot be resolved")
            else:
                resolved = resolution.stdout.strip()
                if resolved != before["revision"]:
                    errors.append(f"{name}: requested ref differs from HEAD")
        if expected is not None:
            if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", expected):
                errors.append(f"{name}: expected revision must be a full commit object ID")
            elif expected.lower() != before["revision"]:
                errors.append(f"{name}: expected revision differs from HEAD")
        if require_clean and before["source_dirty"]:
            errors.append(f"{name}: uncommitted source changes before execution")
        repositories[name] = {
            "requested_ref": requested, "requested_ref_revision": resolved,
            "expected_revision": expected, "before": before,
        }
    if errors:
        code, receipt = 1, {"status": "failed", "checks": [],
                            "not_run": [check.name for check in pending]}
    else:
        code, receipt = execute(pending, locations)
    for name, root in roots.items():
        binding = repositories[name]
        try:
            after = repository_state(name, root, git, excluded_reports)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
            binding["after"] = None
            errors.append(f"{name}: cannot identify source after execution")
            continue
        binding["after"] = after
        before = binding["before"]
        if (before["revision"], before["input_sha256"]) != (
                after["revision"], after["input_sha256"]):
            errors.append(f"{name}: source or HEAD changed during execution")
        if require_clean and after["source_dirty"]:
            errors.append(f"{name}: uncommitted source changes after execution")
    if errors:
        code = code or 1
        receipt["status"] = "failed"
    receipt.update(repositories=repositories, integrity_errors=errors,
                   source_integrity="failed" if errors else "unchanged",
                   require_clean=require_clean)
    return code, receipt


def plan(math_root: Path, symer_root: Path, craft_root: Path, scratch: Path,
         python: str, uv: str, node: str, cmake: str, ctest: str, cxx: str, bash: str,
         npm: str | None = None) -> list[Check]:
    """Describe the real repository commands; do not emulate their test logic."""
    math_env, symer_env = scratch / "quadmath-env", scratch / "symergetics-env"
    executable_dir = "Scripts" if os.name == "nt" else "bin"
    executable_name = "python.exe" if os.name == "nt" else "python"
    mp = str(math_env / executable_dir / executable_name)
    sp = str(symer_env / executable_dir / executable_name)
    common = {"MPLBACKEND": "Agg", "UV_PYTHON": python, "UV_FROZEN": "true"}
    math_common = {**common, "UV_PROJECT_ENVIRONMENT": str(math_env)}
    symer_common = {**common, "UV_PROJECT_ENVIRONMENT": str(symer_env)}
    build = scratch / "native"
    checks = [
        Check("QuadMath locked environment", (uv, "sync", "--frozen", "--python", python,
              "--group", "dev"), math_root,
              math_common),
        Check("symergetics locked environment", (uv, "sync", "--frozen", "--python", python,
              "--extra", "scientific", "--extra", "test"), symer_root,
              symer_common),
        Check("QuadMath tests and coverage", (bash, str(math_root / "run_all.sh"),
              "--test-only"), scratch, math_common),
        Check("symergetics tests and coverage", (sp, "-m", "pytest", "tests/", "-q", "-o",
              "addopts=--strict-markers --strict-config --cov=symergetics "
              "--cov-report=term-missing --cov-fail-under=85"), symer_root, symer_common),
        Check("QuadCraft game and shared tests", (sp, "games/run_games.py", "--test"),
              craft_root, {**common, "PYTHONPATH": str(craft_root / "games")}),
        Check("QuadCraft structural validation", (sp, "games/run_games.py", "--validate"),
              craft_root, common),
        Check("QuadCraft native configure", (cmake, "-S", str(craft_root), "-B", str(build),
              "-DQUADCRAFT_BUILD_GAME=OFF", "-DBUILD_TESTING=ON", "-DCMAKE_BUILD_TYPE=Release"),
              craft_root),
        Check("QuadCraft native build", (cmake, "--build", str(build), "--parallel", "2"),
              craft_root),
        Check("QuadCraft native tests", (ctest, "--output-on-failure", "--no-tests=error"), build),
        Check("Five-implementation interoperability", (sp, "examples/suite_interoperability.py",
              "--quadmath-root", str(math_root), "--quadcraft-root", str(craft_root),
              "--node", node, "--cxx", cxx), symer_root, symer_common),
        Check("QuadMath API glossary freshness", (mp, "quadmath/scripts/generate_glossary.py",
              "--check"), math_root, math_common),
        Check("QuadMath manuscript validation", (mp, "quadmath/scripts/validate_markdown.py",
              "--strict"), math_root, math_common),
        Check("Suite authored-documentation links", (mp, str(math_root / "docs/development/check_links.py"),
              str(math_root), str(symer_root), str(craft_root)),
              math_root, math_common),
        Check("symergetics manuscript assets and structure", (sp, "-I",
              str(symer_root / "paper/validation/validate_paper.py"), "--paper-dir",
              str(symer_root / "paper"), "--output", str(scratch / "paper-validation.md")),
              symer_root, symer_common),
        Check("symergetics manuscript image integration", (sp, "-I",
              str(symer_root / "paper/validation/check_images.py"), "--paper-dir",
              str(symer_root / "paper"), "--output", str(scratch / "paper-images.md")),
              symer_root, symer_common),
    ]
    if npm is not None:
        checks.append(Check(
            "QuadCraft live Chromium gate", (npm, "test"),
            craft_root / "games/tests/browser", {**common, "PYTHON": sp},
        ))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--quadmath-root", type=Path, default=here.parent / "QuadMath")
    parser.add_argument("--quadcraft-root", type=Path, default=here.parent / "QuadCraft")
    parser.add_argument("--python", default="3.13", help="Python version supplied to uv (default: 3.13)")
    parser.add_argument("--report", type=Path, help="Write the execution receipt as JSON")
    parser.add_argument("--browser", action="store_true",
                        help="Include the real Chromium gate; install its locked tooling first")
    parser.add_argument("--require-clean", action="store_true",
                        help="Require clean tracked inputs and non-output untracked inputs before/after")
    for repository in ("quadmath", "symergetics", "quadcraft"):
        parser.add_argument(f"--{repository}-ref", help="Requested Git ref; must resolve to HEAD")
        parser.add_argument(f"--{repository}-expected-revision",
                            help="Expected full commit object ID; must equal HEAD")
    args = parser.parse_args()
    python_selector = args.python
    interpreter = Path(args.python).expanduser()
    explicit_interpreter = interpreter.is_absolute() or "/" in args.python or "\\" in args.python
    if explicit_interpreter:
        interpreter = interpreter.absolute()
        python_selector = str(interpreter)
    math_root, craft_root = args.quadmath_root.resolve(), args.quadcraft_root.resolve()
    for root, required in ((math_root, "pyproject.toml"), (math_root, "src/quadmath/core/quadray.py"),
                           (craft_root, "CMakeLists.txt"), (craft_root, "games/run_games.py")):
        if not (root / required).is_file():
            parser.error(f"missing checkout file: {root / required}")
    tools = {}
    required_tools = {name: name for name in ("bash", "git", "uv", "node", "cmake", "ctest")}
    required_tools["cxx"] = os.environ.get("CXX", "c++")
    if args.browser:
        required_tools["npm"] = "npm"
    for name, command in required_tools.items():
        executable = shutil.which(command)
        if executable is None:
            parser.error(f"required executable not found: {command}")
        tools[name] = executable
    roots = {"QuadMath": math_root, "symergetics": here, "QuadCraft": craft_root}
    with tempfile.TemporaryDirectory(prefix="synergetics-verify-") as temporary:
        checks = plan(math_root, here, craft_root, Path(temporary), python_selector,
                      tools["uv"], tools["node"], tools["cmake"], tools["ctest"],
                      tools["cxx"], tools["bash"], tools.get("npm"))
        locations = {**roots, "scratch": Path(temporary),
                     **{f"tool_{name}": Path(path) for name, path in tools.items()}}
        if explicit_interpreter:
            locations["interpreter"] = interpreter
        try:
            code, receipt = verify(
                checks, roots, tools["git"], locations, require_clean=args.require_clean,
                requested_refs={name: getattr(args, name.lower() + "_ref")
                                for name in roots},
                expected_revisions={name: getattr(args, name.lower() + "_expected_revision")
                                    for name in roots},
                excluded_reports=[args.report] if args.report else [],
            )
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            parser.error(f"cannot verify source checkouts: {exc}")
    receipt = {"schema_version": 2, "profile": "full" if args.browser else "core",
               "python_requested": "${interpreter}" if explicit_interpreter else args.python,
               **receipt}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "profile": receipt["profile"],
                      "executed": len(receipt["checks"]), "not_run": receipt["not_run"]},
                     indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
