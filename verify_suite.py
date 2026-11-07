#!/usr/bin/env python3
"""Verify all three Synergetics repositories without replacing their environments.

Requires bash, uv, Node.js, CMake and a C++17 compiler. Locks are honored;
Python environments and the native build live in an automatically removed
scratch directory. Use --report for a machine-readable execution receipt.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Check:
    name: str
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str] = field(default_factory=dict)


def execute(checks: Iterable[Check]) -> tuple[int, dict]:
    """Run checks in order, stopping on failure and identifying unexecuted checks."""
    pending = list(checks)
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
            "name": check.name, "command": list(check.command),
            "cwd": str(check.cwd), "returncode": code,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        if error is not None:
            entry["error"] = error
        results.append(entry)
        if code:
            exit_code = code if code > 0 else 128 - code
            break
    return exit_code, {
        "status": "failed" if exit_code else "passed",
        "checks": results,
        "not_run": [check.name for check in pending[len(results):]],
    }


def plan(math_root: Path, symer_root: Path, craft_root: Path, scratch: Path,
         python: str, uv: str, node: str, cmake: str, ctest: str, cxx: str, bash: str) -> list[Check]:
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
    return [
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
        Check("QuadCraft native tests", (ctest, "--output-on-failure"), build),
        Check("Five-implementation interoperability", (sp, "examples/suite_interoperability.py",
              "--quadmath-root", str(math_root), "--quadcraft-root", str(craft_root),
              "--node", node, "--cxx", cxx), symer_root, symer_common),
        Check("QuadMath API glossary freshness", (mp, "quadmath/scripts/generate_glossary.py",
              "--check"), math_root, math_common),
        Check("QuadMath manuscript validation", (mp, "quadmath/scripts/validate_markdown.py",
              "--strict"), math_root, math_common),
        Check("QuadMath documentation links", (mp, "docs/development/check_links.py"),
              math_root, math_common),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--quadmath-root", type=Path, default=here.parent / "QuadMath")
    parser.add_argument("--quadcraft-root", type=Path, default=here.parent / "QuadCraft")
    parser.add_argument("--python", default="3.13", help="Python version supplied to uv (default: 3.13)")
    parser.add_argument("--report", type=Path, help="Write the execution receipt as JSON")
    args = parser.parse_args()
    math_root, craft_root = args.quadmath_root.resolve(), args.quadcraft_root.resolve()
    for root, required in ((math_root, "pyproject.toml"), (math_root, "src/quadmath/core/quadray.py"),
                           (craft_root, "CMakeLists.txt"), (craft_root, "games/run_games.py")):
        if not (root / required).is_file():
            parser.error(f"missing checkout file: {root / required}")
    tools = {}
    for name in ("bash", "uv", "node", "cmake", "ctest", os.environ.get("CXX", "c++")):
        executable = shutil.which(name)
        if executable is None:
            parser.error(f"required executable not found: {name}")
        tools[name] = executable
    with tempfile.TemporaryDirectory(prefix="synergetics-verify-") as temporary:
        checks = plan(math_root, here, craft_root, Path(temporary), args.python,
                      tools["uv"], tools["node"], tools["cmake"], tools["ctest"],
                      tools[os.environ.get("CXX", "c++")], tools["bash"])
        code, receipt = execute(checks)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "executed": len(receipt["checks"]),
                      "not_run": receipt["not_run"]}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
