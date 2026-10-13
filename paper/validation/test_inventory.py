"""Report source definitions, not pytest cases or pass/coverage results.

Run from any directory: python paper/validation/test_inventory.py
No test modules are imported or executed. Source hashes bind the inventory to
its inputs even if the checkout is dirty or the report predates a new commit.
"""

import ast
import hashlib
import json
from pathlib import Path


def inventory(root: Path) -> dict:
    """Count test-prefixed AST function definitions in tests/test_*.py recursively."""
    files = []
    source_digest = hashlib.sha256()
    for path in sorted((root / "tests").rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        source = path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        tree = ast.parse(source, filename=relative)
        functions = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
        files.append({"path": relative, "sha256": digest, "functions": functions})
        source_digest.update(relative.encode("utf-8") + b"\0" + source + b"\0")
    return {
        "scope": "tests/**/test_*.py; all test-prefixed AST function definitions",
        "semantics": "definitions, including nested functions; not pytest collection or execution",
        "source_sha256": source_digest.hexdigest(),
        "test_files": len(files),
        "test_function_definitions": sum(item["functions"] for item in files),
        "files": files,
    }


if __name__ == "__main__":
    print(json.dumps(inventory(Path(__file__).resolve().parents[2]), indent=2))
