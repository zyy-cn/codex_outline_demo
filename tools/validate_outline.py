#!/usr/bin/env python3
"""
Machine-checkable spec validation.

This demonstrates the pattern: keep your outline as structured data (outline.yaml),
and validate the code and artifacts automatically.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_spec() -> dict:
    # outline.yaml is actually JSON for simplicity (keeps stdlib-only).
    p = REPO_ROOT / "outline.yaml"
    return json.loads(p.read_text(encoding="utf-8"))


def _assert_functions_exist(module_path: Path, required: list[str]) -> None:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    missing = [f for f in required if f not in defs]
    if missing:
        raise SystemExit(f"Missing required functions in {module_path}: {missing}")


def _assert_cli_flags_present(module_path: Path, flags: list[str]) -> None:
    txt = module_path.read_text(encoding="utf-8")
    missing = [f for f in flags if f not in txt]
    if missing:
        raise SystemExit(f"CLI flags not found in code text (expected in argparse): {missing}")


def main() -> None:
    spec = _load_spec()
    module_path = REPO_ROOT / "src" / "pipeline.py"

    _assert_functions_exist(module_path, spec["required_functions"])
    _assert_cli_flags_present(module_path, spec["cli"]["flags"])

    # Artifact schema checks are enforced by tests, but we keep them here as a pattern.
    artifacts = spec["artifacts"]
    for name, rule in artifacts.items():
        if "required_keys" not in rule:
            raise SystemExit(f"artifact rule for {name} missing required_keys")

    print("outline validation: OK")


if __name__ == "__main__":
    main()
