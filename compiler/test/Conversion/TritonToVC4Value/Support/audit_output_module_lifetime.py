#!/usr/bin/env python3
"""Audit TTIR importer output-module lifetime invariants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"

FORBIDDEN_SOURCE = (
    "getFuncOp()->remove",
    "->remove()",
    ".remove()",
    "Operation::remove",
    "unlinking is enough",
    "Destroying this generic Triton function",
    "trips assertions",
)

REQUIRED_SOURCE = (
    "class TTIRToValueModuleBuilder",
    "OwningOpRef<ModuleOp>",
    "ModuleOp::create",
    "verifyValueOutputModule",
    "commitOutputModule",
    "builder.setInsertionPointToEnd(outputModule.getBody())",
    "op->erase()",
    "splice",
)


def fail(message: str) -> None:
    print(f"TTIR_OUTPUT_MODULE_LIFETIME_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    for needle in FORBIDDEN_SOURCE:
        if needle in source:
            fail(f"forbidden remove/unlink lifetime marker in implementation: {needle}")

    for needle in REQUIRED_SOURCE:
        if needle not in source:
            fail(f"required output-module lifetime marker missing: {needle}")

    print("TTIR_OUTPUT_MODULE_LIFETIME_AUDIT=PASS")
    print("TT_FUNC_REMOVE_WORKAROUND=NO")
    print("OUTPUT_MODULE_CONSTRUCTION=YES")
    print("ATOMIC_IMPORT_POLICY=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
