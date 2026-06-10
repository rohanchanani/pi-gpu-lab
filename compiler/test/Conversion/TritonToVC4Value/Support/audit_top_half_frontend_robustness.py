#!/usr/bin/env python3
"""Audit TTIR importer source for brittle top-half frontend semantics."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"

FORBIDDEN_SOURCE_PATTERNS = (
    "std::regex",
    "#include <regex>",
    "raw_string_ostream",
    "stringify(Type",
    "stringify(Attribute",
    "op->print(",
    ".print(",
    "Operation::remove",
    "->remove()",
    ".remove()",
    "unlink()",
    "eraseWithout",
    "saxpy",
    "vector_add",
    "persistent_loop_skeleton",
    "scalar_if_probe",
    "static_range_probe",
    "tl_range_loop_skeleton",
    "while_probe",
)

REQUIRED_SOURCE_PATTERNS = (
    "LoweringOutcome",
    "finishLowering",
    "IgnoredDeadProofOp",
    "isDeadProofWhitelistOp",
    "verifyValueOutputModule",
    "emitForbiddenValueOutputOp",
    "builtin.unrealized_conversion_cast",
    "isAllowedValueOutputBodyDialect",
    "Argument names are metadata only",
    "READY_FOR_TRITON remains NO",
)

ALLOWED_LOWER_HALF_BOUNDARY_CONTEXT = (
    "dialect == \"vc4kernel\"",
    "dialect == \"ssavc4\"",
    "dialect == \"vc4\"",
)


def fail(message: str) -> None:
    print(f"TOP_HALF_FRONTEND_ROBUSTNESS_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")

    for needle in FORBIDDEN_SOURCE_PATTERNS:
        if needle in source:
            fail(f"forbidden brittle frontend pattern remains: {needle}")

    if "regex" in source:
        fail("regex marker remains in importer source")

    for needle in REQUIRED_SOURCE_PATTERNS:
        if needle not in source:
            fail(f"required hardening marker missing: {needle}")

    accounting_calls = (
        source.count("finishLowering(op") +
        source.count("finishLowering(op.getOperation()")
    )
    if accounting_calls < 19:
        fail("per-op lowering does not consistently use result accounting helper")

    risky_name_patterns = (
        "fixtureName",
        "candidatePath",
        "sourcePath",
        "fileName",
        "filename",
        "getFilename",
    )
    for needle in risky_name_patterns:
        if needle in source:
            fail(f"source/path/fixture semantic hook remains: {needle}")

    broad_drop_patterns = (
        "op->getNumResults() == 0 ||",
        "!planner.isRequiredDataValue(op->getResult(0)))\n      return finishLowering",
        "endswith(\"_ptr\")",
        "ends_with(\"_ptr\")",
        "n_elements -> n",
    )
    for needle in broad_drop_patterns:
        if needle in source:
            fail(f"possible broad success/drop pattern remains: {needle}")

    lower_half_mentions = [
        line.strip()
        for line in source.splitlines()
        if not line.strip().startswith("//")
        and ("vc4kernel" in line or "ssavc4" in line or "dialect == \"vc4\"" in line)
    ]
    for line in lower_half_mentions:
        if not any(allowed in line for allowed in ALLOWED_LOWER_HALF_BOUNDARY_CONTEXT):
            fail(f"lower-half marker appears outside boundary rejection: {line}")

    print("TOP_HALF_FRONTEND_ROBUSTNESS_AUDIT=PASS")
    print("NO_REGEX_SEMANTICS=YES")
    print("NO_FIXTURE_NAME_SEMANTICS=YES")
    print("NO_SOURCE_PATH_SEMANTICS=YES")
    print("NO_PRINTED_IR_OR_TYPE_SEMANTICS=YES")
    print("NO_OPERATION_REMOVE_UNLINK_WORKAROUND=YES")
    print("NO_DIRECT_LOWER_HALF_EMISSION=YES")
    print("NO_BROAD_SUCCESS_WITHOUT_ACCOUNTING=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
