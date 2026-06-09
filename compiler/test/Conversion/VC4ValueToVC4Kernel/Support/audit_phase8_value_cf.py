#!/usr/bin/env python3
"""Phase 8 value control-flow static foundation audit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def fail(message: str) -> None:
    raise SystemExit(f"PHASE8_VALUE_CF_AUDIT=FAIL: {message}")


def quoted_strings(text: str) -> set[str]:
    return set(re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()

    value_converter_path = (
        root
        / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"
    )
    ttir_converter_path = (
        root / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"
    )
    vc4value_ops_path = (
        root / "compiler/include/vc4/Dialect/VC4Value/IR/VC4ValueOps.td"
    )
    vc4_opt_path = root / "compiler/tools/vc4-opt/vc4-opt.cpp"
    vc4_triton_opt_path = root / "compiler/tools/vc4-triton-opt/vc4-triton-opt.cpp"

    value_converter = read(value_converter_path)
    value_code = strip_comments(value_converter)
    ttir_code = strip_comments(read(ttir_converter_path))
    vc4value_ops = strip_comments(read(vc4value_ops_path))
    vc4_opt = strip_comments(read(vc4_opt_path))
    vc4_triton_opt = strip_comments(read(vc4_triton_opt_path))

    direct_value_lower_half_hits = []
    for needle in ("ssavc4.", "vc4.qpu.", "vc4.module", "ssavc4::", "vc4::FuncOp"):
        if needle in value_code:
            direct_value_lower_half_hits.append(needle)
    if direct_value_lower_half_hits:
        fail(f"direct value-to-lower-half emission: {direct_value_lower_half_hits}")

    direct_ttir_vc4kernel_hits = []
    for needle in ("vc4kernel.", "VC4Kernel", "vc4kernel::"):
        if needle in ttir_code:
            direct_ttir_vc4kernel_hits.append(needle)
    if direct_ttir_vc4kernel_hits:
        fail(f"direct TTIR-to-VC4Kernel path: {direct_ttir_vc4kernel_hits}")

    if "Operation::remove" in value_code or ".remove()" in value_code:
        fail("value converter uses remove/unlink-style operation ownership")

    custom_scf_hits = []
    for needle in (
        "scf::IfOp",
        "scf::ForOp",
        "scf::WhileOp",
        "populateSCFToControlFlowConversionPatterns",
        "createSCFToControlFlowPass",
    ):
        if needle in value_code or needle in ttir_code:
            custom_scf_hits.append(needle)
    if custom_scf_hits:
        fail(f"custom SCF lowering found outside upstream pass boundary: {custom_scf_hits}")

    for tool_name, tool_code in (
        ("vc4-opt", vc4_opt),
        ("vc4-triton-opt", vc4_triton_opt),
    ):
        if "registerSCFToControlFlowPass" not in tool_code:
            fail(f"{tool_name} does not register upstream SCF-to-CF pass")

    if "builder.create<cf::BranchOp>" not in value_code:
        fail("cf.br lowering does not create a real cf::BranchOp")
    if "builder.create<cf::CondBranchOp>" not in value_code:
        fail("cf.cond_br lowering does not create a real cf::CondBranchOp")
    if "emitRawSCFDiagnostic" not in value_code or "--convert-scf-to-cf" not in value_converter:
        fail("raw SCF diagnostic does not point to explicit upstream scf-to-cf")

    vc4value_cf_hits = []
    for pattern in (
        r"\bVC4Value_.*(Branch|CondBranch|If|For|While|Yield|Switch)",
        r"\bdef .*vc4value.*(branch|br|if|for|while|yield|switch)",
    ):
        vc4value_cf_hits.extend(re.findall(pattern, vc4value_ops, flags=re.I))
    if vc4value_cf_hits:
        fail(f"vc4value grew control-flow ops: {vc4value_cf_hits}")

    phase8_impl_text = "\n".join(
        [
            value_converter,
            read(root / "compiler/docs/vc4_vector_triton_phase8_control_flow_lock.md"),
            read(root / "compiler/docs/vc4_value_surface_spec.md"),
            read(root / "compiler/docs/vc4_value_to_vc4kernel_planning.md"),
        ]
    )
    forbidden_workaround_patterns = {
        "select-only control-flow flattening": r"select[- ]only|flatten control|flattening control",
        "branch semantics by comment": r"comment[- ]only|comments only|branch semantics.*comment",
        "avoid branch lowering": r"avoid branch lowering|avoid branches",
        "fixture-name special case": r"fixture[-_ ]name special|candidate[-_ ]name special|path special case",
    }
    for label, pattern in forbidden_workaround_patterns.items():
        if re.search(pattern, phase8_impl_text, flags=re.I):
            fail(label)

    source_strings = quoted_strings(value_code + "\n" + ttir_code)
    test_names = {
        p.stem
        for p in (root / "compiler/test/Conversion/VC4ValueToVC4Kernel").glob("*.mlir")
    }
    fixture_name_hits = sorted(source_strings.intersection(test_names))
    if fixture_name_hits:
        fail(f"converter references fixture/test names: {fixture_name_hits}")

    print("PHASE8_VALUE_CF_AUDIT=PASS")
    print("DIRECT_VALUE_TO_LOWER_HALF=NO")
    print("DIRECT_TTIR_TO_VC4KERNEL=NO")
    print("RAW_SCF_IN_VC4KERNEL_POLICY=NO")
    print("PRODUCER_DIALECTS_IN_VC4KERNEL_POLICY=NO")
    print("FIXTURE_NAME_SPECIAL_CASES=NO")
    print("CUSTOM_SCF_TO_CF_LOWERING=NO")
    print("VC4VALUE_CONTROL_FLOW_OPS=NO")
    print("BRANCH_SEMANTICS_COMMENTS_ONLY=NO")
    print("READY_FOR_TRITON=NO")


if __name__ == "__main__":
    main()
