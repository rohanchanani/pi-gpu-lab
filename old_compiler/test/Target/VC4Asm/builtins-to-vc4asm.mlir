// Translation smoke test for physical VC4 execution builtins only.
// RUN: vc4-translate --mlir-to-vc4asm %s | FileCheck %s

module {
  %qpu = vc4.get_builtin qpu_id : index
  %numQpus = vc4.get_builtin num_qpus : index
}

// CHECK: ; inspection-only VC4 assembly sketch generated from lowered vc4 IR
// CHECK: ; contract: intermediate inspection output, not the final artifact boundary
// CHECK: ; placeholders are marked explicitly where exact VC4 assembly syntax is still TBD
// CHECK: ; Builtin suffix reads
// CHECK: ; builtin_qpu_id = vc4.get_builtin qpu_id : index
// CHECK: mov builtin_qpu_id, unif
// CHECK: ; builtin_num_qpus = vc4.get_builtin num_qpus : index
// CHECK: mov builtin_num_qpus, unif
