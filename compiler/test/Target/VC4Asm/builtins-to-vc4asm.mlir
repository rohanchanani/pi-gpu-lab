// Translation smoke test for physical VC4 execution builtins only.
// RUN: vc4-translate --mlir-to-vc4asm %s | FileCheck %s

module {
  %qpu = vc4.get_builtin qpu_id : index
  %numQpus = vc4.get_builtin num_qpus : index
}

// CHECK: ; vc4asm-like output generated from staged vc4 IR
// CHECK: ; contract: schematic printer only, not executable codegen
// CHECK: ; placeholders are marked explicitly where exact vc4asm syntax is TBD
// CHECK: ; Execution builtins
// CHECK: ; builtin_qpu_id = vc4.get_builtin qpu_id : index
// CHECK: mov builtin_qpu_id, qpu_id
// CHECK: ; builtin_num_qpus = vc4.get_builtin num_qpus : index
// CHECK: mov builtin_num_qpus, num_qpus
