// Dialect smoke test for physical VC4 execution builtins only.
// RUN: vc4-opt %s | FileCheck %s

module {
  %qpu = vc4.get_builtin qpu_id : index
  %numQpus = vc4.get_builtin num_qpus : index
}

// CHECK: module {
// CHECK:   %[[QPU:.*]] = vc4.get_builtin qpu_id : index
// CHECK:   %[[NUMQPUS:.*]] = vc4.get_builtin num_qpus : index
// CHECK: }
