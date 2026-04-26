// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Exploratory final-stage metadata for qpu_num_register.
//
// The current vc4 dialect may not yet have a scheduled sink operation that
// directly represents reading the hardware B-regfile QPU_NUMBER register.  The
// trusted reference qasm therefore remains the source of truth for this test.
// Candidate/codegen execution is disabled until the dialect and codegen can
// represent a register-materialized qpu_num builtin faithfully.

module attributes {
  "vc4.hardware_run_test.name" = "qpu_num_register",
  "vc4.hardware_run_test.kind" = "hardware-run",
  "vc4.hardware_run_test.reference_kernel" = "reference/qpu_num_register.qasm",
  "vc4.hardware_run_test.launch_abi" = {
    public_name = "qpu_num_register_launch",
    tail_policy = "exact_16_requests",
    uniform_words_per_qpu = 1 : i32,
    args = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32}
    ],
    builtins = [
      {name = "qpu_num", kind = "qpu_register", materialization = "register"}
    ]
  }
} {
}
