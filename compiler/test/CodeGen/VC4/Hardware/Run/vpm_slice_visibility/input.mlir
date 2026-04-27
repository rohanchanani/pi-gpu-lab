// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Exploratory final-stage metadata for vpm_slice_visibility.
//
// This hardware-run test intentionally exercises direct QPU hardware features
// that are not yet represented cleanly by the final-stage vc4 dialect as a
// scheduled sink body: register-materialized QPU_NUMBER/ELEMENT_NUMBER,
// VPM generic reads/writes, VDW stores, global mutex I/O, and QPU semaphores.
// The trusted handwritten reference qasm is therefore the hardware source of
// truth for this exploratory test.  Candidate/codegen execution remains
// disabled until those operations can be represented and emitted faithfully.

module attributes {
  "vc4.hardware_run_test.name" = "vpm_slice_visibility",
  "vc4.hardware_run_test.kind" = "hardware-run-exploratory",
  "vc4.hardware_run_test.reference_kernel" = "reference/vpm_slice_visibility.qasm",
  "vc4.hardware_run_test.launch_abi" = {
    public_name = "vpm_slice_visibility_launch",
    tail_policy = "fixed_representative_pairs",
    uniform_words_per_qpu = 9 : i32,
    args = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32}
    ],
    builtins = [
      {name = "qpu_num", kind = "qpu_register", materialization = "register"},
      {name = "elem_num", kind = "qpu_register", materialization = "register"},
      {name = "mutex", kind = "qpu_io", materialization = "register"},
      {name = "semaphore", kind = "qpu_signal", materialization = "instruction"},
      {name = "vpm_vdw", kind = "qpu_io", materialization = "register"}
    ],
    modes = [
      {id = 0 : i32, name = "single_qpu_vpm_sanity"},
      {id = 1 : i32, name = "visibility_pair"},
      {id = 2 : i32, name = "collision_pair"}
    ]
  }
} {
}
