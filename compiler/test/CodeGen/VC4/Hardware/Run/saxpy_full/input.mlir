// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for saxpy_full.
//
// This test is the tail-safe SAXPY successor to saxpy_16.  The trusted
// reference qasm computes y[i] = alpha * x[i] + y[i] for arbitrary tested n,
// distributes vector chunks by qpu_id/num_qpus, and dynamically programs VDW
// DEPTH for the final partial vector.  Candidate/codegen execution remains
// disabled until the backend can emit the qasm/launcher bundle.

module attributes {
  "vc4.hardware_run_test.name" = "saxpy_full",
  "vc4.hardware_run_test.kind" = "hardware-run-reference",
  "vc4.hardware_run_test.reference_kernel" = "reference/saxpy_full.qasm",
  "vc4.hardware_run_test.launch_abi" = {
    public_name = "saxpy_full_launch",
    tail_policy = "tail_safe",
    uniform_words_per_qpu = 6 : i32,
    args = [
      {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
      {name = "y", kind = "buffer", direction = "inout", elem_type = "f32", uniform_index = 1 : i32},
      {name = "alpha", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 2 : i32},
      {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
    ],
    builtins = [
      {name = "qpu_id", kind = "qpu_num", materialization = "uniform_suffix", uniform_index = 4 : i32},
      {name = "num_qpus", kind = "num_qpus", materialization = "uniform_suffix", uniform_index = 5 : i32}
    ],
    work_distribution = {
      base_element = "qpu_id * 16",
      stride_elements = "num_qpus * 16",
      tail_store = "dynamic_vdw_depth"
    }
  }
} {
}
