// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for matmul_naive.
//
// This test models the CUDA-like independent-vector lowering for a naive f32
// row-major matmul:
//
//   C[M,N] = A[M,K] * B[K,N]
//
// One QPU request is one logical 16-lane warp.  ELEMENT_NUMBER supplies the
// lane/column within a 16-column tile.  qpu_id and num_qpus are logical launch
// builtins carried as uniforms for grid-stride row distribution.  Candidate
// codegen execution remains disabled until the backend can emit the qasm and
// launcher bundle.

module attributes {
  "vc4.hardware_run_test.name" = "matmul_naive",
  "vc4.hardware_run_test.kind" = "hardware-run-reference",
  "vc4.hardware_run_test.reference_kernel" = "reference/matmul_naive.qasm",
  "vc4.hardware_run_test.launch_abi" = {
    public_name = "matmul_naive_launch",
    tail_policy = "tail_safe",
    uniform_words_per_qpu = 8 : i32,
    args = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
      {name = "c", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 2 : i32},
      {name = "m", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
      {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
      {name = "k", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32}
    ],
    builtins = [
      {name = "qpu_id", kind = "qpu_num", materialization = "uniform_suffix", uniform_index = 6 : i32},
      {name = "num_qpus", kind = "num_qpus", materialization = "uniform_suffix", uniform_index = 7 : i32}
    ],
    work_distribution = {
      logical_warp = "one QPU request",
      lane = "ELEMENT_NUMBER",
      row = "qpu_id + t * num_qpus",
      column = "column_tile_base + lane",
      column_tile_width = 16 : i32,
      store_tail = "dynamic_vdw_depth"
    },
    memory_paths = {
      global_loads = "TMU direct memory lookup",
      global_stores = "VPM staging plus VDW DMA store",
      vpm_vdw_serialization = "global_mutex"
    }
  }
} {
}
