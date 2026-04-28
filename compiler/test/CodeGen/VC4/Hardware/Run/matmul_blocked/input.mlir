// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Final-stage hardware-run metadata for matmul_blocked.
//
// This test models the CUDA-like cooperative-block lowering for a blocked f32
// row-major matmul:
//
//   C[M,N] = A[M,K] * B[K,N]
//
// One resident block contains 12 logical QPU warps.  Each QPU warp computes one
// output row of a 12x16 C tile, and each SIMD lane computes one column.  The
// block stages a 12x16 B tile in VPM rows and uses the locked four-semaphore
// reusable barrier as the implementation of gpu.barrier / __syncthreads().
// Candidate codegen execution remains disabled until the backend can emit the
// qasm and launcher bundle.

module attributes {
  "vc4.hardware_run_test.name" = "matmul_blocked",
  "vc4.hardware_run_test.kind" = "hardware-run-reference",
  "vc4.hardware_run_test.reference_kernel" = "reference/matmul_blocked.qasm",
  "vc4.hardware_run_test.launch_abi" = {
    public_name = "matmul_blocked_launch",
    tail_policy = "tail_safe",
    uniform_words_per_qpu = 14 : i32,
    args = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
      {name = "c", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 2 : i32},
      {name = "m", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
      {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
      {name = "k", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32}
    ],
    builtins = [
      {name = "padded_k_stride", kind = "runtime_internal", materialization = "uniform_suffix", uniform_index = 6 : i32},
      {name = "padded_n_stride", kind = "runtime_internal", materialization = "uniform_suffix", uniform_index = 7 : i32},
      {name = "tile_row_base", kind = "block_id_y", materialization = "uniform_suffix", uniform_index = 8 : i32},
      {name = "tile_col_base", kind = "block_id_x", materialization = "uniform_suffix", uniform_index = 9 : i32},
      {name = "logical_warp_id", kind = "warp_id", materialization = "uniform_suffix", uniform_index = 10 : i32},
      {name = "warps_per_block", kind = "warps_per_block", materialization = "uniform_suffix", uniform_index = 11 : i32},
      {name = "padded_k_limit", kind = "runtime_internal", materialization = "uniform_suffix", uniform_index = 12 : i32},
      {name = "vpm_base_row", kind = "workgroup_memory_base", materialization = "uniform_suffix", uniform_index = 13 : i32}
    ],
    work_distribution = {
      device_model = "one_cuda_like_sm",
      block_shape = "12_qpu_warps_x_16_lanes",
      output_tile = "12_rows_x_16_columns",
      row = "tile_row_base + logical_warp_id",
      column = "tile_col_base + ELEMENT_NUMBER",
      k_tile_rows = 12 : i32,
      shared_memory = "B tile in VPM rows vpm_base_row..vpm_base_row+11",
      barriers = "four_semaphore_reusable_syncthreads_after_loads_and_after_reads",
      store_tail = "dynamic_vdw_depth"
    },
    memory_paths = {
      global_loads = "TMU direct memory lookup",
      workgroup_memory = "VPM horizontal 32-bit rows",
      global_stores = "VPM staging plus VDW DMA store",
      vpm_vdw_serialization = "global_mutex"
    },
    runtime_resource_policy = {
      gpu_allocations_per_boot = 1 : i32,
      code_copies_per_boot = 1 : i32,
      resident_blocks_per_wave = 1 : i32,
      qpu_requests_per_wave = 12 : i32,
      semaphore_ids = "0,1,2,3"
    }
  }
} {
}
