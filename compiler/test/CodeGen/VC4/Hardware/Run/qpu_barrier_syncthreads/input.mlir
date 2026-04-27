// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

// Exploratory final-stage metadata for qpu_barrier_syncthreads.
//
// This hardware-run test intentionally exercises direct QPU hardware features
// that are not yet represented cleanly by the final-stage vc4 dialect as a
// scheduled sink body: register-materialized QPU_NUMBER/ELEMENT_NUMBER, generic
// VPM reads/writes, VDW stores, global mutex I/O, and QPU semaphore barrier
// instructions. The trusted handwritten reference qasm is therefore the
// hardware source of truth for this exploratory gpu.barrier / __syncthreads
// lowering test. Candidate/codegen execution remains disabled until those
// operations can be represented and emitted faithfully.

module attributes {
  "vc4.hardware_run_test.name" = "qpu_barrier_syncthreads",
  "vc4.hardware_run_test.kind" = "hardware-run-exploratory",
  "vc4.hardware_run_test.reference_kernel" = "reference/qpu_barrier_syncthreads.qasm",
  "vc4.hardware_run_test.launch_abi" = {
    public_name = "qpu_barrier_syncthreads_launch",
    tail_policy = "fixed_barrier_modes",
    uniform_words_per_qpu = 14 : i32,
    args = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 12 : i32}
    ],
    builtins = [
      {name = "qpu_num", kind = "qpu_register", materialization = "register"},
      {name = "elem_num", kind = "qpu_register", materialization = "register"},
      {name = "mutex", kind = "qpu_io", materialization = "register"},
      {name = "semaphore", kind = "qpu_signal", materialization = "instruction"},
      {name = "vpm_vdw", kind = "qpu_io", materialization = "register"}
    ],
    physical_uniform_stream = [
      {index = 0 : i32, name = "mode"},
      {index = 1 : i32, name = "run_id"},
      {index = 2 : i32, name = "block_id"},
      {index = 3 : i32, name = "logical_warp_id"},
      {index = 4 : i32, name = "warps_per_block"},
      {index = 5 : i32, name = "iterations"},
      {index = 6 : i32, name = "vpm_base_row"},
      {index = 7 : i32, name = "vpm_rows_per_block"},
      {index = 8 : i32, name = "arrive_sem"},
      {index = 9 : i32, name = "release_sem"},
      {index = 10 : i32, name = "depart_sem"},
      {index = 11 : i32, name = "reset_sem"},
      {index = 12 : i32, name = "result_warp_ptr"},
      {index = 13 : i32, name = "run_result_ptr"}
    ],
    modes = [
      {id = 0 : i32, name = "same_slice_smoke", blocks = 1 : i32, warps_per_block = 2 : i32, iterations = 4 : i32},
      {id = 1 : i32, name = "cross_slice_0_1", blocks = 1 : i32, warps_per_block = 2 : i32, iterations = 4 : i32},
      {id = 2 : i32, name = "cross_slice_0_2", blocks = 1 : i32, warps_per_block = 2 : i32, iterations = 4 : i32},
      {id = 3 : i32, name = "full_block_stress", blocks = 1 : i32, warps_per_block = 12 : i32, iterations = 64 : i32},
      {id = 4 : i32, name = "two_block_partition", blocks = 2 : i32, warps_per_block = 4 : i32, iterations = 32 : i32}
    ],
    barrier_protocol = {
      kind = "four_semaphore_reusable",
      barriers_per_iteration = 2 : i32,
      semaphores_per_block = 4 : i32,
      block0_semaphores = [0 : i32, 1 : i32, 2 : i32, 3 : i32],
      block1_semaphores = [4 : i32, 5 : i32, 6 : i32, 7 : i32]
    }
  }
} {
}
