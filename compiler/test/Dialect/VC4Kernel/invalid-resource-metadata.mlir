// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @uses_vpm_false attributes {
    public_name = "uses_vpm_false",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = false,
      uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32,
      vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32,
      semaphores_per_block = 0 : i32
    }
  } {
    // CHECK: VPM operations require uses_vpm = true
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @too_many_rows attributes {
    public_name = "too_many_rows",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = true,
      uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32,
      vpm_rows_per_block = 1 : i32,
      vpm_bytes_per_block = 64 : i32,
      semaphores_per_block = 0 : i32
    }
  } {
    %tile0 = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: VPM allocations exceed vpm_rows_per_block
    %tile1 = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.return
  }
}
