// RUN: not vc4-opt %s --verify-vc4kernel --allow-unregistered-dialect --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @tt_bad attributes {
    public_name = "tt_bad",
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
    // CHECK: dialect 'tt' is forbidden inside vc4kernel
    "tt.fake"() : () -> ()
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @gpu_bad attributes {
    public_name = "gpu_bad",
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
    // CHECK: dialect 'gpu' is forbidden inside vc4kernel
    "gpu.fake"() : () -> ()
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @linalg_bad attributes {
    public_name = "linalg_bad",
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
    // CHECK: dialect 'linalg' is forbidden inside vc4kernel
    "linalg.fake"() : () -> ()
    vc4kernel.return
  }
}
