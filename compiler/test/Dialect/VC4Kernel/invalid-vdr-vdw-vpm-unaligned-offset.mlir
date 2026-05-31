// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%ptr : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"}
    ],
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
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vdr_load_to_vpm byte_offset must be statically 4-byte aligned
    vc4kernel.vdr_load_to_vpm %ptr, %c2, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.return
  }
}
