// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
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
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: cf successor operands may not carry !vc4kernel.vpm_tile in Stage 1
    cf.br ^use(%tile : !vc4kernel.vpm_tile)
  ^use(%arg_tile: !vc4kernel.vpm_tile):
    vc4kernel.return
  }
}
