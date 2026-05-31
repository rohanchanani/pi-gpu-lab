// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%row : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"}
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
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %c0 : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: VPM row must be a scalar i32 constant in Stage 1
    vc4kernel.vpm_write_fragment %tile, %row, %v, %full {orientation = #vc4kernel.vpm_orientation<row>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
