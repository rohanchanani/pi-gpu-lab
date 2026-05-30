// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @vpm
// CHECK: ssavc4.vpm.write
// CHECK: ssavc4.vpm.read
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @vpm attributes {
    public_name = "vpm",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = true, uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32, vpm_rows_per_block = 1 : i32,
      vpm_bytes_per_block = 64 : i32, semaphores_per_block = 0 : i32
    }
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %c1 : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %c0, %v, %full {orientation = #vc4kernel.vpm_orientation<row>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %r = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<row>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
