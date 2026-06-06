// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @vpm_multiple_allocations
// CHECK-SAME: total_vpm_rows_per_block = 3 : i32
// CHECK-SAME: user_vpm_rows_per_block = 3 : i32
// CHECK-COUNT-2: ssavc4.vpm.write
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @vpm_multiple_allocations attributes {
    public_name = "vpm_multiple_allocations",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v11 = vc4kernel.fragment_const {value = dense<11> : vector<16xi32>} : vector<16xi32>
    %v22 = vc4kernel.fragment_const {value = dense<22> : vector<16xi32>} : vector<16xi32>
    %tile0 = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %tile1 = vc4kernel.vpm_alloc {rows = 2 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile0, %c0, %v11, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile1, %c1, %v22, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
