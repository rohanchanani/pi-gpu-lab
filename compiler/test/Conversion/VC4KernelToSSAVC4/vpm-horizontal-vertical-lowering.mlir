// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @vpm_horizontal_vertical
// CHECK-SAME: requires_vpm_base_row_builtin = true
// CHECK-SAME: total_vpm_rows_per_block = 16 : i32
// CHECK-SAME: user_vpm_rows_per_block = 16 : i32
// CHECK: ssavc4.vpm.write
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK-SAME: subword = #ssavc4.vpm_subword<none>
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w32>
// CHECK: ssavc4.vpm.read
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK-SAME: x = 3 : i32
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @vpm_horizontal_vertical attributes {
    public_name = "vpm_horizontal_vertical",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %c0, %value, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %read = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 3 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
