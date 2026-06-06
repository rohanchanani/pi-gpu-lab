// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @dynamic_rect_vdw_lowering
// CHECK-SAME: builtins = [
// CHECK-SAME: vpm_base_row
// CHECK-SAME: vc4.resource = {
// CHECK-SAME: requires_vpm_base_row_builtin = true
// CHECK-SAME: total_vpm_rows_per_block = 4 : i32
// CHECK-SAME: uses_vdw = true
// CHECK-SAME: uses_vpm = true
// CHECK: %[[VPM_BASE:[0-9]+]] = ssavc4.uniform.read 4 : i32
// CHECK: %[[ZERO:[0-9]+]] = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
// CHECK: %[[ADDRESS:[0-9]+]] = ssavc4.alu.add %0, %[[ZERO]]
// CHECK: %[[VPM_ROW:[0-9]+]] = ssavc4.alu.add %[[VPM_BASE]], %[[ZERO]]
// CHECK: ssavc4.vdw.store_rect.dynamic
// CHECK-SAME: %[[ADDRESS]], %[[VPM_ROW]], %1, %2, %3
// CHECK-SAME: elem_bytes = 4 : i32
// CHECK-SAME: max_cols = 4 : i32
// CHECK-SAME: max_rows = 4 : i32
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK-SAME: preserve_inactive = true
// CHECK-SAME: serialize = "mutex"
// CHECK-SAME: src_x = 0 : i32
// CHECK-SAME: vpm_pitch = 1 : i32
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w32>
// CHECK-NOT: dynamic rectangular VDR/VDW lowering is not implemented yet
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @dynamic_rect_vdw_lowering(%ptr : i32, %rows : i32, %cols : i32, %stride : i32) attributes {
    public_name = "dynamic_rect_vdw_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %ptr, %c0, %rows, %cols, %stride {max_rows = 4 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
