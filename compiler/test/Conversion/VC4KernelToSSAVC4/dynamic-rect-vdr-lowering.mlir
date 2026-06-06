// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @dynamic_rect_vdr_lowering
// CHECK-SAME: builtins = [
// CHECK-SAME: vpm_base_row
// CHECK-SAME: vc4.resource = {
// CHECK-SAME: requires_vpm_base_row_builtin = true
// CHECK-SAME: total_vpm_rows_per_block = 4 : i32
// CHECK-SAME: uses_vdr = true
// CHECK-SAME: uses_vpm = true
// CHECK: %[[VPM_BASE:[0-9]+]] = ssavc4.uniform.read 4 : i32
// CHECK: %[[ZERO:[0-9]+]] = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
// CHECK: %[[ADDRESS:[0-9]+]] = ssavc4.alu.add %0, %[[ZERO]]
// CHECK: %[[VPM_ROW:[0-9]+]] = ssavc4.alu.add %[[VPM_BASE]], %[[ZERO]]
// CHECK: ssavc4.vdr.load_rect.dynamic
// CHECK-SAME: %[[ADDRESS]], %[[VPM_ROW]], %1, %2, %3
// CHECK-SAME: dst_x = 0 : i32
// CHECK-SAME: elem_bytes = 4 : i32
// CHECK-SAME: max_cols = 4 : i32
// CHECK-SAME: max_rows = 4 : i32
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK-SAME: serialize = "mutex"
// CHECK-SAME: vpm_pitch = 1 : i32
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w32>
// CHECK-SAME: zero_fill = true
// CHECK-NOT: dynamic rectangular VDR/VDW lowering is not implemented yet
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @dynamic_rect_vdr_lowering(%ptr : i32, %rows : i32, %cols : i32, %pitch : i32) attributes {
    public_name = "dynamic_rect_vdr_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "pitch", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c0, %rows, %cols, %pitch {max_rows = 4 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.return
  }
}
