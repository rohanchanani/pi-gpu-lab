// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --split-input-file | FileCheck %s

// CHECK-LABEL: ssavc4.func @dynamic_vdr_dst_row_lowering
// CHECK: %[[ARG_ROW:[0-9]+]] = ssavc4.uniform.read 1 : i32
// CHECK: %[[VPM_BASE:[0-9]+]] = ssavc4.uniform.read 5 : i32
// CHECK: %[[ROW:[0-9]+]] = ssavc4.alu.add %[[VPM_BASE]], %[[ARG_ROW]]
// CHECK: ssavc4.vdr.load_rect.dynamic
// CHECK-SAME: %[[ROW]], %2, %3, %4
// CHECK-NOT: dynamic rectangular VPM row must be a scalar i32 constant
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @dynamic_vdr_dst_row_lowering(%ptr : i32, %row : i32, %rows : i32, %cols : i32, %pitch : i32) attributes {
    public_name = "dynamic_vdr_dst_row_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "pitch", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 8 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %row, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.return
  }
}

// -----

// CHECK-LABEL: ssavc4.func @dynamic_vdw_src_row_lowering
// CHECK: %[[ARG_ROW:[0-9]+]] = ssavc4.uniform.read 1 : i32
// CHECK: %[[VPM_BASE:[0-9]+]] = ssavc4.uniform.read 5 : i32
// CHECK: %[[ROW:[0-9]+]] = ssavc4.alu.add %[[VPM_BASE]], %[[ARG_ROW]]
// CHECK: ssavc4.vdw.store_rect.dynamic
// CHECK-SAME: %[[ROW]], %2, %3, %4
// CHECK-NOT: dynamic rectangular VPM row must be a scalar i32 constant
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @dynamic_vdw_src_row_lowering(%ptr : i32, %row : i32, %rows : i32, %cols : i32, %stride : i32) attributes {
    public_name = "dynamic_vdw_src_row_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 8 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdw_store_rect_from_vpm %tile, %row, %ptr, %c0, %rows, %cols, %stride {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}

// -----

// CHECK-LABEL: ssavc4.func @dynamic_vpm_read_write_row_lowering
// CHECK: %[[ARG_ROW:[0-9]+]] = ssavc4.uniform.read 0 : i32
// CHECK: %[[VPM_BASE:[0-9]+]] = ssavc4.uniform.read 1 : i32
// CHECK: %[[WRITE_ROW:[0-9]+]] = ssavc4.alu.add %[[VPM_BASE]], %[[ARG_ROW]]
// CHECK-NEXT: ssavc4.vpm.write %[[WRITE_ROW]]
// CHECK: %[[READ_ROW:[0-9]+]] = ssavc4.alu.add %[[VPM_BASE]], %[[ARG_ROW]]
// CHECK-NEXT: %{{[0-9]+}} = ssavc4.vpm.read %[[READ_ROW]]
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @dynamic_vpm_read_write_row_lowering(%row : i32) attributes {
    public_name = "dynamic_vpm_read_write_row_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c1 = arith.constant 1 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 8 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %row, %value, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %read = vc4kernel.vpm_read_fragment %tile, %row, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

// CHECK-LABEL: ssavc4.func @static_rect_row_lowering
// CHECK: %[[VPM_BASE:[0-9]+]] = ssavc4.uniform.read 4 : i32
// CHECK: %[[ONE:[0-9]+]] = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
// CHECK: %[[ROW:[0-9]+]] = ssavc4.alu.add %[[VPM_BASE]], %[[ONE]]
// CHECK: ssavc4.vdr.load_rect.dynamic
// CHECK-SAME: %[[ROW]], %1, %2, %3
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @static_rect_row_lowering(%ptr : i32, %rows : i32, %cols : i32, %pitch : i32) attributes {
    public_name = "static_rect_row_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "pitch", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %tile = vc4kernel.vpm_alloc {rows = 8 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c1, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.return
  }
}
