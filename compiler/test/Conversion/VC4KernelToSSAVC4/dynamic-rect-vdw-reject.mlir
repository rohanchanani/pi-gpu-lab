// RUN: not vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @dynamic_rect_vdw_reject(%ptr : i32, %rows : i32, %cols : i32, %stride : i32) attributes {
    public_name = "dynamic_rect_vdw_reject",
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
    // CHECK: dynamic rectangular VDR/VDW lowering is not implemented yet
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %ptr, %c0, %rows, %cols, %stride {max_rows = 4 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
