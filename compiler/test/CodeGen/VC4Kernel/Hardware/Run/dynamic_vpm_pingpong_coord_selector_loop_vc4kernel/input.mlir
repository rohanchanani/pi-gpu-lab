module {
  vc4kernel.kernel @dynamic_vpm_pingpong_coord_selector_loop_vc4kernel(%in : i32, %out : i32, %iters : i32, %active_cols : i32) attributes {
    public_name = "dynamic_vpm_pingpong_coord_selector_loop_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "u8"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "u8"},
      {name = "iters", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c5 = arith.constant 5 : i32
    %c32 = arith.constant 32 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 8 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^loop(%c0 : i32)

  ^loop(%j : i32):
    %more = arith.cmpi ult, %j, %iters : i32
    cf.cond_br %more, ^body(%j : i32), ^done

  ^body(%j_body : i32):
    %ping = arith.andi %j_body, %c1 : i32
    %row = arith.shli %ping, %c2 : i32
    %byte_off = arith.shli %j_body, %c5 : i32
    vc4kernel.vdr_load_rect_to_vpm %in, %byte_off, %tile, %row dynamic_dst_x %ping dynamic_subword_selector %ping, %c1, %active_cols, %c32 {max_rows = 1 : i32, max_cols = 7 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32, i32, i32, i32
    vc4kernel.vdw_store_rect_from_vpm %tile, %row dynamic_src_x %ping dynamic_subword_selector %ping, %out, %byte_off, %c1, %active_cols, %c32 {max_rows = 1 : i32, max_cols = 7 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32
    %next = arith.addi %j_body, %c1 : i32
    cf.br ^loop(%next : i32)

  ^done:
    vc4kernel.return
  }
}
