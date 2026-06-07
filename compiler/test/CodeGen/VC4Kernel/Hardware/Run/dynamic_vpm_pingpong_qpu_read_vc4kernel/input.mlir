module {
  vc4kernel.kernel @dynamic_vpm_pingpong_qpu_read_vc4kernel(%out : i32, %iters : i32, %active_cols : i32) attributes {
    public_name = "dynamic_vpm_pingpong_qpu_read_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "iters", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c5 = arith.constant 5 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %sum_base = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %sum_offsets = vc4kernel.fragment_alu.add %sum_base, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %base = vc4kernel.fragment_const {value = dense<16640> : vector<16xi32>} : vector<16xi32>
    %active_v = vc4kernel.splat %active_cols : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 8 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^loop(%c0 : i32)

  ^loop(%j : i32):
    %more = arith.cmpi ult, %j, %iters : i32
    cf.cond_br %more, ^body(%j : i32), ^done

  ^body(%j_body : i32):
    %ping = arith.andi %j_body, %c1 : i32
    %row = arith.shli %ping, %c2 : i32
    %iter_bias_s = arith.shli %j_body, %c5 : i32
    %iter_bias = vc4kernel.splat %iter_bias_s : i32 -> vector<16xi32>
    %biased = vc4kernel.fragment_alu.add %base, %iter_bias {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %values = vc4kernel.fragment_alu.add %biased, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %packed = vc4kernel.fragment_pack %values {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row dynamic_subword_selector %ping, %packed, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %next = arith.addi %j_body, %c1 : i32
    cf.br ^loop(%next : i32)

  ^done:
    %last_iter = arith.subi %iters, %c1 : i32
    %last_ping = arith.andi %last_iter, %c1 : i32
    %read_row = arith.shli %last_ping, %c2 : i32
    %raw = vc4kernel.vpm_read_fragment %tile, %read_row dynamic_subword_selector %last_ping, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %u16 = vc4kernel.fragment_unpack %raw {source = #vc4kernel.subword_type<s16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<sign_extend>} : vector<16xi32> -> vector<16xi32>
    %active = vc4kernel.fragment_cmp %lanes, %active_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %selected = vc4kernel.fragment_select %active, %u16, %zero : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sum = vc4kernel.fragment_reduce %selected, %full {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %selected, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %sum_offsets, %sum, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
