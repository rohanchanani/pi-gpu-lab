module {
  vc4kernel.kernel @vpm_qpu_dynamic_selector_loop_branch_vc4kernel(%out : i32, %control : i32) attributes {
    public_name = "vpm_qpu_dynamic_selector_loop_branch_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "control", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %c500 = arith.constant 500 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %seg1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %seg1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 32 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^loop(%c0, %zero : i32, vector<16xi32>)

  ^loop(%j : i32, %acc : vector<16xi32>):
    %more = arith.cmpi ult, %j, %c2 : i32
    cf.cond_br %more, ^step(%j : i32), ^after_loop(%acc : vector<16xi32>)

  ^step(%j_step : i32):
    %base_scalar = arith.addi %c500, %j_step : i32
    %base_vec = vc4kernel.splat %base_scalar : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %base_vec, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %packed = vc4kernel.fragment_pack %value {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %j_step dynamic_subword_selector %j_step, %packed, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %raw = vc4kernel.vpm_read_fragment %tile, %j_step dynamic_subword_selector %j_step, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %unpacked = vc4kernel.fragment_unpack %raw {source = #vc4kernel.subword_type<s16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<sign_extend>} : vector<16xi32> -> vector<16xi32>
    %j_next = arith.addi %j_step, %c1 : i32
    cf.br ^loop(%j_next, %unpacked : i32, vector<16xi32>)

  ^after_loop(%loop_result : vector<16xi32>):
    %use_alt = arith.cmpi ne, %control, %c0 : i32
    cf.cond_br %use_alt, ^alt_path(%loop_result : vector<16xi32>), ^main_path(%loop_result : vector<16xi32>)

  ^main_path(%loop_main : vector<16xi32>):
    %base_main = vc4kernel.fragment_const {value = dense<70> : vector<16xi32>} : vector<16xi32>
    %value_main = vc4kernel.fragment_alu.add %base_main, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %pack_main = vc4kernel.fragment_pack %value_main {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %c16 dynamic_x %control dynamic_subword_selector %control, %pack_main, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %raw_main = vc4kernel.vpm_read_fragment %tile, %c16 dynamic_x %control dynamic_subword_selector %control, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %out_main = vc4kernel.fragment_unpack %raw_main {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    cf.br ^store(%loop_main, %out_main : vector<16xi32>, vector<16xi32>)

  ^alt_path(%loop_alt : vector<16xi32>):
    %sel_alt = arith.addi %control, %c2 : i32
    %x_alt = arith.addi %control, %control : i32
    %base_alt = vc4kernel.fragment_const {value = dense<80> : vector<16xi32>} : vector<16xi32>
    %value_alt = vc4kernel.fragment_alu.add %base_alt, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %pack_alt = vc4kernel.fragment_pack %value_alt {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %c16 dynamic_x %x_alt dynamic_subword_selector %sel_alt, %pack_alt, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<laned>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %raw_alt = vc4kernel.vpm_read_fragment %tile, %c16 dynamic_x %x_alt dynamic_subword_selector %sel_alt, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<laned>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %out_alt = vc4kernel.fragment_unpack %raw_alt {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    cf.br ^store(%loop_alt, %out_alt : vector<16xi32>, vector<16xi32>)

  ^store(%loop_out : vector<16xi32>, %branch_out : vector<16xi32>):
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %loop_out, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off1, %branch_out, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
