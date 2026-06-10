module {
  // Assignment Part 2 shape, scaled to VC4:
  //   X      : [in_channels, input_height, input_width]
  //   W      : [out_channels, in_channels, 3, 3]
  //   bias   : [out_channels]
  //   output : [out_channels, out_pool_height, out_pool_width]
  //
  // One program instance computes one output channel, one output row, and a
  // 16-column spatial vector.  pool_size is the assignment's only pooling
  // parameter and must be 1 or 2.
  vc4kernel.kernel @fused_conv2d_3x3_maxpool_vc4kernel(
      %input : i32,
      %weights : i32,
      %bias : i32,
      %out : i32,
      %in_channels : i32,
      %in_h : i32,
      %in_w : i32,
      %out_h : i32,
      %out_w : i32,
      %pool_size : i32) attributes {
    public_name = "fused_conv2d_3x3_maxpool_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "weights", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "bias", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "in_channels", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "in_h", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "in_w", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "out_h", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "out_w", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "pool_size", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c9 = arith.constant 9 : i32
    %safe0 = arith.constant 0 : i32

    %pid_x = vc4kernel.program_id {axis = 0 : i32} : i32
    %pid_y = vc4kernel.program_id {axis = 1 : i32} : i32
    %oc = vc4kernel.program_id {axis = 2 : i32} : i32

    %lanes = vc4kernel.lane_range : vector<16xi32>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %base_col = arith.shli %pid_x, %c4 : i32
    %tail = vc4kernel.pred.tail %base_col, %out_w : i32, i32 -> !vc4kernel.pred<16>
    %base_col_v = vc4kernel.splat %base_col : i32 -> vector<16xi32>
    %pool_x_unscaled = vc4kernel.fragment_alu.add %base_col_v, %lanes
        {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %pool_size_v = vc4kernel.splat %pool_size : i32 -> vector<16xi32>
    %pool_x_base = vc4kernel.fragment_alu.mul %pool_x_unscaled, %pool_size_v
        {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %pool_y_base = arith.muli %pid_y, %pool_size : i32

    %one_i_v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %four_i_v = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %neg_inf = vc4kernel.fragment_const {value = dense<-3.4028234663852886E+38> : vector<16xf32>} : vector<16xf32>

    %bias_byte = arith.shli %oc, %c2 : i32
    %bias_off = vc4kernel.splat %bias_byte : i32 -> vector<16xi32>
    %bias_v = vc4kernel.tmu_load_fragment %bias, %bias_off, %full, %safe0
        {inactive_load = #vc4kernel.inactive_load<zero>,
         memory_path = #vc4kernel.memory_path<tmu_global_read>,
         coherency = #vc4kernel.coherency<readonly_tmu>}
        : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>

    %is_pool2 = arith.cmpi eq, %pool_size, %c2 : i32
    %pool_positions = arith.select %is_pool2, %c4, %c1 : i32
    cf.br ^pool_loop(%c0, %neg_inf : i32, vector<16xf32>)

  ^pool_loop(%pool_idx : i32, %best : vector<16xf32>):
    %more_pool = arith.cmpi ult, %pool_idx, %pool_positions : i32
    cf.cond_br %more_pool, ^start_conv(%best : vector<16xf32>), ^store(%best : vector<16xf32>)

  ^start_conv(%best_in : vector<16xf32>):
    %idx_lt_2 = arith.cmpi ult, %pool_idx, %c2 : i32
    %idx_eq_0 = arith.cmpi eq, %pool_idx, %c0 : i32
    %idx_eq_2 = arith.cmpi eq, %pool_idx, %c2 : i32
    %pw_is_zero = arith.ori %idx_eq_0, %idx_eq_2 : i1
    %ph = arith.select %idx_lt_2, %c0, %c1 : i32
    %pw = arith.select %pw_is_zero, %c0, %c1 : i32
    cf.br ^channel_loop(%c0, %bias_v : i32, vector<16xf32>)

  ^channel_loop(%ic : i32, %acc_ch : vector<16xf32>):
    %more_channels = arith.cmpi ult, %ic, %in_channels : i32
    cf.cond_br %more_channels,
      ^ky_loop(%c0, %acc_ch : i32, vector<16xf32>),
      ^finish_conv(%acc_ch : vector<16xf32>)

  ^ky_loop(%ky : i32, %acc_ky : vector<16xf32>):
    %more_ky = arith.cmpi ult, %ky, %c3 : i32
    cf.cond_br %more_ky,
      ^kx_loop(%c0, %acc_ky : i32, vector<16xf32>),
      ^next_channel(%acc_ky : vector<16xf32>)

  ^kx_loop(%kx : i32, %acc_kx : vector<16xf32>):
    %more_kx = arith.cmpi ult, %kx, %c3 : i32
    cf.cond_br %more_kx,
      ^tap(%acc_kx : vector<16xf32>),
      ^next_ky(%acc_kx : vector<16xf32>)

  ^tap(%acc_tap : vector<16xf32>):
    %hw = arith.muli %in_h, %in_w : i32
    %channel_elems = arith.muli %ic, %hw : i32
    %conv_y0 = arith.addi %pool_y_base, %ph : i32
    %iy = arith.addi %conv_y0, %ky : i32
    %row_elems = arith.muli %iy, %in_w : i32
    %input_base_elems = arith.addi %channel_elems, %row_elems : i32
    %input_base_bytes = arith.shli %input_base_elems, %c2 : i32
    %input_base_v = vc4kernel.splat %input_base_bytes : i32 -> vector<16xi32>

    %pw_v = vc4kernel.splat %pw : i32 -> vector<16xi32>
    %kx_v = vc4kernel.splat %kx : i32 -> vector<16xi32>
    %x_with_pool = vc4kernel.fragment_alu.add %pool_x_base, %pw_v
        {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %ix = vc4kernel.fragment_alu.add %x_with_pool, %kx_v
        {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %x_bytes = vc4kernel.fragment_alu.mul %ix, %four_i_v
        {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %input_off = vc4kernel.fragment_alu.add %input_base_v, %x_bytes
        {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %x_v = vc4kernel.tmu_load_fragment %input, %input_off, %tail, %safe0
        {inactive_load = #vc4kernel.inactive_load<zero>,
         memory_path = #vc4kernel.memory_path<tmu_global_read>,
         coherency = #vc4kernel.coherency<readonly_tmu>}
        : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>

    %oc_ic = arith.muli %oc, %in_channels : i32
    %w_channel = arith.addi %oc_ic, %ic : i32
    %w_channel_base = arith.muli %w_channel, %c9 : i32
    %ky3 = arith.muli %ky, %c3 : i32
    %tap_index = arith.addi %ky3, %kx : i32
    %w_index = arith.addi %w_channel_base, %tap_index : i32
    %w_byte = arith.shli %w_index, %c2 : i32
    %w_off = vc4kernel.splat %w_byte : i32 -> vector<16xi32>
    %w_v = vc4kernel.tmu_load_fragment %weights, %w_off, %full, %safe0
        {inactive_load = #vc4kernel.inactive_load<zero>,
         memory_path = #vc4kernel.memory_path<tmu_global_read>,
         coherency = #vc4kernel.coherency<readonly_tmu>}
        : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>

    %prod = vc4kernel.fragment_alu.mul %x_v, %w_v
        {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc_next = vc4kernel.fragment_alu.add %acc_tap, %prod
        {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %kx_next = arith.addi %kx, %c1 : i32
    cf.br ^kx_loop(%kx_next, %acc_next : i32, vector<16xf32>)

  ^next_ky(%acc_after_kx : vector<16xf32>):
    %ky_next = arith.addi %ky, %c1 : i32
    cf.br ^ky_loop(%ky_next, %acc_after_kx : i32, vector<16xf32>)

  ^next_channel(%acc_after_ky : vector<16xf32>):
    %ic_next = arith.addi %ic, %c1 : i32
    cf.br ^channel_loop(%ic_next, %acc_after_ky : i32, vector<16xf32>)

  ^finish_conv(%conv_value : vector<16xf32>):
    %new_best = vc4kernel.fragment_alu.add %best_in, %conv_value
        {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %pool_next = arith.addi %pool_idx, %c1 : i32
    cf.br ^pool_loop(%pool_next, %new_best : i32, vector<16xf32>)

  ^store(%result : vector<16xf32>):
    %oc_rows = arith.muli %oc, %out_h : i32
    %out_row = arith.addi %oc_rows, %pid_y : i32
    %out_row_base = arith.muli %out_row, %out_w : i32
    %out_base_elem = arith.addi %out_row_base, %base_col : i32
    %out_base_byte = arith.shli %out_base_elem, %c2 : i32
    %out_base_v = vc4kernel.splat %out_base_byte : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %out_offsets = vc4kernel.fragment_alu.add %out_base_v, %lane_bytes
        {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %out_offsets, %result, %tail
        {memory_path = #vc4kernel.memory_path<vdw_global_store>,
         coherency = #vc4kernel.coherency<dma_ordered>,
         inactive_store = #vc4kernel.inactive_store<preserve>}
        : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
