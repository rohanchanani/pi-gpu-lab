module {
  vc4kernel.kernel @fragment_rotate_dynamic_loop_branch_vc4kernel(%out : i32, %iters : i32, %control : i32, %base_value : i32) attributes {
    public_name = "fragment_rotate_dynamic_loop_branch_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "iters", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "control", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "base_value", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %c7 = arith.constant 7 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %base_vec = vc4kernel.splat %base_value : i32 -> vector<16xi32>
    %init = vc4kernel.fragment_alu.add %base_vec, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    cf.br ^loop(%c0, %init : i32, vector<16xi32>)

  ^loop(%j : i32, %acc : vector<16xi32>):
    %more = arith.cmpi ult, %j, %iters : i32
    cf.cond_br %more, ^branch(%j, %acc : i32, vector<16xi32>), ^store(%acc : vector<16xi32>)

  ^branch(%j_step : i32, %acc_step : vector<16xi32>):
    %use_alt = arith.cmpi ne, %control, %c0 : i32
    cf.cond_br %use_alt, ^alt_path(%j_step, %acc_step : i32, vector<16xi32>), ^main_path(%j_step, %acc_step : i32, vector<16xi32>)

  ^main_path(%j_main : i32, %acc_main : vector<16xi32>):
    %amount_main = arith.addi %j_main, %c3 : i32
    %rot_main = vc4kernel.fragment_rotate %acc_main, %amount_main : vector<16xi32>, i32 -> vector<16xi32>
    %next_j_main = arith.addi %j_main, %c1 : i32
    cf.br ^loop(%next_j_main, %rot_main : i32, vector<16xi32>)

  ^alt_path(%j_alt : i32, %acc_alt : vector<16xi32>):
    %amount_alt = arith.addi %j_alt, %c7 : i32
    %rot_alt = vc4kernel.fragment_rotate %acc_alt, %amount_alt : vector<16xi32>, i32 -> vector<16xi32>
    %next_j_alt = arith.addi %j_alt, %c1 : i32
    cf.br ^loop(%next_j_alt, %rot_alt : i32, vector<16xi32>)

  ^store(%result : vector<16xi32>):
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %result, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
