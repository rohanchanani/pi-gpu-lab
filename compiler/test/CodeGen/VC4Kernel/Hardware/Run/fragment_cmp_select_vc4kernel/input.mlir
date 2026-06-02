module {
  vc4kernel.kernel @fragment_cmp_select_vc4kernel(%out : i32, %threshold : i32) attributes {
    public_name = "fragment_cmp_select_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "threshold", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %off0 = arith.constant 0 : i32
    %off64 = arith.constant 64 : i32
    %off128 = arith.constant 128 : i32
    %off192 = arith.constant 192 : i32
    %off256 = arith.constant 256 : i32
    %off320 = arith.constant 320 : i32
    %true_base = arith.constant 805306368 : i32
    %false_base = arith.constant 1073741824 : i32
    %row_step = arith.constant 4096 : i32
    %one_step = arith.constant 1 : i32
    %two_step = arith.constant 2 : i32
    %three_step = arith.constant 3 : i32
    %four_step = arith.constant 4 : i32
    %five_step = arith.constant 5 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %threshold_v = vc4kernel.splat %threshold : i32 -> vector<16xi32>
    %true_base_v = vc4kernel.splat %true_base : i32 -> vector<16xi32>
    %false_base_v = vc4kernel.splat %false_base : i32 -> vector<16xi32>
    %step_v = vc4kernel.splat %row_step : i32 -> vector<16xi32>
    %one_v = vc4kernel.splat %one_step : i32 -> vector<16xi32>
    %two_v = vc4kernel.splat %two_step : i32 -> vector<16xi32>
    %three_v = vc4kernel.splat %three_step : i32 -> vector<16xi32>
    %four_v = vc4kernel.splat %four_step : i32 -> vector<16xi32>
    %five_v = vc4kernel.splat %five_step : i32 -> vector<16xi32>

    %true0 = vc4kernel.fragment_add %true_base_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false0 = vc4kernel.fragment_add %false_base_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true1_base = vc4kernel.fragment_add %true_base_v, %step_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false1_base = vc4kernel.fragment_add %false_base_v, %step_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true1 = vc4kernel.fragment_add %true1_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false1 = vc4kernel.fragment_add %false1_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true2_base_mul = vc4kernel.fragment_mul %step_v, %two_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false2_base_mul = vc4kernel.fragment_mul %step_v, %two_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true2_base = vc4kernel.fragment_add %true_base_v, %true2_base_mul : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false2_base = vc4kernel.fragment_add %false_base_v, %false2_base_mul : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true2 = vc4kernel.fragment_add %true2_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false2 = vc4kernel.fragment_add %false2_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true3_base_mul = vc4kernel.fragment_mul %step_v, %three_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false3_base_mul = vc4kernel.fragment_mul %step_v, %three_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true3_base = vc4kernel.fragment_add %true_base_v, %true3_base_mul : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false3_base = vc4kernel.fragment_add %false_base_v, %false3_base_mul : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true3 = vc4kernel.fragment_add %true3_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false3 = vc4kernel.fragment_add %false3_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true4_base_mul = vc4kernel.fragment_mul %step_v, %four_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false4_base_mul = vc4kernel.fragment_mul %step_v, %four_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true4_base = vc4kernel.fragment_add %true_base_v, %true4_base_mul : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false4_base = vc4kernel.fragment_add %false_base_v, %false4_base_mul : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true4 = vc4kernel.fragment_add %true4_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false4 = vc4kernel.fragment_add %false4_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true5_base_mul = vc4kernel.fragment_mul %step_v, %five_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false5_base_mul = vc4kernel.fragment_mul %step_v, %five_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true5_base = vc4kernel.fragment_add %true_base_v, %true5_base_mul : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false5_base = vc4kernel.fragment_add %false_base_v, %false5_base_mul : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %true5 = vc4kernel.fragment_add %true5_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %false5 = vc4kernel.fragment_add %false5_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>

    %eq = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ne = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ne>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ult = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ule = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ule>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ugt = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ugt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %uge = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<uge>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel0 = vc4kernel.fragment_select %eq, %true0, %false0 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel1 = vc4kernel.fragment_select %ne, %true1, %false1 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel2 = vc4kernel.fragment_select %ult, %true2, %false2 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel3 = vc4kernel.fragment_select %ule, %true3, %false3 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel4 = vc4kernel.fragment_select %ugt, %true4, %false4 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel5 = vc4kernel.fragment_select %uge, %true5, %false5 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>

    %off0v = vc4kernel.splat %off0 : i32 -> vector<16xi32>
    %off64v = vc4kernel.splat %off64 : i32 -> vector<16xi32>
    %off128v = vc4kernel.splat %off128 : i32 -> vector<16xi32>
    %off192v = vc4kernel.splat %off192 : i32 -> vector<16xi32>
    %off256v = vc4kernel.splat %off256 : i32 -> vector<16xi32>
    %off320v = vc4kernel.splat %off320 : i32 -> vector<16xi32>
    %offs0 = vc4kernel.fragment_add %off0v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs1 = vc4kernel.fragment_add %off64v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs2 = vc4kernel.fragment_add %off128v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs3 = vc4kernel.fragment_add %off192v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs4 = vc4kernel.fragment_add %off256v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs5 = vc4kernel.fragment_add %off320v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs0, %sel0, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs1, %sel1, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs2, %sel2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs3, %sel3, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs4, %sel4, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs5, %sel5, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
