module {
  vc4kernel.kernel @gemm_shape_vc4kernel_micro_gemm_vc4kernel(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i32, %arg4 : i32, %arg5 : i32, %arg6 : i32, %arg7 : i32, %arg8 : i32) attributes {arg_attrs = [{direction = "in", elem_type = "f32", kind = "buffer", name = "a"}, {direction = "in", elem_type = "f32", kind = "buffer", name = "b"}, {direction = "inout", elem_type = "f32", kind = "buffer", name = "c"}, {direction = "by_value", kind = "scalar", name = "m", type = "i32"}, {direction = "by_value", kind = "scalar", name = "n", type = "i32"}, {direction = "by_value", kind = "scalar", name = "k", type = "i32"}, {direction = "by_value", kind = "scalar", name = "lda", type = "i32"}, {direction = "by_value", kind = "scalar", name = "ldb", type = "i32"}, {direction = "by_value", kind = "scalar", name = "ldc", type = "i32"}], public_name = "gemm_shape_vc4kernel_micro_gemm_vc4kernel", schedule_mode = #vc4kernel.schedule_mode<independent_vector>, warps_per_block = 1 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = vc4kernel.program_id {axis = 0 : i32} : i32
    %1 = vc4kernel.program_id {axis = 1 : i32} : i32
    %2 = arith.shli %0, %c4_i32 : i32
    %3 = arith.cmpi ult, %1, %arg3 : i32
    cf.cond_br %3, ^bb1, ^bb6
  ^bb1:  // pred: ^bb0
    %4 = arith.cmpi ult, %2, %arg4 : i32
    cf.cond_br %4, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %5 = vc4kernel.lane_range : vector<16xi32>
    %6 = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %7 = vc4kernel.pred.tail %2, %arg4 : i32, i32 -> <16>
    %8 = vc4kernel.pred.full : <16>
    %9 = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    cf.br ^bb3(%c0_i32, %9 : i32, vector<16xf32>)
  ^bb3(%10: i32, %11: vector<16xf32>):  // 2 preds: ^bb2, ^bb4
    %12 = arith.cmpi ult, %10, %arg5 : i32
    cf.cond_br %12, ^bb4(%10, %11 : i32, vector<16xf32>), ^bb5(%11 : vector<16xf32>)
  ^bb4(%13: i32, %14: vector<16xf32>):  // pred: ^bb3
    %15 = arith.muli %1, %arg6 : i32
    %16 = arith.addi %15, %13 : i32
    %17 = arith.shli %16, %c2_i32 : i32
    %18 = vc4kernel.splat %17 : i32 -> vector<16xi32>
    %19 = arith.muli %13, %arg7 : i32
    %20 = arith.addi %19, %2 : i32
    %21 = arith.shli %20, %c2_i32 : i32
    %22 = vc4kernel.splat %21 : i32 -> vector<16xi32>
    %23 = vc4kernel.fragment_alu.add %22, %6 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %24 = vc4kernel.tmu_load_fragment %arg0, %18, %8 : i32, vector<16xi32>, <16> -> vector<16xf32>
    %25 = vc4kernel.tmu_load_fragment %arg1, %23, %7 : i32, vector<16xi32>, <16> -> vector<16xf32>
    %26 = vc4kernel.fragment_alu.mul %24, %25 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %27 = vc4kernel.fragment_alu.add %14, %26 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %28 = arith.addi %13, %c1_i32 : i32
    cf.br ^bb3(%28, %27 : i32, vector<16xf32>)
  ^bb5(%29: vector<16xf32>):  // pred: ^bb3
    %30 = arith.muli %1, %arg8 : i32
    %31 = arith.addi %30, %2 : i32
    %32 = arith.shli %31, %c2_i32 : i32
    %33 = vc4kernel.splat %32 : i32 -> vector<16xi32>
    %34 = vc4kernel.fragment_alu.add %33, %6 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %arg2, %34, %29, %7 : i32, vector<16xi32>, vector<16xf32>, <16>
    vc4kernel.return
  ^bb6:  // 2 preds: ^bb0, ^bb1
    vc4kernel.return
  }
}
