module {
  ssavc4.module @vc4kernel_lowered {
    ssavc4.func @gemm_shape_vc4kernel_micro_gemm_vc4kernel() attributes {kernel, threading = #vc4.threading_mode<single>, vc4.launch_abi = {args = [{direction = "in", elem_type = "f32", kind = "buffer", name = "a", uniform_index = 0 : i32}, {direction = "in", elem_type = "f32", kind = "buffer", name = "b", uniform_index = 1 : i32}, {direction = "inout", elem_type = "f32", kind = "buffer", name = "c", uniform_index = 2 : i32}, {direction = "by_value", kind = "scalar", name = "m", type = "i32", uniform_index = 3 : i32}, {direction = "by_value", kind = "scalar", name = "n", type = "i32", uniform_index = 4 : i32}, {direction = "by_value", kind = "scalar", name = "k", type = "i32", uniform_index = 5 : i32}, {direction = "by_value", kind = "scalar", name = "lda", type = "i32", uniform_index = 6 : i32}, {direction = "by_value", kind = "scalar", name = "ldb", type = "i32", uniform_index = 7 : i32}, {direction = "by_value", kind = "scalar", name = "ldc", type = "i32", uniform_index = 8 : i32}], builtins = [{kind = #vc4.builtin_kind<program_id_x>, materialization = "uniform_suffix", name = "program_id_x", uniform_index = 9 : i32}, {kind = #vc4.builtin_kind<program_id_y>, materialization = "uniform_suffix", name = "program_id_y", uniform_index = 10 : i32}, {kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", name = "vpm_base_row", uniform_index = 11 : i32}], code_symbol = "gemm_shape_vc4kernel_micro_gemm_vc4kernel_shader", public_name = "gemm_shape_vc4kernel_micro_gemm_vc4kernel", symbol_name = "gemm_shape_vc4kernel_micro_gemm_vc4kernel", tail_policy = "exact_multiple", uniform_words_per_qpu = 12 : i32}, vc4.resource = {compiler_vpm_staging_rows_per_block = 0 : i32, compiler_vpm_staging_rows_per_warp = 1 : i32, requires_semaphore_base_builtin = false, requires_vpm_base_row_builtin = true, schedule_mode = "independent_vector", semaphore_count_per_block = 0 : i32, total_vpm_rows_per_block = 1 : i32, user_vpm_rows_per_block = 0 : i32, uses_barrier = false, uses_tmu = true, uses_vdr = false, uses_vdw = true, uses_vpm = true, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, warps_per_block = 1 : i32}} {
      %0 = ssavc4.uniform.read 0 : i32
      %1 = ssavc4.uniform.read 1 : i32
      %2 = ssavc4.uniform.read 2 : i32
      %3 = ssavc4.uniform.read 3 : i32
      %4 = ssavc4.uniform.read 4 : i32
      %5 = ssavc4.uniform.read 5 : i32
      %6 = ssavc4.uniform.read 6 : i32
      %7 = ssavc4.uniform.read 7 : i32
      %8 = ssavc4.uniform.read 8 : i32
      %9 = ssavc4.uniform.read 9 : i32
      %10 = ssavc4.uniform.read 10 : i32
      %11 = ssavc4.uniform.read 11 : i32
      %12 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
      %13 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
      %14 = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
      %15 = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
      %16 = ssavc4.load_imm <splat32> {value = 0.000000e+00 : f32} : f32
      %17 = ssavc4.alu.add %9, %15 {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
      %18 = ssavc4.make_flags %10, %3 {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
      ssavc4.cond_br %18, ^bb1, ^bb6 {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
    ^bb1:  // pred: ^bb0
      %19 = ssavc4.make_flags %17, %4 {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
      ssavc4.cond_br %19, ^bb2, ^bb6 {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
    ^bb2:  // pred: ^bb1
      %20 = ssavc4.element_number : vector<16xi32>
      %21 = ssavc4.splat %14 : i32 -> vector<16xi32>
      %22 = ssavc4.alu.add %20, %21 {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %23 = ssavc4.splat %16 : f32 -> vector<16xf32>
      ssavc4.br ^bb3(%12, %23 : i32, vector<16xf32>)
    ^bb3(%24: i32, %25: vector<16xf32>):  // 2 preds: ^bb2, ^bb8
      %26 = ssavc4.make_flags %24, %5 {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
      ssavc4.cond_br %26, ^bb4(%24, %25 : i32, vector<16xf32>), ^bb5(%25 : vector<16xf32>) {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
    ^bb4(%27: i32, %28: vector<16xf32>):  // pred: ^bb3
      %29 = ssavc4.load_imm <splat32> {value = 65535 : i32} : i32
      %30 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
      %31 = ssavc4.alu.add %10, %29 {opcode = #vc4.add_opcode<and>} : (i32, i32) -> i32
      %32 = ssavc4.alu.add %6, %29 {opcode = #vc4.add_opcode<and>} : (i32, i32) -> i32
      %33 = ssavc4.alu.add %10, %30 {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
      %34 = ssavc4.alu.add %6, %30 {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
      %35 = ssavc4.alu.mul %31, %32 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %36 = ssavc4.alu.mul %31, %34 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %37 = ssavc4.alu.mul %33, %32 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %38 = ssavc4.alu.add %36, %37 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %39 = ssavc4.alu.add %38, %30 {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
      %40 = ssavc4.alu.add %35, %39 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %41 = ssavc4.alu.add %40, %27 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %42 = ssavc4.alu.add %41, %14 {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
      %43 = ssavc4.splat %42 : i32 -> vector<16xi32>
      %44 = ssavc4.load_imm <splat32> {value = 65535 : i32} : i32
      %45 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
      %46 = ssavc4.alu.add %27, %44 {opcode = #vc4.add_opcode<and>} : (i32, i32) -> i32
      %47 = ssavc4.alu.add %7, %44 {opcode = #vc4.add_opcode<and>} : (i32, i32) -> i32
      %48 = ssavc4.alu.add %27, %45 {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
      %49 = ssavc4.alu.add %7, %45 {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
      %50 = ssavc4.alu.mul %46, %47 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %51 = ssavc4.alu.mul %46, %49 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %52 = ssavc4.alu.mul %48, %47 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %53 = ssavc4.alu.add %51, %52 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %54 = ssavc4.alu.add %53, %45 {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
      %55 = ssavc4.alu.add %50, %54 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %56 = ssavc4.alu.add %55, %17 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %57 = ssavc4.alu.add %56, %14 {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
      %58 = ssavc4.splat %57 : i32 -> vector<16xi32>
      %59 = ssavc4.alu.add %58, %22 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %60 = ssavc4.splat %0 : i32 -> vector<16xi32>
      %61 = ssavc4.alu.add %60, %43 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %62 = ssavc4.tmu.request %61 {mode = "direct", unit = "tmu0"} : vector<16xi32> -> !ssavc4.async.token
      %63 = ssavc4.tmu.read %62 {part = "raw32", unit = "tmu0"} : !ssavc4.async.token -> vector<16xf32>
      %64 = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xf32>
      %65 = ssavc4.element_number : vector<16xi32>
      %66 = ssavc4.splat %57 : i32 -> vector<16xi32>
      %67 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
      %68 = ssavc4.alu.add %17, %67 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %69 = ssavc4.alu.add %4, %17 {opcode = #vc4.add_opcode<sub>} : (i32, i32) -> i32
      %70 = ssavc4.make_flags %4, %68 {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
      ssavc4.cond_br %70, ^bb8(%64 : vector<16xf32>), ^bb7 {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
    ^bb5(%71: vector<16xf32>):  // pred: ^bb3
      %72 = ssavc4.load_imm <splat32> {value = 65535 : i32} : i32
      %73 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
      %74 = ssavc4.alu.add %10, %72 {opcode = #vc4.add_opcode<and>} : (i32, i32) -> i32
      %75 = ssavc4.alu.add %8, %72 {opcode = #vc4.add_opcode<and>} : (i32, i32) -> i32
      %76 = ssavc4.alu.add %10, %73 {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
      %77 = ssavc4.alu.add %8, %73 {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
      %78 = ssavc4.alu.mul %74, %75 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %79 = ssavc4.alu.mul %74, %77 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %80 = ssavc4.alu.mul %76, %75 {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
      %81 = ssavc4.alu.add %79, %80 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %82 = ssavc4.alu.add %81, %73 {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
      %83 = ssavc4.alu.add %78, %82 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %84 = ssavc4.alu.add %83, %17 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %85 = ssavc4.alu.add %84, %14 {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
      %86 = ssavc4.splat %85 : i32 -> vector<16xi32>
      %87 = ssavc4.alu.add %86, %22 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %88 = ssavc4.alu.add %2, %85 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %89 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
      %90 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
      %91 = ssavc4.alu.add %17, %90 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %92 = ssavc4.alu.add %17, %89 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      %93 = ssavc4.alu.add %4, %17 {opcode = #vc4.add_opcode<sub>} : (i32, i32) -> i32
      %94 = ssavc4.make_flags %4, %91 {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
      ssavc4.cond_br %94, ^bb12, ^bb9 {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
    ^bb6:  // 2 preds: ^bb0, ^bb1
      ssavc4.thread_end
    ^bb7:  // pred: ^bb4
      %95 = ssavc4.splat %69 : i32 -> vector<16xi32>
      %96 = ssavc4.make_flags %65, %95 {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
      %97 = ssavc4.cond_select %96, %59, %66 {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
      %98 = ssavc4.splat %1 : i32 -> vector<16xi32>
      %99 = ssavc4.alu.add %98, %97 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %100 = ssavc4.tmu.request %99 {mode = "direct", unit = "tmu0"} : vector<16xi32> -> !ssavc4.async.token
      %101 = ssavc4.tmu.read %100 {part = "raw32", unit = "tmu0"} : !ssavc4.async.token -> vector<16xf32>
      %102 = ssavc4.make_flags %65, %95 {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
      %103 = ssavc4.cond_select %102, %101, %64 {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xf32>, vector<16xf32> -> vector<16xf32>
      ssavc4.br ^bb8(%103 : vector<16xf32>)
    ^bb8(%104: vector<16xf32>):  // 2 preds: ^bb4, ^bb7
      %105 = ssavc4.alu.mul %63, %104 {opcode = #vc4.mul_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
      %106 = ssavc4.alu.add %28, %105 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
      %107 = ssavc4.alu.add %27, %13 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
      ssavc4.br ^bb3(%107, %106 : i32, vector<16xf32>)
    ^bb9:  // pred: ^bb5
      %108 = ssavc4.make_flags %4, %92 {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
      ssavc4.cond_br %108, ^bb11, ^bb10 {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags
    ^bb10:  // pred: ^bb9
      ssavc4.vdw.store %88, %71, %93, %11 {serialize = "mutex", subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, width = #ssavc4.vpm_elem_width<w32>} : i32, vector<16xf32>, i32, i32
      ssavc4.br ^bb12
    ^bb11:  // pred: ^bb9
      ssavc4.vdw.store %88, %71, %89, %11 {active_lanes = 16 : i32, serialize = "mutex", subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, width = #ssavc4.vpm_elem_width<w32>} : i32, vector<16xf32>, i32, i32
      ssavc4.br ^bb12
    ^bb12:  // 3 preds: ^bb5, ^bb10, ^bb11
      ssavc4.thread_end
    }
  }
}
