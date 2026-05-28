// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh cute_gemm_tail_17x13x9_vc4tile generate

vc4tile.kernel @cute_gemm_tail_17x13x9_vc4tile(%out : i32, %a : i32, %b : i32) attributes {
  public_name = "cute_gemm_tail_17x13x9_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "a", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "b", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %three = arith.constant 3 : i32
  %four = arith.constant 4 : i32
  %eight = arith.constant 8 : i32
  %thirteen = arith.constant 13 : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %mask_a_k_tail = vc4tile.tile_rect_mask {active_rows = 4 : i32, active_cols = 1 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %mask_b_k_tail = vc4tile.tile_rect_mask {active_rows = 1 : i32, active_cols = 4 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %pid = vc4tile.program_id : i32
  %tile_i = arith.shrui %pid, %two : i32
  %tile_j = arith.andi %pid, %three : i32
  %tile_i_row_base = arith.shli %tile_i, %two : i32
  %tile_i_times3 = arith.muli %tile_i, %three : i32
  %tile_i_row_words = arith.muli %tile_i_row_base, %thirteen : i32
  %tile_j_col_words = arith.shli %tile_j, %two : i32
  %out_tile_base = arith.addi %tile_i_row_words, %tile_j_col_words : i32
  %is_m_tail = arith.cmpi eq, %tile_i, %four : i32
  %is_n_tail = arith.cmpi eq, %tile_j, %three : i32
  %store_rows = scf.if %is_m_tail -> (i32) {
    scf.yield %one : i32
  } else {
    scf.yield %four : i32
  }
  %store_cols = scf.if %is_n_tail -> (i32) {
    scf.yield %one : i32
  } else {
    scf.yield %four : i32
  }
  %store_mask = vc4tile.tile_bounds_mask %store_rows, %store_cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>

  %acc0 = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 0 : i32} : () -> vector<16xi32>
  %a_tile_0 = arith.addi %tile_i_times3, %zero : i32
  %a_off_0 = arith.shli %a_tile_0, %four : i32
  %b_tile_0 = arith.addi %tile_j, %zero : i32
  %b_off_0 = arith.shli %b_tile_0, %four : i32
  %lhs_0 = "vc4tile.tile_load"(%a, %a_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %rhs_0 = "vc4tile.tile_load"(%b, %b_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %acc1 = "vc4tile.tile_matmul"(%lhs_0, %rhs_0, %acc0, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>

  %a_tile_1 = arith.addi %tile_i_times3, %one : i32
  %a_off_1 = arith.shli %a_tile_1, %four : i32
  %b_tile_1 = arith.addi %tile_j, %four : i32
  %b_off_1 = arith.shli %b_tile_1, %four : i32
  %lhs_1 = "vc4tile.tile_load"(%a, %a_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %rhs_1 = "vc4tile.tile_load"(%b, %b_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %acc2 = "vc4tile.tile_matmul"(%lhs_1, %rhs_1, %acc1, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>

  %a_tile_2 = arith.addi %tile_i_times3, %two : i32
  %a_off_2 = arith.shli %a_tile_2, %four : i32
  %b_tile_2 = arith.addi %tile_j, %eight : i32
  %b_off_2 = arith.shli %b_tile_2, %four : i32
  %lhs_2 = "vc4tile.tile_load"(%a, %a_off_2, %mask_a_k_tail) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %rhs_2 = "vc4tile.tile_load"(%b, %b_off_2, %mask_b_k_tail) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %acc3 = "vc4tile.tile_matmul"(%lhs_2, %rhs_2, %acc2, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>

  "vc4tile.tile_store"(%acc3, %out, %out_tile_base, %store_mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>, memory_pitch_bytes = 52 : i32} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()

  vc4tile.return
}
