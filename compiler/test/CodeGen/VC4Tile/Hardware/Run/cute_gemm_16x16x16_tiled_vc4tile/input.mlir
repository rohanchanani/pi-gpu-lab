// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh cute_gemm_16x16x16_tiled_vc4tile generate

vc4tile.kernel @cute_gemm_16x16x16_tiled_vc4tile(%out : i32, %a : i32, %b : i32) attributes {
  public_name = "cute_gemm_16x16x16_tiled_vc4tile",
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
  %two = arith.constant 2 : i32
  %three = arith.constant 3 : i32
  %four = arith.constant 4 : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %mask4 = vc4tile.tail_mask %zero, %four : i32, i32 -> vector<16xi1>
  %pid = vc4tile.program_id : i32
  %tile_i = arith.shrui %pid, %two : i32
  %tile_j = arith.andi %pid, %three : i32
  %tile_i_times4 = arith.shli %tile_i, %two : i32
  %tile_i_row_base = arith.shli %tile_i, %two : i32
  %tile_i_row_words = arith.shli %tile_i_row_base, %four : i32
  %tile_j_col_words = arith.shli %tile_j, %two : i32
  %out_tile_base = arith.addi %tile_i_row_words, %tile_j_col_words : i32

  %acc0 = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 0 : i32} : () -> vector<16xi32>
  %a_tile_0 = arith.addi %tile_i_times4, %zero : i32
  %a_off_0 = arith.shli %a_tile_0, %four : i32
  %b_tile_0 = arith.addi %tile_j, %zero : i32
  %b_off_0 = arith.shli %b_tile_0, %four : i32
  %lhs_0 = "vc4tile.tile_load"(%a, %a_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %rhs_0 = "vc4tile.tile_load"(%b, %b_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %acc1 = "vc4tile.tile_matmul"(%lhs_0, %rhs_0, %acc0, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>

  %k1 = arith.constant 1 : i32
  %a_tile_1 = arith.addi %tile_i_times4, %k1 : i32
  %a_off_1 = arith.shli %a_tile_1, %four : i32
  %k1_times4 = arith.constant 4 : i32
  %b_tile_1 = arith.addi %tile_j, %k1_times4 : i32
  %b_off_1 = arith.shli %b_tile_1, %four : i32
  %lhs_1 = "vc4tile.tile_load"(%a, %a_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %rhs_1 = "vc4tile.tile_load"(%b, %b_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %acc2 = "vc4tile.tile_matmul"(%lhs_1, %rhs_1, %acc1, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>

  %k2 = arith.constant 2 : i32
  %a_tile_2 = arith.addi %tile_i_times4, %k2 : i32
  %a_off_2 = arith.shli %a_tile_2, %four : i32
  %k2_times4 = arith.constant 8 : i32
  %b_tile_2 = arith.addi %tile_j, %k2_times4 : i32
  %b_off_2 = arith.shli %b_tile_2, %four : i32
  %lhs_2 = "vc4tile.tile_load"(%a, %a_off_2, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %rhs_2 = "vc4tile.tile_load"(%b, %b_off_2, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %acc3 = "vc4tile.tile_matmul"(%lhs_2, %rhs_2, %acc2, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>

  %k3 = arith.constant 3 : i32
  %a_tile_3 = arith.addi %tile_i_times4, %k3 : i32
  %a_off_3 = arith.shli %a_tile_3, %four : i32
  %k3_times4 = arith.constant 12 : i32
  %b_tile_3 = arith.addi %tile_j, %k3_times4 : i32
  %b_off_3 = arith.shli %b_tile_3, %four : i32
  %lhs_3 = "vc4tile.tile_load"(%a, %a_off_3, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %rhs_3 = "vc4tile.tile_load"(%b, %b_off_3, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %acc4 = "vc4tile.tile_matmul"(%lhs_3, %rhs_3, %acc3, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>

  %out_off_0 = arith.addi %out_tile_base, %zero : i32
  "vc4tile.tile_store"(%acc4, %out, %out_off_0, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %row1 = vc4tile.rotate %acc4 {amount = 4 : i32} : vector<16xi32> -> vector<16xi32>
  %row_stride_1 = arith.constant 16 : i32
  %out_off_1 = arith.addi %out_tile_base, %row_stride_1 : i32
  "vc4tile.tile_store"(%row1, %out, %out_off_1, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %row2 = vc4tile.rotate %acc4 {amount = 8 : i32} : vector<16xi32> -> vector<16xi32>
  %row_stride_2 = arith.constant 32 : i32
  %out_off_2 = arith.addi %out_tile_base, %row_stride_2 : i32
  "vc4tile.tile_store"(%row2, %out, %out_off_2, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %row3 = vc4tile.rotate %acc4 {amount = 12 : i32} : vector<16xi32> -> vector<16xi32>
  %row_stride_3 = arith.constant 48 : i32
  %out_off_3 = arith.addi %out_tile_base, %row_stride_3 : i32
  "vc4tile.tile_store"(%row3, %out, %out_off_3, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()

  vc4tile.return
}
