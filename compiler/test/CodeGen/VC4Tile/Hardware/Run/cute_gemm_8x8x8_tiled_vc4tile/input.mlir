// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh cute_gemm_8x8x8_tiled_vc4tile generate

vc4tile.kernel @cute_gemm_8x8x8_tiled_vc4tile(%out : i32, %a : i32, %b : i32) attributes {
  public_name = "cute_gemm_8x8x8_tiled_vc4tile",
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
  %four = arith.constant 4 : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %mask4 = vc4tile.tail_mask %zero, %four : i32, i32 -> vector<16xi1>

  %t00_acc0 = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 0 : i32} : () -> vector<16xi32>
  %t00_a_off_0 = arith.constant 0 : i32
  %t00_b_off_0 = arith.constant 0 : i32
  %t00_lhs_0 = "vc4tile.tile_load"(%a, %t00_a_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t00_rhs_0 = "vc4tile.tile_load"(%b, %t00_b_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t00_acc1 = "vc4tile.tile_matmul"(%t00_lhs_0, %t00_rhs_0, %t00_acc0, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %t00_a_off_1 = arith.constant 16 : i32
  %t00_b_off_1 = arith.constant 32 : i32
  %t00_lhs_1 = "vc4tile.tile_load"(%a, %t00_a_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t00_rhs_1 = "vc4tile.tile_load"(%b, %t00_b_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t00_acc2 = "vc4tile.tile_matmul"(%t00_lhs_1, %t00_rhs_1, %t00_acc1, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %t00_out_off_0 = arith.constant 0 : i32
  "vc4tile.tile_store"(%t00_acc2, %out, %t00_out_off_0, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t00_out_off_1 = arith.constant 8 : i32
  %t00_row1 = vc4tile.rotate %t00_acc2 {amount = 4 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t00_row1, %out, %t00_out_off_1, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t00_out_off_2 = arith.constant 16 : i32
  %t00_row2 = vc4tile.rotate %t00_acc2 {amount = 8 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t00_row2, %out, %t00_out_off_2, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t00_out_off_3 = arith.constant 24 : i32
  %t00_row3 = vc4tile.rotate %t00_acc2 {amount = 12 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t00_row3, %out, %t00_out_off_3, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()

  %t01_acc0 = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 0 : i32} : () -> vector<16xi32>
  %t01_a_off_0 = arith.constant 0 : i32
  %t01_b_off_0 = arith.constant 16 : i32
  %t01_lhs_0 = "vc4tile.tile_load"(%a, %t01_a_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t01_rhs_0 = "vc4tile.tile_load"(%b, %t01_b_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t01_acc1 = "vc4tile.tile_matmul"(%t01_lhs_0, %t01_rhs_0, %t01_acc0, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %t01_a_off_1 = arith.constant 16 : i32
  %t01_b_off_1 = arith.constant 48 : i32
  %t01_lhs_1 = "vc4tile.tile_load"(%a, %t01_a_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t01_rhs_1 = "vc4tile.tile_load"(%b, %t01_b_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t01_acc2 = "vc4tile.tile_matmul"(%t01_lhs_1, %t01_rhs_1, %t01_acc1, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %t01_out_off_0 = arith.constant 4 : i32
  "vc4tile.tile_store"(%t01_acc2, %out, %t01_out_off_0, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t01_out_off_1 = arith.constant 12 : i32
  %t01_row1 = vc4tile.rotate %t01_acc2 {amount = 4 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t01_row1, %out, %t01_out_off_1, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t01_out_off_2 = arith.constant 20 : i32
  %t01_row2 = vc4tile.rotate %t01_acc2 {amount = 8 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t01_row2, %out, %t01_out_off_2, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t01_out_off_3 = arith.constant 28 : i32
  %t01_row3 = vc4tile.rotate %t01_acc2 {amount = 12 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t01_row3, %out, %t01_out_off_3, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()

  %t10_acc0 = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 0 : i32} : () -> vector<16xi32>
  %t10_a_off_0 = arith.constant 32 : i32
  %t10_b_off_0 = arith.constant 0 : i32
  %t10_lhs_0 = "vc4tile.tile_load"(%a, %t10_a_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t10_rhs_0 = "vc4tile.tile_load"(%b, %t10_b_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t10_acc1 = "vc4tile.tile_matmul"(%t10_lhs_0, %t10_rhs_0, %t10_acc0, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %t10_a_off_1 = arith.constant 48 : i32
  %t10_b_off_1 = arith.constant 32 : i32
  %t10_lhs_1 = "vc4tile.tile_load"(%a, %t10_a_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t10_rhs_1 = "vc4tile.tile_load"(%b, %t10_b_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t10_acc2 = "vc4tile.tile_matmul"(%t10_lhs_1, %t10_rhs_1, %t10_acc1, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %t10_out_off_0 = arith.constant 32 : i32
  "vc4tile.tile_store"(%t10_acc2, %out, %t10_out_off_0, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t10_out_off_1 = arith.constant 40 : i32
  %t10_row1 = vc4tile.rotate %t10_acc2 {amount = 4 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t10_row1, %out, %t10_out_off_1, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t10_out_off_2 = arith.constant 48 : i32
  %t10_row2 = vc4tile.rotate %t10_acc2 {amount = 8 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t10_row2, %out, %t10_out_off_2, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t10_out_off_3 = arith.constant 56 : i32
  %t10_row3 = vc4tile.rotate %t10_acc2 {amount = 12 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t10_row3, %out, %t10_out_off_3, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()

  %t11_acc0 = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 0 : i32} : () -> vector<16xi32>
  %t11_a_off_0 = arith.constant 32 : i32
  %t11_b_off_0 = arith.constant 16 : i32
  %t11_lhs_0 = "vc4tile.tile_load"(%a, %t11_a_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t11_rhs_0 = "vc4tile.tile_load"(%b, %t11_b_off_0, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t11_acc1 = "vc4tile.tile_matmul"(%t11_lhs_0, %t11_rhs_0, %t11_acc0, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %t11_a_off_1 = arith.constant 48 : i32
  %t11_b_off_1 = arith.constant 48 : i32
  %t11_lhs_1 = "vc4tile.tile_load"(%a, %t11_a_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t11_rhs_1 = "vc4tile.tile_load"(%b, %t11_b_off_1, %mask) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %t11_acc2 = "vc4tile.tile_matmul"(%t11_lhs_1, %t11_rhs_1, %t11_acc1, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  %t11_out_off_0 = arith.constant 36 : i32
  "vc4tile.tile_store"(%t11_acc2, %out, %t11_out_off_0, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t11_out_off_1 = arith.constant 44 : i32
  %t11_row1 = vc4tile.rotate %t11_acc2 {amount = 4 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t11_row1, %out, %t11_out_off_1, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t11_out_off_2 = arith.constant 52 : i32
  %t11_row2 = vc4tile.rotate %t11_acc2 {amount = 8 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t11_row2, %out, %t11_out_off_2, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  %t11_out_off_3 = arith.constant 60 : i32
  %t11_row3 = vc4tile.rotate %t11_acc2 {amount = 12 : i32} : vector<16xi32> -> vector<16xi32>
  "vc4tile.tile_store"(%t11_row3, %out, %t11_out_off_3, %mask4) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()

  vc4tile.return
}
