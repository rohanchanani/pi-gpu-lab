// RUN: vc4-opt %s --canonicalize-vc4tile-surface -o - | FileCheck %s --check-prefix=CORE
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s --check-prefix=SSAVC4

// CORE-LABEL: vc4tile.kernel @tile_matmul_active_k_lowering
// CORE-NOT: vc4tile.tile_matmul
// SSAVC4-LABEL: ssavc4.func @tile_matmul_active_k_lowering
// SSAVC4-NOT: vc4tile.tile_matmul
// SSAVC4: ssavc4.vdw.store
vc4tile.kernel @tile_matmul_active_k_lowering(%out : i32, %lhs_base : i32, %rhs_base : i32) attributes {
  public_name = "tile_matmul_active_k_lowering",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "lhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
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
  %mask = vc4tile.mask_all
  %lhs = "vc4tile.tile_load"(%lhs_base, %zero, %mask) {shape = [4, 4], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %rhs = "vc4tile.tile_load"(%rhs_base, %zero, %mask) {shape = [4, 4], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %acc = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 0 : i32} : () -> vector<16xi32>
  %result = "vc4tile.tile_matmul"(%lhs, %rhs, %acc, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, active_k = 1 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4_active_k_1"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, !vc4tile.predicate) -> vector<16xi32>
  "vc4tile.tile_store"(%result, %out, %zero, %mask) {shape = [4, 4], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
