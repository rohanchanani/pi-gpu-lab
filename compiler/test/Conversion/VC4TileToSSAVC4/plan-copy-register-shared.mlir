// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @plan_copy_register_shared
// CHECK-NOT: vc4tile.copy_tile
// CHECK-NOT: vc4tile.shared_tile_alloc
// CHECK: vc4tile.shared_alloc
// CHECK: vc4tile.shared_store
vc4tile.kernel @plan_copy_register_shared(%out : i32) attributes {
  public_name = "plan_copy_register_shared",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [{abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %zero = arith.constant 0 : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  %shared = "vc4tile.shared_tile_alloc"() {rows = 1 : i32, elem_bytes = 4 : i32, shape = [1, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  "vc4tile.copy_tile"(%values, %shared, %zero, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.return
}
