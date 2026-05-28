// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @tile_load_composed_mask_lowering
// CHECK-NOT: vc4tile.mask_and
// CHECK-NOT: vc4tile.tile_bounds_mask
// CHECK-NOT: vc4tile.tail_mask
// CHECK-NOT: vc4tile.tile_load
// CHECK: ssavc4.alu.add {{.*}}opcode = #vc4.add_opcode<and>
// CHECK: ssavc4.tmu.request
// CHECK: ssavc4.tmu.read
// CHECK: ssavc4.make_flags {{.*}}kind = #ssavc4.flag_kind<zero_test>
// CHECK: ssavc4.cond_select
// CHECK-SAME: cond = #vc4.cond<zc>
// CHECK: ssavc4.vdw.store
vc4tile.kernel @tile_load_composed_mask_lowering(%out : i32, %in : i32, %rows : i32, %cols : i32, %n : i32) attributes {
  public_name = "tile_load_composed_mask_lowering",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
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
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %composed = vc4tile.mask_and %bounds, %tail : vector<16xi1>, vector<16xi1> -> vector<16xi1>
  %tile = "vc4tile.tile_load"(%in, %zero, %composed) {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %all = vc4tile.mask_all : vector<16xi1>
  "vc4tile.tile_store"(%tile, %out, %zero, %all) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
