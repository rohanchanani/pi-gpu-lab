// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @tile_reduce_vc4tile
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.tmu.read
// SSAVC4: ssavc4.rotate
// SSAVC4-SAME: amount = 8 : i32
// SSAVC4: ssavc4.alu.add
// SSAVC4: ssavc4.vdw.store
// SSAVC4: ssavc4.thread_end
// VC4-LABEL: vc4.func @tile_reduce_vc4tile
// VC4: vc4.qpu.bundle
// VC4: small_imm = 56 : i32
// VC4: vc4.qpu.vpmvcd_setup
vc4tile.kernel @tile_reduce_vc4tile(%out : i32, %in : i32) attributes {
  public_name = "tile_reduce_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
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
  %mask = vc4tile.mask_all : vector<16xi1>
  %tile = "vc4tile.tile_load"(%in, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %sum = "vc4tile.tile_reduce"(%tile, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, kind = #vc4tile.reduce_kind<add>, axis = 1 : i32} : (vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  "vc4tile.tile_store"(%sum, %out, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
