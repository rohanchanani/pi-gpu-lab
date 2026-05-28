// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @vector_store_smoke_vc4tile
// SSAVC4: ssavc4.uniform.read
// SSAVC4: ssavc4.element_number
// SSAVC4: ssavc4.vdw.store
// SSAVC4: active_lanes = 16 : i32
// SSAVC4: elem_bytes = 4 : i32
// SSAVC4: serialize = "mutex"
// SSAVC4: ssavc4.thread_end
// VC4-LABEL: vc4.func @vector_store_smoke_vc4tile
// VC4: vc4.qpu.bundle
// VC4: vc4.qpu.vpmvcd_setup
// VC4: vc4.qpu.vpmvcd_addr
// VC4: vc4.qpu.vpmvcd_wait
vc4tile.kernel @vector_store_smoke_vc4tile attributes {
  public_name = "vector_store_smoke_vc4tile",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%base: i32):
  %lanes = vc4tile.lane_range : vector<16xi32>
  %bias = arith.constant 100 : i32
  %bias_vec = vector.broadcast %bias : i32 to vector<16xi32>
  %value = arith.addi %lanes, %bias_vec : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
