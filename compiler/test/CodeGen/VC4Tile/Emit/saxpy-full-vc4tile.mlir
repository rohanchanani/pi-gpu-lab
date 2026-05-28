// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @saxpy_full_vc4tile
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.tmu.read
// SSAVC4: ssavc4.alu.mul
// SSAVC4: ssavc4.alu.add
// SSAVC4: ssavc4.vdw.store
// SSAVC4: ssavc4.thread_end
// VC4-LABEL: vc4.func @saxpy_full_vc4tile
// VC4: vc4.qpu.bundle
// VC4: ldtmu0
// VC4: vc4.qpu.vpmvcd_setup
// VC4: vc4.qpu.vpmvcd_addr
// VC4: vc4.qpu.vpmvcd_wait
vc4tile.kernel @saxpy_full_vc4tile attributes {
  public_name = "saxpy_full_vc4tile",
  arg_attrs = [{name = "buffer", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%base: i32):
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  %loaded = vc4tile.masked_load_global %base, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  %two = arith.constant 2 : i32
  %two_vec = vector.broadcast %two : i32 to vector<16xi32>
  %scaled = arith.muli %loaded, %two_vec : vector<16xi32>
  %result = arith.addi %scaled, %lanes : vector<16xi32>
  vc4tile.masked_store_global %base, %lanes, %result, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
