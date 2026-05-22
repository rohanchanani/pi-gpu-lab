// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @warp_reduce_sum_vc4tile
// SSAVC4: ssavc4.element_number
// SSAVC4: ssavc4.rotate
// SSAVC4-SAME: amount = 8 : i32
// SSAVC4: ssavc4.alu.add
// SSAVC4: ssavc4.rotate
// SSAVC4-SAME: amount = 4 : i32
// SSAVC4: ssavc4.alu.add
// SSAVC4: ssavc4.rotate
// SSAVC4-SAME: amount = 2 : i32
// SSAVC4: ssavc4.alu.add
// SSAVC4: ssavc4.rotate
// SSAVC4-SAME: amount = 1 : i32
// SSAVC4: ssavc4.alu.add
// SSAVC4: ssavc4.vdw.store
// SSAVC4: ssavc4.thread_end
// VC4-LABEL: vc4.func @warp_reduce_sum_vc4tile
// VC4: vc4.qpu.bundle
// VC4: small_imm = 56 : i32
// VC4: small_imm = 52 : i32
// VC4: small_imm = 50 : i32
// VC4: small_imm = 49 : i32
// VC4: vc4.qpu.vpmvcd_setup
// VC4: vc4.qpu.vpmvcd_addr
// VC4: vc4.qpu.vpmvcd_wait
vc4tile.kernel @warp_reduce_sum_vc4tile attributes {
  public_name = "warp_reduce_sum_vc4tile",
  launch_abi = {
    public_name = "warp_reduce_sum_vc4tile",
    code_symbol = "warp_reduce_sum_vc4tile_shader",
    tail_policy = "exact_multiple",
    uniform_words_per_qpu = 1 : i32,
    args = [{direction = "by_value", kind = "scalar", name = "out", type = "u32", uniform_index = 0 : i32}],
    builtins = []
  }
} {
  %base = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  %sum = vc4tile.reduce %lanes, %mask {kind = #vc4tile.reduce_kind<add>} : vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.masked_store_global %base, %lanes, %sum, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
