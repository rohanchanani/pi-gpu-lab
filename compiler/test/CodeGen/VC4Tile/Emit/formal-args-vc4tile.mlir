// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @formal_args_vc4tile
// SSAVC4-SAME: name = "out"
// SSAVC4-SAME: name = "n"
// SSAVC4: ssavc4.uniform.read 0 : i32
// SSAVC4: ssavc4.uniform.read 1 : i32
// SSAVC4: ssavc4.vdw.store
// VC4-LABEL: vc4.func @formal_args_vc4tile
// VC4: vc4.qpu.bundle
// VC4: vc4.qpu.vpmvcd_wait
vc4tile.kernel @formal_args_vc4tile attributes {
  public_name = "formal_args_vc4tile",
  arg_attrs = [
    {abi_name = "out", direction = "out", elem_type = "u32", kind = "buffer", type = "u32"},
    {abi_name = "n", direction = "by_value", kind = "scalar", type = "u32"}
  ]
} {
^entry(%out: i32, %n: i32):
  %zero = arith.constant 0 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
