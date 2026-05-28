// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_boundary(%out : i32, %in : i32, %n : i32) attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  %zero = arith.constant 0 : i32
  %mask = vc4tile.mask_all
  // CHECK: boundary policy zero/clamp/reject is not implemented in M5
  %bad = "vc4tile.tile_load"(%in, %zero, %mask) {
    shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<zero>
  } : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  vc4tile.return
}
