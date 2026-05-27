// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_affine attributes { public_name = "bad_affine" } {
  // CHECK: strides attribute is required
  %bad = vc4tile.tile_descriptor {
    shape = [16, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<global>, layout = #vc4tile.layout<affine_2d>,
    role = #vc4tile.role<input>, offset_unit = #vc4tile.offset_unit<byte>,
    precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  vc4tile.return
}
