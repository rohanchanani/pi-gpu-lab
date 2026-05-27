// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_precision attributes { public_name = "bad_precision" } {
  // CHECK: M5 supports only 32-bit executable tile storage; got storage_type = 'f16'
  %bad = vc4tile.tile_descriptor {
    shape = [1, 16], rank = 2 : i32,
    element_type = f32, storage_type = f16, expressed_type = f32, accumulator_type = f32,
    memory_space = #vc4tile.memory_space<register>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<input>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  vc4tile.return
}
