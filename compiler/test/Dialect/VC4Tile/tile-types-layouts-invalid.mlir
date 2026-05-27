// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_rank attributes { public_name = "bad_rank" } {
  // CHECK: rank attribute must match logical shape rank
  %bad = vc4tile.tile_descriptor {
    shape = [1, 16], rank = 1 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<register>, layout = #vc4tile.layout<row_major>,
    role = #vc4tile.role<input>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : !vc4tile.tile
  vc4tile.return
}
