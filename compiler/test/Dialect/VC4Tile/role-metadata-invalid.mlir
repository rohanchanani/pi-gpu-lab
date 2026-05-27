// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_role_metadata attributes {schedule_mode = #vc4tile.schedule_mode<independent_vector>} {
  // CHECK: role
  %bad = "vc4tile.tile_descriptor"() {
    shape = [1, 16], rank = 2 : i32,
    element_type = i32, storage_type = i32, expressed_type = i32, accumulator_type = i32,
    memory_space = #vc4tile.memory_space<global>, layout = #vc4tile.layout<row_major>,
    role = "input", precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : () -> !vc4tile.tile
  vc4tile.return
}
