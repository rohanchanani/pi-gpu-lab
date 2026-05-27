// RUN: not vc4-opt --legalize-vc4tile-core-cfg %s 2>&1 | FileCheck %s

// CHECK: positive static

vc4tile.kernel @legalize_scf_for_dynamic_step_invalid attributes {
  public_name = "legalize_scf_for_dynamic_step_invalid",
  arg_attrs = [{name = "step", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%step_i32: i32):
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.index_cast %step_i32 : i32 to index
  scf.for %i = %lb to %ub step %step {
  }
  vc4tile.return
}
