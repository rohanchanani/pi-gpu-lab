// RUN: vc4-opt --legalize-vc4tile-core-cfg %s | FileCheck %s

// CHECK-NOT: scf.for
// CHECK-NOT: index
// CHECK: cf.cond_br
vc4tile.kernel @legalize_scf_for_dynamic_i32_bound_core(%lo: i32, %hi: i32) attributes {
  public_name = "legalize_scf_for_dynamic_i32_bound_core",
  arg_attrs = [
    {name = "lo", kind = "scalar", direction = "by_value", type = "u32"},
    {name = "hi", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  %lb = arith.index_cast %lo : i32 to index
  %ub = arith.index_cast %hi : i32 to index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %ii = arith.index_cast %i : index to i32
    %one = arith.constant 1 : i32
    %x = arith.addi %ii, %one : i32
  }
  vc4tile.return
}
