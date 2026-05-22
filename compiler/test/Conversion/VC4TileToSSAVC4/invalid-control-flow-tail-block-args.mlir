// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @invalid_control_flow_tail_block_args attributes {public_name = "invalid_control_flow_tail_block_args"} {
  %not_a_condition = arith.constant 0 : i32
  cf.cond_br %not_a_condition, ^then, ^else
^then:
  vc4tile.return
^else:
  vc4tile.return
}

// CHECK: error:
// CHECK: cf.cond_br
