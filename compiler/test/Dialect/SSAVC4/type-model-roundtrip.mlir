// RUN: vc4-opt %s | FileCheck %s

// CHECK: !ssavc4.async.token
// CHECK: !ssavc4.tmu.desc
// CHECK: !ssavc4.vpm.desc
// CHECK: !ssavc4.flags
module {
  %tok = "builtin.unrealized_conversion_cast"() : () -> !ssavc4.async.token
  %tmu = "builtin.unrealized_conversion_cast"() : () -> !ssavc4.tmu.desc
  %vpm = "builtin.unrealized_conversion_cast"() : () -> !ssavc4.vpm.desc
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  %vx = ssavc4.splat %x : i32 -> vector<16xi32>
  %flags = ssavc4.make_flags %vx {kind = #ssavc4.flag_kind<zero_test>} : (vector<16xi32>) -> !ssavc4.flags
}
