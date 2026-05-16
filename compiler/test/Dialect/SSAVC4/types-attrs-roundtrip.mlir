// RUN: vc4-opt %s | FileCheck %s

// CHECK-DAG: !ssavc4.async.token
// CHECK-DAG: !ssavc4.tmu.desc
// CHECK-DAG: !ssavc4.vpm.desc
// CHECK-DAG: !ssavc4.flags
// CHECK-DAG: #ssavc4.tmu_unit<tmu0>
// CHECK-DAG: #ssavc4.tmu_mode<direct>
// CHECK-DAG: #ssavc4.tmu_read_part<raw32>
// CHECK-DAG: #ssavc4.vpm_orientation<horizontal>
// CHECK-DAG: #ssavc4.vpm_lane_mode<laned>
// CHECK-DAG: #ssavc4.vpm_elem_width<w32>
// CHECK-DAG: #ssavc4.vdw_store_serialize<mutex>
// CHECK-DAG: #ssavc4.flag_kind<zero_test>
module attributes {
  ssavc4.tmu_unit = #ssavc4.tmu_unit<tmu0>,
  ssavc4.tmu_mode = #ssavc4.tmu_mode<direct>,
  ssavc4.tmu_part = #ssavc4.tmu_read_part<raw32>,
  ssavc4.vpm_orientation = #ssavc4.vpm_orientation<horizontal>,
  ssavc4.vpm_lane_mode = #ssavc4.vpm_lane_mode<laned>,
  ssavc4.vpm_elem_width = #ssavc4.vpm_elem_width<w32>,
  ssavc4.vdw_serialize = #ssavc4.vdw_store_serialize<mutex>,
  ssavc4.flag_kind = #ssavc4.flag_kind<zero_test>
} {
  %tok = "builtin.unrealized_conversion_cast"() : () -> !ssavc4.async.token
  %tmu = "builtin.unrealized_conversion_cast"() : () -> !ssavc4.tmu.desc
  %vpm = "builtin.unrealized_conversion_cast"() : () -> !ssavc4.vpm.desc
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  %vx = ssavc4.splat %x : i32 -> vector<16xi32>
  %flags = ssavc4.make_flags %vx {kind = #ssavc4.flag_kind<zero_test>} : (vector<16xi32>) -> !ssavc4.flags
}
