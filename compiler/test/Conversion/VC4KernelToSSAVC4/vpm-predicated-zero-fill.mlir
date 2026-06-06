// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @vpm_predicated_zero_fill
// CHECK: ssavc4.element_number
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.vpm.write
// CHECK: ssavc4.vpm.read
// CHECK: ssavc4.cond_select
// CHECK-NOT: vpm_write_fragment lowering currently supports only pred.full
// CHECK-NOT: vpm_read_fragment lowering currently supports only pred.full
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @vpm_predicated_zero_fill(%limit : i32) attributes {
    public_name = "vpm_predicated_zero_fill",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "limit", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tail = vc4kernel.pred.tail %c0, %limit : i32, i32 -> !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %five = vc4kernel.fragment_const {value = dense<5> : vector<16xi32>} : vector<16xi32>
    %mask = vc4kernel.fragment_cmp %lanes, %five {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %c0, %lanes, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %read = vc4kernel.vpm_read_fragment %tile, %c0, %mask {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
