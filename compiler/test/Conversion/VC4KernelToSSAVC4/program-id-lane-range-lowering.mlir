// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @program_id_lane_range
// CHECK-SAME: builtins = [
// CHECK-SAME: {kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", name = "logical_request", uniform_index = 1 : i32}
// CHECK-SAME: {kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", name = "total_requests", uniform_index = 2 : i32}
// CHECK-SAME: {kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", name = "vpm_base_row", uniform_index = 3 : i32}
// CHECK-SAME: uniform_words_per_qpu = 4 : i32
// CHECK: %[[OUT:.*]] = ssavc4.uniform.read 0 : i32
// CHECK-NEXT: %[[REQ:.*]] = ssavc4.uniform.read 1 : i32
// CHECK-NEXT: %[[TOTAL:.*]] = ssavc4.uniform.read 2 : i32
// CHECK-NEXT: %[[VPM:.*]] = ssavc4.uniform.read 3 : i32
// CHECK: %[[C2:.*]] = ssavc4.load_imm
// CHECK: %[[C6:.*]] = ssavc4.load_imm
// CHECK: %[[FULL:.*]] = ssavc4.element_number : vector<16xi32>
// CHECK: %[[REQ_BYTES:.*]] = ssavc4.alu.add %[[REQ]], %{{.*}} {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
// CHECK: %[[LANE_BYTES:.*]] = ssavc4.alu.add %[[FULL]], %{{.*}} {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
// CHECK: ssavc4.vdw.store
// CHECK-NOT: QPU_NUMBER
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @program_id_lane_range(%out : i32) attributes {
    public_name = "program_id_lane_range",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %tag = arith.constant 1644167168 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %pid = vc4kernel.program_id : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %request_bytes = arith.shli %pid, %c6 : i32
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %request_vec = vc4kernel.splat %request_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_add %request_vec, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %tag_vec = vc4kernel.splat %tag : i32 -> vector<16xi32>
    %pid_vec = vc4kernel.splat %pid : i32 -> vector<16xi32>
    %pid_scaled = vc4kernel.fragment_shl %pid_vec, %c4 : vector<16xi32>, i32 -> vector<16xi32>
    %value_base = vc4kernel.fragment_add %tag_vec, %pid_scaled : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value = vc4kernel.fragment_add %value_base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
