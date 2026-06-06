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
// CHECK: %[[C6:.*]] = ssavc4.load_imm
// CHECK: %[[REQ_BYTES:.*]] = ssavc4.alu.add %[[REQ]], %{{.*}} {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
// CHECK: %[[LANES:.*]] = ssavc4.element_number : vector<16xi32>
// CHECK: %[[C2:.*]] = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
// CHECK: %[[LANE_SHIFTED:.*]] = ssavc4.alu.add %[[LANES]], %[[C2]] {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
// CHECK: %[[ZERO:.*]] = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
// CHECK: %[[LANE_BYTES:.*]] = ssavc4.alu.add %[[ZERO]], %[[LANE_SHIFTED]] {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
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
    %c6 = arith.constant 6 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %pid = vc4kernel.program_id {axis = 0 : i32} : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %request_bytes = arith.shli %pid, %c6 : i32
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %request_vec = vc4kernel.splat %request_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %request_vec, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tag_vec = vc4kernel.fragment_const {value = dense<1644167168> : vector<16xi32>} : vector<16xi32>
    %pid_vec = vc4kernel.splat %pid : i32 -> vector<16xi32>
    %pid_scaled_shift = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %pid_scaled = vc4kernel.fragment_alu.add %pid_vec, %pid_scaled_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_base = vc4kernel.fragment_alu.add %tag_vec, %pid_scaled {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %value_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
