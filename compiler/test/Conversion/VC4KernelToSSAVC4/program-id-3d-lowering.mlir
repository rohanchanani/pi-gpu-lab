// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @program_id_3d
// CHECK-SAME: {kind = #vc4.builtin_kind<program_id_x>, materialization = "uniform_suffix", name = "program_id_x", uniform_index = 1 : i32}
// CHECK-SAME: {kind = #vc4.builtin_kind<program_id_y>, materialization = "uniform_suffix", name = "program_id_y", uniform_index = 2 : i32}
// CHECK-SAME: {kind = #vc4.builtin_kind<program_id_z>, materialization = "uniform_suffix", name = "program_id_z", uniform_index = 3 : i32}
// CHECK-SAME: {kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", name = "vpm_base_row", uniform_index = 4 : i32}
// CHECK-SAME: uniform_words_per_qpu = 5 : i32
// CHECK: %[[X:.*]] = ssavc4.uniform.read 1 : i32
// CHECK: %[[Y:.*]] = ssavc4.uniform.read 2 : i32
// CHECK: %[[Z:.*]] = ssavc4.uniform.read 3 : i32
// CHECK: ssavc4.alu.add %[[X]], %[[Y]]
// CHECK: ssavc4.alu.add %{{.*}}, %[[Z]]
// CHECK-NOT: QPU_NUMBER
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @program_id_3d(%out : i32) attributes {
    public_name = "program_id_3d",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %x = vc4kernel.program_id {axis = 0 : i32} : i32
    %y = vc4kernel.program_id {axis = 1 : i32} : i32
    %z = vc4kernel.program_id {axis = 2 : i32} : i32
    %xy = arith.addi %x, %y : i32
    %xyz = arith.addi %xy, %z : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %offs = vc4kernel.fragment_alu.add %lanes, %offs_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value = vc4kernel.splat %xyz : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs, %value, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
