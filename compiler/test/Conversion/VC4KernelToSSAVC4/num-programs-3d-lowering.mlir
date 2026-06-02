// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @num_programs_3d
// CHECK-SAME: {kind = #vc4.builtin_kind<num_programs_x>, materialization = "uniform_suffix", name = "num_programs_x", uniform_index = 1 : i32}
// CHECK-SAME: {kind = #vc4.builtin_kind<num_programs_y>, materialization = "uniform_suffix", name = "num_programs_y", uniform_index = 2 : i32}
// CHECK-SAME: {kind = #vc4.builtin_kind<num_programs_z>, materialization = "uniform_suffix", name = "num_programs_z", uniform_index = 3 : i32}
// CHECK-SAME: {kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", name = "vpm_base_row", uniform_index = 4 : i32}
// CHECK-SAME: uniform_words_per_qpu = 5 : i32
// CHECK: %[[NX:.*]] = ssavc4.uniform.read 1 : i32
// CHECK: %[[NY:.*]] = ssavc4.uniform.read 2 : i32
// CHECK: %[[NZ:.*]] = ssavc4.uniform.read 3 : i32
// CHECK: ssavc4.alu.add %[[NX]], %[[NY]]
// CHECK: ssavc4.alu.add %{{.*}}, %[[NZ]]
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @num_programs_3d(%out : i32) attributes {
    public_name = "num_programs_3d",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %nx = vc4kernel.num_programs {axis = 0 : i32} : i32
    %ny = vc4kernel.num_programs {axis = 1 : i32} : i32
    %nz = vc4kernel.num_programs {axis = 2 : i32} : i32
    %nxy = arith.addi %nx, %ny : i32
    %nxyz = arith.addi %nxy, %nz : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %value = vc4kernel.splat %nxyz : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs, %value, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
