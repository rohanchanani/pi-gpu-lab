// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @launch_lane(%n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %np = vc4value.num_programs {axis = 0 : i32} : index
  %lanes = vector.step : vector<16xindex>
  %pid_v = vector.broadcast %pid : index to vector<16xindex>
  %sum = arith.addi %lanes, %pid_v : vector<16xindex>
  return
}

// CHECK-LABEL: vc4kernel.kernel @launch_lane
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 0
// CHECK-SAME: : i32
// CHECK: vc4kernel.num_programs
// CHECK-SAME: axis = 0
// CHECK-SAME: : i32
// CHECK: vc4kernel.lane_range : vector<16xi32>
// CHECK: vc4kernel.splat {{.*}} : i32 -> vector<16xi32>
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<add>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
