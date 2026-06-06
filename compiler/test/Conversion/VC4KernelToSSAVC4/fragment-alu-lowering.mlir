// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_alu_lowering
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fadd>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fsub>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fmin>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fmax>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fminabs>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<fmaxabs>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<ftoi>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<itof>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<add>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<sub>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<shl>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<shr>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<asr>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<ror>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<min>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<max>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<and>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<or>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<not>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<clz>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<v8adds>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<v8subs>}
// CHECK: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<fmul>}
// CHECK: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<mul24>}
// CHECK: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<v8muld>}
// CHECK: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<v8min>}
// CHECK: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<v8max>}
// CHECK: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<v8adds>}
// CHECK: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<v8subs>}
// CHECK-NOT: vc4.qpu
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_alu_lowering(%i : i32, %f : f32) attributes {
    public_name = "fragment_alu_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "i", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "f", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c1 = arith.constant 1 : i32
    %iv = vc4kernel.splat %i : i32 -> vector<16xi32>
    %shift = vc4kernel.splat %c1 : i32 -> vector<16xi32>
    %fv = vc4kernel.splat %f : f32 -> vector<16xf32>

    %fadd = vc4kernel.fragment_alu.add %fv, %fv {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %fsub = vc4kernel.fragment_alu.add %fadd, %fv {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %fmin = vc4kernel.fragment_alu.add %fsub, %fv {opcode = #vc4kernel.add_alu_opcode<fmin>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %fmax = vc4kernel.fragment_alu.add %fmin, %fv {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %fminabs = vc4kernel.fragment_alu.add %fmax, %fv {opcode = #vc4kernel.add_alu_opcode<fminabs>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %fmaxabs = vc4kernel.fragment_alu.add %fminabs, %fv {opcode = #vc4kernel.add_alu_opcode<fmaxabs>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %ftoi = vc4kernel.fragment_alu.add %fmaxabs {opcode = #vc4kernel.add_alu_opcode<ftoi>} : (vector<16xf32>) -> vector<16xi32>
    %itof = vc4kernel.fragment_alu.add %ftoi {opcode = #vc4kernel.add_alu_opcode<itof>} : (vector<16xi32>) -> vector<16xf32>

    %add = vc4kernel.fragment_alu.add %iv, %iv {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sub = vc4kernel.fragment_alu.add %add, %iv {opcode = #vc4kernel.add_alu_opcode<sub>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %shl = vc4kernel.fragment_alu.add %sub, %shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %shr = vc4kernel.fragment_alu.add %shl, %shift {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %asr = vc4kernel.fragment_alu.add %shr, %shift {opcode = #vc4kernel.add_alu_opcode<asr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %ror = vc4kernel.fragment_alu.add %asr, %shift {opcode = #vc4kernel.add_alu_opcode<ror>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %min = vc4kernel.fragment_alu.add %ror, %iv {opcode = #vc4kernel.add_alu_opcode<min>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %max = vc4kernel.fragment_alu.add %min, %iv {opcode = #vc4kernel.add_alu_opcode<max>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %and = vc4kernel.fragment_alu.add %max, %iv {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %or = vc4kernel.fragment_alu.add %and, %iv {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %xor = vc4kernel.fragment_alu.add %or, %iv {opcode = #vc4kernel.add_alu_opcode<xor>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %not = vc4kernel.fragment_alu.add %xor {opcode = #vc4kernel.add_alu_opcode<not>} : (vector<16xi32>) -> vector<16xi32>
    %clz = vc4kernel.fragment_alu.add %not {opcode = #vc4kernel.add_alu_opcode<clz>} : (vector<16xi32>) -> vector<16xi32>
    %v8adds = vc4kernel.fragment_alu.add %clz, %iv {opcode = #vc4kernel.add_alu_opcode<v8adds>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8subs = vc4kernel.fragment_alu.add %v8adds, %iv {opcode = #vc4kernel.add_alu_opcode<v8subs>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    %fmul = vc4kernel.fragment_alu.mul %itof, %fv {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %mul24 = vc4kernel.fragment_alu.mul %v8subs, %iv {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8muld = vc4kernel.fragment_alu.mul %mul24, %iv {opcode = #vc4kernel.mul_alu_opcode<v8muld>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8min = vc4kernel.fragment_alu.mul %v8muld, %iv {opcode = #vc4kernel.mul_alu_opcode<v8min>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8max = vc4kernel.fragment_alu.mul %v8min, %iv {opcode = #vc4kernel.mul_alu_opcode<v8max>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %mv8adds = vc4kernel.fragment_alu.mul %v8max, %iv {opcode = #vc4kernel.mul_alu_opcode<v8adds>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %mv8subs = vc4kernel.fragment_alu.mul %mv8adds, %iv {opcode = #vc4kernel.mul_alu_opcode<v8subs>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    vc4kernel.return
  }
}
