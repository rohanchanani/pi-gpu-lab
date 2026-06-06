// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @fragment_alu(%i : i32, %f : f32) attributes {
    public_name = "fragment_alu",
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

    // CHECK: #vc4kernel.add_alu_opcode<fadd>
    %fadd = vc4kernel.fragment_alu.add %fv, %fv {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    // CHECK: #vc4kernel.add_alu_opcode<fsub>
    %fsub = vc4kernel.fragment_alu.add %fadd, %fv {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    // CHECK: #vc4kernel.add_alu_opcode<fmin>
    %fmin = vc4kernel.fragment_alu.add %fsub, %fv {opcode = #vc4kernel.add_alu_opcode<fmin>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    // CHECK: #vc4kernel.add_alu_opcode<fmax>
    %fmax = vc4kernel.fragment_alu.add %fmin, %fv {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    // CHECK: #vc4kernel.add_alu_opcode<fminabs>
    %fminabs = vc4kernel.fragment_alu.add %fmax, %fv {opcode = #vc4kernel.add_alu_opcode<fminabs>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    // CHECK: #vc4kernel.add_alu_opcode<fmaxabs>
    %fmaxabs = vc4kernel.fragment_alu.add %fminabs, %fv {opcode = #vc4kernel.add_alu_opcode<fmaxabs>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    // CHECK: #vc4kernel.add_alu_opcode<ftoi>
    %ftoi = vc4kernel.fragment_alu.add %fmaxabs {opcode = #vc4kernel.add_alu_opcode<ftoi>} : (vector<16xf32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<itof>
    %itof = vc4kernel.fragment_alu.add %ftoi {opcode = #vc4kernel.add_alu_opcode<itof>} : (vector<16xi32>) -> vector<16xf32>

    // CHECK: #vc4kernel.add_alu_opcode<add>
    %add = vc4kernel.fragment_alu.add %iv, %iv {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<sub>
    %sub = vc4kernel.fragment_alu.add %add, %iv {opcode = #vc4kernel.add_alu_opcode<sub>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<shl>
    %shl = vc4kernel.fragment_alu.add %sub, %shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<shr>
    %shr = vc4kernel.fragment_alu.add %shl, %shift {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<asr>
    %asr = vc4kernel.fragment_alu.add %shr, %shift {opcode = #vc4kernel.add_alu_opcode<asr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<ror>
    %ror = vc4kernel.fragment_alu.add %asr, %shift {opcode = #vc4kernel.add_alu_opcode<ror>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<min>
    %min = vc4kernel.fragment_alu.add %ror, %iv {opcode = #vc4kernel.add_alu_opcode<min>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<max>
    %max = vc4kernel.fragment_alu.add %min, %iv {opcode = #vc4kernel.add_alu_opcode<max>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<and>
    %and = vc4kernel.fragment_alu.add %max, %iv {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<or>
    %or = vc4kernel.fragment_alu.add %and, %iv {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<xor>
    %xor = vc4kernel.fragment_alu.add %or, %iv {opcode = #vc4kernel.add_alu_opcode<xor>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<not>
    %not = vc4kernel.fragment_alu.add %xor {opcode = #vc4kernel.add_alu_opcode<not>} : (vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<clz>
    %clz = vc4kernel.fragment_alu.add %not {opcode = #vc4kernel.add_alu_opcode<clz>} : (vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<v8adds>
    %v8adds = vc4kernel.fragment_alu.add %clz, %iv {opcode = #vc4kernel.add_alu_opcode<v8adds>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.add_alu_opcode<v8subs>
    %v8subs = vc4kernel.fragment_alu.add %v8adds, %iv {opcode = #vc4kernel.add_alu_opcode<v8subs>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    // CHECK: #vc4kernel.mul_alu_opcode<fmul>
    %fmul = vc4kernel.fragment_alu.mul %itof, %fv {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    // CHECK: #vc4kernel.mul_alu_opcode<mul24>
    %mul24 = vc4kernel.fragment_alu.mul %v8subs, %iv {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.mul_alu_opcode<v8muld>
    %v8muld = vc4kernel.fragment_alu.mul %mul24, %iv {opcode = #vc4kernel.mul_alu_opcode<v8muld>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.mul_alu_opcode<v8min>
    %v8min = vc4kernel.fragment_alu.mul %v8muld, %iv {opcode = #vc4kernel.mul_alu_opcode<v8min>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.mul_alu_opcode<v8max>
    %v8max = vc4kernel.fragment_alu.mul %v8min, %iv {opcode = #vc4kernel.mul_alu_opcode<v8max>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.mul_alu_opcode<v8adds>
    %mv8adds = vc4kernel.fragment_alu.mul %v8max, %iv {opcode = #vc4kernel.mul_alu_opcode<v8adds>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: #vc4kernel.mul_alu_opcode<v8subs>
    %mv8subs = vc4kernel.fragment_alu.mul %mv8adds, %iv {opcode = #vc4kernel.mul_alu_opcode<v8subs>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    vc4kernel.return
  }
}
