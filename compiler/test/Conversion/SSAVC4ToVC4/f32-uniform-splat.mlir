// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @f32_uniform_splat_ssavc4
// CHECK: vc4.func @kernel
// CHECK: vc4.qpu.bundle {{.*}}raddr_a = 32 : i32{{.*}}sig = #vc4.qpu_signal<small_imm>
// CHECK: vc4.qpu.bundle {{.*}}raddr_a = 32 : i32{{.*}}sig = #vc4.qpu_signal<small_imm>
// CHECK: op_mul = #vc4.mul_opcode<fmul>
// CHECK: op_add = #vc4.add_opcode<fadd>
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: ssavc4.
ssavc4.module @f32_uniform_splat_ssavc4 {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "f32_uniform_splat_ssavc4",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "alpha", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 0 : i32},
        {name = "beta", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 1 : i32}
      ],
      builtins = []
    }
  } {
    %alpha = ssavc4.uniform.read 0 : f32
    %beta = ssavc4.uniform.read 1 : f32
    %alpha_v = ssavc4.splat %alpha : f32 -> vector<16xf32>
    %beta_v = ssavc4.splat %beta : f32 -> vector<16xf32>
    %x = ssavc4.load_imm <splat32> {value = 1065353216 : i32} : vector<16xf32>
    %scaled = ssavc4.alu.mul %alpha_v, %x {opcode = #vc4.mul_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %sum = ssavc4.alu.add %scaled, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.thread_end
  }
}
