// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --input-file=%t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/manifest.json
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/warp_reduce_sum_ssavc4.qasm

// CHECK: public_name = "warp_reduce_sum_ssavc4"
// CHECK: sig = #vc4.qpu_signal<ldtmu0>
// CHECK: small_imm = 56 : i32
// CHECK: op_add = #vc4.add_opcode<fadd>
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @warp_reduce_sum_ssavc4 {
  ssavc4.func @warp_reduce_sum_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "warp_reduce_sum_ssavc4",
      code_symbol = "warp_reduce_sum_ssavc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 3 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 4 : i32}
      ],
      work_distribution = {
        base_element = "qpu_id * 16",
        stride_elements = "num_qpus * 16",
        tail_store = "dynamic_vdw_depth"
      }
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block_max = 1 : i32,
      uses_shared_vpm = false,
      uses_barrier = false,
      semaphores_per_block = 0 : i32,
      require_full_block_residency = false
    }
  } {
    %zero_addr = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %out_base = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %tok = ssavc4.tmu.request %zero_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %x = ssavc4.tmu.read %tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    %r8 = ssavc4.rotate %x {amount = 8 : i32} : vector<16xf32> -> vector<16xf32>
    %s8 = ssavc4.alu.add %x, %r8 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r4 = ssavc4.rotate %s8 {amount = 4 : i32} : vector<16xf32> -> vector<16xf32>
    %s4 = ssavc4.alu.add %s8, %r4 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r2 = ssavc4.rotate %s4 {amount = 2 : i32} : vector<16xf32> -> vector<16xf32>
    %s2 = ssavc4.alu.add %s4, %r2 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r1 = ssavc4.rotate %s2 {amount = 1 : i32} : vector<16xf32> -> vector<16xf32>
    %sum = ssavc4.alu.add %s2, %r1 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.vdw.store %out_base, %sum {
      elem_bytes = 4 : i32,
      active_lanes = 16 : i32,
      vpm_row = 0 : i32,
      serialize = "mutex"
    } : i32, vector<16xf32>
    ssavc4.thread_end
  }
}
