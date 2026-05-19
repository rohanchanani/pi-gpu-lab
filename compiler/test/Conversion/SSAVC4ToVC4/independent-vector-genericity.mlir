// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @neutral_vector_module
// CHECK: vc4.func @neutral_vector_kernel
// CHECK: code_symbol = "neutral_vector_shader"
// CHECK: public_name = "neutral_vector_entry"
// CHECK: waddr_add = 56 : i32
// CHECK: sig = #vc4.qpu_signal<ldtmu0>
// CHECK: op_mul = #vc4.mul_opcode<fmul>
// CHECK: op_add = #vc4.add_opcode<fadd>
// CHECK: waddr_add = 48 : i32
// CHECK: vc4.qpu.vpmvcd_addr
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_wait
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.
ssavc4.module @neutral_vector_module {
  ssavc4.func @neutral_vector_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "neutral_vector_entry",
      code_symbol = "neutral_vector_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ],
      work_distribution = {
        base_element = "qpu_id * 16",
        stride_elements = "num_qpus * 16",
        tail_store = "exact_multiple"
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
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %out = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %scale = ssavc4.load_imm <splat32> {value = 1065353216 : i32} : vector<16xf32>
    %tok = ssavc4.tmu.request %addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %x = ssavc4.tmu.read %tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    %scaled = ssavc4.alu.mul %x, %scale {opcode = #vc4.mul_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %sum = ssavc4.alu.add %scaled, %x {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.vdw.store %out, %sum {elem_bytes = 4 : i32, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>
    ssavc4.thread_end
  }
}
