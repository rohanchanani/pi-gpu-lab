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
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ],
      work_distribution = {
        base_element = "qpu_id * 16",
        stride_elements = "num_qpus * 16",
        tail_store = "exact_multiple"
      }
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 0 : i32,
      compiler_vpm_staging_rows_per_warp = 1 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 1 : i32,
      uses_tmu = true,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
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
