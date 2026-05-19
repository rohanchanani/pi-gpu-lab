// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @vdw_store_lowering
// CHECK: vc4.func @vdw_store_kernel
// CHECK: domain = #vc4.execution_domain<qpu>
// CHECK: form = #vc4.function_form<scheduled>
// CHECK: vc4.launch_abi =
// CHECK: public_name = "vdw_store_lowering"
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: waddr_add = 48 : i32
// CHECK: vc4.qpu.vpmvcd_wait
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: value = -1073741824 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: sig = #vc4.qpu_signal<small_imm>
// CHECK: vc4.qpu.vpmvcd_addr
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_wait
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.
ssavc4.module @vdw_store_lowering {
  ssavc4.func @vdw_store_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_store_lowering",
      code_symbol = "vdw_store_lowering_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "case_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 3 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 4 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      uses_barrier = false,
      uses_shared_vpm = false,
      require_full_block_residency = false,
      semaphores_per_block = 0 : i32,
      warps_per_block_max = 1 : i32
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    ssavc4.vdw.store %addr, %value {elem_bytes = 4 : i32, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
