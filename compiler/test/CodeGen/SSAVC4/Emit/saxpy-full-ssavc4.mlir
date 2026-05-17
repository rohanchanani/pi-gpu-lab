// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/manifest.json
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/saxpy_full_ssavc4.qasm

ssavc4.module @saxpy_full_ssavc4 {
  ssavc4.func @saxpy_full_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    lowering_template = "tmu0_raw32_f32_axpy_tail_safe",
    "vc4.launch_abi" = {
      public_name = "saxpy_full_ssavc4",
      code_symbol = "saxpy_full_ssavc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
        {name = "y", kind = "buffer", direction = "inout", elem_type = "f32", uniform_index = 1 : i32},
        {name = "alpha", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 2 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 5 : i32}
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
    %base = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %tok = ssavc4.tmu.request %zero_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %x = ssavc4.tmu.read %tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    ssavc4.vdw.store %base, %x {
      elem_bytes = 4 : i32,
      active_lanes = 16 : i32,
      vpm_row = 0 : i32,
      serialize = "mutex"
    } : i32, vector<16xf32>
    ssavc4.thread_end
  }
}
