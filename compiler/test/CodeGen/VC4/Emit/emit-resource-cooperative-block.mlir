// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: python3 %S/../Support/check_resource_runtime_descriptor.py %t.bundle/manifest.json %t.bundle/kernel_launch.c --public-name cooperative_block_smoke --schedule-mode cooperative_block --semantic-total-vpm-rows 16 --semantic-compiler-vpm-staging-rows-per-warp 0 --semantic-user-vpm-rows 16 --semantic-semaphore-count 4

// MANIFEST-DAG: "schedule_mode": "cooperative_block"
// MANIFEST-DAG: "uses_barrier": true
// MANIFEST-DAG: "warps_per_block": 12
// MANIFEST-DAG: "user_vpm_rows_per_block": 16
// MANIFEST-DAG: "total_vpm_rows_per_block": 16
// MANIFEST-DAG: "semaphore_count_per_block": 4
// SOURCE-DAG: #define KERNEL_0_TOTAL_VPM_ROWS_PER_BLOCK 16u
// SOURCE-DAG: .resource = {
// SOURCE-DAG: .schedule_mode = VC4_SCHEDULE_COOPERATIVE_BLOCK
// SOURCE-DAG: .user_vpm_rows_per_block = KERNEL_0_USER_VPM_ROWS_PER_BLOCK
// SOURCE-DAG: .semaphore_count_per_block = KERNEL_0_SEMAPHORE_COUNT_PER_BLOCK

vc4.module @resource_cooperative_block {

  vc4.func @cooperative_block_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "cooperative_block_smoke",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 2 : i32,
      args = [

      ],
      builtins = [
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "warps_per_block", kind = #vc4.builtin_kind<warps_per_block>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 12 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    // Thread end plus two non-branch scheduled delay-slot instructions.
    // The two trailing bundles are hardware nops: both ALU pipes are inactive.
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }

}
