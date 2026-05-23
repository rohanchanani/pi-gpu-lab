// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4.module @scheduled_sink_contract
// CHECK-LABEL: vc4.func @kernel
// CHECK-SAME: domain = #vc4.execution_domain<qpu>
// CHECK-SAME: form = #vc4.function_form<scheduled>
// CHECK-SAME: kernel
// CHECK-SAME: threading = #vc4.threading_mode<threadable>
// CHECK-SAME: vc4.launch_abi =
// CHECK-SAME: vc4.resource =
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.ldi <splat32>
// CHECK: vc4.qpu.sema <release>
// CHECK: vc4.qpu.branch
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.ldi <splat32>
// CHECK: vc4.qpu.sema <acquire>

vc4.module @scheduled_sink_contract {
  vc4.func @kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<threadable>,
    "vc4.launch_abi" = {
      public_name = "scheduled_sink_contract",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "warps_per_block", kind = #vc4.builtin_kind<warps_per_block>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      uses_barrier = true,
      uses_shared_vpm = true,
      shared_vpm_bytes = 1024 : i32,
      require_full_block_residency = true,
      warps_per_block_max = 2 : i32,
      semaphores_per_block = 4 : i32
    }
  } {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 0 : i32,
      waddr_mul = 1 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.ldi <splat32> {
      value = 42 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 2 : i32,
      waddr_mul = 3 : i32
    }
    vc4.qpu.sema <release> {
      id = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 4 : i32,
      waddr_mul = 5 : i32
    }
    vc4.qpu.branch attributes {
      cond = #vc4.branch_cond<always>,
      relative = true,
      use_reg = false,
      raddr_a = 0 : i32,
      immediate = 16 : i32,
      waddr_add = 0 : i32,
      waddr_mul = 0 : i32
    } {
      vc4.qpu.bundle {
        sig = #vc4.qpu_signal<none>,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 0 : i32,
        waddr_mul = 1 : i32,
        op_add = #vc4.add_opcode<nop>,
        op_mul = #vc4.mul_opcode<nop>,
        raddr_a = 0 : i32,
        raddr_b = 1 : i32,
        add_a = #vc4.qpu_mux<a>,
        add_b = #vc4.qpu_mux<b>,
        mul_a = #vc4.qpu_mux<r0>,
        mul_b = #vc4.qpu_mux<r1>
      }
      vc4.qpu.ldi <splat32> {
        value = 7 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<never>,
        waddr_add = 2 : i32,
        waddr_mul = 3 : i32
      }
      vc4.qpu.sema <acquire> {
        id = 1 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 4 : i32,
        waddr_mul = 5 : i32
      }
    }
  }
}
