// RUN: vc4-opt %s --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null

vc4.module @qpu_barrier_syncthreads {
  vc4.func @qpu_barrier_syncthreads_kernel() attributes {
    domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = { public_name = "qpu_barrier_syncthreads", tail_policy = "tail_safe", uniform_words_per_qpu = 2 : i32, args = [], builtins = [{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32}, {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}] },
    "vc4.resource" = {schedule_mode = "cooperative_block", uses_barrier = true, uses_shared_vpm = false, require_full_block_residency = true, warps_per_block_max = 12 : i32, semaphores_per_block = 4 : i32}
  } {
    // Uniforms: ra0 = logical_warp_id, ra1 = warps_per_block.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 0 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 1 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}

    // A single-warp block has no peers and must skip every semaphore access.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 31 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 1 : i32, small_imm = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<all_z_set>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 488 : i32, waddr_add = 31 : i32, waddr_mul = 30 : i32} {
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    }

    // logical_warp_id 0 is the leader; every other warp takes the nonleader path.
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, set_flags, waddr_add = 31 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<sub>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<all_z_set>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 96 : i32, waddr_add = 31 : i32, waddr_mul = 30 : i32} {
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    }
    vc4.qpu.sema <release> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 384 : i32, waddr_add = 31 : i32, waddr_mul = 30 : i32} {
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    }

    // Leader path for the 12-warp full-residency launch used by this fixture.
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.sema <release> {id = 3 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 33 : i32}

    vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<never>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}
