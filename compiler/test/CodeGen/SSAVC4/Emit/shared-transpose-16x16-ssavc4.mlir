// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --input-file=%t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/manifest.json
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/shared_transpose_16x16_ssavc4.qasm

// CHECK: public_name = "shared_transpose_16x16_ssavc4"
// CHECK: tail_policy = "exact_multiple"
// CHECK: schedule_mode = "cooperative_block"
// CHECK: uses_barrier = true
// CHECK: uses_shared_vpm = true
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.sema <release>
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: raddr_a = 48 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: waddr_add = 48 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @shared_transpose_16x16_ssavc4 {
  ssavc4.func @shared_transpose_16x16_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "shared_transpose_16x16_ssavc4",
      code_symbol = "shared_transpose_16x16_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 7 : i32,
      args = [
        {name = "input", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "output", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
        {name = "logical_warp_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "warps_per_block", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
        {name = "vpm_base_row", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 5 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 6 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      uses_barrier = true,
      uses_shared_vpm = true,
      shared_vpm_bytes = 1024 : i32,
      require_full_block_residency = true,
      warps_per_block_max = 4 : i32,
      semaphores_per_block = 4 : i32
    }
  } {
    %input_base = ssavc4.uniform.read 0 : i32
    %out_base = ssavc4.uniform.read 1 : i32
    %logical_warp_id = ssavc4.uniform.read 2 : i32
    %warps_per_block = ssavc4.uniform.read 3 : i32
    %vpm_base_row = ssavc4.uniform.read 4 : i32
    %qpu_id = ssavc4.uniform.read 5 : i32
    %num_qpus = ssavc4.uniform.read 6 : i32

    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %three = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %store_row = ssavc4.load_imm <splat32> {value = 63 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %lane_bytes = ssavc4.alu.add %lane, %shift_two {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %row_base = ssavc4.alu.add %logical_warp_id, %two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32

    %row0 = ssavc4.mov %row_base : i32 -> i32
    %row1 = ssavc4.alu.add %row_base, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row2 = ssavc4.alu.add %row_base, %two {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row3 = ssavc4.alu.add %row_base, %three {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32

    %row0_bytes = ssavc4.alu.add %row0, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %row0_input = ssavc4.alu.add %input_base, %row0_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row0_input_vec = ssavc4.splat %row0_input : i32 -> vector<16xi32>
    %row0_addr = ssavc4.alu.add %row0_input_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tok0 = ssavc4.tmu.request %row0_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %tile0 = ssavc4.tmu.read %tok0 {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %vpm_row0 = ssavc4.alu.add %vpm_base_row, %row0 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %vpm_row0, %tile0 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>

    %row1_bytes = ssavc4.alu.add %row1, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %row1_input = ssavc4.alu.add %input_base, %row1_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row1_input_vec = ssavc4.splat %row1_input : i32 -> vector<16xi32>
    %row1_addr = ssavc4.alu.add %row1_input_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tok1 = ssavc4.tmu.request %row1_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %tile1 = ssavc4.tmu.read %tok1 {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %vpm_row1 = ssavc4.alu.add %vpm_base_row, %row1 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %vpm_row1, %tile1 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>

    %row2_bytes = ssavc4.alu.add %row2, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %row2_input = ssavc4.alu.add %input_base, %row2_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row2_input_vec = ssavc4.splat %row2_input : i32 -> vector<16xi32>
    %row2_addr = ssavc4.alu.add %row2_input_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tok2 = ssavc4.tmu.request %row2_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %tile2 = ssavc4.tmu.read %tok2 {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %vpm_row2 = ssavc4.alu.add %vpm_base_row, %row2 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %vpm_row2, %tile2 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>

    %row3_bytes = ssavc4.alu.add %row3, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %row3_input = ssavc4.alu.add %input_base, %row3_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row3_input_vec = ssavc4.splat %row3_input : i32 -> vector<16xi32>
    %row3_addr = ssavc4.alu.add %row3_input_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tok3 = ssavc4.tmu.request %row3_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %tile3 = ssavc4.tmu.read %tok3 {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %vpm_row3 = ssavc4.alu.add %vpm_base_row, %row3 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %vpm_row3, %tile3 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>

    ssavc4.barrier %logical_warp_id, %warps_per_block : i32, i32 {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}

    %out0 = ssavc4.alu.add %out_base, %row0_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %read0 = ssavc4.vpm.read %row0 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical", serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out0, %read0, %active_full, %store_row {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32

    %out1 = ssavc4.alu.add %out_base, %row1_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %read1 = ssavc4.vpm.read %row1 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical", serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out1, %read1, %active_full, %store_row {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32

    %out2 = ssavc4.alu.add %out_base, %row2_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %read2 = ssavc4.vpm.read %row2 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical", serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out2, %read2, %active_full, %store_row {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32

    %out3 = ssavc4.alu.add %out_base, %row3_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %read3 = ssavc4.vpm.read %row3 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical", serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out3, %read3, %active_full, %store_row {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32

    ssavc4.thread_end
  }
}
