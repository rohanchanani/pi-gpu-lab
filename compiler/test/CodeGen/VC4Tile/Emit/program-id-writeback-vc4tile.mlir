// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @program_id_writeback_vc4tile
// SSAVC4-SAME: args = [
// SSAVC4-SAME: abi_name = "out"
// SSAVC4-SAME: uniform_index = 0 : i32
// SSAVC4-SAME: abi_name = "n"
// SSAVC4-SAME: uniform_index = 1 : i32
// SSAVC4-SAME: builtins = [
// SSAVC4-SAME: kind = #vc4.builtin_kind<logical_request>
// SSAVC4-SAME: name = "logical_request"
// SSAVC4-SAME: uniform_index = 2 : i32
// SSAVC4-SAME: tail_policy = "tail_safe"
// SSAVC4-SAME: uniform_words_per_qpu = 3 : i32
// SSAVC4: %[[OUT:.*]] = ssavc4.uniform.read 0 {abi_name = "out", name = "out"} : i32
// SSAVC4: ssavc4.uniform.read 1 {abi_name = "n", name = "n"} : i32
// SSAVC4: %[[PID:.*]] = ssavc4.uniform.read 2 {abi_name = "logical_request", name = "logical_request"} : i32
// SSAVC4: %[[LANES:.*]] = ssavc4.element_number : vector<16xi32>
// SSAVC4: ssavc4.alu.add %[[PID]]
// SSAVC4-SAME: opcode = #vc4.add_opcode<shl>
// SSAVC4: ssavc4.splat
// SSAVC4: ssavc4.alu.add
// SSAVC4: ssavc4.vdw.store
// VC4-LABEL: vc4.func @program_id_writeback_vc4tile
// VC4: vc4.qpu.bundle
// VC4: vc4.qpu.vpmvcd_setup
// VC4: vc4.qpu.vpmvcd_addr
// VC4: vc4.qpu.vpmvcd_wait
vc4tile.kernel @program_id_writeback_vc4tile(%out : i32, %n : i32) attributes {
  public_name = "program_id_writeback_vc4tile",
  tail_policy = "tail_safe",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", elem_type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  %pid = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %four = arith.constant 4 : i32
  %six = arith.constant 6 : i32
  %pid_elem_base = arith.shli %pid, %four : i32
  %pid_vec = vector.broadcast %pid_elem_base : i32 to vector<16xi32>
  %value = arith.addi %pid_vec, %lanes : vector<16xi32>
  %pid_byte_offset = arith.shli %pid, %six : i32
  %base = arith.addi %out, %pid_byte_offset : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
