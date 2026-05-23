// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @program_id_writeback_vc4tile
// SSAVC4-SAME: vc4.launch_abi =
// SSAVC4-SAME: name = "out"
// SSAVC4-SAME: name = "n"
// SSAVC4-SAME: #vc4.builtin_kind<logical_request>
// SSAVC4-SAME: #vc4.builtin_kind<total_requests>
// SSAVC4: ssavc4.uniform.read 0 : i32
// SSAVC4: ssavc4.uniform.read 1 : i32
// SSAVC4: ssavc4.uniform.read 2 : i32
// SSAVC4: ssavc4.element_number
// SSAVC4: ssavc4.vdw.store
// SSAVC4: ssavc4.thread_end
// VC4-LABEL: vc4.func @program_id_writeback_vc4tile
// VC4: vc4.qpu.bundle
// VC4: vc4.qpu.vpmvcd_setup
// VC4: vc4.qpu.vpmvcd_wait
vc4tile.kernel @program_id_writeback_vc4tile attributes {
  public_name = "program_id_writeback_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", elem_type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
^entry(%out: i32, %n: i32):
  %pid = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %four = arith.constant 4 : i32
  %six = arith.constant 6 : i32
  %pid_lanes = arith.shli %pid, %four : i32
  %pid_bytes = arith.shli %pid, %six : i32
  %base = arith.addi %out, %pid_bytes : i32
  %mask = vc4tile.tail_mask %pid_lanes, %n : i32, i32 -> vector<16xi1>
  %pid_vec = vector.broadcast %pid_lanes : i32 to vector<16xi32>
  %value = arith.addi %pid_vec, %lanes : vector<16xi32>
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
