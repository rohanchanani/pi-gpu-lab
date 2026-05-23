// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @cooperative_ids
// CHECK-SAME: builtins = [
// CHECK-SAME: kind = #vc4.builtin_kind<logical_block_id>
// CHECK-SAME: uniform_index = 0 : i32
// CHECK-SAME: kind = #vc4.builtin_kind<logical_warp_id>
// CHECK-SAME: uniform_index = 1 : i32
// CHECK-SAME: uniform_words_per_qpu = 2 : i32
// CHECK: ssavc4.uniform.read 0 {abi_name = "logical_block_id", name = "logical_block_id"} : i32
// CHECK: ssavc4.uniform.read 1 {abi_name = "logical_warp_id", name = "logical_warp_id"} : i32
// CHECK: ssavc4.thread_end
vc4tile.kernel @cooperative_ids attributes {
  public_name = "cooperative_ids",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>
} {
  %block = vc4tile.block_id : i32
  %warp = vc4tile.warp_id : i32
  vc4tile.return
}
