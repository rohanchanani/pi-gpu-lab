// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @formal_args_rejected(%out : i32, %n : i32) attributes {
  public_name = "formal_args_rejected",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  // CHECK: formal vc4tile.kernel arguments require ABI formal-argument lowering
  vc4tile.return
}
