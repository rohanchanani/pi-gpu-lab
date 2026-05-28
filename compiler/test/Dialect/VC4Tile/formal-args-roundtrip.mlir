// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @formal_args(
// CHECK-SAME: %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f32
// CHECK-SAME: )
// CHECK-SAME: arg_attrs = [
// CHECK-SAME: abi_name = "out"
// CHECK-SAME: abi_name = "n"
// CHECK-SAME: abi_name = "alpha"
vc4tile.kernel @formal_args(%out : i32, %n : i32, %alpha : f32) attributes {
  public_name = "formal_args",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", elem_type = "f32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "alpha", kind = "scalar", direction = "by_value", type = "f32"}
  ]
} {
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  %sum = arith.addf %alpha, %alpha : f32
  // CHECK: vc4tile.masked_store_global
  vc4tile.masked_store_global %out, %lanes, %lanes, %mask {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>, memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<coalesced>} : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  %zero = arith.constant 0 : i32
  %tail = vc4tile.core_tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %all = arith.andi %tail, %mask : vector<16xi1>
  %cmp = arith.cmpf oeq, %sum, %alpha : f32
  vc4tile.return
}

// CHECK-LABEL: vc4tile.kernel @explicit_empty
vc4tile.kernel @explicit_empty() attributes {
  public_name = "explicit_empty",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  vc4tile.return
}

// CHECK-LABEL: vc4tile.kernel @legacy_empty
vc4tile.kernel @legacy_empty attributes {
  public_name = "legacy_empty",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  vc4tile.return
}
