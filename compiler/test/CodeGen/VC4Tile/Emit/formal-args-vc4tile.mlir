// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses > %t.vc4.mlir
// RUN: vc4-codegen %t.vc4.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c --implicit-check-not='requestInfo->logical_request' --implicit-check-not='builtin logical_request'
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// HEADER: int formal_args_vc4tile_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t out, uint32_t n);

// SOURCE: #define KERNEL_0_NUM_UNIFS 2u
// SOURCE: *   [0] arg out
// SOURCE: *   [1] arg n
// SOURCE: uniformWords[0] = (uint32_t)ctx->out; /* arg out */
// SOURCE: uniformWords[1] = (uint32_t)ctx->n; /* arg n */
// SOURCE: int formal_args_vc4tile_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t out, uint32_t n)

// MANIFEST: "name": "out"
// MANIFEST: "direction": "out"
// MANIFEST: "name": "n"
// MANIFEST: "direction": "by_value"

vc4tile.kernel @formal_args_vc4tile(%out : i32, %n : i32) attributes {
  public_name = "formal_args_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", elem_type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  %zero = arith.constant 0 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %bias = arith.constant 300 : i32
  %bias_vec = vector.broadcast %bias : i32 to vector<16xi32>
  %value = arith.addi %lanes, %bias_vec : vector<16xi32>
  vc4tile.masked_store_global %out, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
