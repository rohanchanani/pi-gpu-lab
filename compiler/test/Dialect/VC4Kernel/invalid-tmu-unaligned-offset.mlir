// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%ptr : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    // CHECK: tmu_load_fragment byte_offsets must be statically 4-byte aligned
    %p7_safe0 = arith.constant 0 : i32
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    vc4kernel.return
  }
}
