// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @old_signature_rejected(%ptr : i32) attributes {
    public_name = "old_signature_rejected",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    // CHECK: expected ','
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @vector_safe_offset_rejected(%ptr : i32) attributes {
    public_name = "vector_safe_offset_rejected",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    // CHECK: operand #3 must be i32
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full, %offs {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @missing_inactive_rejected(%ptr : i32) attributes {
    public_name = "missing_inactive_rejected",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %safe = arith.constant 0 : i32
    // CHECK: requires attribute 'inactive_load'
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full, %safe {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @wrong_inactive_policy_rejected(%ptr : i32) attributes {
    public_name = "wrong_inactive_policy_rejected",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %safe = arith.constant 0 : i32
    // CHECK: attribute 'inactive_load' failed to satisfy constraint
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full, %safe {inactive_load = #vc4kernel.inactive_store<preserve>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    vc4kernel.return
  }
}
