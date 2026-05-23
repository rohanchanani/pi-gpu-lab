// RUN: vc4-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{arg_attrs length must match formal argument count}}
vc4tile.kernel @arg_attr_mismatch(%out : i32) attributes {
  arg_attrs = []
} {
  vc4tile.return
}

// -----

// expected-error @+1 {{duplicate abi_name 'out' in arg_attrs}}
vc4tile.kernel @duplicate_name(%out : i32, %other : i32) attributes {
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32"},
    {abi_name = "out", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  vc4tile.return
}

// -----

// expected-error @+1 {{requires non-empty abi_name}}
vc4tile.kernel @missing_name(%out : i32) attributes {
  arg_attrs = [
    {kind = "buffer", direction = "out", type = "u32"}
  ]
} {
  vc4tile.return
}

// -----

// expected-error @+1 {{has invalid abi_name '9out'}}
vc4tile.kernel @invalid_name(%out : i32) attributes {
  arg_attrs = [
    {abi_name = "9out", kind = "buffer", direction = "out", type = "u32"}
  ]
} {
  vc4tile.return
}

// -----

// expected-error @+1 {{abi_name 'logical_request' is reserved}}
vc4tile.kernel @reserved_name(%request : i32) attributes {
  arg_attrs = [
    {abi_name = "logical_request", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  vc4tile.return
}

// -----

// expected-error @+1 {{must not contain uniform_index}}
vc4tile.kernel @uniform_index_arg(%out : i32) attributes {
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", uniform_index = 0 : i32}
  ]
} {
  vc4tile.return
}

// -----

// expected-error @+1 {{formal argument 0 type must be i32 or f32}}
vc4tile.kernel @vector_formal(%value : vector<16xi32>) attributes {
  arg_attrs = [
    {abi_name = "value", kind = "scalar", direction = "by_value", type = "u32"}
  ]
} {
  vc4tile.return
}

// -----

// expected-error @+1 {{function_type must have zero results}}
"vc4tile.kernel"() ({
  vc4tile.return
}) {sym_name = "result_type", function_type = (i32) -> i32} : () -> ()

// -----

// expected-error @+1 {{kernels with formal arguments must not carry manual launch_abi}}
vc4tile.kernel @manual_launch_abi(%out : i32) attributes {
  launch_abi = {},
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32"}
  ]
} {
  vc4tile.return
}

// -----

vc4tile.kernel @bad_program_id_attr attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>
} {
  // expected-error @+1 {{must not carry 'uniform_index'}}
  %pid = vc4tile.program_id {uniform_index = 0 : i32} : i32
  vc4tile.return
}

// -----

vc4tile.kernel @bad_block_id_attr attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32
} {
  // expected-error @+1 {{must not carry 'uniform_index'}}
  %bid = vc4tile.block_id {uniform_index = 0 : i32} : i32
  vc4tile.return
}

// -----

vc4tile.kernel @bad_warp_id_attr attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32
} {
  // expected-error @+1 {{must not carry 'uniform_index'}}
  %wid = vc4tile.warp_id {uniform_index = 0 : i32} : i32
  vc4tile.return
}
