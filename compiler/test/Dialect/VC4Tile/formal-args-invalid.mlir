// RUN: vc4-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{arg_attrs length must match formal argument count}}
vc4tile.kernel @arg_attr_mismatch attributes {arg_attrs = []} {
^entry(%out: i32):
  vc4tile.return
}

// -----

// expected-error @+1 {{duplicate abi_name 'out' in arg_attrs}}
vc4tile.kernel @duplicate_name attributes {arg_attrs = [{abi_name = "out", direction = "out", kind = "buffer", type = "u32"}, {abi_name = "out", direction = "by_value", kind = "scalar", type = "u32"}]} {
^entry(%out: i32, %n: i32):
  vc4tile.return
}

// -----

// expected-error @+1 {{requires non-empty abi_name}}
vc4tile.kernel @missing_name attributes {arg_attrs = [{direction = "out", kind = "buffer", type = "u32"}]} {
^entry(%out: i32):
  vc4tile.return
}

// -----

// expected-error @+1 {{has invalid abi_name '9out'}}
vc4tile.kernel @invalid_name attributes {arg_attrs = [{abi_name = "9out", direction = "out", kind = "buffer", type = "u32"}]} {
^entry(%out: i32):
  vc4tile.return
}

// -----

// expected-error @+1 {{abi_name 'logical_request' is reserved}}
vc4tile.kernel @reserved_name attributes {arg_attrs = [{abi_name = "logical_request", direction = "by_value", kind = "scalar", type = "u32"}]} {
^entry(%request: i32):
  vc4tile.return
}

// -----

// expected-error @+1 {{must not contain uniform_index}}
vc4tile.kernel @uniform_index_arg attributes {arg_attrs = [{abi_name = "out", direction = "out", kind = "buffer", type = "u32", uniform_index = 0 : i32}]} {
^entry(%out: i32):
  vc4tile.return
}

// -----

// expected-error @+1 {{formal argument 0 type must be i32 or f32}}
vc4tile.kernel @vector_formal attributes {arg_attrs = [{abi_name = "value", direction = "by_value", kind = "scalar", type = "u32"}]} {
^entry(%value: vector<16xi32>):
  vc4tile.return
}

// -----

// expected-error @+1 {{function_type must have zero results}}
vc4tile.kernel @result_type attributes {function_type = () -> i32} {
  vc4tile.return
}

// -----

// expected-error @+1 {{kernels with formal arguments must not carry manual launch_abi}}
vc4tile.kernel @manual_launch_abi attributes {arg_attrs = [{abi_name = "out", direction = "out", kind = "buffer", type = "u32"}], launch_abi = {}} {
^entry(%out: i32):
  vc4tile.return
}

// -----

vc4tile.kernel @program_id_uniform attributes {public_name = "program_id_uniform"} {
  // expected-error @+1 {{must not carry 'uniform_index'}}
  %pid = vc4tile.program_id {uniform_index = 0 : i32} : i32
  vc4tile.return
}

// -----

vc4tile.kernel @block_id_uniform attributes {public_name = "block_id_uniform", schedule_mode = #vc4tile.schedule_mode<cooperative_block>} {
  // expected-error @+1 {{must not carry 'uniform_index'}}
  %bid = vc4tile.block_id {uniform_index = 0 : i32} : i32
  vc4tile.return
}

// -----

vc4tile.kernel @warp_id_uniform attributes {public_name = "warp_id_uniform", schedule_mode = #vc4tile.schedule_mode<cooperative_block>} {
  // expected-error @+1 {{must not carry 'uniform_index'}}
  %wid = vc4tile.warp_id {uniform_index = 0 : i32} : i32
  vc4tile.return
}
