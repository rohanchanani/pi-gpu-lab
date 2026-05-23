// RUN: vc4-opt %s --verify-diagnostics

vc4.module @non_kernel {
  // expected-error@+1 {{"vc4.launch_abi" may appear only on vc4.func with 'kernel'}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}]}}
}

vc4.module @empty_public_name {
  // expected-error@+1 {{"vc4.launch_abi" requires a non-empty string 'public_name'}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}]}}
}

vc4.module @bad_buffer_direction {
  // expected-error@+1 {{"vc4.launch_abi" buffer argument entry requires direction = "in", "out", or "inout"}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [{name = "x", kind = "buffer", direction = "by_value", elem_type = "f32", uniform_index = 0 : i32}], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 1 : i32}]}}
}

vc4.module @scalar_missing_type {
  // expected-error@+1 {{"vc4.launch_abi" scalar argument entry requires type = "i32", "u32", "f32", or "index"}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", uniform_index = 0 : i32}], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 1 : i32}]}}
}

vc4.module @duplicate_uniform_indices {
  // expected-error@+1 {{"vc4.launch_abi" uniform indices must be unique and dense in [0, uniform_words_per_qpu)}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}]}}
}

vc4.module @builtin_register_materialization {
  // expected-error@+1 {{"vc4.launch_abi" builtin entry requires materialization = "uniform_suffix"}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "register"}]}}
}

vc4.module @element_number_builtin {
  // expected-error@+1 {{"vc4.launch_abi" builtin entry 'element_number' is lane identity and must not be launch ABI metadata}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [], builtins = [{name = "element_number", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}]}}
}

vc4.module @runtime_name_as_arg {
  // expected-error@+1 {{"vc4.launch_abi" argument entry 'logical_request' is runtime metadata and must be a builtin}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [{name = "logical_request", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = []}}
}

vc4.module @lane_name_as_arg {
  // expected-error@+1 {{"vc4.launch_abi" argument entry 'lane_id' is lane identity and must not be launch ABI metadata}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [{name = "lane_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = []}}
}

vc4.module @bad_builtin_materialization_with_uniform_index {
  // expected-error@+1 {{"vc4.launch_abi" builtin entry requires materialization = "uniform_suffix"}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "register", uniform_index = 1 : i32}]}}
}

vc4.module @non_dense_uniform_indices {
  // expected-error@+1 {{"vc4.launch_abi" uniform indices must be unique and dense in [0, uniform_words_per_qpu)}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 3 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 2 : i32}]}}
}
