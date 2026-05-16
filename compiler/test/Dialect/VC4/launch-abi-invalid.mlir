// RUN: vc4-opt %s --verify-diagnostics

vc4.module @non_kernel {
  // expected-error@+1 {{"vc4.launch_abi" may appear only on vc4.func with 'kernel'}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [], builtins = [{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32}]}}
}

vc4.module @empty_public_name {
  // expected-error@+1 {{"vc4.launch_abi" requires a non-empty string 'public_name'}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [], builtins = [{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32}]}}
}

vc4.module @bad_buffer_direction {
  // expected-error@+1 {{"vc4.launch_abi" buffer argument entry requires direction = "in", "out", or "inout"}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [{name = "x", kind = "buffer", direction = "by_value", elem_type = "f32", uniform_index = 0 : i32}], builtins = [{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 1 : i32}]}}
}

vc4.module @scalar_missing_type {
  // expected-error@+1 {{"vc4.launch_abi" scalar argument entry requires type = "i32", "u32", "f32", or "index"}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", uniform_index = 0 : i32}], builtins = [{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 1 : i32}]}}
}

vc4.module @duplicate_uniform_indices {
  // expected-error@+1 {{"vc4.launch_abi" uniform indices must be unique and dense in [0, uniform_words_per_qpu)}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32}]}}
}

vc4.module @num_qpus_register {
  // expected-error@+1 {{"vc4.launch_abi" builtin kind #vc4.builtin_kind<num_qpus> must use materialization = "uniform_suffix"}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "register"}]}}
}

vc4.module @elem_num_builtin {
  // expected-error@+1 {{"vc4.launch_abi" builtin kind #vc4.builtin_kind<elem_num> must not appear}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "elem", kind = #vc4.builtin_kind<elem_num>, materialization = "register"}]}}
}

vc4.module @register_uniform_index {
  // expected-error@+1 {{"vc4.launch_abi" register-materialized builtin entry must not specify 'uniform_index'}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 1 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "register", uniform_index = 1 : i32}]}}
}

vc4.module @non_dense_uniform_indices {
  // expected-error@+1 {{"vc4.launch_abi" uniform indices must be unique and dense in [0, uniform_words_per_qpu)}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.launch_abi" = {public_name = "launch", tail_policy = "exact_multiple", uniform_words_per_qpu = 3 : i32, args = [{name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 0 : i32}], builtins = [{name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 2 : i32}]}}
}
