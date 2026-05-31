// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  "vc4kernel.kernel"() ({
    // CHECK: function_type must be a FunctionType with no results
    "vc4kernel.return"() : () -> ()
  }) {
    sym_name = "result_type",
    function_type = () -> i32,
    public_name = "result_type",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = false,
      uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32,
      vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32,
      semaphores_per_block = 0 : i32
    }
  } : () -> ()
}
