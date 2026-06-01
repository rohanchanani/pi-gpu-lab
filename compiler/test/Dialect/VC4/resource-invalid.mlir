// RUN: vc4-opt %s --verify-diagnostics

vc4.module @missing_warps_per_block {
  // expected-error@+1 {{"vc4.resource" requires semantic field 'warps_per_block'}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.resource" = {schedule_mode = "independent_vector", user_vpm_rows_per_block = 0 : i32, compiler_vpm_staging_rows_per_warp = 0 : i32, compiler_vpm_staging_rows_per_block = 0 : i32, total_vpm_rows_per_block = 0 : i32, uses_tmu = false, uses_vpm = false, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, uses_vdr = false, uses_vdw = false, uses_barrier = false, semaphore_count_per_block = 0 : i32, requires_vpm_base_row_builtin = false, requires_semaphore_base_builtin = false}}
}

vc4.module @zero_warps_per_block {
  // expected-error@+1 {{"vc4.resource" warps_per_block must be in range [1, 12]; got 0}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.resource" = {schedule_mode = "independent_vector", warps_per_block = 0 : i32, user_vpm_rows_per_block = 0 : i32, compiler_vpm_staging_rows_per_warp = 0 : i32, compiler_vpm_staging_rows_per_block = 0 : i32, total_vpm_rows_per_block = 0 : i32, uses_tmu = false, uses_vpm = false, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, uses_vdr = false, uses_vdw = false, uses_barrier = false, semaphore_count_per_block = 0 : i32, requires_vpm_base_row_builtin = false, requires_semaphore_base_builtin = false}}
}

vc4.module @too_many_warps_per_block {
  // expected-error@+1 {{"vc4.resource" warps_per_block must be in range [1, 12]; got 13}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.resource" = {schedule_mode = "cooperative_block", warps_per_block = 13 : i32, user_vpm_rows_per_block = 0 : i32, compiler_vpm_staging_rows_per_warp = 0 : i32, compiler_vpm_staging_rows_per_block = 0 : i32, total_vpm_rows_per_block = 0 : i32, uses_tmu = false, uses_vpm = false, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, uses_vdr = false, uses_vdw = false, uses_barrier = false, semaphore_count_per_block = 0 : i32, requires_vpm_base_row_builtin = false, requires_semaphore_base_builtin = false}}
}

vc4.module @inconsistent_total_vpm_rows {
  // expected-error@+1 {{"vc4.resource" total_vpm_rows_per_block must equal user + compiler_block + warps_per_block * compiler_per_warp + spill rows; expected 3, got 2}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.resource" = {schedule_mode = "independent_vector", warps_per_block = 1 : i32, user_vpm_rows_per_block = 1 : i32, compiler_vpm_staging_rows_per_warp = 1 : i32, compiler_vpm_staging_rows_per_block = 1 : i32, total_vpm_rows_per_block = 2 : i32, uses_tmu = false, uses_vpm = true, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, uses_vdr = false, uses_vdw = false, uses_barrier = false, semaphore_count_per_block = 0 : i32, requires_vpm_base_row_builtin = true, requires_semaphore_base_builtin = false}}
}

vc4.module @too_many_total_vpm_rows {
  // expected-error@+1 {{"vc4.resource" total_vpm_rows_per_block must fit the 64 row VPM; got 65}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.resource" = {schedule_mode = "cooperative_block", warps_per_block = 1 : i32, user_vpm_rows_per_block = 65 : i32, compiler_vpm_staging_rows_per_warp = 0 : i32, compiler_vpm_staging_rows_per_block = 0 : i32, total_vpm_rows_per_block = 65 : i32, uses_tmu = false, uses_vpm = true, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, uses_vdr = false, uses_vdw = false, uses_barrier = false, semaphore_count_per_block = 0 : i32, requires_vpm_base_row_builtin = true, requires_semaphore_base_builtin = false}}
}

vc4.module @uses_vpm_false_with_rows {
  // expected-error@+1 {{"vc4.resource" uses_vpm must equal VPM feature usage or nonzero total VPM rows}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.resource" = {schedule_mode = "independent_vector", warps_per_block = 1 : i32, user_vpm_rows_per_block = 1 : i32, compiler_vpm_staging_rows_per_warp = 0 : i32, compiler_vpm_staging_rows_per_block = 0 : i32, total_vpm_rows_per_block = 1 : i32, uses_tmu = false, uses_vpm = false, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, uses_vdr = false, uses_vdw = false, uses_barrier = false, semaphore_count_per_block = 0 : i32, requires_vpm_base_row_builtin = true, requires_semaphore_base_builtin = false}}
}

vc4.module @missing_vpm_base_builtin {
  // expected-error@+1 {{"vc4.resource" requires_vpm_base_row_builtin must equal total_vpm_rows_per_block > 0}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.resource" = {schedule_mode = "independent_vector", warps_per_block = 1 : i32, user_vpm_rows_per_block = 1 : i32, compiler_vpm_staging_rows_per_warp = 0 : i32, compiler_vpm_staging_rows_per_block = 0 : i32, total_vpm_rows_per_block = 1 : i32, uses_tmu = false, uses_vpm = true, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, uses_vdr = false, uses_vdw = false, uses_barrier = false, semaphore_count_per_block = 0 : i32, requires_vpm_base_row_builtin = false, requires_semaphore_base_builtin = false}}
}

vc4.module @missing_semaphore_base_builtin {
  // expected-error@+1 {{"vc4.resource" requires_semaphore_base_builtin must equal semaphore_count_per_block > 0}}
  vc4.func private @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>, "vc4.resource" = {schedule_mode = "cooperative_block", warps_per_block = 1 : i32, user_vpm_rows_per_block = 0 : i32, compiler_vpm_staging_rows_per_warp = 0 : i32, compiler_vpm_staging_rows_per_block = 0 : i32, total_vpm_rows_per_block = 0 : i32, uses_tmu = false, uses_vpm = false, uses_vpm_qpu_read = false, uses_vpm_qpu_write = false, uses_vdr = false, uses_vdw = false, uses_barrier = true, semaphore_count_per_block = 4 : i32, requires_vpm_base_row_builtin = false, requires_semaphore_base_builtin = false}}
}
