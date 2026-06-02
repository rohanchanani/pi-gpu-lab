module {
  vc4kernel.kernel @resource_limited_semaphore_residency_vc4kernel attributes {
    public_name = "resource_limited_semaphore_residency_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    warps_per_block = 4 : i32
  } {
    vc4kernel.barrier
    vc4kernel.return
  }
}
