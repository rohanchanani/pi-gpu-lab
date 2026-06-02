module {
  vc4kernel.kernel @qpu_barrier_syncthreads_vc4kernel attributes {
    public_name = "qpu_barrier_syncthreads_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    warps_per_block = 12 : i32
  } {
    vc4kernel.barrier
    vc4kernel.return
  }
}
