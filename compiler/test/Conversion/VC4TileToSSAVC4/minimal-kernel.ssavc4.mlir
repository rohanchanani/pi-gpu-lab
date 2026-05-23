// RUN: true

// This checked-in lowered-IR fixture documents the M4-03 minimal lowering
// contract for the typed verifier.  The executable lowering path remains
// vc4tile -> ssavc4 -> scheduled vc4; this file is not an artifact, not QASM,
// and not a replacement for the conversion pass.

module {
  ssavc4.module @vc4tile_lowered {
    ssavc4.func @minimal() attributes {
      function_type = () -> (),
      kernel,
      threading = #vc4.threading_mode<single>,
      vc4.launch_abi = {
        args = [],
        builtins = [
          {kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", name = "total_requests", uniform_index = 0 : i32}
        ],
        code_symbol = "minimal_vc4tile_shader",
        public_name = "minimal_vc4tile",
        symbol_name = "minimal",
        tail_policy = "exact_multiple",
        uniform_words_per_qpu = 1 : i32
      },
      vc4.resource = {
        require_full_block_residency = false,
        schedule_mode = "independent_vector",
        semaphores_per_block = 0 : i32,
        shared_vpm_bytes = 0 : i32,
        user_shared_vpm_rows_per_block = 0 : i32,
        uses_barrier = false,
        uses_shared_vpm = false,
        vpm_bytes_per_block = 0 : i32,
        vpm_rows_per_block = 0 : i32,
        warps_per_block_max = 1 : i32
      }
    } {
      ssavc4.thread_end
    }
  }
}
