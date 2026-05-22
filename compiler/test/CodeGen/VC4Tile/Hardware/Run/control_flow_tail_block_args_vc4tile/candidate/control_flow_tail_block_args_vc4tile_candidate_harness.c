#include "vc4_m2_candidate_test_helpers.h"
#include "kernel_launch.h"

#include <stdint.h>

#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_LANES 16u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_MAX_N 8u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_BUFFER_N 32u
#define CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_SENTINEL 0x7cb10c07u

static uint32_t out_values[CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_BUFFER_N];

static void fill_host_buffer(void) {
  for (unsigned i = 0; i < CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_BUFFER_N; ++i)
    out_values[i] = CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_SENTINEL;
}

static uint32_t expected_word(unsigned lane) { return 700u + lane; }

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("control_flow_tail_block_args_vc4tile vc4_program_create failed");

  const uint32_t activeQpus = 1u;
  const uint32_t laneWidth = CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_LANES;
  const uint32_t maxN = CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_MAX_N;
  const uint32_t bytes = CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_BUFFER_N * sizeof(uint32_t);

  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("control_flow_tail_block_args_vc4tile device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;

  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

  printk("Running VC4 control_flow_tail_block_args_vc4tile candidate bundle...\n");
  fill_host_buffer();
  int start = timer_get_usec();
  if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
      control_flow_tail_block_args_vc4tile_launch(program, grid, block, out_dev) < 0 ||
      vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
    printk("ERROR: control_flow_tail_block_args_vc4tile launch/copy failed\n");
    launch_failures++;
  } else {
    for (unsigned lane = 0; lane < maxN; ++lane) {
      uint32_t expected = expected_word(lane);
      checksum_accum += out_values[lane];
      if (out_values[lane] != expected) {
        if (total_mismatches < 8) {
          printk("MISMATCH control_flow_tail_block_args_vc4tile lane=%u observed=%x expected=%x\n",
                 lane, out_values[lane], expected);
        }
        total_mismatches++;
      }
    }
    for (unsigned i = maxN; i < CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_BUFFER_N; ++i) {
      if (out_values[i] != CONTROL_FLOW_TAIL_BLOCK_ARGS_VC4TILE_SENTINEL)
        sentinel_mismatches++;
    }
  }

  int elapsed = timer_get_usec() - start;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=control_flow_tail_block_args_vc4tile status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%u runtime_allocations=%u runtime_launches=%u elapsed_usec=%d\n",
         status, 1, total_mismatches, sentinel_mismatches, launch_failures,
         (int)activeQpus, (int)laneWidth, (int)maxN, checksum_accum,
         control_flow_tail_block_args_vc4tile_runtime_allocations(),
         control_flow_tail_block_args_vc4tile_runtime_launches(), elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
