#include "vc4_m2_candidate_test_helpers.h"
#include "kernel_launch.h"

#include <stdint.h>

#define SAXPY_FULL_VC4TILE_LANES 16u
#define SAXPY_FULL_VC4TILE_MAX_N 16u
#define SAXPY_FULL_VC4TILE_BUFFER_N 32u
#define SAXPY_FULL_VC4TILE_SENTINEL 0x5a7c06a9u

static uint32_t values[SAXPY_FULL_VC4TILE_BUFFER_N];

static uint32_t initial_word(unsigned lane) { return 200u + 3u * lane; }
static uint32_t expected_word(unsigned lane) { return 2u * initial_word(lane) + lane; }

static void fill_host_buffer(void) {
  for (unsigned i = 0; i < SAXPY_FULL_VC4TILE_BUFFER_N; ++i)
    values[i] = SAXPY_FULL_VC4TILE_SENTINEL;
  for (unsigned lane = 0; lane < SAXPY_FULL_VC4TILE_LANES; ++lane)
    values[lane] = initial_word(lane);
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("saxpy_full_vc4tile vc4_program_create failed");

  const uint32_t activeQpus = 1u;
  const uint32_t laneWidth = SAXPY_FULL_VC4TILE_LANES;
  const uint32_t bytes = SAXPY_FULL_VC4TILE_BUFFER_N * sizeof(uint32_t);

  vc4_deviceptr_t buffer_dev = 0;
  if (vc4_m2_malloc(program, &buffer_dev, bytes) < 0)
    panic("saxpy_full_vc4tile device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;

  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

  printk("Running VC4 saxpy_full_vc4tile candidate bundle...\n");
  fill_host_buffer();
  int start = timer_get_usec();
  if (vc4_m2_copy_htod(program, buffer_dev, values, bytes) < 0 ||
      saxpy_full_vc4tile_launch(program, grid, block, buffer_dev) < 0 ||
      vc4_m2_copy_dtoh(program, values, buffer_dev, bytes) < 0) {
    printk("ERROR: saxpy_full_vc4tile launch/copy failed\n");
    launch_failures++;
  } else {
    for (unsigned lane = 0; lane < SAXPY_FULL_VC4TILE_LANES; ++lane) {
      uint32_t expected = expected_word(lane);
      checksum_accum += values[lane];
      if (values[lane] != expected) {
        if (total_mismatches < 8) {
          printk("MISMATCH saxpy_full_vc4tile lane=%u observed=%x expected=%x initial=%x\n",
                 lane, values[lane], expected, initial_word(lane));
        }
        total_mismatches++;
      }
    }
    for (unsigned i = SAXPY_FULL_VC4TILE_LANES;
         i < SAXPY_FULL_VC4TILE_BUFFER_N; ++i) {
      if (values[i] != SAXPY_FULL_VC4TILE_SENTINEL)
        sentinel_mismatches++;
    }
  }

  int elapsed = timer_get_usec() - start;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=saxpy_full_vc4tile status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%u runtime_allocations=%u runtime_launches=%u elapsed_usec=%d\n",
         status, 1, total_mismatches, sentinel_mismatches, launch_failures,
         (int)activeQpus, (int)laneWidth, (int)SAXPY_FULL_VC4TILE_MAX_N,
         checksum_accum, saxpy_full_vc4tile_runtime_allocations(),
         saxpy_full_vc4tile_runtime_launches(), elapsed);

  vc4Free(program, buffer_dev);
  vc4_program_destroy(program);
}
