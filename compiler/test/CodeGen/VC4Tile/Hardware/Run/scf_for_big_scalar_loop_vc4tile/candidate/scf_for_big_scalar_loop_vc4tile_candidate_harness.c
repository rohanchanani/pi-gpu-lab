#include "vc4_m2_candidate_test_helpers.h"
#include "kernel_launch.h"

#include <stdint.h>

#define FIXTURE_LANES 16u
#define FIXTURE_ACTIVE_QPUS 12u
#define FIXTURE_ACTIVE_REQUESTS 12u
#define FIXTURE_LOOP_ITERATIONS 512u
#define FIXTURE_ACTIVE_WORDS (FIXTURE_ACTIVE_REQUESTS * FIXTURE_LANES)
#define FIXTURE_BUFFER_N (FIXTURE_ACTIVE_WORDS + FIXTURE_LANES)
#define FIXTURE_SENTINEL 0x515cf001u
#define FIXTURE_SEMANTIC_TOTAL 130816u
#define FIXTURE_EXPECTED_CHECKSUM 25135008u

static uint32_t out_values[FIXTURE_BUFFER_N];

static void fill_host_buffer(void) {
  for (unsigned i = 0; i < FIXTURE_BUFFER_N; ++i)
    out_values[i] = FIXTURE_SENTINEL;
}

static uint32_t expected_word(unsigned request, unsigned lane) {
  return FIXTURE_SEMANTIC_TOTAL + request * FIXTURE_LANES + lane;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("scf_for_big_scalar_loop_vc4tile vc4_program_create failed");

  const uint32_t bytes = FIXTURE_BUFFER_N * sizeof(uint32_t);

  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("scf_for_big_scalar_loop_vc4tile device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;

  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(FIXTURE_ACTIVE_QPUS * FIXTURE_LANES, 1, 1);

  printk("Running VC4 scf_for_big_scalar_loop_vc4tile candidate bundle...\n");
  fill_host_buffer();
  int start = timer_get_usec();
  if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
      scf_for_big_scalar_loop_vc4tile_launch(program, grid, block, out_dev,
                                             FIXTURE_LOOP_ITERATIONS) < 0 ||
      vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
    printk("ERROR: scf_for_big_scalar_loop_vc4tile launch/copy failed\n");
    launch_failures++;
  } else {
    for (unsigned request = 0; request < FIXTURE_ACTIVE_REQUESTS; ++request) {
      for (unsigned lane = 0; lane < FIXTURE_LANES; ++lane) {
        unsigned index = request * FIXTURE_LANES + lane;
        uint32_t expected = expected_word(request, lane);
        checksum_accum += out_values[index];
        if (out_values[index] != expected) {
          if (total_mismatches < 8) {
            printk("MISMATCH scf_for_big_scalar_loop_vc4tile request=%u lane=%u observed=%x expected=%x\n",
                   request, lane, out_values[index], expected);
          }
          total_mismatches++;
        }
      }
    }
    for (unsigned i = FIXTURE_ACTIVE_WORDS; i < FIXTURE_BUFFER_N; ++i) {
      if (out_values[i] != FIXTURE_SENTINEL)
        sentinel_mismatches++;
    }
    if (checksum_accum != FIXTURE_EXPECTED_CHECKSUM) {
      printk("MISMATCH scf_for_big_scalar_loop_vc4tile checksum observed=%u expected=%u\n",
             checksum_accum, FIXTURE_EXPECTED_CHECKSUM);
      total_mismatches++;
    }
  }

  int elapsed = timer_get_usec() - start;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=scf_for_big_scalar_loop_vc4tile status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d active_requests=%d loop_iterations=%d semantic_total=%u checksum_accum=%u runtime_allocations=%u runtime_launches=%u elapsed_usec=%d\n",
         status, 1, total_mismatches, sentinel_mismatches, launch_failures,
         (int)FIXTURE_ACTIVE_QPUS, (int)FIXTURE_LANES,
         (int)FIXTURE_ACTIVE_REQUESTS, (int)FIXTURE_LOOP_ITERATIONS,
         FIXTURE_SEMANTIC_TOTAL, checksum_accum,
         scf_for_big_scalar_loop_vc4tile_runtime_allocations(),
         scf_for_big_scalar_loop_vc4tile_runtime_launches(), elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
