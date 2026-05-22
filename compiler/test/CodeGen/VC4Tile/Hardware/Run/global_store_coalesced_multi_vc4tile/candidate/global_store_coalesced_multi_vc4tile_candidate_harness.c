#include "vc4_m2_candidate_test_helpers.h"
#include "kernel_launch.h"

#include <stdint.h>

#define GLOBAL_STORE_COALESCED_MULTI_VC4TILE_LANES 16u
#define GLOBAL_STORE_COALESCED_MULTI_VC4TILE_CASES 7u
#define GLOBAL_STORE_COALESCED_MULTI_VC4TILE_MAX_N 33u
#define GLOBAL_STORE_COALESCED_MULTI_VC4TILE_BUFFER_N 48u
#define GLOBAL_STORE_COALESCED_MULTI_VC4TILE_SENTINEL 0x60741e5du

static const uint32_t case_n[GLOBAL_STORE_COALESCED_MULTI_VC4TILE_CASES] = {
    1u, 15u, 16u, 17u, 31u, 32u, 33u};
static uint32_t out_values[GLOBAL_STORE_COALESCED_MULTI_VC4TILE_BUFFER_N];

static void fill_host_buffer(void) {
  for (unsigned i = 0; i < GLOBAL_STORE_COALESCED_MULTI_VC4TILE_BUFFER_N; ++i)
    out_values[i] = GLOBAL_STORE_COALESCED_MULTI_VC4TILE_SENTINEL;
}

static uint32_t expected_word(unsigned lane) { return 700u + lane; }

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("global_store_coalesced_multi_vc4tile vc4_program_create failed");

  const uint32_t activeQpus = 1u;
  const uint32_t laneWidth = GLOBAL_STORE_COALESCED_MULTI_VC4TILE_LANES;
  const uint32_t bytes =
      GLOBAL_STORE_COALESCED_MULTI_VC4TILE_BUFFER_N * sizeof(uint32_t);

  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("global_store_coalesced_multi_vc4tile device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;

  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

  printk("Running VC4 global_store_coalesced_multi_vc4tile candidate bundle...\n");
  int start = timer_get_usec();
  for (unsigned c = 0; c < GLOBAL_STORE_COALESCED_MULTI_VC4TILE_CASES; ++c) {
    uint32_t n = case_n[c];
    uint32_t active = n < GLOBAL_STORE_COALESCED_MULTI_VC4TILE_LANES
                          ? n
                          : GLOBAL_STORE_COALESCED_MULTI_VC4TILE_LANES;
    fill_host_buffer();
    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        global_store_coalesced_multi_vc4tile_launch(program, grid, block,
                                                    out_dev, active) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
      printk("ERROR: global_store_coalesced_multi_vc4tile launch/copy failed n=%u\n",
             n);
      launch_failures++;
      continue;
    }

    for (unsigned lane = 0; lane < active; ++lane) {
      uint32_t expected = expected_word(lane);
      checksum_accum += out_values[lane];
      if (out_values[lane] != expected) {
        if (total_mismatches < 8) {
          printk("MISMATCH global_store_coalesced_multi_vc4tile n=%u lane=%u observed=%x expected=%x\n",
                 n, lane, out_values[lane], expected);
        }
        total_mismatches++;
      }
    }
    for (unsigned i = active; i < GLOBAL_STORE_COALESCED_MULTI_VC4TILE_BUFFER_N;
         ++i) {
      if (out_values[i] != GLOBAL_STORE_COALESCED_MULTI_VC4TILE_SENTINEL)
        sentinel_mismatches++;
    }
  }

  int elapsed = timer_get_usec() - start;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=global_store_coalesced_multi_vc4tile status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%u runtime_allocations=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)GLOBAL_STORE_COALESCED_MULTI_VC4TILE_CASES,
         total_mismatches, sentinel_mismatches, launch_failures,
         (int)activeQpus, (int)laneWidth,
         (int)GLOBAL_STORE_COALESCED_MULTI_VC4TILE_MAX_N, checksum_accum,
         global_store_coalesced_multi_vc4tile_runtime_allocations(),
         global_store_coalesced_multi_vc4tile_runtime_launches(), elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
