#include "vc4_m2_candidate_test_helpers.h"
#include "kernel_launch.h"

#include <stdint.h>

#define PROGRAM_ID_WRITEBACK_VC4TILE_LANES 16u
#define PROGRAM_ID_WRITEBACK_VC4TILE_CASES 4u
#define PROGRAM_ID_WRITEBACK_VC4TILE_MAX_REQUESTS 25u
#define PROGRAM_ID_WRITEBACK_VC4TILE_MAX_N \
  (PROGRAM_ID_WRITEBACK_VC4TILE_MAX_REQUESTS * PROGRAM_ID_WRITEBACK_VC4TILE_LANES)
#define PROGRAM_ID_WRITEBACK_VC4TILE_BUFFER_N \
  (PROGRAM_ID_WRITEBACK_VC4TILE_MAX_N + PROGRAM_ID_WRITEBACK_VC4TILE_LANES)
#define PROGRAM_ID_WRITEBACK_VC4TILE_SENTINEL 0x5142ab16u

static const uint32_t case_requests[PROGRAM_ID_WRITEBACK_VC4TILE_CASES] = {
    1u, 12u, 13u, 25u};
static uint32_t out_values[PROGRAM_ID_WRITEBACK_VC4TILE_BUFFER_N];

static void fill_host_buffer(void) {
  for (unsigned i = 0; i < PROGRAM_ID_WRITEBACK_VC4TILE_BUFFER_N; ++i)
    out_values[i] = PROGRAM_ID_WRITEBACK_VC4TILE_SENTINEL;
}

static uint32_t expected_word(unsigned request, unsigned lane) {
  return request * PROGRAM_ID_WRITEBACK_VC4TILE_LANES + lane;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("program_id_writeback_vc4tile vc4_program_create failed");

  const uint32_t activeQpus = 12u;
  const uint32_t laneWidth = PROGRAM_ID_WRITEBACK_VC4TILE_LANES;
  const uint32_t bytes =
      PROGRAM_ID_WRITEBACK_VC4TILE_BUFFER_N * sizeof(uint32_t);

  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("program_id_writeback_vc4tile device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int proof_request12 = 0;
  int proof_request24 = 0;
  uint32_t checksum_accum = 0;

  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(activeQpus * laneWidth, 1, 1);

  printk("Running VC4 program_id_writeback_vc4tile candidate bundle...\n");
  int start = timer_get_usec();
  for (unsigned c = 0; c < PROGRAM_ID_WRITEBACK_VC4TILE_CASES; ++c) {
    uint32_t requests = case_requests[c];
    uint32_t n = requests * PROGRAM_ID_WRITEBACK_VC4TILE_LANES;

    fill_host_buffer();
    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        program_id_writeback_vc4tile_launch(program, grid, block, out_dev, n) <
            0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
      printk("ERROR: program_id_writeback_vc4tile launch/copy failed requests=%u n=%u\n",
             requests, n);
      launch_failures++;
      continue;
    }

    for (unsigned request = 0; request < requests; ++request) {
      for (unsigned lane = 0; lane < PROGRAM_ID_WRITEBACK_VC4TILE_LANES;
           ++lane) {
        unsigned index = request * PROGRAM_ID_WRITEBACK_VC4TILE_LANES + lane;
        uint32_t expected = expected_word(request, lane);
        checksum_accum += out_values[index];
        if (out_values[index] != expected) {
          if (total_mismatches < 8) {
            printk("MISMATCH program_id_writeback_vc4tile requests=%u request=%u lane=%u observed=%x expected=%x\n",
                   requests, request, lane, out_values[index], expected);
          }
          total_mismatches++;
        }
      }
    }

    if (requests > 12u && out_values[12u * PROGRAM_ID_WRITEBACK_VC4TILE_LANES] ==
                             expected_word(12u, 0u))
      proof_request12 = 1;
    if (requests > 24u &&
        out_values[24u * PROGRAM_ID_WRITEBACK_VC4TILE_LANES + 15u] ==
            expected_word(24u, 15u))
      proof_request24 = 1;

    for (unsigned i = n; i < PROGRAM_ID_WRITEBACK_VC4TILE_BUFFER_N; ++i) {
      if (out_values[i] != PROGRAM_ID_WRITEBACK_VC4TILE_SENTINEL)
        sentinel_mismatches++;
    }
  }

  int elapsed = timer_get_usec() - start;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0 && proof_request12 && proof_request24)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=program_id_writeback_vc4tile status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d proof_request12=%d proof_request24=%d checksum_accum=%u runtime_allocations=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)PROGRAM_ID_WRITEBACK_VC4TILE_CASES, total_mismatches,
         sentinel_mismatches, launch_failures, (int)activeQpus, (int)laneWidth,
         (int)PROGRAM_ID_WRITEBACK_VC4TILE_MAX_N, proof_request12,
         proof_request24, checksum_accum,
         program_id_writeback_vc4tile_runtime_allocations(),
         program_id_writeback_vc4tile_runtime_launches(), elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
