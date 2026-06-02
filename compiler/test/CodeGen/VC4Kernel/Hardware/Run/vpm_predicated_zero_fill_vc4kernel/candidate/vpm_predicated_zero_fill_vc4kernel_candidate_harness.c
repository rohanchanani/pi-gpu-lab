#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ROWS 2u
#define WORDS (ROWS * LANES)
#define GUARD_WORDS 16u
#define BUFFER_WORDS (WORDS + GUARD_WORDS)
#define SENTINEL 0xdeadbeefu
#define BASE_TAIL 0x73000000u
#define BASE_MASK 0x74000000u

struct test_case {
  uint32_t limit;
  uint32_t threshold;
};

static const struct test_case cases[] = {
    {0u, 0u},
    {5u, 7u},
    {15u, 9u},
    {16u, 16u},
};

static uint32_t out_values[BUFFER_WORDS];

static void fill_output(void) {
  for (uint32_t i = 0; i < BUFFER_WORDS; ++i)
    out_values[i] = SENTINEL;
}

static uint32_t expected_value(uint32_t row, uint32_t lane,
                               const struct test_case *tc) {
  if (row == 0u)
    return lane < tc->limit ? BASE_TAIL + lane : 0u;
  return lane < tc->threshold ? BASE_MASK + lane : 0u;
}

static int verify_active(const struct test_case *tc, uint32_t *checksum) {
  int mismatches = 0;
  *checksum = 0;
  for (uint32_t row = 0; row < ROWS; ++row) {
    for (uint32_t lane = 0; lane < LANES; ++lane) {
      uint32_t index = row * LANES + lane;
      uint32_t expected = expected_value(row, lane, tc);
      *checksum += out_values[index];
      if (out_values[index] != expected) {
        if (mismatches < 8)
          printk("ERROR: vpm_predicated_zero_fill row=%d lane=%d limit=%d threshold=%d actual=%x expected=%x\n",
                 (int)row, (int)lane, (int)tc->limit, (int)tc->threshold,
                 out_values[index], expected);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = WORDS; i < BUFFER_WORDS; ++i) {
    if (out_values[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: vpm_predicated_zero_fill sentinel=%d actual=%x expected=%x\n",
               (int)i, out_values[i], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vpm_predicated_zero_fill_vc4kernel program create failed");

  vc4_deviceptr_t out_dev = 0;
  uint32_t bytes = BUFFER_WORDS * sizeof(uint32_t);
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("vpm_predicated_zero_fill_vc4kernel allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]);
       ++case_id) {
    fill_output();
    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        vpm_predicated_zero_fill_vc4kernel_launch(
            program, grid, block, out_dev, cases[case_id].limit,
            cases[case_id].threshold) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
      printk("ERROR: vpm_predicated_zero_fill_vc4kernel launch/copy failed case=%d\n",
             (int)case_id);
      ++launch_failures;
      continue;
    }
    uint32_t checksum = 0;
    int mismatches = verify_active(&cases[case_id], &checksum);
    int sentinels = verify_sentinels();
    total_mismatches += mismatches;
    sentinel_mismatches += sentinels;
    checksum_accum += checksum;
    printk("VPM_PREDICATED_ZERO_FILL_CASE case=%d limit=%d threshold=%d mismatches=%d sentinel_mismatches=%d checksum=%u\n",
           (int)case_id, (int)cases[case_id].limit,
           (int)cases[case_id].threshold, mismatches, sentinels, checksum);
  }

  launch_failures +=
      (int)vpm_predicated_zero_fill_vc4kernel_runtime_launch_failures();
  uint32_t runtime_launches =
      vpm_predicated_zero_fill_vc4kernel_runtime_launches();
  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vpm_predicated_zero_fill_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d rows=%d checksum_accum=%u runtime_allocations=1 runtime_launches=%u elapsed_usec=%d\n",
         status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
         sentinel_mismatches, launch_failures, (int)LANES, (int)ROWS,
         checksum_accum, runtime_launches, elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
