#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define COND_SELECT_LANE_MASK_SSAVC4_ACTIVE_QPUS 12u
#define COND_SELECT_LANE_MASK_SSAVC4_LANE_WIDTH 16u
#define COND_SELECT_LANE_MASK_SSAVC4_MAX_N \
  (COND_SELECT_LANE_MASK_SSAVC4_ACTIVE_QPUS * COND_SELECT_LANE_MASK_SSAVC4_LANE_WIDTH)
#define COND_SELECT_LANE_MASK_SSAVC4_GUARD 32u
#define COND_SELECT_LANE_MASK_SSAVC4_BUFFER_N \
  (COND_SELECT_LANE_MASK_SSAVC4_MAX_N + COND_SELECT_LANE_MASK_SSAVC4_GUARD)
#define COND_SELECT_LANE_MASK_SSAVC4_SENTINEL 0xdeadbeefu

static uint32_t out_values[COND_SELECT_LANE_MASK_SSAVC4_BUFFER_N];
static const uint32_t thresholds[] = {0u, 1u, 7u, 8u, 15u, 16u};

static void fill_host_buffer(void) {
  for (uint32_t i = 0; i < COND_SELECT_LANE_MASK_SSAVC4_BUFFER_N; ++i)
    out_values[i] = COND_SELECT_LANE_MASK_SSAVC4_SENTINEL;
}

static uint32_t expected_value(uint32_t threshold, uint32_t index) {
  uint32_t lane = index & (COND_SELECT_LANE_MASK_SSAVC4_LANE_WIDTH - 1u);
  if (lane < threshold)
    return 0x11000000u + index;
  return 0x22000000u + index * 3u;
}

static int verify_results(uint32_t threshold) {
  int mismatches = 0;
  for (uint32_t i = 0; i < COND_SELECT_LANE_MASK_SSAVC4_MAX_N; ++i) {
    uint32_t expected = expected_value(threshold, i);
    if (out_values[i] != expected) {
      if (mismatches < 8)
        printk("ERROR: cond_select_lane_mask_ssavc4 threshold=%d i=%d gpu=%x expected=%x\n",
               (int)threshold, (int)i, out_values[i], expected);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_sentinel_tail(void) {
  int mismatches = 0;
  for (uint32_t i = COND_SELECT_LANE_MASK_SSAVC4_MAX_N;
       i < COND_SELECT_LANE_MASK_SSAVC4_BUFFER_N; ++i) {
    if (out_values[i] != COND_SELECT_LANE_MASK_SSAVC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: cond_select_lane_mask_ssavc4 sentinel changed i=%d value=%x expected=%x\n",
               (int)i, out_values[i], COND_SELECT_LANE_MASK_SSAVC4_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

static int checksum_low16(const uint32_t *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] & 0xffffu);
  return checksum;
}

void notmain(void) {
  struct vc4_program *program = 0;
  const uint32_t active_qpus = COND_SELECT_LANE_MASK_SSAVC4_ACTIVE_QPUS;
  const uint32_t lane_width = COND_SELECT_LANE_MASK_SSAVC4_LANE_WIDTH;
  const uint32_t case_count = sizeof(thresholds) / sizeof(thresholds[0]);
  const uint32_t bytes = COND_SELECT_LANE_MASK_SSAVC4_BUFFER_N * sizeof(uint32_t);
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(active_qpus * lane_width, 1, 1);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vc4_program_create failed");

  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("cond_select_lane_mask_ssavc4 device allocation failed");

  printk("Running VC4 cond_select_lane_mask_ssavc4 candidate bundle...\n");
  int start = timer_get_usec();
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;

  for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
    uint32_t threshold = thresholds[case_index];
    fill_host_buffer();

    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        cond_select_lane_mask_ssavc4_launch(program, grid, block, out_dev,
                                            threshold) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
      printk("ERROR: cond_select_lane_mask_ssavc4 launch/copy failed threshold=%d\n",
             (int)threshold);
      ++launch_failures;
      continue;
    }

    int mismatches = verify_results(threshold);
    int case_sentinel_mismatches = verify_sentinel_tail();
    int checksum = checksum_low16(out_values, COND_SELECT_LANE_MASK_SSAVC4_MAX_N);
    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinel_mismatches;
    checksum_accum += checksum;
    printk("COND_SELECT_LANE_MASK_SSAVC4_CASE threshold=%d n=%d qpus=%d lanes=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
           (int)threshold, (int)COND_SELECT_LANE_MASK_SSAVC4_MAX_N,
           (int)active_qpus, (int)lane_width, mismatches,
           case_sentinel_mismatches, checksum);
  }

  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=cond_select_lane_mask_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, (int)active_qpus, (int)lane_width,
         (int)COND_SELECT_LANE_MASK_SSAVC4_MAX_N, checksum_accum, 1,
         (int)case_count, elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
