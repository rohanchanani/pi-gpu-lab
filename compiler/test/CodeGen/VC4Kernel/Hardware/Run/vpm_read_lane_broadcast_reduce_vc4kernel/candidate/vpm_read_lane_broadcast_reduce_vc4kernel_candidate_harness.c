#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define OUTPUT_ROWS 2u
#define OUTPUT_WORDS (OUTPUT_ROWS * LANES)
#define GUARD_WORDS 16u
#define BUFFER_WORDS (OUTPUT_WORDS + GUARD_WORDS)
#define SENTINEL (-4321.0f)

static const uint32_t cases[] = {0u, 1u, 2u, 3u, 7u, 15u, 99u};
static float input_values[LANES];
static float output_values[BUFFER_WORDS];

static float pattern(uint32_t case_id, uint32_t lane) {
  int raw = (int)(case_id * 37u + lane * 5u + 11u) - 20;
  return (float)raw * 0.25f;
}

static void fill_buffers(uint32_t case_id) {
  for (uint32_t lane = 0; lane < LANES; ++lane)
    input_values[lane] = pattern(case_id, lane);
  for (uint32_t i = 0; i < BUFFER_WORDS; ++i)
    output_values[i] = SENTINEL;
}

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static int verify_values(uint32_t case_id, uint32_t kk, float *max_abs_diff) {
  int mismatches = 0;
  float selected = kk < LANES ? input_values[kk] : 0.0f;
  *max_abs_diff = 0.0f;
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    float got = output_values[lane];
    float diff = got - selected;
    float ad = absf_local(diff);
    if (ad > *max_abs_diff)
      *max_abs_diff = ad;
    if (ad != 0.0f) {
      if (mismatches < 8)
        printk("ERROR: vpm_read_lane_broadcast_reduce case=%d kk=%d lane=%d got=%f want=%f diff=%f\n",
               (int)case_id, (int)kk, (int)lane, got, selected, diff);
      ++mismatches;
    }
  }
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    float got = output_values[LANES + lane];
    float diff = got - 0.0f;
    float ad = absf_local(diff);
    if (ad > *max_abs_diff)
      *max_abs_diff = ad;
    if (ad != 0.0f) {
      if (mismatches < 8)
        printk("ERROR: vpm_read_lane_broadcast_reduce empty case=%d lane=%d got=%f want=0.000000 diff=%f\n",
               (int)case_id, (int)lane, got, diff);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = OUTPUT_WORDS; i < BUFFER_WORDS; ++i) {
    if (output_values[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: vpm_read_lane_broadcast_reduce sentinel=%d got=%f want=%f\n",
               (int)i, output_values[i], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

static int scaled_checksum(void) {
  int checksum = 0;
  for (uint32_t i = 0; i < OUTPUT_WORDS; ++i)
    checksum += (int)(output_values[i] * 1024.0f);
  return checksum;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vpm_read_lane_broadcast_reduce_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, LANES * sizeof(float)) < 0 ||
      vc4_m2_malloc(program, &out_dev, BUFFER_WORDS * sizeof(float)) < 0)
    panic("vpm_read_lane_broadcast_reduce_vc4kernel allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  int saw_lane0 = 0;
  int saw_lane15 = 0;
  int saw_empty = 0;
  float max_abs_diff_overall = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  printk("Running VC4 vpm_read_lane_broadcast_reduce_vc4kernel candidate bundle...\n");
  for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
    uint32_t kk = cases[case_id];
    fill_buffers(case_id);
    if (vc4_m2_copy_htod(program, in_dev, input_values,
                         LANES * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_values,
                         BUFFER_WORDS * sizeof(float)) < 0 ||
        vpm_read_lane_broadcast_reduce_vc4kernel_launch(program, grid, block,
                                                        in_dev, out_dev, kk) < 0 ||
        vc4_m2_copy_dtoh(program, output_values, out_dev,
                         BUFFER_WORDS * sizeof(float)) < 0) {
      printk("ERROR: vpm_read_lane_broadcast_reduce_vc4kernel launch/copy failed case=%d kk=%d\n",
             (int)case_id, (int)kk);
      ++launch_failures;
      continue;
    }

    float max_abs_diff = 0.0f;
    int mismatches = verify_values(case_id, kk, &max_abs_diff);
    int sentinels = verify_sentinels();
    int checksum = scaled_checksum();
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += mismatches;
    sentinel_mismatches += sentinels;
    checksum_accum += checksum;
    if (kk == 0u)
      saw_lane0 = 1;
    if (kk == 15u)
      saw_lane15 = 1;
    if (kk >= LANES)
      saw_empty = 1;
    printk("VPM_READ_LANE_BROADCAST_REDUCE_CASE case=%d kk=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
           (int)case_id, (int)kk, mismatches, sentinels, checksum,
           max_abs_diff);
  }

  launch_failures +=
      (int)vpm_read_lane_broadcast_reduce_vc4kernel_runtime_launch_failures();
  uint32_t launches = vpm_read_lane_broadcast_reduce_vc4kernel_runtime_launches();
  int elapsed = timer_get_usec() - start;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0 && saw_lane0 && saw_lane15 && saw_empty &&
       launches == sizeof(cases) / sizeof(cases[0]))
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=vpm_read_lane_broadcast_reduce_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d lanes=%d saw_lane0=%d saw_lane15=%d saw_empty=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%u elapsed_usec=%d\n",
         status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
         sentinel_mismatches, launch_failures, (int)LANES, saw_lane0,
         saw_lane15, saw_empty, checksum_accum, max_abs_diff_overall, 2,
         launches, elapsed);

  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
