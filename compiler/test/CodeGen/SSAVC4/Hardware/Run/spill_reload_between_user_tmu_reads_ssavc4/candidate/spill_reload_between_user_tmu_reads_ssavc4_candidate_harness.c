#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_EPSILON 0.0002f
#define SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_CHECKSUM_SCALE 4096.0f
#define SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES 16u
#define SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_GUARD 32u
#define SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_SENTINEL (-12345.0f)
#define SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_BETA 0.03125f
#define SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_SPILL_ADJUST \
  (1224.0f * SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_BETA)

static float a_values[SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES];
static float b_values[SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES];
static float out_values[SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES +
                        SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_GUARD];
static float expected_values[SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES +
                             SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_GUARD];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float a_value(uint32_t lane) {
  return ((float)((lane * 7u + 5u) % 31u) - 15.0f) * 0.0625f;
}

static float b_value(uint32_t lane) {
  return ((float)((lane * 11u + 3u) % 37u) - 18.0f) * 0.03125f;
}

static void fill_inputs(void) {
  for (uint32_t lane = 0; lane < SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES; ++lane) {
    a_values[lane] = a_value(lane);
    b_values[lane] = b_value(lane);
  }
}

static void fill_output(void) {
  for (uint32_t i = 0;
       i < SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES +
               SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_GUARD;
       ++i) {
    out_values[i] = SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_SENTINEL;
    expected_values[i] = SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_SENTINEL;
  }
}

static void run_cpu_reference(void) {
  for (uint32_t lane = 0; lane < SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES; ++lane)
    expected_values[lane] = a_value(lane) + b_value(lane) +
                            SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_SPILL_ADJUST;
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] *
                      SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_CHECKSUM_SCALE);
  return checksum;
}

static void verify_results(int *mismatches, float *max_abs_diff) {
  *mismatches = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t i = 0; i < SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES; ++i) {
    float diff = out_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
      *max_abs_diff = abs_diff;
    if (abs_diff > SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_EPSILON) {
      if (*mismatches < 8)
        printk("ERROR: spill_reload_between_user_tmu_reads_ssavc4 lane=%d gpu=%f expected=%f diff=%f\n",
               (int)i, out_values[i], expected_values[i], diff);
      ++*mismatches;
    }
  }
}

static int verify_sentinel_region(void) {
  int mismatches = 0;
  for (uint32_t i = SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES;
       i < SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES +
               SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_GUARD;
       ++i) {
    if (out_values[i] != SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: spill_reload_between_user_tmu_reads_ssavc4 sentinel changed i=%d value=%f\n",
               (int)i, out_values[i]);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t a_dev = 0, b_dev = 0, out_dev = 0;
  const uint32_t out_count = SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES +
                             SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_GUARD;
  const uint32_t out_bytes = out_count * sizeof(float);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("spill_reload_between_user_tmu_reads_ssavc4 program create failed");
  if (vc4_m2_malloc(program, &a_dev, sizeof(a_values)) < 0 ||
      vc4_m2_malloc(program, &b_dev, sizeof(b_values)) < 0 ||
      vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
    panic("spill_reload_between_user_tmu_reads_ssavc4 allocation failed");

  fill_inputs();
  fill_output();
  run_cpu_reference();
  int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
  float max_abs_diff = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES, 1, 1);

  if (vc4_m2_copy_htod(program, a_dev, a_values, sizeof(a_values)) < 0 ||
      vc4_m2_copy_htod(program, b_dev, b_values, sizeof(b_values)) < 0 ||
      vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
      spill_reload_between_user_tmu_reads_ssavc4_launch(
          program, grid, block, a_dev, b_dev, out_dev,
          SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_BETA) < 0 ||
      vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
    printk("ERROR: spill_reload_between_user_tmu_reads_ssavc4 launch/copy failed\n");
    ++launch_failures;
  }

  verify_results(&total_mismatches, &max_abs_diff);
  sentinel_mismatches = verify_sentinel_region();
  int checksum = scaled_checksum(out_values, SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES);
  int expected_checksum = scaled_checksum(expected_values, SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES);
  if (checksum != expected_checksum) {
    printk("ERROR: spill_reload_between_user_tmu_reads_ssavc4 checksum mismatch gpu=%d expected=%d\n",
           checksum, expected_checksum);
    ++total_mismatches;
  }

  launch_failures += (int)spill_reload_between_user_tmu_reads_ssavc4_runtime_launch_failures();
  uint32_t launches = spill_reload_between_user_tmu_reads_ssavc4_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0 && launches == 1u)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=spill_reload_between_user_tmu_reads_ssavc4 status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d forced_spill_terms=48 checksum_accum=%d max_abs_diff=%f runtime_allocations=3 runtime_launches=%d elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)SPILL_RELOAD_BETWEEN_USER_TMU_READS_SSAVC4_LANES,
         checksum, max_abs_diff, launches, timer_get_usec() - start);

  vc4Free(program, a_dev);
  vc4Free(program, b_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
