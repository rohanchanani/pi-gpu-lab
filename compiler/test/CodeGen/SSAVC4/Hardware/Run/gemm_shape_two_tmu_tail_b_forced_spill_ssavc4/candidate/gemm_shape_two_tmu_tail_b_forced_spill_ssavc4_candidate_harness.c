#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_EPSILON 0.0002f
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_CHECKSUM_SCALE 4096.0f
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_REQUESTS 12u
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_WAVES 1u
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_MAX_K 17u
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_ACTIVE_COLS 15u
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES 16u
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_GUARD 32u
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_SENTINEL (-12345.0f)
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_BETA 0.03125f
#define GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_SPILL_ADJUST \
  (324.0f * GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_BETA)

static float a_values[GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_MAX_K];
static float b_values[GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_MAX_K *
                      GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES];
static float out_values[GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_REQUESTS *
                            GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES +
                        GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_GUARD];
static float expected_values[GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_REQUESTS *
                                 GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES +
                             GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_GUARD];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float a_value(uint32_t j) {
  return ((float)((j * 5u + 3u) % 19u) - 9.0f) * 0.0625f;
}

static float b_value(uint32_t j, uint32_t lane) {
  return ((float)((j * 11u + lane * 7u + 1u) % 29u) - 14.0f) * 0.03125f;
}

static void fill_inputs(void) {
  for (uint32_t j = 0; j < GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_MAX_K; ++j) {
    a_values[j] = a_value(j);
    for (uint32_t lane = 0; lane < GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES; ++lane)
      b_values[j * GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES + lane] =
          b_value(j, lane);
  }
}

static void fill_output(void) {
  const uint32_t total_count =
      GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_REQUESTS *
          GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES +
      GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_GUARD;
  for (uint32_t i = 0; i < total_count; ++i) {
    out_values[i] = GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_SENTINEL;
    expected_values[i] = GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_SENTINEL;
  }
}

static void run_cpu_reference(uint32_t k) {
  for (uint32_t req = 0; req < GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_REQUESTS; ++req) {
    for (uint32_t lane = 0; lane < GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES; ++lane) {
      float acc = 0.0f;
      for (uint32_t j = 0; j < k; ++j)
        acc += a_value(j) *
               (lane < GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_ACTIVE_COLS
                    ? b_value(j, lane)
                    : 0.0f);
      expected_values[req * GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES + lane] =
          acc + GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_SPILL_ADJUST;
    }
  }
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] * GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_CHECKSUM_SCALE);
  return checksum;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t a_dev = 0, b_dev = 0, out_dev = 0;
  const uint32_t active_count =
      GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_REQUESTS *
      GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES;
  const uint32_t total_count =
      active_count + GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_GUARD;
  const uint32_t out_bytes = total_count * sizeof(float);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("gemm_shape_two_tmu_tail_b_forced_spill_ssavc4 program create failed");
  if (vc4_m2_malloc(program, &a_dev, sizeof(a_values)) < 0 ||
      vc4_m2_malloc(program, &b_dev, sizeof(b_values)) < 0 ||
      vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
    panic("gemm_shape_two_tmu_tail_b_forced_spill_ssavc4 allocation failed");

  fill_inputs();
  fill_output();
  run_cpu_reference(GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_MAX_K);

  int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
  float max_abs_diff = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(active_count, 1, 1);

  if (vc4_m2_copy_htod(program, a_dev, a_values, sizeof(a_values)) < 0 ||
      vc4_m2_copy_htod(program, b_dev, b_values, sizeof(b_values)) < 0 ||
      vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
      gemm_shape_two_tmu_tail_b_forced_spill_ssavc4_launch(
          program, grid, block, a_dev, b_dev, out_dev,
          GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_MAX_K,
          GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_BETA,
          GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_ACTIVE_COLS) < 0 ||
      vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
    printk("ERROR: gemm_shape_two_tmu_tail_b_forced_spill_ssavc4 launch/copy failed\n");
    ++launch_failures;
  } else {
    for (uint32_t i = 0; i < active_count; ++i) {
      float diff = out_values[i] - expected_values[i];
      float abs_diff = absf_local(diff);
      if (abs_diff > max_abs_diff)
        max_abs_diff = abs_diff;
      if (abs_diff > GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_EPSILON) {
        if (total_mismatches < 8)
          printk("ERROR: gemm_shape_two_tmu_tail_b_forced_spill_ssavc4 value i=%d gpu=%f expected=%f diff=%f\n",
                 (int)i, out_values[i], expected_values[i], diff);
        ++total_mismatches;
      }
    }
    for (uint32_t i = active_count; i < total_count; ++i) {
      if (out_values[i] != GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_SENTINEL) {
        if (sentinel_mismatches < 8)
          printk("ERROR: gemm_shape_two_tmu_tail_b_forced_spill_ssavc4 sentinel changed i=%d value=%f\n",
                 (int)i, out_values[i]);
        ++sentinel_mismatches;
      }
    }
  }

  int checksum = scaled_checksum(out_values, active_count);
  int expected_checksum = scaled_checksum(expected_values, active_count);
  if (checksum != expected_checksum) {
    printk("ERROR: gemm_shape_two_tmu_tail_b_forced_spill_ssavc4 checksum mismatch gpu=%d expected=%d\n",
           checksum, expected_checksum);
    ++total_mismatches;
  }
  launch_failures +=
      (int)gemm_shape_two_tmu_tail_b_forced_spill_ssavc4_runtime_launch_failures();
  uint32_t launches = gemm_shape_two_tmu_tail_b_forced_spill_ssavc4_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0 && launches == 1)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=gemm_shape_two_tmu_tail_b_forced_spill_ssavc4 status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d requests=%d waves=%d resident_blocks=12 lanes=%d k=%d active_cols=%d forced_spill_live_adjust=%d checksum=%d max_abs_diff=%f runtime_allocations=3 runtime_launches=%d elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_REQUESTS,
         (int)GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_WAVES,
         (int)GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_LANES,
         (int)GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_MAX_K,
         (int)GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_ACTIVE_COLS,
         (int)GEMM_SHAPE_TWO_TMU_TAIL_B_FORCED_SPILL_SSAVC4_SPILL_ADJUST,
         checksum, max_abs_diff, launches, timer_get_usec() - start);

  vc4Free(program, a_dev);
  vc4Free(program, b_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
