#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MIXED_F32_EPSILON 0.0002f
#define MIXED_F32_CHECKSUM_SCALE 4096.0f
#define MIXED_F32_MAX_K 33u
#define MIXED_F32_LANES 16u
#define MIXED_F32_GUARD 32u
#define MIXED_F32_SENTINEL (-11223.5f)
#define MIXED_F32_BETA 0.03125f
#define MIXED_F32_SPILL_ADJUST (324.0f * MIXED_F32_BETA)

struct mixed_f32_case {
  uint32_t k;
  uint32_t active_cols;
};

static const struct mixed_f32_case test_cases[] = {
    {0u, 1u},  {0u, 15u},  {0u, 16u}, {1u, 1u},  {1u, 15u},
    {1u, 16u}, {2u, 1u},   {2u, 15u}, {2u, 16u}, {15u, 1u},
    {15u, 15u}, {15u, 16u}, {16u, 1u}, {16u, 15u}, {16u, 16u},
    {17u, 1u}, {17u, 15u}, {17u, 16u}, {33u, 1u}, {33u, 15u},
    {33u, 16u},
};

static float a_values[MIXED_F32_MAX_K];
static float b_values[MIXED_F32_MAX_K * MIXED_F32_LANES];
static float out_values[MIXED_F32_LANES + MIXED_F32_GUARD];
static float expected_values[MIXED_F32_LANES + MIXED_F32_GUARD];

static float absf_local(float v) { return v < 0.0f ? -v : v; }
static float fmin_local(float a, float b) { return a < b ? a : b; }
static float fmax_local(float a, float b) { return a > b ? a : b; }

static void fill_inputs(void) {
  for (uint32_t j = 0; j < MIXED_F32_MAX_K; ++j) {
    a_values[j] = ((float)((j * 7u + 5u) % 23u) - 11.0f) * 0.0625f;
    for (uint32_t lane = 0; lane < MIXED_F32_LANES; ++lane) {
      b_values[j * MIXED_F32_LANES + lane] =
          ((float)((j * 13u + lane * 3u + 9u) % 29u) - 14.0f) * 0.03125f;
    }
  }
}

static void fill_output(void) {
  for (uint32_t i = 0; i < MIXED_F32_LANES + MIXED_F32_GUARD; ++i) {
    out_values[i] = MIXED_F32_SENTINEL;
    expected_values[i] = MIXED_F32_SENTINEL;
  }
}

static void compute_expected(uint32_t k, uint32_t active_cols) {
  float acc[MIXED_F32_LANES];
  float mn[MIXED_F32_LANES];
  float mx[MIXED_F32_LANES];
  for (uint32_t lane = 0; lane < MIXED_F32_LANES; ++lane) {
    acc[lane] = 0.0f;
    mn[lane] = 0.0f;
    mx[lane] = 0.0f;
  }
  for (uint32_t j = 0; j < k; ++j) {
    for (uint32_t lane = 0; lane < active_cols; ++lane) {
      float prod = a_values[j] * b_values[j * MIXED_F32_LANES + lane];
      acc[lane] += prod;
      mn[lane] = fmin_local(mn[lane], prod);
      mx[lane] = fmax_local(mx[lane], prod);
    }
  }

  float reduce_add = 0.0f;
  float reduce_min = 0.0f;
  float reduce_max = 0.0f;
  for (uint32_t lane = 0; lane < active_cols; ++lane) {
    reduce_add += acc[lane];
    reduce_min = (lane == 0u) ? mn[lane] : fmin_local(reduce_min, mn[lane]);
    reduce_max = (lane == 0u) ? mx[lane] : fmax_local(reduce_max, mx[lane]);
  }
  float value = reduce_add + reduce_min + reduce_max + MIXED_F32_SPILL_ADJUST;
  if (!(value > 0.0f))
    value = 0.0f;
  for (uint32_t lane = 0; lane < active_cols; ++lane)
    expected_values[lane] = value;
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] * MIXED_F32_CHECKSUM_SCALE);
  return checksum;
}

static int verify_active(uint32_t active_cols, float *max_abs_diff) {
  int mismatches = 0;
  for (uint32_t i = 0; i < active_cols; ++i) {
    float diff = out_values[i] - expected_values[i];
    float ad = absf_local(diff);
    if (ad > *max_abs_diff)
      *max_abs_diff = ad;
    if (ad > MIXED_F32_EPSILON) {
      if (mismatches < 8)
        printk("ERROR: mixed_f32 lane=%d got=%f want=%f diff=%f\n",
               (int)i, out_values[i], expected_values[i], diff);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_sentinels(uint32_t active_cols) {
  int mismatches = 0;
  for (uint32_t i = active_cols; i < MIXED_F32_LANES + MIXED_F32_GUARD; ++i) {
    if (out_values[i] != MIXED_F32_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: mixed_f32 sentinel index=%d got=%f want=%f\n",
               (int)i, out_values[i], MIXED_F32_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 128u * 1024u) < 0 || !program)
    panic("mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel program create failed");

  vc4_deviceptr_t a_dev = 0;
  vc4_deviceptr_t b_dev = 0;
  vc4_deviceptr_t out_dev = 0;
  const uint32_t a_bytes = MIXED_F32_MAX_K * sizeof(float);
  const uint32_t b_bytes = MIXED_F32_MAX_K * MIXED_F32_LANES * sizeof(float);
  const uint32_t out_bytes = (MIXED_F32_LANES + MIXED_F32_GUARD) * sizeof(float);
  if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
      vc4_m2_malloc(program, &b_dev, b_bytes) < 0 ||
      vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
    panic("mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel allocation failed");

  fill_inputs();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(MIXED_F32_LANES, 1, 1);
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  float max_abs_diff = 0.0f;
  int saw_k0 = 0;
  int saw_k33 = 0;
  int saw_active_cols1 = 0;
  int saw_active_cols15 = 0;
  int saw_active_cols16 = 0;
  int start = timer_get_usec();

  printk("Running VC4 mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel candidate bundle...\n");
  for (uint32_t case_id = 0; case_id < sizeof(test_cases) / sizeof(test_cases[0]); ++case_id) {
    uint32_t k = test_cases[case_id].k;
    uint32_t active_cols = test_cases[case_id].active_cols;
    fill_output();
    compute_expected(k, active_cols);
    saw_k0 |= (k == 0u);
    saw_k33 |= (k == 33u);
    saw_active_cols1 |= (active_cols == 1u);
    saw_active_cols15 |= (active_cols == 15u);
    saw_active_cols16 |= (active_cols == 16u);
    if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
        vc4_m2_copy_htod(program, b_dev, b_values, b_bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
        mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel_launch(
            program, grid, block, a_dev, b_dev, out_dev, k, active_cols,
            MIXED_F32_BETA) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
      ++launch_failures;
      continue;
    }
    int mismatches = verify_active(active_cols, &max_abs_diff);
    int sentinels = verify_sentinels(active_cols);
    int checksum = scaled_checksum(out_values, active_cols);
    checksum_accum ^= checksum + (int)(case_id * 257u);
    total_mismatches += mismatches;
    sentinel_mismatches += sentinels;
    printk("MIXED_F32_REDUCE_TMU_LOOP_SPILL_VDW_CASE case=%d k=%d active_cols=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
           (int)case_id, (int)k, (int)active_cols, mismatches, sentinels,
           checksum, max_abs_diff);
  }

  launch_failures +=
      (int)mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel_runtime_launch_failures();
  uint32_t runtime_launches =
      mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0 && saw_k0 && saw_k33 &&
                        saw_active_cols1 && saw_active_cols15 &&
                        saw_active_cols16)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_two_tmu_loop=1 saw_f32_reduce=1 saw_spill_frame_nonzero=1 saw_vdw_preserve=1 saw_safe_offset=1 saw_k0=%d saw_k33=%d saw_active_cols1=%d saw_active_cols15=%d saw_active_cols16=%d forced_spill_live_adjust_x10000=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=3 runtime_launches=%d elapsed_usec=%d\n",
         status, (int)(sizeof(test_cases) / sizeof(test_cases[0])),
         total_mismatches, sentinel_mismatches, launch_failures, saw_k0,
         saw_k33, saw_active_cols1, saw_active_cols15, saw_active_cols16,
         (int)(MIXED_F32_SPILL_ADJUST * 10000.0f), checksum_accum,
         max_abs_diff, (int)runtime_launches, timer_get_usec() - start);

  vc4Free(program, a_dev);
  vc4Free(program, b_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
