#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_EPSILON 0.0002f
#define TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_CHECKSUM_SCALE 4096.0f
#define TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_MAX_K 17u
#define TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES 16u
#define TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_GUARD 32u
#define TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_SENTINEL (-12345.0f)

static const uint32_t k_cases[] = {0u, 1u, 2u, 15u, 16u, 17u};
static float a_values[TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_MAX_K];
static float b_values[TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_MAX_K *
                      TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES];
static float out_values[TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES +
                        TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_GUARD];
static float expected_values[TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES +
                             TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_GUARD];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float a_value(uint32_t j) {
  return ((float)((j * 5u + 3u) % 19u) - 9.0f) * 0.0625f;
}

static float b_value(uint32_t j, uint32_t lane) {
  return ((float)((j * 11u + lane * 7u + 1u) % 29u) - 14.0f) * 0.03125f;
}

static void fill_inputs(void) {
  for (uint32_t j = 0; j < TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_MAX_K; ++j) {
    a_values[j] = a_value(j);
    for (uint32_t lane = 0; lane < TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES; ++lane)
      b_values[j * TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES + lane] =
          b_value(j, lane);
  }
}

static void fill_output(void) {
  for (uint32_t i = 0;
       i < TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES +
               TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_GUARD;
       ++i) {
    out_values[i] = TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_SENTINEL;
    expected_values[i] = TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_SENTINEL;
  }
}

static void run_cpu_reference(uint32_t k) {
  for (uint32_t lane = 0; lane < TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES; ++lane) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j)
      acc += a_value(j) * b_value(j, lane);
    expected_values[lane] = acc;
  }
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] * TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_CHECKSUM_SCALE);
  return checksum;
}

static void verify_results(int *mismatches, float *max_abs_diff) {
  *mismatches = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t i = 0; i < TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES; ++i) {
    float diff = out_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
      *max_abs_diff = abs_diff;
    if (abs_diff > TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_EPSILON) {
      if (*mismatches < 8)
        printk("ERROR: two_tmu_f32_loop_no_spill_ssavc4 lane=%d gpu=%f expected=%f diff=%f\n",
               (int)i, out_values[i], expected_values[i], diff);
      ++*mismatches;
    }
  }
}

static int verify_sentinel_region(void) {
  int mismatches = 0;
  for (uint32_t i = TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES;
       i < TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES +
               TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_GUARD;
       ++i) {
    if (out_values[i] != TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: two_tmu_f32_loop_no_spill_ssavc4 sentinel changed i=%d value=%f\n",
               (int)i, out_values[i]);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t a_dev = 0, b_dev = 0, out_dev = 0;
  const uint32_t case_count = sizeof(k_cases) / sizeof(k_cases[0]);
  const uint32_t out_count = TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES +
                             TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_GUARD;
  const uint32_t out_bytes = out_count * sizeof(float);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("two_tmu_f32_loop_no_spill_ssavc4 program create failed");
  if (vc4_m2_malloc(program, &a_dev, sizeof(a_values)) < 0 ||
      vc4_m2_malloc(program, &b_dev, sizeof(b_values)) < 0 ||
      vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
    panic("two_tmu_f32_loop_no_spill_ssavc4 allocation failed");

  fill_inputs();
  int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
  int checksum_accum = 0, saw_k0 = 0, saw_k17 = 0;
  float max_abs_diff_overall = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES, 1, 1);

  for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
    uint32_t k = k_cases[case_index];
    saw_k0 |= (k == 0u);
    saw_k17 |= (k == 17u);
    fill_output();
    run_cpu_reference(k);
    if (vc4_m2_copy_htod(program, a_dev, a_values, sizeof(a_values)) < 0 ||
        vc4_m2_copy_htod(program, b_dev, b_values, sizeof(b_values)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
        two_tmu_f32_loop_no_spill_ssavc4_launch(program, grid, block, a_dev, b_dev,
                                                out_dev, k) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
      printk("ERROR: two_tmu_f32_loop_no_spill_ssavc4 launch/copy failed case=%d k=%d\n",
             (int)case_index, (int)k);
      ++launch_failures;
      continue;
    }
    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(&mismatches, &max_abs_diff);
    int case_sentinel_mismatches = verify_sentinel_region();
    int checksum = scaled_checksum(out_values, TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES);
    int expected_checksum = scaled_checksum(expected_values, TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES);
    if (checksum != expected_checksum) {
      printk("ERROR: two_tmu_f32_loop_no_spill_ssavc4 checksum mismatch k=%d gpu=%d expected=%d\n",
             (int)k, checksum, expected_checksum);
      ++mismatches;
    }
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinel_mismatches;
    checksum_accum += checksum;
    printk("TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_CASE case=%d k=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
           (int)case_index, (int)k, mismatches, case_sentinel_mismatches,
           checksum, max_abs_diff);
  }

  launch_failures += (int)two_tmu_f32_loop_no_spill_ssavc4_runtime_launch_failures();
  uint32_t launches = two_tmu_f32_loop_no_spill_ssavc4_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=two_tmu_f32_loop_no_spill_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_k0=%d saw_k17=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=3 runtime_launches=%d elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, (int)TWO_TMU_F32_LOOP_NO_SPILL_SSAVC4_LANES,
         saw_k0, saw_k17, checksum_accum, max_abs_diff_overall, launches,
         timer_get_usec() - start);

  vc4Free(program, a_dev);
  vc4Free(program, b_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
