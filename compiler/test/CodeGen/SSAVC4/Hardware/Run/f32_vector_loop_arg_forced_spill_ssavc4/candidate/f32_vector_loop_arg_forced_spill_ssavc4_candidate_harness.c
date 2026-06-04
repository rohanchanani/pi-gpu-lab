#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_EPSILON 0.0002f
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_CHECKSUM_SCALE 1024.0f
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_MAX_N 192u
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_GUARD 32u
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ACTIVE_QPUS 12u
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_LANE_WIDTH 16u
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ITERS 5u
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_PRESSURE_TERMS 20u
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ELEMENTS_PER_WAVE (F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ACTIVE_QPUS * F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_LANE_WIDTH)
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_MAX_WAVES ((F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_MAX_N + F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ELEMENTS_PER_WAVE - 1u) / F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ELEMENTS_PER_WAVE)
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_MAX_COVERAGE_N (F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_MAX_WAVES * F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ELEMENTS_PER_WAVE)
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_BUFFER_N (F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_MAX_COVERAGE_N + F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_GUARD)
#define F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_SENTINEL (-12345.0f)

struct f32_vector_loop_arg_forced_spill_ssavc4_case {
  uint32_t n;
  float alpha;
  float beta;
};

static float x_values[F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_BUFFER_N];
static float out_values[F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_BUFFER_N];
static float expected_values[F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_BUFFER_N];

static const struct f32_vector_loop_arg_forced_spill_ssavc4_case test_cases[] = {
  {0u, 0.0f, 0.0f},
  {1u, 0.25f, -0.5f},
  {17u, -0.125f, 1.0f},
  {65u, 0.5f, -1.25f},
  {192u, -0.75f, 2.0f}
};

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float input_value(uint32_t i) {
  return ((float)((i * 7u + 3u) % 101u) * 0.125f) - 4.0f;
}

static float expected_value(uint32_t i, float alpha, float beta) {
  float x = input_value(i);
  float acc = x;
  float carry = beta;
  for (uint32_t iter = 0; iter < F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ITERS; ++iter) {
    acc += carry;
    carry += alpha;
  }
  float total = acc + carry;
  float pressure = x;
  for (uint32_t term = 0; term < F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_PRESSURE_TERMS; ++term) {
    pressure += beta;
    total += pressure;
  }
  return total;
}

static void fill_buffers(void) {
  for (uint32_t i = 0; i < F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_BUFFER_N; ++i) {
    x_values[i] = input_value(i);
    out_values[i] = F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_SENTINEL;
    expected_values[i] = F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_SENTINEL;
  }
}

static void run_cpu_reference(uint32_t n, float alpha, float beta) {
  for (uint32_t i = 0; i < n; ++i)
    expected_values[i] = expected_value(i, alpha, beta);
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] * F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_CHECKSUM_SCALE);
  return checksum;
}

static void verify_results(uint32_t n, int *mismatches, float *max_abs_diff) {
  *mismatches = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t i = 0; i < n; ++i) {
    float diff = out_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
      *max_abs_diff = abs_diff;
    if (abs_diff > F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_EPSILON) {
      if (*mismatches < 8)
        printk("ERROR: f32_vector_loop_arg_forced_spill_ssavc4 i=%d gpu=%f expected=%f diff=%f\n",
               (int)i, out_values[i], expected_values[i], diff);
      ++*mismatches;
    }
  }
}

static int verify_sentinel_region(uint32_t n) {
  int mismatches = 0;
  for (uint32_t i = n; i < F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_BUFFER_N; ++i) {
    if (out_values[i] != F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: f32_vector_loop_arg_forced_spill_ssavc4 sentinel changed i=%d value=%f expected=%f\n",
               (int)i, out_values[i], F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vc4_program_create failed");

  const uint32_t active_qpus = F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ACTIVE_QPUS;
  const uint32_t lane_width = F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_LANE_WIDTH;
  const uint32_t case_count = sizeof(test_cases) / sizeof(test_cases[0]);
  const uint32_t bytes = F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_BUFFER_N * sizeof(float);
  vc4_dim3 block = vc4_m2_dim3(active_qpus * lane_width, 1, 1);

  vc4_deviceptr_t x_dev = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
      vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("f32_vector_loop_arg_forced_spill_ssavc4 device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  float max_abs_diff_overall = 0.0f;
  int start = timer_get_usec();

  printk("Running VC4 f32_vector_loop_arg_forced_spill_ssavc4 candidate bundle...\n");
  printk("F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_SETUP max_n=%d active_qpus=%d lanes=%d iters=%d pressure_terms=%d allocations=%d cases=%d\n",
         (int)F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_MAX_N, (int)active_qpus, (int)lane_width,
         (int)F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ITERS, (int)F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_PRESSURE_TERMS, 1,
         (int)case_count);

  for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
    uint32_t n = test_cases[case_index].n;
    float alpha = test_cases[case_index].alpha;
    float beta = test_cases[case_index].beta;
    uint32_t waves = n == 0u ? 0u : (n + F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ELEMENTS_PER_WAVE - 1u) / F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_ELEMENTS_PER_WAVE;
    vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
    fill_buffers();
    run_cpu_reference(n, alpha, beta);

    if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        f32_vector_loop_arg_forced_spill_ssavc4_launch(program, grid, block, x_dev, out_dev, alpha, beta, n) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
      printk("ERROR: f32_vector_loop_arg_forced_spill_ssavc4 launch/copy failed case=%d n=%d\n",
             (int)case_index, (int)n);
      ++launch_failures;
      continue;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(n, &mismatches, &max_abs_diff);
    int case_sentinel_mismatches = verify_sentinel_region(n);
    int checksum = scaled_checksum(out_values, n);
    int expected_checksum = scaled_checksum(expected_values, n);
    if (checksum != expected_checksum) {
      printk("ERROR: f32_vector_loop_arg_forced_spill_ssavc4 checksum mismatch n=%d gpu=%d expected=%d\n",
             (int)n, checksum, expected_checksum);
      ++mismatches;
    }
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinel_mismatches;
    checksum_accum += checksum;
    printk("F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_CASE case=%d n=%d alpha=%f beta=%f mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f waves=%d\n",
           (int)case_index, (int)n, alpha, beta, mismatches,
           case_sentinel_mismatches, checksum, max_abs_diff, (int)waves);
  }

  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=f32_vector_loop_arg_forced_spill_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, (int)active_qpus, (int)lane_width, (int)F32_VECTOR_LOOP_ARG_FORCED_SPILL_SSAVC4_MAX_N,
         checksum_accum, max_abs_diff_overall, 1, (int)case_count, elapsed);

  vc4Free(program, x_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
