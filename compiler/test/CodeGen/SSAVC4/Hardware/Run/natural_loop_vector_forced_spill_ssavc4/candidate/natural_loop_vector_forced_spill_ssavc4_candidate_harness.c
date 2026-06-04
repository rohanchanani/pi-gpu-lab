#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_EPSILON 0.001f
#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_CHECKSUM_SCALE 1024.0f
#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_MAX_N 80u
#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_GUARD 32u
#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_BUFFER_N \
  (NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_MAX_N + NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_GUARD)
#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_SENTINEL (-12345.0f)
#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_ACTIVE_QPUS 5u
#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_LANE_WIDTH 16u
#define NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_PRESSURE_TERMS 20u

static float out_values[NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_BUFFER_N];
static float expected_values[NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_BUFFER_N];
static const uint32_t test_trips[] = {0u, 1u, 2u, 15u, 16u, 17u, 65u};

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float seed_for_case(uint32_t case_id) {
  return (float)case_id * 0.125f;
}

static float alpha_for_case(uint32_t case_id) {
  return (float)((case_id % 3u) + 1u) * 0.0625f;
}

static float beta_for_case(uint32_t case_id) {
  return ((float)case_id - 4.0f) * 0.125f;
}

static float expected_value(uint32_t case_id, uint32_t trips) {
  float seed = seed_for_case(case_id);
  float alpha = alpha_for_case(case_id);
  float beta = beta_for_case(case_id);
  float acc = seed;
  float carry = beta;
  for (uint32_t iter = 0; iter < trips; ++iter) {
    acc += carry;
    carry += alpha;
  }
  float total = acc + carry;
  for (uint32_t term = 1; term <= NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_PRESSURE_TERMS; ++term)
    total += seed + (float)term * beta;
  return total;
}

static void fill_buffers(void) {
  for (uint32_t i = 0; i < NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_BUFFER_N; ++i) {
    out_values[i] = NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_SENTINEL;
    expected_values[i] = NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_SENTINEL;
  }
}

static void run_cpu_reference(uint32_t case_id, uint32_t n, uint32_t trips) {
  float value = expected_value(case_id, trips);
  for (uint32_t i = 0; i < n; ++i)
    expected_values[i] = value;
}

static void verify_results(uint32_t n, int *mismatches, float *max_abs_diff) {
  *mismatches = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t i = 0; i < n; ++i) {
    float diff = out_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
      *max_abs_diff = abs_diff;
    if (abs_diff > NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_EPSILON) {
      if (*mismatches < 8)
        printk("ERROR: natural_loop_vector_forced_spill_ssavc4 i=%d gpu=%f expected=%f diff=%f\n",
               (int)i, out_values[i], expected_values[i], diff);
      ++*mismatches;
    }
  }
}

static int verify_sentinel_tail(uint32_t n) {
  int mismatches = 0;
  for (uint32_t i = n; i < NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_BUFFER_N; ++i) {
    if (out_values[i] != NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: natural_loop_vector_forced_spill_ssavc4 sentinel changed i=%d value=%f expected=%f\n",
               (int)i, out_values[i], NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; ++i)
    checksum += (int)(values[i] * NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_CHECKSUM_SCALE);
  return checksum;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vc4_program_create failed");

  const uint32_t active_qpus = NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_ACTIVE_QPUS;
  const uint32_t lane_width = NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_LANE_WIDTH;
  const uint32_t case_count = sizeof(test_trips) / sizeof(test_trips[0]);
  const uint32_t bytes = NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_BUFFER_N * sizeof(float);
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(active_qpus * lane_width, 1, 1);

  vc4_deviceptr_t out_dev = 0;
  if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
    panic("natural_loop_vector_forced_spill_ssavc4 device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  float max_abs_diff_overall = 0.0f;
  int start = timer_get_usec();

  printk("Running VC4 natural_loop_vector_forced_spill_ssavc4 candidate bundle...\n");
  printk("NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_SETUP max_n=%d active_qpus=%d lanes=%d pressure_terms=%d allocations=%d cases=%d\n",
         (int)NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_MAX_N, (int)active_qpus,
         (int)lane_width, (int)NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_PRESSURE_TERMS,
         1, (int)case_count);

  for (uint32_t case_index = 0; case_index < case_count; ++case_index) {
    uint32_t n = test_trips[case_index];
    uint32_t trips = test_trips[case_index];
    uint32_t case_id = case_index + 1u;
    float seed = seed_for_case(case_id);
    float alpha = alpha_for_case(case_id);
    float beta = beta_for_case(case_id);
    fill_buffers();
    run_cpu_reference(case_id, n, trips);

    if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
        natural_loop_vector_forced_spill_ssavc4_launch(program, grid, block,
                                                       out_dev, n, trips, seed,
                                                       alpha, beta, case_id) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
      printk("ERROR: natural_loop_vector_forced_spill_ssavc4 launch/copy failed case=%d n=%d trips=%d\n",
             (int)case_id, (int)n, (int)trips);
      ++launch_failures;
      continue;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(n, &mismatches, &max_abs_diff);
    int case_sentinel_mismatches = verify_sentinel_tail(n);
    int checksum = scaled_checksum(out_values, n);
    int expected_checksum = scaled_checksum(expected_values, n);
    if (checksum != expected_checksum) {
      printk("ERROR: natural_loop_vector_forced_spill_ssavc4 checksum mismatch n=%d gpu=%d expected=%d\n",
             (int)n, checksum, expected_checksum);
      ++mismatches;
    }
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinel_mismatches;
    checksum_accum += checksum;
    printk("NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_CASE case=%d n=%d trips=%d seed=%f alpha=%f beta=%f mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
           (int)case_id, (int)n, (int)trips, seed, alpha, beta, mismatches,
           case_sentinel_mismatches, checksum, max_abs_diff);
  }

  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=natural_loop_vector_forced_spill_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, (int)active_qpus, (int)lane_width,
         (int)NATURAL_LOOP_VECTOR_FORCED_SPILL_SSAVC4_MAX_N, checksum_accum,
         max_abs_diff_overall, 1, (int)case_count, elapsed);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
