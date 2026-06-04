#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GEMV_NAIVE_VC4KERNEL_EPSILON 0.0001f
#define GEMV_NAIVE_VC4KERNEL_CHECKSUM_SCALE 1024.0f
#define GEMV_NAIVE_VC4KERNEL_MAX_M 65u
#define GEMV_NAIVE_VC4KERNEL_MAX_N 65u
#define GEMV_NAIVE_VC4KERNEL_MAX_LDA 72u
#define GEMV_NAIVE_VC4KERNEL_ACTIVE_QPUS 12u
#define GEMV_NAIVE_VC4KERNEL_LANES 16u
#define GEMV_NAIVE_VC4KERNEL_MAX_ROW_COVERAGE                                  \
  (((GEMV_NAIVE_VC4KERNEL_MAX_M + GEMV_NAIVE_VC4KERNEL_LANES - 1u) /          \
    GEMV_NAIVE_VC4KERNEL_LANES) *                                              \
   GEMV_NAIVE_VC4KERNEL_LANES)
#define GEMV_NAIVE_VC4KERNEL_GUARD 32u
#define GEMV_NAIVE_VC4KERNEL_A_WORDS                                            \
  (GEMV_NAIVE_VC4KERNEL_MAX_ROW_COVERAGE * GEMV_NAIVE_VC4KERNEL_MAX_LDA)
#define GEMV_NAIVE_VC4KERNEL_Y_WORDS                                            \
  (GEMV_NAIVE_VC4KERNEL_MAX_M + GEMV_NAIVE_VC4KERNEL_GUARD)
#define GEMV_NAIVE_VC4KERNEL_SENTINEL (-4321.25f)

struct gemv_naive_vc4kernel_case {
  uint32_t m;
  uint32_t n;
  uint32_t lda;
};

static const struct gemv_naive_vc4kernel_case test_cases[] = {
    {0u, 0u, 1u},    {1u, 1u, 1u},    {15u, 15u, 16u},
    {16u, 16u, 23u}, {17u, 17u, 17u}, {31u, 31u, 32u},
    {32u, 32u, 39u}, {33u, 33u, 33u}, {65u, 65u, 66u},
    {1u, 65u, 72u},  {65u, 17u, 24u}, {7u, 65u, 65u},
    {33u, 1u, 2u},
};

static float a_values[GEMV_NAIVE_VC4KERNEL_A_WORDS];
static float x_values[GEMV_NAIVE_VC4KERNEL_MAX_N];
static float y_values[GEMV_NAIVE_VC4KERNEL_Y_WORDS];
static float expected_values[GEMV_NAIVE_VC4KERNEL_Y_WORDS];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static uint32_t rounded_waves(uint32_t m) {
  uint32_t row_blocks = (m + GEMV_NAIVE_VC4KERNEL_LANES - 1u) /
                        GEMV_NAIVE_VC4KERNEL_LANES;
  uint32_t waves = (row_blocks + GEMV_NAIVE_VC4KERNEL_ACTIVE_QPUS - 1u) /
                   GEMV_NAIVE_VC4KERNEL_ACTIVE_QPUS;
  return waves == 0u ? 1u : waves;
}

static void fill_inputs(uint32_t m, uint32_t n, uint32_t lda) {
  for (uint32_t i = 0; i < GEMV_NAIVE_VC4KERNEL_A_WORDS; i++)
    a_values[i] = 99.0f;
  for (uint32_t row = 0; row < GEMV_NAIVE_VC4KERNEL_MAX_ROW_COVERAGE; row++) {
    for (uint32_t col = 0; col < lda; col++) {
      uint32_t idx = row * lda + col;
      a_values[idx] = ((float)((row * 17u + col * 5u + 13u) % 41u) - 20.0f) *
                      0.03125f;
    }
  }
  for (uint32_t i = 0; i < GEMV_NAIVE_VC4KERNEL_MAX_N; i++)
    x_values[i] = ((float)((i * 11u + 7u) % 29u) - 14.0f) * 0.0625f;
  for (uint32_t i = 0; i < GEMV_NAIVE_VC4KERNEL_Y_WORDS; i++) {
    y_values[i] = GEMV_NAIVE_VC4KERNEL_SENTINEL;
    expected_values[i] = GEMV_NAIVE_VC4KERNEL_SENTINEL;
  }
  for (uint32_t row = 0; row < m; row++) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < n; col++)
      sum += a_values[row * lda + col] * x_values[col];
    expected_values[row] = sum;
  }
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; i++)
    checksum += (int)(values[i] * GEMV_NAIVE_VC4KERNEL_CHECKSUM_SCALE);
  return checksum;
}

static void verify_results(uint32_t m, int *mismatch_count,
                           float *max_abs_diff) {
  *mismatch_count = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t row = 0; row < m; row++) {
    float diff = y_values[row] - expected_values[row];
    float ad = absf_local(diff);
    if (ad > *max_abs_diff)
      *max_abs_diff = ad;
    if (ad > GEMV_NAIVE_VC4KERNEL_EPSILON) {
      if (*mismatch_count < 8)
        printk("ERROR: gemv_naive_vc4kernel row=%d gpu=%f cpu=%f diff=%f\n",
               (int)row, y_values[row], expected_values[row], diff);
      (*mismatch_count)++;
    }
  }
}

static int verify_sentinel_region(uint32_t m) {
  int mismatches = 0;
  for (uint32_t i = m; i < GEMV_NAIVE_VC4KERNEL_Y_WORDS; i++) {
    if (y_values[i] != GEMV_NAIVE_VC4KERNEL_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: gemv_naive_vc4kernel sentinel changed row=%d value=%f expected=%f\n",
               (int)i, y_values[i], GEMV_NAIVE_VC4KERNEL_SENTINEL);
      mismatches++;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vc4_program_create failed");

  uint32_t a_bytes = GEMV_NAIVE_VC4KERNEL_A_WORDS * sizeof(float);
  uint32_t x_bytes = GEMV_NAIVE_VC4KERNEL_MAX_N * sizeof(float);
  uint32_t y_bytes = GEMV_NAIVE_VC4KERNEL_Y_WORDS * sizeof(float);
  vc4_deviceptr_t a_dev = 0;
  vc4_deviceptr_t x_dev = 0;
  vc4_deviceptr_t y_dev = 0;
  if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
      vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
      vc4_m2_malloc(program, &y_dev, y_bytes) < 0)
    panic("gemv_naive_vc4kernel device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  float max_abs_diff_overall = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 block = vc4_m2_dim3(GEMV_NAIVE_VC4KERNEL_ACTIVE_QPUS *
                                   GEMV_NAIVE_VC4KERNEL_LANES,
                               1, 1);

  printk("Running VC4 gemv_naive_vc4kernel candidate bundle...\n");
  for (uint32_t case_id = 0; case_id < sizeof(test_cases) / sizeof(test_cases[0]);
       case_id++) {
    uint32_t m = test_cases[case_id].m;
    uint32_t n = test_cases[case_id].n;
    uint32_t lda = test_cases[case_id].lda;
    uint32_t waves = rounded_waves(m);
    vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
    fill_inputs(m, n, lda);
    if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
        vc4_m2_copy_htod(program, x_dev, x_values, x_bytes) < 0 ||
        vc4_m2_copy_htod(program, y_dev, y_values, y_bytes) < 0 ||
        gemv_naive_vc4kernel_launch(program, grid, block, a_dev, x_dev, y_dev,
                                    m, n, lda) < 0 ||
        vc4_m2_copy_dtoh(program, y_values, y_dev, y_bytes) < 0) {
      printk("ERROR: gemv_naive_vc4kernel launch/copy failed case=%d m=%d n=%d lda=%d\n",
             (int)case_id, (int)m, (int)n, (int)lda);
      launch_failures++;
      continue;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(m, &mismatches, &max_abs_diff);
    int case_sentinels = verify_sentinel_region(m);
    int checksum = scaled_checksum(y_values, m);
    int expected_checksum = scaled_checksum(expected_values, m);
    if (checksum != expected_checksum) {
      printk("ERROR: gemv_naive_vc4kernel checksum mismatch case=%d gpu=%d cpu=%d\n",
             (int)case_id, checksum, expected_checksum);
      mismatches++;
    }
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinels;
    checksum_accum += checksum;
    printk("GEMV_NAIVE_VC4KERNEL_CASE case=%d m=%d n=%d lda=%d waves=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
           (int)case_id, (int)m, (int)n, (int)lda, (int)waves, mismatches,
           case_sentinels, checksum, max_abs_diff);
  }

  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=gemv_naive_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_m=%d max_n=%d max_lda=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
         status, (int)(sizeof(test_cases) / sizeof(test_cases[0])),
         total_mismatches, sentinel_mismatches, launch_failures,
         GEMV_NAIVE_VC4KERNEL_ACTIVE_QPUS, GEMV_NAIVE_VC4KERNEL_LANES,
         GEMV_NAIVE_VC4KERNEL_MAX_M, GEMV_NAIVE_VC4KERNEL_MAX_N,
         GEMV_NAIVE_VC4KERNEL_MAX_LDA, checksum_accum, max_abs_diff_overall, 3,
         (int)(sizeof(test_cases) / sizeof(test_cases[0])), elapsed);

  vc4Free(program, a_dev);
  vc4Free(program, x_dev);
  vc4Free(program, y_dev);
  vc4_program_destroy(program);
}
