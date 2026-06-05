#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_EPSILON 0.0001f
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_CHECKSUM_SCALE 1024.0f
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_M 33u
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_N 33u
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_K 33u
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDA 96u
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDB 96u
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDC 96u
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_LANES 16u
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_GUARD_ROWS 2u
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_A_WORDS \
  (GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_M * GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDA)
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_B_WORDS \
  (GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_K * GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDB)
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_C_ROWS \
  (GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_M + GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_GUARD_ROWS)
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_C_WORDS \
  (GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_C_ROWS * GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDC)
#define GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_SENTINEL (-7654.25f)

struct gemm_shape_vc4kernel_micro_gemm_vc4kernel_case {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t lda;
  uint32_t ldb;
  uint32_t ldc;
};

static const struct gemm_shape_vc4kernel_micro_gemm_vc4kernel_case test_cases[] = {
    {1u, 1u, 1u, 1u, 1u, 1u},
    {15u, 15u, 15u, 15u, 15u, 15u},
};

static float a_values[GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_A_WORDS];
static float b_values[GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_B_WORDS];
static float c_values[GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_C_WORDS];
static float expected_values[GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_C_WORDS];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static uint32_t rounded_col_blocks(uint32_t n) {
  uint32_t blocks = (n + GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_LANES - 1u) /
                    GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_LANES;
  return blocks == 0u ? 1u : blocks;
}

static uint32_t rounded_rows(uint32_t m) { return m == 0u ? 1u : m; }

static void fill_inputs(uint32_t m, uint32_t n, uint32_t k, uint32_t lda,
                        uint32_t ldb, uint32_t ldc) {
  for (uint32_t i = 0; i < GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_A_WORDS; i++)
    a_values[i] = 77.0f;
  for (uint32_t i = 0; i < GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_B_WORDS; i++)
    b_values[i] = -55.0f;
  for (uint32_t i = 0; i < GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_C_WORDS; i++) {
    c_values[i] = GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_SENTINEL;
    expected_values[i] = GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_SENTINEL;
  }

  for (uint32_t row = 0; row < m; row++) {
    for (uint32_t col = 0; col < k; col++) {
      a_values[row * lda + col] =
          ((float)((row * 17u + col * 5u + 3u) % 43u) - 21.0f) * 0.03125f;
    }
  }
  for (uint32_t row = 0; row < k; row++) {
    for (uint32_t col = 0; col < n; col++) {
      b_values[row * ldb + col] =
          ((float)((row * 7u + col * 11u + 9u) % 37u) - 18.0f) * 0.0625f;
    }
  }
  for (uint32_t row = 0; row < m; row++) {
    for (uint32_t col = 0; col < n; col++) {
      float sum = 0.0f;
      for (uint32_t kk = 0; kk < k; kk++)
        sum += a_values[row * lda + kk] * b_values[kk * ldb + col];
      expected_values[row * ldc + col] = sum;
      c_values[row * ldc + col] = 123.0f;
    }
  }
}

static int scaled_checksum(uint32_t m, uint32_t n, uint32_t ldc,
                           const float *values) {
  int checksum = 0;
  for (uint32_t row = 0; row < m; row++)
    for (uint32_t col = 0; col < n; col++)
      checksum +=
          (int)(values[row * ldc + col] * GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_CHECKSUM_SCALE);
  return checksum;
}

static void verify_results(uint32_t m, uint32_t n, uint32_t ldc,
                           int *mismatch_count, float *max_abs_diff) {
  *mismatch_count = 0;
  *max_abs_diff = 0.0f;
  for (uint32_t row = 0; row < m; row++) {
    for (uint32_t col = 0; col < n; col++) {
      uint32_t idx = row * ldc + col;
      float diff = c_values[idx] - expected_values[idx];
      float ad = absf_local(diff);
      if (ad > *max_abs_diff)
        *max_abs_diff = ad;
      if (ad > GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_EPSILON) {
        if (*mismatch_count < 8)
          printk("ERROR: gemm_shape_vc4kernel_micro_gemm_vc4kernel row=%d col=%d gpu=%f cpu=%f diff=%f\n",
                 (int)row, (int)col, c_values[idx], expected_values[idx],
                 diff);
        (*mismatch_count)++;
      }
    }
  }
}

static int verify_sentinels(uint32_t m, uint32_t n, uint32_t ldc) {
  int mismatches = 0;
  for (uint32_t row = 0; row < GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_C_ROWS; row++) {
    for (uint32_t col = 0; col < GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDC; col++) {
      uint32_t active = row < m && col < n && col < ldc;
      uint32_t allocated = col < ldc;
      uint32_t idx = row * GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDC + col;
      if (!allocated)
        continue;
      idx = row * ldc + col;
      if (!active && c_values[idx] != GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_SENTINEL) {
        if (mismatches < 8)
          printk("ERROR: gemm_shape_vc4kernel_micro_gemm_vc4kernel sentinel row=%d col=%d value=%f expected=%f\n",
                 (int)row, (int)col, c_values[idx],
                 GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_SENTINEL);
        mismatches++;
      }
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vc4_program_create failed");

  uint32_t a_bytes = GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_A_WORDS * sizeof(float);
  uint32_t b_bytes = GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_B_WORDS * sizeof(float);
  uint32_t c_bytes = GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_C_WORDS * sizeof(float);
  vc4_deviceptr_t a_dev = 0;
  vc4_deviceptr_t b_dev = 0;
  vc4_deviceptr_t c_dev = 0;
  if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
      vc4_m2_malloc(program, &b_dev, b_bytes) < 0 ||
      vc4_m2_malloc(program, &c_dev, c_bytes) < 0)
    panic("gemm_shape_vc4kernel_micro_gemm_vc4kernel device allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  float max_abs_diff_overall = 0.0f;
  int saw_zero = 0;
  int saw_non_square = 0;
  int saw_k_diff = 0;
  int saw_tail_n = 0;
  int saw_runtime_ld = 0;
  int saw_contiguous_ld = 0;
  int saw_plus1_ld = 0;
  int saw_plus7_ld = 0;
  int saw_large_ld = 0;
  int saw_old_timeout_shape = 0;
  int start = timer_get_usec();
  vc4_dim3 block = vc4_m2_dim3(GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_LANES, 1, 1);

  printk("Running VC4 gemm_shape_vc4kernel_micro_gemm_vc4kernel candidate bundle...\n");
  for (uint32_t case_id = 0; case_id < sizeof(test_cases) / sizeof(test_cases[0]);
       case_id++) {
    uint32_t m = test_cases[case_id].m;
    uint32_t n = test_cases[case_id].n;
    uint32_t k = test_cases[case_id].k;
    uint32_t lda = test_cases[case_id].lda;
    uint32_t ldb = test_cases[case_id].ldb;
    uint32_t ldc = test_cases[case_id].ldc;
    uint32_t grid_x = rounded_col_blocks(n);
    uint32_t grid_y = rounded_rows(m);
    vc4_dim3 grid = vc4_m2_dim3(grid_x, grid_y, 1);

    fill_inputs(m, n, k, lda, ldb, ldc);
    if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
        vc4_m2_copy_htod(program, b_dev, b_values, b_bytes) < 0 ||
        vc4_m2_copy_htod(program, c_dev, c_values, c_bytes) < 0) {
      printk("ERROR: gemm_shape_vc4kernel_micro_gemm_vc4kernel copy-in failed case=%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d\n",
             (int)case_id, (int)m, (int)n, (int)k, (int)lda, (int)ldb,
             (int)ldc);
      launch_failures++;
      continue;
    }

    if (gemm_shape_vc4kernel_micro_gemm_vc4kernel_launch(program, grid, block, a_dev, b_dev, c_dev,
                                    m, n, k, lda, ldb, ldc) < 0 ||
        vc4_m2_copy_dtoh(program, c_values, c_dev, c_bytes) < 0) {
      printk("ERROR: gemm_shape_vc4kernel_micro_gemm_vc4kernel launch/copy failed case=%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d\n",
             (int)case_id, (int)m, (int)n, (int)k, (int)lda, (int)ldb,
             (int)ldc);
      launch_failures++;
      continue;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(m, n, ldc, &mismatches, &max_abs_diff);
    int case_sentinels = verify_sentinels(m, n, ldc);
    int checksum = scaled_checksum(m, n, ldc, c_values);
    int expected_checksum = scaled_checksum(m, n, ldc, expected_values);
    if (checksum != expected_checksum) {
      printk("ERROR: gemm_shape_vc4kernel_micro_gemm_vc4kernel checksum mismatch case=%d gpu=%d cpu=%d\n",
             (int)case_id, checksum, expected_checksum);
      mismatches++;
    }
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinels;
    checksum_accum ^= checksum + (int)(case_id * 131u);
    saw_zero |= (m == 0u || n == 0u || k == 0u);
    saw_non_square |= (m != n || n != k);
    saw_k_diff |= (k != m && k != n);
    saw_tail_n |= (n % GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_LANES) != 0u;
    saw_runtime_ld |= (lda != k || ldb != n || ldc != n);
    saw_contiguous_ld |= (lda == k && ldb == n && ldc == n);
    saw_plus1_ld |= (lda == k + 1u && ldb == n + 1u && ldc == n + 1u);
    saw_plus7_ld |= (lda == k + 7u && ldb == n + 7u && ldc == n + 7u);
    saw_large_ld |= (lda == 96u || ldb == 96u || ldc == 96u);
    saw_old_timeout_shape |= (m == 17u && n == 31u && k == 33u);
    printk("GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_CASE case=%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d grid_x=%d grid_y=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
           (int)case_id, (int)m, (int)n, (int)k, (int)lda, (int)ldb,
           (int)ldc, (int)grid_x, (int)grid_y, mismatches, case_sentinels,
           checksum, max_abs_diff);
  }

  int elapsed = timer_get_usec() - start;
  int runtime_launches = (int)gemm_shape_vc4kernel_micro_gemm_vc4kernel_runtime_launches();
  launch_failures += (int)gemm_shape_vc4kernel_micro_gemm_vc4kernel_runtime_launch_failures();
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0 && saw_tail_n && saw_contiguous_ld)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=gemm_shape_vc4kernel_micro_gemm_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d block_x=%d lanes=%d max_m=%d max_n=%d max_k=%d max_ld=%d saw_zero=%d saw_non_square=%d saw_k_diff=%d saw_tail_n=%d saw_runtime_ld=%d saw_contiguous_ld=%d saw_plus1_ld=%d saw_plus7_ld=%d saw_large_ld=%d saw_old_timeout_shape=%d contiguous_layout_only=0 launch_geometry_2d_block16=1 checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
         status, (int)(sizeof(test_cases) / sizeof(test_cases[0])),
         total_mismatches, sentinel_mismatches, launch_failures,
         GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_LANES, GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_LANES,
         GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_M, GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_N,
         GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_K, GEMM_SHAPE_VC4KERNEL_MICRO_GEMM_VC4KERNEL_MAX_LDC, saw_zero,
         saw_non_square, saw_k_diff, saw_tail_n, saw_runtime_ld,
         saw_contiguous_ld, saw_plus1_ld, saw_plus7_ld, saw_large_ld,
         saw_old_timeout_shape, checksum_accum, max_abs_diff_overall, 3,
         runtime_launches, elapsed);

  vc4Free(program, a_dev);
  vc4Free(program, b_dev);
  vc4Free(program, c_dev);
  vc4_program_destroy(program);
}
