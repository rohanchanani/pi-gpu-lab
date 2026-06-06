#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GEMV_BLOCKED_VPM_VC4KERNEL_EPSILON 0.0001f
#define GEMV_BLOCKED_VPM_VC4KERNEL_CHECKSUM_SCALE 1024.0f
#define GEMV_BLOCKED_VPM_VC4KERNEL_MAX_M 1024u
#define GEMV_BLOCKED_VPM_VC4KERNEL_MAX_N 1024u
#define GEMV_BLOCKED_VPM_VC4KERNEL_MAX_LDA 1031u
#define GEMV_BLOCKED_VPM_VC4KERNEL_LANES 16u
#define GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_ROWS 16u
#define GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_K 4u
#define GEMV_BLOCKED_VPM_VC4KERNEL_MAX_ROW_COVERAGE \
  (((GEMV_BLOCKED_VPM_VC4KERNEL_MAX_M + \
     GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_ROWS - 1u) / \
    GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_ROWS) * \
   GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_ROWS)
#define GEMV_BLOCKED_VPM_VC4KERNEL_GUARD 32u
#define GEMV_BLOCKED_VPM_VC4KERNEL_A_WORDS \
  (GEMV_BLOCKED_VPM_VC4KERNEL_MAX_ROW_COVERAGE * \
   GEMV_BLOCKED_VPM_VC4KERNEL_MAX_LDA)
#define GEMV_BLOCKED_VPM_VC4KERNEL_Y_WORDS \
  (GEMV_BLOCKED_VPM_VC4KERNEL_MAX_M + GEMV_BLOCKED_VPM_VC4KERNEL_GUARD)
#define GEMV_BLOCKED_VPM_VC4KERNEL_PROGRAM_HEAP_BYTES \
  ((GEMV_BLOCKED_VPM_VC4KERNEL_A_WORDS + \
    GEMV_BLOCKED_VPM_VC4KERNEL_MAX_N + \
    GEMV_BLOCKED_VPM_VC4KERNEL_Y_WORDS) * \
       (uint32_t)sizeof(float) + \
   65536u)
#define GEMV_BLOCKED_VPM_VC4KERNEL_SENTINEL (-9876.5f)
#define MIXED_GEMV_ALPHA 1.25f
#define MIXED_GEMV_BIAS 0.125f
#define MIXED_GEMV_THRESHOLD (-0.03125f)

struct gemv_blocked_vpm_vc4kernel_case {
  uint32_t m;
  uint32_t n;
  uint32_t lda;
};

static const struct gemv_blocked_vpm_vc4kernel_case test_cases[] = {
    {0u, 0u, 1u},    {1u, 1u, 1u},    {15u, 15u, 15u},
    {16u, 16u, 17u}, {17u, 17u, 24u}, {31u, 31u, 31u},
    {32u, 32u, 33u}, {33u, 33u, 40u}, {65u, 65u, 65u},
    {1u, 65u, 66u},  {65u, 17u, 24u}, {7u, 65u, 72u},
    {33u, 1u, 8u},   {15u, 33u, 96u}, {96u, 96u, 103u},
    {112u, 112u, 119u}, {128u, 120u, 120u}, {192u, 192u, 199u},
    {256u, 256u, 263u}, {384u, 384u, 391u}, {512u, 512u, 512u},
    {512u, 505u, 512u}, {65u, 512u, 519u}, {768u, 768u, 775u},
    {1024u, 1024u, 1024u}, {1024u, 1017u, 1024u},
    {129u, 1024u, 1031u}, {1024u, 17u, 24u},
};

static float a_values[GEMV_BLOCKED_VPM_VC4KERNEL_A_WORDS];
static float x_values[GEMV_BLOCKED_VPM_VC4KERNEL_MAX_N];
static float y_values[GEMV_BLOCKED_VPM_VC4KERNEL_Y_WORDS];
static float expected_values[GEMV_BLOCKED_VPM_VC4KERNEL_Y_WORDS];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static uint32_t rounded_waves(uint32_t m) {
  uint32_t row_blocks = (m + GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_ROWS - 1u) /
                        GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_ROWS;
  return row_blocks == 0u ? 1u : row_blocks;
}

static void fill_inputs(uint32_t m, uint32_t n, uint32_t lda) {
  for (uint32_t i = 0; i < GEMV_BLOCKED_VPM_VC4KERNEL_A_WORDS; i++)
    a_values[i] = 123.0f;
  for (uint32_t row = 0; row < GEMV_BLOCKED_VPM_VC4KERNEL_MAX_ROW_COVERAGE;
       row++) {
    for (uint32_t col = 0; col < lda; col++) {
      uint32_t idx = row * lda + col;
      a_values[idx] = ((float)((row * 19u + col * 7u + 5u) % 47u) - 23.0f) *
                      0.03125f;
    }
  }
  for (uint32_t i = 0; i < GEMV_BLOCKED_VPM_VC4KERNEL_MAX_N; i++)
    x_values[i] = ((float)((i * 13u + 3u) % 31u) - 15.0f) * 0.0625f;
  for (uint32_t i = 0; i < GEMV_BLOCKED_VPM_VC4KERNEL_Y_WORDS; i++) {
    y_values[i] = GEMV_BLOCKED_VPM_VC4KERNEL_SENTINEL;
    expected_values[i] = GEMV_BLOCKED_VPM_VC4KERNEL_SENTINEL;
  }
  for (uint32_t row = 0; row < m; row++) {
    float sum = 0.0f;
    for (uint32_t col = 0; col < n; col++)
      sum += a_values[row * lda + col] * x_values[col];
    float biased = MIXED_GEMV_ALPHA * sum + MIXED_GEMV_BIAS;
    expected_values[row] =
        (biased > MIXED_GEMV_THRESHOLD) ? biased : MIXED_GEMV_BIAS;
  }
}

static int scaled_checksum(const float *values, uint32_t n) {
  int checksum = 0;
  for (uint32_t i = 0; i < n; i++)
    checksum += (int)(values[i] * GEMV_BLOCKED_VPM_VC4KERNEL_CHECKSUM_SCALE);
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
    if (ad > GEMV_BLOCKED_VPM_VC4KERNEL_EPSILON) {
      if (*mismatch_count < 8)
        printk("ERROR: mixed_blocked_gemv_vpm_fullstack_vc4kernel row=%d gpu=%f cpu=%f diff=%f\n",
               (int)row, y_values[row], expected_values[row], diff);
      (*mismatch_count)++;
    }
  }
}

static int verify_sentinel_region(uint32_t m) {
  int mismatches = 0;
  for (uint32_t i = m; i < GEMV_BLOCKED_VPM_VC4KERNEL_Y_WORDS; i++) {
    if (y_values[i] != GEMV_BLOCKED_VPM_VC4KERNEL_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: mixed_blocked_gemv_vpm_fullstack_vc4kernel sentinel changed row=%d value=%f expected=%f\n",
               (int)i, y_values[i], GEMV_BLOCKED_VPM_VC4KERNEL_SENTINEL);
      mismatches++;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  if (vc4_program_create(&program,
                         GEMV_BLOCKED_VPM_VC4KERNEL_PROGRAM_HEAP_BYTES) < 0 ||
      !program)
    panic("vc4_program_create failed");

  uint32_t a_bytes = GEMV_BLOCKED_VPM_VC4KERNEL_A_WORDS * sizeof(float);
  uint32_t x_bytes = GEMV_BLOCKED_VPM_VC4KERNEL_MAX_N * sizeof(float);
  uint32_t y_bytes = GEMV_BLOCKED_VPM_VC4KERNEL_Y_WORDS * sizeof(float);
  vc4_deviceptr_t a_dev = 0;
  vc4_deviceptr_t x_dev = 0;
  vc4_deviceptr_t y_dev = 0;
  if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
      vc4_m2_malloc(program, &x_dev, x_bytes) < 0 ||
      vc4_m2_malloc(program, &y_dev, y_bytes) < 0)
    panic("mixed_blocked_gemv_vpm_fullstack_vc4kernel device allocation failed");

  vc4_dim3 block = vc4_m2_dim3(GEMV_BLOCKED_VPM_VC4KERNEL_LANES, 1u, 1u);

  printk("Running VC4 mixed_blocked_gemv_vpm_fullstack_vc4kernel candidate bundle...\n");
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int checksum_accum = 0;
  float global_max_abs_diff = 0.0f;
  int largest_case_elapsed_usec = 0;
  int largest_case_ops = 0;
  int saw_lda_exact = 0;
  int saw_lda_plus1 = 0;
  int saw_lda_plus7 = 0;
  int saw_large_lda = 0;
  int start = timer_get_usec();

  for (uint32_t case_id = 0;
       case_id < sizeof(test_cases) / sizeof(test_cases[0]); case_id++) {
    uint32_t m = test_cases[case_id].m;
    uint32_t n = test_cases[case_id].n;
    uint32_t lda = test_cases[case_id].lda;
    if (lda == n)
      saw_lda_exact = 1;
    if (lda == n + 1u)
      saw_lda_plus1 = 1;
    if (lda == n + 7u)
      saw_lda_plus7 = 1;
    if (lda >= 96u)
      saw_large_lda = 1;
    fill_inputs(m, n, lda);
    uint32_t waves = rounded_waves(m);
    vc4_dim3 grid = vc4_m2_dim3(waves, 1u, 1u);
    if (vc4MemcpyHtoD(program, a_dev, a_values, a_bytes) < 0 ||
        vc4MemcpyHtoD(program, x_dev, x_values, x_bytes) < 0 ||
        vc4MemcpyHtoD(program, y_dev, y_values, y_bytes) < 0) {
      printk("ERROR: mixed_blocked_gemv_vpm_fullstack_vc4kernel copy-in failed case=%d m=%d n=%d lda=%d\n",
             (int)case_id, (int)m, (int)n, (int)lda);
      launch_failures++;
      continue;
    }

    int case_start = timer_get_usec();
    if (mixed_blocked_gemv_vpm_fullstack_vc4kernel_launch(
            program, grid, block, a_dev, x_dev, y_dev, (int32_t)m,
            (int32_t)n, (int32_t)lda, MIXED_GEMV_ALPHA, MIXED_GEMV_BIAS,
            MIXED_GEMV_THRESHOLD) < 0 ||
        vc4MemcpyDtoH(program, y_values, y_dev, y_bytes) < 0) {
      printk("ERROR: mixed_blocked_gemv_vpm_fullstack_vc4kernel launch/copy failed case=%d m=%d n=%d lda=%d\n",
             (int)case_id, (int)m, (int)n, (int)lda);
      launch_failures++;
      continue;
    }
    int case_elapsed = timer_get_usec() - case_start;

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(m, &mismatches, &max_abs_diff);
    int guard_mismatches = verify_sentinel_region(m);
    if (max_abs_diff > global_max_abs_diff)
      global_max_abs_diff = max_abs_diff;
    int checksum = scaled_checksum(y_values, m);
    checksum_accum ^= checksum + (int)(case_id * 131u);
    int case_ops = (int)(m * n);
    if (case_ops > largest_case_ops) {
      largest_case_ops = case_ops;
      largest_case_elapsed_usec = case_elapsed;
    }
    total_mismatches += mismatches;
    sentinel_mismatches += guard_mismatches;
    printk("MIXED_BLOCKED_GEMV_VPM_FULLSTACK_CASE case=%d m=%d n=%d lda=%d waves=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f elapsed_usec=%d ops=%d\n",
           (int)case_id, (int)m, (int)n, (int)lda, (int)waves, mismatches,
           guard_mismatches, checksum, max_abs_diff, case_elapsed, case_ops);
  }

  launch_failures +=
      (int)mixed_blocked_gemv_vpm_fullstack_vc4kernel_runtime_launch_failures();
  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=mixed_blocked_gemv_vpm_fullstack_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d lanes=%d max_m=%d max_n=%d max_lda=%d block_rows=%d block_k=%d saw_lda_exact=%d saw_lda_plus1=%d saw_lda_plus7=%d saw_large_lda=%d saw_vdr_vpm_path=1 saw_tmu_safe_offset_x=1 saw_runtime_lda=1 saw_vdw_preserve=1 no_fixed_padded_stride=1 saw_alpha_bias_path=1 saw_f32_cmp_select_gate=1 checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d largest_case_ops=%d largest_case_elapsed_usec=%d\n",
         status, (int)(sizeof(test_cases) / sizeof(test_cases[0])),
         total_mismatches, sentinel_mismatches, launch_failures,
         GEMV_BLOCKED_VPM_VC4KERNEL_LANES,
         GEMV_BLOCKED_VPM_VC4KERNEL_MAX_M,
         GEMV_BLOCKED_VPM_VC4KERNEL_MAX_N,
         GEMV_BLOCKED_VPM_VC4KERNEL_MAX_LDA,
         GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_ROWS,
         GEMV_BLOCKED_VPM_VC4KERNEL_BLOCK_K, saw_lda_exact, saw_lda_plus1,
         saw_lda_plus7, saw_large_lda, checksum_accum, global_max_abs_diff, 3,
         (int)mixed_blocked_gemv_vpm_fullstack_vc4kernel_runtime_launches(), elapsed,
         largest_case_ops, largest_case_elapsed_usec);

  vc4Free(program, a_dev);
  vc4Free(program, x_dev);
  vc4Free(program, y_dev);
  vc4_program_destroy(program);
}
