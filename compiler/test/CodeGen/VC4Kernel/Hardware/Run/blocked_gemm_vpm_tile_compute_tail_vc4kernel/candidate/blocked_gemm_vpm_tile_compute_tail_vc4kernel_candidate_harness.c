#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define BN 16u
#define BK 4u
#define GUARD_WORDS 16u
#define BUFFER_WORDS (BN + GUARD_WORDS)
#define SENTINEL (-8765.0f)

struct case_desc {
  uint32_t active_cols;
  uint32_t active_k;
};

static const struct case_desc cases[] = {
    {1u, 1u}, {7u, 1u}, {15u, 1u}, {1u, 3u}, {7u, 3u}, {15u, 3u}};

static float a_values[BK];
static float b_values[BK * BN];
static float c_values[BUFFER_WORDS];

static float a_pattern(uint32_t case_id, uint32_t kk) {
  int raw = (int)(case_id * 3u + kk * 5u) - 7;
  return (float)raw * 0.25f;
}

static float b_pattern(uint32_t case_id, uint32_t kk, uint32_t col) {
  int raw = (int)(case_id * 11u + kk * 13u + col * 3u + 1u) - 19;
  return (float)raw * 0.125f;
}

static void fill_buffers(uint32_t case_id) {
  for (uint32_t kk = 0; kk < BK; ++kk)
    a_values[kk] = a_pattern(case_id, kk);
  for (uint32_t kk = 0; kk < BK; ++kk)
    for (uint32_t col = 0; col < BN; ++col)
      b_values[kk * BN + col] = b_pattern(case_id, kk, col);
  for (uint32_t i = 0; i < BUFFER_WORDS; ++i)
    c_values[i] = SENTINEL;
}

static float expected(uint32_t active_k, uint32_t col) {
  float sum = 0.0f;
  for (uint32_t kk = 0; kk < active_k; ++kk)
    sum += a_values[kk] * b_values[kk * BN + col];
  return sum;
}

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static int verify_values(uint32_t case_id, uint32_t active_cols,
                         uint32_t active_k, float *max_abs_diff,
                         int *checksum) {
  int mismatches = 0;
  *max_abs_diff = 0.0f;
  *checksum = 0;
  for (uint32_t col = 0; col < BN; ++col) {
    float got = c_values[col];
    if (col < active_cols) {
      float want = expected(active_k, col);
      float diff = got - want;
      float ad = absf_local(diff);
      if (ad > *max_abs_diff)
        *max_abs_diff = ad;
      *checksum += (int)(got * 1024.0f);
      if (ad != 0.0f) {
        if (mismatches < 8)
          printk("ERROR: blocked_gemm_vpm_tile_compute_tail case=%d active_cols=%d active_k=%d col=%d got=%f want=%f diff=%f\n",
                 (int)case_id, (int)active_cols, (int)active_k, (int)col,
                 got, want, diff);
        ++mismatches;
      }
    } else if (got != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: blocked_gemm_vpm_tile_compute_tail inactive col=%d got=%f want=%f\n",
               (int)col, got, SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_guard(void) {
  int mismatches = 0;
  for (uint32_t i = BN; i < BUFFER_WORDS; ++i) {
    if (c_values[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: blocked_gemm_vpm_tile_compute_tail guard=%d got=%f want=%f\n",
               (int)i, c_values[i], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t a_dev = 0;
  vc4_deviceptr_t b_dev = 0;
  vc4_deviceptr_t c_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("blocked_gemm_vpm_tile_compute_tail_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &a_dev, BK * sizeof(float)) < 0 ||
      vc4_m2_malloc(program, &b_dev, BK * BN * sizeof(float)) < 0 ||
      vc4_m2_malloc(program, &c_dev, BUFFER_WORDS * sizeof(float)) < 0)
    panic("blocked_gemm_vpm_tile_compute_tail_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int checksum_accum = 0;
  int saw_cols1 = 0;
  int saw_cols7 = 0;
  int saw_cols15 = 0;
  int saw_k1 = 0;
  int saw_k3 = 0;
  float max_abs_diff_overall = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

  for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]);
       ++case_id) {
    uint32_t active_cols = cases[case_id].active_cols;
    uint32_t active_k = cases[case_id].active_k;
    fill_buffers(case_id);
    if (vc4_m2_copy_htod(program, a_dev, a_values, BK * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, b_dev, b_values, BK * BN * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, c_dev, c_values,
                         BUFFER_WORDS * sizeof(float)) < 0 ||
        blocked_gemm_vpm_tile_compute_tail_vc4kernel_launch(
            program, grid, block, a_dev, b_dev, c_dev, active_cols,
            active_k) < 0 ||
        vc4_m2_copy_dtoh(program, c_values, c_dev,
                         BUFFER_WORDS * sizeof(float)) < 0) {
      printk("ERROR: blocked_gemm_vpm_tile_compute_tail_vc4kernel launch/copy failed case=%d active_cols=%d active_k=%d\n",
             (int)case_id, (int)active_cols, (int)active_k);
      ++launch_failures;
      continue;
    }

    float max_abs_diff = 0.0f;
    int checksum = 0;
    int mismatches =
        verify_values(case_id, active_cols, active_k, &max_abs_diff, &checksum);
    int sentinels = verify_guard();
    if (max_abs_diff > max_abs_diff_overall)
      max_abs_diff_overall = max_abs_diff;
    total_mismatches += mismatches;
    sentinel_mismatches += sentinels;
    checksum_accum += checksum;
    if (active_cols == 1u)
      saw_cols1 = 1;
    if (active_cols == 7u)
      saw_cols7 = 1;
    if (active_cols == 15u)
      saw_cols15 = 1;
    if (active_k == 1u)
      saw_k1 = 1;
    if (active_k == 3u)
      saw_k3 = 1;
    printk("BLOCKED_GEMM_VPM_TILE_COMPUTE_TAIL_CASE case=%d active_cols=%d active_k=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
           (int)case_id, (int)active_cols, (int)active_k, mismatches,
           sentinels, checksum, max_abs_diff);
  }

  launch_failures +=
      (int)blocked_gemm_vpm_tile_compute_tail_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      blocked_gemm_vpm_tile_compute_tail_vc4kernel_runtime_launches();
  int elapsed = timer_get_usec() - start;
  const char *status =
      (launch_failures == 0 && total_mismatches == 0 &&
       sentinel_mismatches == 0 && saw_cols1 && saw_cols7 && saw_cols15 &&
       saw_k1 && saw_k3 && launches == sizeof(cases) / sizeof(cases[0]))
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=blocked_gemm_vpm_tile_compute_tail_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_cols1=%d saw_cols7=%d saw_cols15=%d saw_k1=%d saw_k3=%d uses_vdr_vpm=1 uses_fragment_reduce=1 checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%u elapsed_usec=%d\n",
         status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
         sentinel_mismatches, launch_failures, saw_cols1, saw_cols7,
         saw_cols15, saw_k1, saw_k3, checksum_accum, max_abs_diff_overall, 3,
         launches, elapsed);

  vc4Free(program, a_dev);
  vc4Free(program, b_dev);
  vc4Free(program, c_dev);
  vc4_program_destroy(program);
}
