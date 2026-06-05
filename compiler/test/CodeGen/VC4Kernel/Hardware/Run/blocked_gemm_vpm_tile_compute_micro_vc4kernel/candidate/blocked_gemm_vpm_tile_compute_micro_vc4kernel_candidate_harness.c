#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define BM 1u
#define BN 16u
#define BK 4u
#define GUARD_WORDS 16u
#define C_WORDS BN
#define BUFFER_WORDS (C_WORDS + GUARD_WORDS)
#define SENTINEL (-7654.0f)

static float a_values[BK];
static float b_values[BK * BN];
static float c_values[BUFFER_WORDS];

static float a_pattern(uint32_t kk) { return (float)((int)kk - 1) * 0.5f; }

static float b_pattern(uint32_t kk, uint32_t col) {
  int raw = (int)(kk * 17u + col * 3u + 5u) - 23;
  return (float)raw * 0.25f;
}

static void fill_buffers(void) {
  for (uint32_t kk = 0; kk < BK; ++kk)
    a_values[kk] = a_pattern(kk);
  for (uint32_t kk = 0; kk < BK; ++kk)
    for (uint32_t col = 0; col < BN; ++col)
      b_values[kk * BN + col] = b_pattern(kk, col);
  for (uint32_t i = 0; i < BUFFER_WORDS; ++i)
    c_values[i] = SENTINEL;
}

static float expected(uint32_t col) {
  float sum = 0.0f;
  for (uint32_t kk = 0; kk < BK; ++kk)
    sum += a_values[kk] * b_values[kk * BN + col];
  return sum;
}

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static int verify_values(float *max_abs_diff, int *checksum) {
  int mismatches = 0;
  *max_abs_diff = 0.0f;
  *checksum = 0;
  for (uint32_t col = 0; col < BN; ++col) {
    float want = expected(col);
    float got = c_values[col];
    float diff = got - want;
    float ad = absf_local(diff);
    if (ad > *max_abs_diff)
      *max_abs_diff = ad;
    *checksum += (int)(got * 1024.0f);
    if (ad != 0.0f) {
      if (mismatches < 8)
        printk("ERROR: blocked_gemm_vpm_tile_compute_micro col=%d got=%f want=%f diff=%f\n",
               (int)col, got, want, diff);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = C_WORDS; i < BUFFER_WORDS; ++i) {
    if (c_values[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: blocked_gemm_vpm_tile_compute_micro sentinel=%d got=%f want=%f\n",
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
    panic("blocked_gemm_vpm_tile_compute_micro_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &a_dev, BK * sizeof(float)) < 0 ||
      vc4_m2_malloc(program, &b_dev, BK * BN * sizeof(float)) < 0 ||
      vc4_m2_malloc(program, &c_dev, BUFFER_WORDS * sizeof(float)) < 0)
    panic("blocked_gemm_vpm_tile_compute_micro_vc4kernel allocation failed");

  fill_buffers();
  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int checksum = 0;
  float max_abs_diff = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

  if (vc4_m2_copy_htod(program, a_dev, a_values, BK * sizeof(float)) < 0 ||
      vc4_m2_copy_htod(program, b_dev, b_values, BK * BN * sizeof(float)) < 0 ||
      vc4_m2_copy_htod(program, c_dev, c_values, BUFFER_WORDS * sizeof(float)) < 0 ||
      blocked_gemm_vpm_tile_compute_micro_vc4kernel_launch(program, grid, block,
                                                           a_dev, b_dev, c_dev) < 0 ||
      vc4_m2_copy_dtoh(program, c_values, c_dev, BUFFER_WORDS * sizeof(float)) < 0) {
    printk("ERROR: blocked_gemm_vpm_tile_compute_micro_vc4kernel launch/copy failed\n");
    ++launch_failures;
  } else {
    total_mismatches = verify_values(&max_abs_diff, &checksum);
    sentinel_mismatches = verify_sentinels();
  }

  launch_failures +=
      (int)blocked_gemm_vpm_tile_compute_micro_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      blocked_gemm_vpm_tile_compute_micro_vc4kernel_runtime_launches();
  int elapsed = timer_get_usec() - start;
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == 1)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=blocked_gemm_vpm_tile_compute_micro_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d bm=%d bn=%d bk=%d uses_vdr_vpm=1 uses_fragment_reduce=1 checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)BM, (int)BN, (int)BK, checksum, max_abs_diff, 3, launches,
         elapsed);

  vc4Free(program, a_dev);
  vc4Free(program, b_dev);
  vc4Free(program, c_dev);
  vc4_program_destroy(program);
}
