#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GEMM_BLOCKED_ONE_TILE_M 1u
#define GEMM_BLOCKED_ONE_TILE_N 16u
#define GEMM_BLOCKED_ONE_TILE_K 4u
#define GEMM_BLOCKED_ONE_TILE_A_WORDS \
  (GEMM_BLOCKED_ONE_TILE_M * GEMM_BLOCKED_ONE_TILE_K)
#define GEMM_BLOCKED_ONE_TILE_B_WORDS \
  (GEMM_BLOCKED_ONE_TILE_K * GEMM_BLOCKED_ONE_TILE_N)
#define GEMM_BLOCKED_ONE_TILE_C_WORDS \
  (GEMM_BLOCKED_ONE_TILE_M * GEMM_BLOCKED_ONE_TILE_N)
#define GEMM_BLOCKED_ONE_TILE_GUARD_WORDS 16u
#define GEMM_BLOCKED_ONE_TILE_BUFFER_WORDS \
  (GEMM_BLOCKED_ONE_TILE_C_WORDS + GEMM_BLOCKED_ONE_TILE_GUARD_WORDS)
#define GEMM_BLOCKED_ONE_TILE_SENTINEL (-7654.0f)
#define GEMM_BLOCKED_ONE_TILE_CHECKSUM_SCALE 1024.0f

static float a_values[GEMM_BLOCKED_ONE_TILE_A_WORDS];
static float b_values[GEMM_BLOCKED_ONE_TILE_B_WORDS];
static float c_values[GEMM_BLOCKED_ONE_TILE_BUFFER_WORDS];
static float expected_values[GEMM_BLOCKED_ONE_TILE_C_WORDS];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float a_pattern(uint32_t row, uint32_t kk) {
  int raw = (int)(row * 19u + kk * 5u + 3u) - 11;
  return (float)raw * 0.125f;
}

static float b_pattern(uint32_t kk, uint32_t col) {
  int raw = (int)(kk * 17u + col * 3u + 5u) - 23;
  return (float)raw * 0.25f;
}

static void fill_buffers(void) {
  for (uint32_t row = 0; row < GEMM_BLOCKED_ONE_TILE_M; ++row)
    for (uint32_t kk = 0; kk < GEMM_BLOCKED_ONE_TILE_K; ++kk)
      a_values[row * GEMM_BLOCKED_ONE_TILE_K + kk] = a_pattern(row, kk);

  for (uint32_t kk = 0; kk < GEMM_BLOCKED_ONE_TILE_K; ++kk)
    for (uint32_t col = 0; col < GEMM_BLOCKED_ONE_TILE_N; ++col)
      b_values[kk * GEMM_BLOCKED_ONE_TILE_N + col] = b_pattern(kk, col);

  for (uint32_t row = 0; row < GEMM_BLOCKED_ONE_TILE_M; ++row) {
    for (uint32_t col = 0; col < GEMM_BLOCKED_ONE_TILE_N; ++col) {
      float sum = 0.0f;
      for (uint32_t kk = 0; kk < GEMM_BLOCKED_ONE_TILE_K; ++kk)
        sum += a_values[row * GEMM_BLOCKED_ONE_TILE_K + kk] *
               b_values[kk * GEMM_BLOCKED_ONE_TILE_N + col];
      expected_values[row * GEMM_BLOCKED_ONE_TILE_N + col] = sum;
    }
  }

  for (uint32_t i = 0; i < GEMM_BLOCKED_ONE_TILE_BUFFER_WORDS; ++i)
    c_values[i] = GEMM_BLOCKED_ONE_TILE_SENTINEL;
}

static int verify_values(float *max_abs_diff, int *checksum) {
  int mismatches = 0;
  *max_abs_diff = 0.0f;
  *checksum = 0;
  for (uint32_t i = 0; i < GEMM_BLOCKED_ONE_TILE_C_WORDS; ++i) {
    float got = c_values[i];
    float want = expected_values[i];
    float diff = got - want;
    float ad = absf_local(diff);
    if (ad > *max_abs_diff)
      *max_abs_diff = ad;
    *checksum += (int)(got * GEMM_BLOCKED_ONE_TILE_CHECKSUM_SCALE);
    if (ad != 0.0f) {
      if (mismatches < 8)
        printk("ERROR: gemm_blocked_vpm_one_tile idx=%d got=%f want=%f diff=%f\n",
               (int)i, got, want, diff);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = GEMM_BLOCKED_ONE_TILE_C_WORDS;
       i < GEMM_BLOCKED_ONE_TILE_BUFFER_WORDS; ++i) {
    if (c_values[i] != GEMM_BLOCKED_ONE_TILE_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: gemm_blocked_vpm_one_tile sentinel=%d got=%f want=%f\n",
               (int)i, c_values[i], GEMM_BLOCKED_ONE_TILE_SENTINEL);
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
    panic("gemm_blocked_vpm_one_tile_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &a_dev,
                    GEMM_BLOCKED_ONE_TILE_A_WORDS * sizeof(float)) < 0 ||
      vc4_m2_malloc(program, &b_dev,
                    GEMM_BLOCKED_ONE_TILE_B_WORDS * sizeof(float)) < 0 ||
      vc4_m2_malloc(program, &c_dev,
                    GEMM_BLOCKED_ONE_TILE_BUFFER_WORDS * sizeof(float)) < 0)
    panic("gemm_blocked_vpm_one_tile_vc4kernel allocation failed");

  fill_buffers();
  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int checksum = 0;
  float max_abs_diff = 0.0f;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

  if (vc4_m2_copy_htod(program, a_dev, a_values,
                       GEMM_BLOCKED_ONE_TILE_A_WORDS * sizeof(float)) < 0 ||
      vc4_m2_copy_htod(program, b_dev, b_values,
                       GEMM_BLOCKED_ONE_TILE_B_WORDS * sizeof(float)) < 0 ||
      vc4_m2_copy_htod(program, c_dev, c_values,
                       GEMM_BLOCKED_ONE_TILE_BUFFER_WORDS * sizeof(float)) < 0 ||
      gemm_blocked_vpm_one_tile_vc4kernel_launch(
          program, grid, block, a_dev, b_dev, c_dev, GEMM_BLOCKED_ONE_TILE_M,
          GEMM_BLOCKED_ONE_TILE_N, GEMM_BLOCKED_ONE_TILE_K) < 0 ||
      vc4_m2_copy_dtoh(program, c_values, c_dev,
                       GEMM_BLOCKED_ONE_TILE_BUFFER_WORDS * sizeof(float)) < 0) {
    printk("ERROR: gemm_blocked_vpm_one_tile_vc4kernel launch/copy failed\n");
    ++launch_failures;
  } else {
    total_mismatches = verify_values(&max_abs_diff, &checksum);
    sentinel_mismatches = verify_sentinels();
  }

  launch_failures +=
      (int)gemm_blocked_vpm_one_tile_vc4kernel_runtime_launch_failures();
  uint32_t launches = gemm_blocked_vpm_one_tile_vc4kernel_runtime_launches();
  int elapsed = timer_get_usec() - start;
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == 1)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=gemm_blocked_vpm_one_tile_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d m=%d n=%d k=%d uses_vdr_vpm=1 uses_vpm_read_compute=1 uses_fragment_reduce=1 runtime_mnk_args=1 contiguous_runtime_strides=1 checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)GEMM_BLOCKED_ONE_TILE_M, (int)GEMM_BLOCKED_ONE_TILE_N,
         (int)GEMM_BLOCKED_ONE_TILE_K, checksum, max_abs_diff, 3, launches,
         elapsed);

  vc4Free(program, a_dev);
  vc4Free(program, b_dev);
  vc4Free(program, c_dev);
  vc4_program_destroy(program);
}
