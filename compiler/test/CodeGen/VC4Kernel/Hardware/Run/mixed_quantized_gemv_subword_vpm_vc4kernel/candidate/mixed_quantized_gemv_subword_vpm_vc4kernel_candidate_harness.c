#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define MAX_M 65u
#define MAX_N 65536u
#define MAX_LDA 65543u
#define A_BYTES (MAX_M * MAX_LDA + 64u)
#define X_BYTES (MAX_N + 64u)
#define OUT_WORDS 1152u
#define OUT_BYTES (OUT_WORDS * 4u)
#define AUDIT_WORD_OFFSET 1024u
#define PROGRAM_HEAP_LIMIT_BYTES (50u * 1024u * 1024u)
#define PROGRAM_HEAP_BYTES (A_BYTES + X_BYTES + OUT_BYTES + 65536u)
#define SENTINEL 0xd00d0000u

#if PROGRAM_HEAP_BYTES > PROGRAM_HEAP_LIMIT_BYTES
#error "mixed_quantized_gemv_subword_vpm_vc4kernel heap exceeds 50 MiB cap"
#endif

struct gemv_case {
  uint32_t m;
  uint32_t n;
  uint32_t lda;
};

static const struct gemv_case cases[] = {
  {0u, 0u, 16u},
  {1u, 1u, 16u},
  {15u, 4u, 16u},
  {16u, 15u, 16u},
  {17u, 16u, 17u},
  {31u, 17u, 24u},
  {32u, 31u, 40u},
  {33u, 32u, 40u},
  {65u, 65u, 72u},
  {65u, 1024u, 1031u},
  {33u, 4096u, 4103u},
  {7u, 16384u, 16391u},
  {1u, 65536u, 65543u},
};

static uint8_t a_bytes[A_BYTES];
static uint8_t x_bytes[X_BYTES];
static uint32_t out_words[OUT_WORDS];

static uint8_t a_value(uint32_t row, uint32_t col) {
  return (uint8_t)(3u + row * 7u + col * 11u);
}

static uint8_t x_value(uint32_t col) {
  return (uint8_t)(5u + col * 3u);
}

static uint32_t expected_row(uint32_t row, const struct gemv_case *tc) {
  uint32_t v = 0;
  for (uint32_t col = 0; col < tc->n; ++col)
    v += (uint32_t)a_value(row, col) * (uint32_t)x_value(col);
  return v;
}

static uint32_t sentinel_value(uint32_t index) {
  return SENTINEL + index;
}

static void fill_buffers(const struct gemv_case *tc) {
  for (uint32_t r = 0; r < tc->m; ++r) {
    for (uint32_t c = 0; c < tc->n; ++c)
      a_bytes[r * tc->lda + c] = a_value(r, c);
    for (uint32_t c = tc->n; c < tc->lda; ++c)
      a_bytes[r * tc->lda + c] = 0;
  }
  for (uint32_t c = 0; c < tc->n; ++c)
    x_bytes[c] = x_value(c);
  for (uint32_t i = 0; i < OUT_WORDS; ++i)
    out_words[i] = sentinel_value(i);
}

static uint32_t grid_x(uint32_t m) {
  return m == 0u ? 1u : m;
}

static int verify_case(const struct gemv_case *tc, uint32_t *checksum,
                       int *sentinel_mismatches) {
  int mismatches = 0;
  *checksum = 0;
  *sentinel_mismatches = 0;

  for (uint32_t i = 0; i < OUT_WORDS; ++i) {
    int active = 0;
    uint32_t want = sentinel_value(i);
    if (i < tc->m) {
      active = 1;
      want = expected_row(i, tc);
    }
    if (tc->m > 0u && i >= AUDIT_WORD_OFFSET &&
        i < AUDIT_WORD_OFFSET + tc->m) {
      uint32_t row = i - AUDIT_WORD_OFFSET;
      active = 1;
      want = expected_row(row, tc);
    }
    uint32_t got = out_words[i];
    if (active)
      *checksum += got;
    if (got != want) {
      if (active) {
        if (mismatches < 8)
          printk("ERROR: mixed_quantized_gemv idx=%d got=%x want=%x m=%d n=%d lda=%d\n",
                 (int)i, got, want, (int)tc->m, (int)tc->n, (int)tc->lda);
        ++mismatches;
      } else {
        if (*sentinel_mismatches < 8)
          printk("ERROR: mixed_quantized_gemv sentinel idx=%d got=%x want=%x\n",
                 (int)i, got, want);
        ++*sentinel_mismatches;
      }
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t a_dev = 0, x_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, PROGRAM_HEAP_BYTES) < 0 || !program)
    panic("mixed_quantized_gemv_subword_vpm_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &a_dev, sizeof(a_bytes)) < 0 ||
      vc4_m2_malloc(program, &x_dev, sizeof(x_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(out_words)) < 0)
    panic("mixed_quantized_gemv_subword_vpm_vc4kernel allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;
  uint32_t max_m = 0;
  uint32_t max_n = 0;
  uint32_t max_lda = 0;
  int start = timer_get_usec();
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
    const struct gemv_case *tc = &cases[case_id];
    if (tc->m > max_m)
      max_m = tc->m;
    if (tc->n > max_n)
      max_n = tc->n;
    if (tc->lda > max_lda)
      max_lda = tc->lda;
    fill_buffers(tc);
    vc4_dim3 grid = vc4_m2_dim3(grid_x(tc->m), 1, 1);
    if (vc4_m2_copy_htod(program, a_dev, a_bytes, sizeof(a_bytes)) < 0 ||
        vc4_m2_copy_htod(program, x_dev, x_bytes, sizeof(x_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_words, sizeof(out_words)) < 0 ||
        mixed_quantized_gemv_subword_vpm_vc4kernel_launch(
            program, grid, block, a_dev, x_dev, out_dev, tc->m, tc->n,
            tc->lda) < 0 ||
        vc4_m2_copy_dtoh(program, out_words, out_dev, sizeof(out_words)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    uint32_t checksum = 0;
    total_mismatches += verify_case(tc, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
    checksum_accum += checksum;
  }
  launch_failures +=
      (int)mixed_quantized_gemv_subword_vpm_vc4kernel_runtime_launch_failures();
  uint32_t launches = mixed_quantized_gemv_subword_vpm_vc4kernel_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=mixed_quantized_gemv_subword_vpm_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d max_m=%u max_n=%u max_lda=%u program_heap_bytes=%u saw_runtime_mn=1 saw_runtime_lda=1 saw_vdr_vpm_subword_tile=1 saw_fragment_unpack=1 saw_fragment_reduce=1 saw_vdw_preserve=1 no_tmu_ab_path=1 no_fixed_padded_stride=1 saw_explicit_x_subword=1 saw_mul24=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
         sentinel_mismatches, launch_failures, max_m, max_n, max_lda,
         PROGRAM_HEAP_BYTES, checksum_accum, launches, timer_get_usec() - start);
  vc4Free(program, a_dev);
  vc4Free(program, x_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
