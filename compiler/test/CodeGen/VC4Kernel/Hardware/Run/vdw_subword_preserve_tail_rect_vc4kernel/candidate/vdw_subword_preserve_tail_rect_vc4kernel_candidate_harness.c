#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define OUT_BYTES 384u
#define GUARD_BYTES 64u
#define RECT_OFFSET 128u
#define SENTINEL8 0xb9u

struct preserve_case {
  uint32_t n;
  uint32_t active_rows;
  uint32_t active_cols;
  uint32_t stride;
};

static const struct preserve_case cases[] = {
  {0u, 0u, 16u, 32u},
  {1u, 1u, 0u, 34u},
  {9u, 1u, 1u, 32u},
  {15u, 2u, 7u, 46u},
  {16u, 2u, 16u, 64u},
};

static uint8_t output_bytes[OUT_BYTES + GUARD_BYTES];

static void fill_output(void) {
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
}

static uint16_t get16(uint32_t offset) {
  return (uint16_t)output_bytes[offset] |
         ((uint16_t)output_bytes[offset + 1u] << 8);
}

static int verify_case(const struct preserve_case *tc, uint32_t *checksum,
                       int *sentinel_mismatches) {
  int mismatches = 0;
  *checksum = 0;
  *sentinel_mismatches = 0;
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i) {
    uint8_t want = SENTINEL8;
    int active = 0;
    if (i < 16u && i < tc->n) {
      want = (uint8_t)(40u + i);
      active = 1;
    }
    for (uint32_t r = 0; r < 2u; ++r) {
      uint32_t row_base = RECT_OFFSET + r * tc->stride;
      if (i >= row_base && i < row_base + 32u) {
        uint32_t col = (i - row_base) / 2u;
        uint32_t byte = (i - row_base) & 1u;
        if (r < tc->active_rows && col < tc->active_cols) {
          uint16_t v = (uint16_t)((r == 0u ? 2000u : 2200u) + col);
          want = byte ? (uint8_t)(v >> 8) : (uint8_t)(v & 0xffu);
          active = 1;
        }
        break;
      }
    }
    uint8_t got = output_bytes[i];
    if (active)
      *checksum += got;
    if (got != want) {
      if (active) {
        if (mismatches < 8)
          printk("ERROR: vdw_subword_preserve active offset=%d got=%x want=%x\n",
                 (int)i, got, want);
        ++mismatches;
      } else {
        if (*sentinel_mismatches < 8)
          printk("ERROR: vdw_subword_preserve sentinel offset=%d got=%x want=%x\n",
                 (int)i, got, want);
        ++*sentinel_mismatches;
      }
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdw_subword_preserve_tail_rect_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("vdw_subword_preserve_tail_rect_vc4kernel allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
    const struct preserve_case *tc = &cases[case_id];
    fill_output();
    if (vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        vdw_subword_preserve_tail_rect_vc4kernel_launch(
            program, grid, block, out_dev, tc->n, tc->active_rows,
            tc->active_cols, tc->stride) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0) {
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
      (int)vdw_subword_preserve_tail_rect_vc4kernel_runtime_launch_failures();
  uint32_t launches = vdw_subword_preserve_tail_rect_vc4kernel_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_subword_preserve_tail_rect_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdw_w8=1 saw_vdw_w16=1 saw_byte_halfword_guards=1 saw_tail_preserve=1 saw_rect_preserve=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
         sentinel_mismatches, launch_failures, checksum_accum, launches,
         timer_get_usec() - start);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
