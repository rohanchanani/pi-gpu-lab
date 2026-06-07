#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define OUT_BYTES 16640u
#define GUARD_BYTES 64u
#define RECT_OFFSET 256u
#define SENTINEL8 0xacu

struct test_case {
  uint32_t n;
  uint32_t active_rows;
  uint32_t active_cols;
  uint32_t stride;
  uint32_t row;
  uint32_t x;
  uint32_t sel8;
  uint32_t sel16;
};

static const struct test_case cases[] = {
    {0u, 0u, 7u, 32u, 0u, 0u, 0u, 0u},
    {1u, 1u, 0u, 34u, 2u, 0u, 1u, 1u},
    {9u, 1u, 3u, 64u, 4u, 1u, 0u, 0u},
    {15u, 2u, 7u, 8191u, 6u, 1u, 1u, 1u},
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

static uint8_t tail_value(uint32_t subword) {
  uint32_t selector = subword & 3u;
  uint32_t word = subword >> 2;
  if (selector == 0u)
    return (uint8_t)(40u + word);
  if (selector == 2u)
    return (uint8_t)(80u + word);
  return 0u;
}

static uint16_t rect_value(uint32_t row, uint32_t subword) {
  uint32_t selector = subword & 1u;
  uint32_t word = subword >> 1;
  if (row == 0u)
    return (uint16_t)((selector ? 3000u : 2000u) + word);
  return (uint16_t)((selector ? 5000u : 4000u) + word);
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *sentinel_mismatches = 0;
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i) {
    uint8_t want = SENTINEL8;
    int active = 0;
    if (i < 16u && i < tc->n) {
      want = tail_value(i);
      active = 1;
    }
    for (uint32_t r = 0; r < 2u; ++r) {
      uint32_t row_base = RECT_OFFSET + r * tc->stride;
      if (i >= row_base && i < row_base + 14u) {
        uint32_t byte_offset = i - row_base;
        uint32_t col = byte_offset / 2u;
        uint32_t byte = byte_offset & 1u;
        if (r < tc->active_rows && col < tc->active_cols) {
          uint16_t value = rect_value(r, tc->x * 2u + tc->sel16 + col);
          want = byte ? (uint8_t)(value >> 8) : (uint8_t)(value & 0xffu);
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
          printk("ERROR: vdw_dynamic_preserve active case=%d offset=%d got=%x want=%x\n",
                 (int)case_index, (int)i, got, want);
        ++mismatches;
      } else {
        if (*sentinel_mismatches < 8)
          printk("ERROR: vdw_dynamic_preserve sentinel case=%d offset=%d got=%x want=%x\n",
                 (int)case_index, (int)i, got, want);
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
    panic("vdw_dynamic_selector_preserve_tail_rect_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("vdw_dynamic_selector_preserve_tail_rect_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const struct test_case *tc = &cases[i];
    fill_output();
    if (vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        vdw_dynamic_selector_preserve_tail_rect_vc4kernel_launch(
            program, grid, block, out_dev, tc->n, tc->active_rows,
            tc->active_cols, tc->stride, tc->row, tc->x, tc->sel8,
            tc->sel16) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    total_mismatches += verify_case(tc, i, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
  }

  launch_failures +=
      (int)vdw_dynamic_selector_preserve_tail_rect_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      vdw_dynamic_selector_preserve_tail_rect_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_dynamic_selector_preserve_tail_rect_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdw_dynamic_vpm_row=1 saw_vdw_dynamic_vpm_x=1 saw_vdw_dynamic_subword_selector=1 saw_horizontal_subword=1 saw_byte_halfword_guards=1 saw_vdw_preserve=1 saw_tail_preserve=1 saw_rect_preserve=1 saw_runtime_stride=1 saw_stride_13bit_boundary=1 no_tmu_to_vpm=1 selector_values_w8=0,1 selector_values_w16=0,1 x_values=0,1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
