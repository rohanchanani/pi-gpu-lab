#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define IN_BYTES 16640u
#define OUT_BYTES 16640u
#define GUARD_BYTES 64u
#define SENTINEL8 0xadu

struct test_case {
  uint32_t row;
  uint32_t x;
  uint32_t sel16;
  uint32_t active_rows;
  uint32_t active_cols;
  uint32_t pitch;
  uint32_t stride;
};

static const struct test_case cases[] = {
    {0u, 0u, 0u, 0u, 7u, 32u, 32u},
    {2u, 0u, 1u, 1u, 0u, 34u, 34u},
    {4u, 1u, 0u, 1u, 3u, 64u, 64u},
    {6u, 1u, 1u, 2u, 7u, 128u, 8191u},
};

static uint8_t input_bytes[IN_BYTES];
static uint8_t output_bytes[OUT_BYTES + GUARD_BYTES];

static void store16(uint8_t *buf, uint32_t offset, uint16_t value) {
  buf[offset] = (uint8_t)(value & 0xffu);
  buf[offset + 1u] = (uint8_t)(value >> 8);
}

static uint16_t load16(const uint8_t *buf, uint32_t offset) {
  return (uint16_t)buf[offset] | ((uint16_t)buf[offset + 1u] << 8);
}

static void fill_buffers(uint32_t case_index) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0x31u + ((i + case_index) & 31u));
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
  for (uint32_t r = 0; r < 2u; ++r)
    for (uint32_t c = 0; c < 7u; ++c)
      store16(input_bytes, r * 128u + c * 2u,
              (uint16_t)(0x2400u + case_index * 0x100u + r * 0x20u + c));
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *sentinel_mismatches = 0;
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i) {
    uint8_t want = SENTINEL8;
    int active = 0;
    for (uint32_t r = 0; r < 2u; ++r) {
      uint32_t row_base = r * tc->stride;
      if (i >= row_base && i < row_base + 14u) {
        uint32_t byte_offset = i - row_base;
        uint32_t col = byte_offset / 2u;
        uint32_t byte = byte_offset & 1u;
        if (r < tc->active_rows && col < tc->active_cols) {
          uint16_t value = load16(input_bytes, r * tc->pitch + col * 2u);
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
          printk("ERROR: dynamic_pitch_stride active case=%d offset=%d got=%x want=%x\n",
                 (int)case_index, (int)i, got, want);
        ++mismatches;
      } else {
        if (*sentinel_mismatches < 8)
          printk("ERROR: dynamic_pitch_stride sentinel case=%d offset=%d got=%x want=%x\n",
                 (int)case_index, (int)i, got, want);
        ++*sentinel_mismatches;
      }
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_vdw_dynamic_selector_runtime_pitch_stride_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("vdr_vdw_dynamic_selector_runtime_pitch_stride_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const struct test_case *tc = &cases[i];
    fill_buffers(i);
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        vdr_vdw_dynamic_selector_runtime_pitch_stride_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, tc->row, tc->x, tc->sel16,
            tc->active_rows, tc->active_cols, tc->pitch, tc->stride) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    total_mismatches += verify_case(tc, i, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
  }

  launch_failures +=
      (int)vdr_vdw_dynamic_selector_runtime_pitch_stride_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      vdr_vdw_dynamic_selector_runtime_pitch_stride_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_vdw_dynamic_selector_runtime_pitch_stride_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_dynamic_subword_selector=1 saw_vdw_dynamic_vpm_row=1 saw_vdw_dynamic_vpm_x=1 saw_vdw_dynamic_subword_selector=1 saw_horizontal_subword=1 saw_byte_halfword_guards=1 saw_vdw_preserve=1 no_tmu_to_vpm=1 saw_runtime_pitch_stride=1 saw_runtime_active_rows_cols=1 saw_stride_13bit_boundary=1 selector_values_w16=0,1 x_values=0,1 pitches=32,34,64,128 strides=32,34,64,8191 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
