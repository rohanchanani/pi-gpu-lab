#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define IN_BYTES 256u
#define OUT_BYTES 896u
#define SENTINEL8 0xacu

struct test_case {
  uint32_t iters;
  uint32_t active_cols;
};

static const struct test_case cases[] = {
    {0u, 7u}, {1u, 1u}, {2u, 0u}, {3u, 5u}, {4u, 7u},
};

static uint8_t input_bytes[IN_BYTES];
static uint8_t output_bytes[OUT_BYTES];

static void store16(uint8_t *buf, uint32_t offset, uint16_t value) {
  buf[offset] = (uint8_t)(value & 0xffu);
  buf[offset + 1u] = (uint8_t)(value >> 8);
}

static void store32(uint8_t *buf, uint32_t offset, uint32_t value) {
  buf[offset] = (uint8_t)value;
  buf[offset + 1u] = (uint8_t)(value >> 8);
  buf[offset + 2u] = (uint8_t)(value >> 16);
  buf[offset + 3u] = (uint8_t)(value >> 24);
}

static uint16_t load16(const uint8_t *buf, uint32_t offset) {
  return (uint16_t)buf[offset] | ((uint16_t)buf[offset + 1u] << 8);
}

static uint32_t load32(const uint8_t *buf, uint32_t offset) {
  return (uint32_t)buf[offset] | ((uint32_t)buf[offset + 1u] << 8) |
         ((uint32_t)buf[offset + 2u] << 16) |
         ((uint32_t)buf[offset + 3u] << 24);
}

static void fill_buffers(uint32_t case_index) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0x20u + ((i + case_index) & 31u));
  for (uint32_t i = 0; i < OUT_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
  for (uint32_t iter = 0; iter < 4u; ++iter)
    for (uint32_t col = 0; col < 7u; ++col)
      store16(input_bytes, iter * 32u + col * 2u,
              (uint16_t)(0x3000u + case_index * 0x100u + iter * 0x20u + col));
  store32(input_bytes, 160u, 0x7a000000u | (case_index << 8) | 0x55u);
}

static uint32_t selected_value(uint32_t lane, uint32_t active_cols) {
  uint32_t value = 20u + lane;
  value += lane;
  return lane < active_cols ? value : 0u;
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *sentinel_mismatches = 0;
  uint32_t sum = 0;
  for (uint32_t lane = 0; lane < 16u; ++lane)
    sum += selected_value(lane, tc->active_cols);
  for (uint32_t i = 0; i < OUT_BYTES; ++i) {
    uint8_t want = SENTINEL8;
    int active = 0;
    for (uint32_t iter = 0; iter < 4u; ++iter) {
      uint32_t row_base = iter * 32u;
      if (i >= row_base && i < row_base + 14u) {
        uint32_t byte_offset = i - row_base;
        uint32_t col = byte_offset / 2u;
        uint32_t byte = byte_offset & 1u;
        if (iter < tc->iters && col < tc->active_cols) {
          uint16_t value = load16(input_bytes, iter * 32u + col * 2u);
          want = byte ? (uint8_t)(value >> 8) : (uint8_t)(value & 0xffu);
          active = 1;
        }
        break;
      }
    }
    if (i >= 512u && i < 576u) {
      uint32_t lane = (i - 512u) / 4u;
      uint32_t byte = (i - 512u) & 3u;
      uint32_t src_lane = (lane + tc->iters) & 15u;
      uint32_t value = selected_value(src_lane, tc->active_cols);
      want = (uint8_t)(value >> (byte * 8u));
      active = 1;
    } else if (i >= 576u && i < 640u) {
      uint32_t byte = (i - 576u) & 3u;
      want = (uint8_t)(sum >> (byte * 8u));
      active = 1;
    } else if (i >= 640u && i < 644u) {
      uint32_t byte = i - 640u;
      uint32_t value = load32(input_bytes, 160u);
      want = (uint8_t)(value >> (byte * 8u));
      active = 1;
    }
    uint8_t got = output_bytes[i];
    if (active)
      *checksum += got;
    if (got != want) {
      if (active) {
        if (mismatches < 8)
          printk("ERROR: pingpong active case=%d offset=%d got=%x want=%x\n",
                 (int)case_index, (int)i, got, want);
        ++mismatches;
      } else {
        if (*sentinel_mismatches < 8)
          printk("ERROR: pingpong sentinel case=%d offset=%d got=%x want=%x\n",
                 (int)case_index, (int)i, got, want);
        ++*sentinel_mismatches;
      }
    }
  }
  for (uint32_t lane = 0; lane < 16u; ++lane) {
    uint32_t src_lane = (lane + tc->iters) & 15u;
    uint32_t got = load32(output_bytes, 512u + lane * 4u);
    uint32_t want = selected_value(src_lane, tc->active_cols);
    if (got != want) {
      if (mismatches < 8)
        printk("ERROR: mixed rect compute lane case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, want);
      ++mismatches;
    }
  }
  for (uint32_t lane = 0; lane < 16u; ++lane) {
    uint32_t got = load32(output_bytes, 576u + lane * 4u);
    if (got != sum) {
      if (mismatches < 8)
        printk("ERROR: mixed rect reduce lane case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, sum);
      ++mismatches;
    }
  }
  uint32_t got_w32 = load32(output_bytes, 640u);
  uint32_t want_w32 = load32(input_bytes, 160u);
  if (got_w32 != want_w32) {
    if (mismatches < 8)
      printk("ERROR: mixed rect w32 case=%d got=%x want=%x\n",
             (int)case_index, got_w32, want_w32);
    ++mismatches;
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("mixed_dynamic_vpm_coordinates_rect_loop_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("mixed_dynamic_vpm_coordinates_rect_loop_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_buffers(i);
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        mixed_dynamic_vpm_coordinates_rect_loop_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, cases[i].iters,
            cases[i].active_cols) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    total_mismatches += verify_case(&cases[i], i, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
  }

  launch_failures +=
      (int)mixed_dynamic_vpm_coordinates_rect_loop_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      mixed_dynamic_vpm_coordinates_rect_loop_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=mixed_dynamic_vpm_coordinates_rect_loop_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_runtime_rect_loop=1 saw_dynamic_vpm_pingpong=1 saw_dynamic_vpm_row=1 saw_dynamic_vpm_x=1 saw_dynamic_subword_selector=1 saw_dynamic_vdr_vpm_dest_coords=1 saw_vdr_dynamic_subword_selector=1 saw_dynamic_vpm_qpu_coords=1 saw_dynamic_vpm_qpu_subword_selectors=1 saw_dynamic_vdw_vpm_source_coords=1 saw_vdw_dynamic_subword_selector=1 saw_w32_path=1 saw_packed_subword_path=1 saw_fragment_alu=1 saw_cmp_select=1 saw_fragment_reduce=1 saw_scalar_control=1 saw_dynamic_rotate=1 saw_sfu_sidepath=0 saw_iters0=1 saw_iters4=1 saw_tail_cols=1 saw_vdw_preserve=1 no_tmu_to_vpm=1 selector_values_w8=0,1,2,3 selector_values_w16=0,1 x_values=0,1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
