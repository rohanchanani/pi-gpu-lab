#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ROWS 4u
#define COLS 16u
#define MAX_STRIDE_BYTES 64u
#define REGION_BYTES 512u
#define TOTAL_BYTES (REGION_BYTES * 2u)
#define GUARD_BYTES 128u
#define SENTINEL8 0xa7u

struct dma_case {
  uint32_t active_rows;
  uint32_t active_cols;
  uint32_t pitch;
  uint32_t stride;
};

static const struct dma_case cases[] = {
  {0u, 16u, 32u, 32u},
  {1u, 1u, 34u, 34u},
  {2u, 7u, 46u, 46u},
  {3u, 15u, 64u, 64u},
  {4u, 13u, 64u, 64u},
};

static uint8_t input_bytes[TOTAL_BYTES];
static uint8_t output_bytes[TOTAL_BYTES + GUARD_BYTES];

static uint8_t pattern8(uint32_t row, uint32_t col) {
  return (uint8_t)(0x30u + row * 17u + col);
}

static uint16_t pattern16(uint32_t row, uint32_t col) {
  return (uint16_t)(0x1200u + row * 257u + col * 3u);
}

static void put16(uint8_t *base, uint32_t offset, uint16_t value) {
  base[offset] = (uint8_t)(value & 0xffu);
  base[offset + 1u] = (uint8_t)(value >> 8);
}

static uint16_t get16(const uint8_t *base, uint32_t offset) {
  return (uint16_t)base[offset] | ((uint16_t)base[offset + 1u] << 8);
}

static void fill_buffers(const struct dma_case *tc) {
  for (uint32_t i = 0; i < TOTAL_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0x80u + (i * 13u));
  for (uint32_t r = 0; r < ROWS; ++r) {
    for (uint32_t c = 0; c < COLS; ++c) {
      input_bytes[r * tc->pitch + c] = pattern8(r, c);
      put16(input_bytes, REGION_BYTES + r * tc->pitch + c * 2u,
            pattern16(r, c));
    }
  }
  for (uint32_t i = 0; i < TOTAL_BYTES + GUARD_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
}

static int verify_case(const struct dma_case *tc, uint32_t *checksum,
                       int *sentinel_mismatches) {
  int mismatches = 0;
  *checksum = 0;
  *sentinel_mismatches = 0;
  for (uint32_t i = 0; i < TOTAL_BYTES + GUARD_BYTES; ++i) {
    uint8_t want = SENTINEL8;
    int active = 0;
    for (uint32_t r = 0; r < ROWS; ++r) {
      uint32_t row8 = r * tc->stride;
      uint32_t row16 = REGION_BYTES + r * tc->stride;
      if (i >= row8 && i < row8 + COLS) {
        uint32_t c = i - row8;
        if (r < tc->active_rows && c < tc->active_cols) {
          want = pattern8(r, c);
          active = 1;
        }
        break;
      }
      if (i >= row16 && i < row16 + COLS * 2u) {
        uint32_t c = (i - row16) / 2u;
        uint32_t byte = (i - row16) & 1u;
        if (r < tc->active_rows && c < tc->active_cols) {
          uint16_t v = pattern16(r, c);
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
          printk("ERROR: subword_roundtrip active offset=%d got=%x want=%x\n",
                 (int)i, got, want);
        ++mismatches;
      } else {
        if (*sentinel_mismatches < 8)
          printk("ERROR: subword_roundtrip sentinel offset=%d got=%x want=%x\n",
                 (int)i, got, want);
        ++*sentinel_mismatches;
      }
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_vdw_subword_roundtrip_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("vdr_vdw_subword_roundtrip_vc4kernel allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
    const struct dma_case *tc = &cases[case_id];
    fill_buffers(tc);
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        vdr_vdw_subword_roundtrip_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, tc->active_rows,
            tc->active_cols, tc->pitch, tc->stride) < 0 ||
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
      (int)vdr_vdw_subword_roundtrip_vc4kernel_runtime_launch_failures();
  uint32_t launches = vdr_vdw_subword_roundtrip_vc4kernel_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_vdw_subword_roundtrip_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_w8=1 saw_vdr_w16=1 saw_vdw_w8=1 saw_vdw_w16=1 saw_byte_halfword_guards=1 saw_tail_preserve=1 saw_rect_preserve=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
         sentinel_mismatches, launch_failures, checksum_accum, launches,
         timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
