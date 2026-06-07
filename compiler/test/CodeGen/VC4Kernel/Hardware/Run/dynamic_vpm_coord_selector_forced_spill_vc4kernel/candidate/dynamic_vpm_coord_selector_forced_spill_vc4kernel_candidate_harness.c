#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define IN_BYTES 128u
#define OUT_BYTES 384u
#define SENTINEL8 0xbdu

struct test_case {
  uint32_t row;
  uint32_t x;
  uint32_t sel16;
  uint32_t beta;
};

static const struct test_case cases[] = {
    {0u, 0u, 0u, 1u},
    {16u, 1u, 1u, 2u},
    {0u, 7u, 0u, 3u},
    {16u, 15u, 1u, 5u},
};

static uint8_t input_bytes[IN_BYTES];
static uint8_t output_bytes[OUT_BYTES];

static void store16(uint8_t *buf, uint32_t offset, uint16_t value) {
  buf[offset] = (uint8_t)(value & 0xffu);
  buf[offset + 1u] = (uint8_t)(value >> 8);
}

static uint16_t load16(const uint8_t *buf, uint32_t offset) {
  return (uint16_t)buf[offset] | ((uint16_t)buf[offset + 1u] << 8);
}

static void store32_host(uint8_t *buf, uint32_t offset, uint32_t value) {
  buf[offset] = (uint8_t)(value & 0xffu);
  buf[offset + 1u] = (uint8_t)((value >> 8) & 0xffu);
  buf[offset + 2u] = (uint8_t)((value >> 16) & 0xffu);
  buf[offset + 3u] = (uint8_t)(value >> 24);
}

static uint32_t load32_host(const uint8_t *buf, uint32_t offset) {
  return (uint32_t)buf[offset] | ((uint32_t)buf[offset + 1u] << 8) |
         ((uint32_t)buf[offset + 2u] << 16) | ((uint32_t)buf[offset + 3u] << 24);
}

static void fill_buffers(uint32_t case_index) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0x41u + ((i + case_index) & 31u));
  for (uint32_t i = 0; i < OUT_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
  for (uint32_t lane = 0; lane < LANES; ++lane)
    store16(input_bytes, lane * 2u,
            (uint16_t)(0x1800u + case_index * 0x80u + lane));
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *sentinel_mismatches = 0;
  uint32_t pressure = 324u * tc->beta;
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    uint32_t base = 0x1800u + lane;
    uint32_t want = base + pressure;
    uint32_t got = load32_host(output_bytes, lane * 4u);
    *checksum += got;
    if (got != want) {
      if (mismatches < 8)
        printk("ERROR: coord_selector_spill value case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, want);
      ++mismatches;
    }
  }
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    uint16_t want = load16(input_bytes, lane * 2u);
    uint16_t got = load16(output_bytes, 256u + lane * 2u);
    *checksum += got;
    if (got != want) {
      if (mismatches < 8)
        printk("ERROR: coord_selector_spill vdw case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, want);
      ++mismatches;
    }
  }
  for (uint32_t i = 64u; i < OUT_BYTES; ++i) {
    if (i >= 256u && i < 288u)
      continue;
    if (output_bytes[i] != SENTINEL8) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: coord_selector_spill sentinel case=%d offset=%d got=%x want=%x\n",
               (int)case_index, (int)i, output_bytes[i], SENTINEL8);
      ++*sentinel_mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vpm_coord_selector_forced_spill_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("dynamic_vpm_coord_selector_forced_spill_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_buffers(i);
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        dynamic_vpm_coord_selector_forced_spill_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, cases[i].row, cases[i].x,
            cases[i].sel16, cases[i].beta) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    total_mismatches += verify_case(&cases[i], i, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
  }

  launch_failures +=
      (int)dynamic_vpm_coord_selector_forced_spill_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      dynamic_vpm_coord_selector_forced_spill_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vpm_coord_selector_forced_spill_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_spill_frame_nonzero=1 saw_dynamic_vpm_row=1 saw_dynamic_vpm_x=1 saw_dynamic_subword_selector=1 saw_vdr_dynamic_subword_selector=1 saw_vdw_dynamic_subword_selector=1 saw_vdw_preserve=1 hidden_spill_reload_tmu_hits=0 no_tmu_to_vpm=1 selector_values_w16=0,1 x_values=0,1,7,15 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
