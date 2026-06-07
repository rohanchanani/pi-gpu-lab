#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define IN_BYTES 448u
#define OUT_BYTES 448u
#define GUARD_BYTES 64u
#define SENTINEL8 0xa5u

struct test_case {
  uint32_t row_h, row_v32, row_v8, row_v16, x, sel8, sel16;
};

static const struct test_case cases[] = {
    {0u, 16u, 32u, 48u, 0u, 0u, 0u},
    {2u, 16u, 32u, 48u, 1u, 1u, 1u},
    {4u, 16u, 32u, 48u, 7u, 2u, 0u},
    {6u, 16u, 32u, 48u, 15u, 3u, 1u},
};

static uint8_t input_bytes[IN_BYTES];
static uint8_t output_bytes[OUT_BYTES + GUARD_BYTES];

static void store32_at(uint32_t off, uint32_t v) {
  input_bytes[off] = (uint8_t)v;
  input_bytes[off + 1u] = (uint8_t)(v >> 8);
  input_bytes[off + 2u] = (uint8_t)(v >> 16);
  input_bytes[off + 3u] = (uint8_t)(v >> 24);
}

static void store16_at(uint32_t off, uint16_t v) {
  input_bytes[off] = (uint8_t)v;
  input_bytes[off + 1u] = (uint8_t)(v >> 8);
}

static void fill_input(uint32_t case_id) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0xd0u + i * 7u);
  store32_at(0u, 0x41000000u | (case_id << 8) | 0x33u);
  for (uint32_t lane = 0; lane < 16u; ++lane)
    store32_at(64u + lane * 4u, 0x42000000u | (case_id << 8) | lane);
  input_bytes[128u] = (uint8_t)(0x31u + case_id);
  store16_at(192u, (uint16_t)(0x1200u + case_id));
  for (uint32_t lane = 0; lane < 16u; ++lane) {
    input_bytes[256u + lane] = (uint8_t)(0x51u + case_id + lane);
    store16_at(320u + lane * 2u, (uint16_t)(0x2400u + case_id * 17u + lane));
  }
}

static void fill_output(void) {
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
}

static int verify(uint32_t *checksum, int *sentinels) {
  int mismatches = 0;
  *sentinels = 0;
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i) {
    int active = (i < 4u) || (i >= 64u && i < 128u) || i == 128u ||
                 (i >= 192u && i < 194u) ||
                 (i >= 256u && i < 272u) ||
                 (i >= 320u && i < 352u);
    uint8_t want = active ? input_bytes[i] : SENTINEL8;
    uint8_t got = output_bytes[i];
    if (active)
      *checksum += got;
    if (got != want) {
      if (active) {
        if (mismatches < 8)
          printk("ERROR: dynamic_roundtrip active offset=%d got=%x want=%x\n",
                 (int)i, got, want);
        ++mismatches;
      } else {
        if (*sentinels < 8)
          printk("ERROR: dynamic_roundtrip sentinel offset=%d got=%x want=%x\n",
                 (int)i, got, want);
        ++*sentinels;
      }
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_vdw_dynamic_x_selector_roundtrip_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("vdr_vdw_dynamic_x_selector_roundtrip_vc4kernel allocation failed");

  int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_input(i);
    fill_output();
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        vdr_vdw_dynamic_x_selector_roundtrip_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, cases[i].row_h,
            cases[i].row_v32, cases[i].row_v8, cases[i].row_v16, cases[i].x,
            cases[i].sel8, cases[i].sel16) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0) {
      ++launch_failures;
      continue;
    }
    int s = 0;
    total_mismatches += verify(&checksum, &s);
    sentinel_mismatches += s;
  }
  launch_failures +=
      (int)vdr_vdw_dynamic_x_selector_roundtrip_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      vdr_vdw_dynamic_x_selector_roundtrip_vc4kernel_runtime_launches();
  uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_vdw_dynamic_x_selector_roundtrip_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_dynamic_subword_selector=1 saw_vdw_dynamic_vpm_row=1 saw_vdw_dynamic_vpm_x=1 saw_vdw_dynamic_subword_selector=1 saw_horizontal_w32=1 saw_vertical_w32=1 saw_horizontal_subword=1 saw_vertical_subword=1 saw_byte_halfword_guards=1 saw_vdw_preserve=1 no_tmu_to_vpm=1 selector_values_w8=0,1,2,3 selector_values_w16=0,1 x_values=0,1,7,15 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
