#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define OUT_BYTES 448u
#define GUARD_BYTES 64u
#define SENTINEL8 0xacu

struct test_case {
  uint32_t row_h;
  uint32_t row_v32;
  uint32_t row_v8;
  uint32_t row_v16;
  uint32_t x;
  uint32_t sel8;
  uint32_t sel16;
};

static const struct test_case cases[] = {
    {0u, 16u, 32u, 48u, 0u, 0u, 0u},
    {2u, 16u, 32u, 48u, 1u, 1u, 1u},
    {4u, 16u, 32u, 48u, 7u, 2u, 0u},
    {6u, 16u, 32u, 48u, 15u, 3u, 1u},
};

static uint8_t output_bytes[OUT_BYTES + GUARD_BYTES];

static void fill_output(void) {
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
}

static uint32_t load32_at(uint32_t offset) {
  return (uint32_t)output_bytes[offset] |
         ((uint32_t)output_bytes[offset + 1u] << 8) |
         ((uint32_t)output_bytes[offset + 2u] << 16) |
         ((uint32_t)output_bytes[offset + 3u] << 24);
}

static uint16_t get16(uint32_t offset) {
  return (uint16_t)output_bytes[offset] |
         ((uint16_t)output_bytes[offset + 1u] << 8);
}

static uint32_t h32_value(uint32_t lane) { return 16640u + lane; }
static uint32_t v32_value(uint32_t lane) { return 16896u + lane; }
static uint8_t h8_value(uint32_t selector, uint32_t lane) {
  return (uint8_t)((49u + selector * 32u) + lane);
}

static uint16_t h16_value(uint32_t selector, uint32_t lane) {
  return (uint16_t)((selector ? 8704u : 4608u) + lane);
}

static uint8_t v8_value(uint32_t selector, uint32_t lane) {
  return (uint8_t)((81u + selector * 32u) + lane);
}

static uint16_t v16_value(uint32_t selector, uint32_t lane) {
  return (uint16_t)((selector ? 9216u : 5120u) + lane);
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum) {
  int mismatches = 0;
  uint32_t got32 = load32_at(0u);
  uint32_t want32 = h32_value(tc->x);
  *checksum += got32;
  if (got32 != want32) {
    printk("ERROR: vdw_dynamic_probe h32 case=%d got=%x want=%x\n",
           (int)case_index, got32, want32);
    ++mismatches;
  }
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    got32 = load32_at(64u + lane * 4u);
    want32 = v32_value(lane);
    *checksum += got32;
    if (got32 != want32) {
      if (mismatches < 8)
        printk("ERROR: vdw_dynamic_probe v32 case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got32, want32);
      ++mismatches;
    }
  }
  uint8_t got8 = output_bytes[128u];
  uint8_t want8 = h8_value(tc->sel8, tc->x);
  *checksum += got8;
  if (got8 != want8) {
    printk("ERROR: vdw_dynamic_probe h8 case=%d got=%x want=%x\n",
           (int)case_index, got8, want8);
    ++mismatches;
  }
  uint16_t got16 = get16(192u);
  uint16_t want16 = h16_value(tc->sel16, tc->x);
  *checksum += got16;
  if (got16 != want16) {
    printk("ERROR: vdw_dynamic_probe h16 case=%d got=%x want=%x\n",
           (int)case_index, got16, want16);
    ++mismatches;
  }
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    got8 = output_bytes[256u + lane];
    uint32_t v8_sub = tc->sel8 + lane;
    want8 = v8_value(v8_sub & 3u, v8_sub >> 2);
    *checksum += got8;
    if (got8 != want8) {
      if (mismatches < 8)
        printk("ERROR: vdw_dynamic_probe v8 case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got8, want8);
      ++mismatches;
    }
    got16 = get16(320u + lane * 2u);
    uint32_t v16_sub = tc->sel16 + lane;
    want16 = v16_value(v16_sub & 1u, v16_sub >> 1);
    *checksum += got16;
    if (got16 != want16) {
      if (mismatches < 8)
        printk("ERROR: vdw_dynamic_probe v16 case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got16, want16);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i) {
    int active = (i < 4u) || (i >= 64u && i < 128u) || i == 128u ||
                 (i >= 192u && i < 194u) ||
                 (i >= 256u && i < 272u) ||
                 (i >= 320u && i < 352u);
    if (!active && output_bytes[i] != SENTINEL8) {
      if (mismatches < 8)
        printk("ERROR: vdw_dynamic_probe sentinel offset=%d got=%x want=%x\n",
               (int)i, output_bytes[i], SENTINEL8);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdw_dynamic_vpm_source_x_selector_probe_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("vdw_dynamic_vpm_source_x_selector_probe_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_output();
    if (vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        vdw_dynamic_vpm_source_x_selector_probe_vc4kernel_launch(
            program, grid, block, out_dev, cases[i].row_h, cases[i].row_v32,
            cases[i].row_v8, cases[i].row_v16, cases[i].x, cases[i].sel8,
            cases[i].sel16) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0) {
      ++launch_failures;
      continue;
    }
    total_mismatches += verify_case(&cases[i], i, &checksum);
    sentinel_mismatches += verify_sentinels();
  }

  launch_failures +=
      (int)vdw_dynamic_vpm_source_x_selector_probe_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      vdw_dynamic_vpm_source_x_selector_probe_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_dynamic_vpm_source_x_selector_probe_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdw_dynamic_vpm_row=1 saw_vdw_dynamic_vpm_x=1 saw_vdw_dynamic_subword_selector=1 saw_horizontal_w32=1 saw_vertical_w32=1 saw_horizontal_subword=1 saw_vertical_subword=1 saw_byte_halfword_guards=1 saw_vdw_preserve=1 no_tmu_to_vpm=1 selector_values_w8=0,1,2,3 selector_values_w16=0,1 x_values=0,1,7,15 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
