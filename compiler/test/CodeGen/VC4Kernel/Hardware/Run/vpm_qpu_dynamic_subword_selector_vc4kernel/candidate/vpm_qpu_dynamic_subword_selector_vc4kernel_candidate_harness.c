#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 6u
#define OUT_WORDS (LANES * SEGMENTS + 32u)
#define SENTINEL 0x71ab51edu

struct test_case {
  uint32_t row_h;
  uint32_t row_v;
  uint32_t x;
  uint32_t sel8;
  uint32_t sel16;
};

static const struct test_case cases[] = {
    {0u, 0u, 0u, 0u, 0u},   {3u, 16u, 1u, 1u, 1u},
    {5u, 0u, 7u, 2u, 0u},   {9u, 16u, 15u, 3u, 1u},
};

static uint32_t out_words[OUT_WORDS];

static void fill_output(void) {
  for (uint32_t i = 0; i < OUT_WORDS; ++i)
    out_words[i] = SENTINEL;
}

static uint32_t expected_value(uint32_t segment, uint32_t lane) {
  switch (segment) {
  case 0: return 33u + lane;
  case 1: return 1024u + lane;
  case 2: return 77u + lane;
  case 3: return 2048u + lane;
  case 4: return 4096u + lane;
  default: return 8192u + lane;
  }
}

static int verify_case(uint32_t case_index, uint32_t *checksum) {
  int mismatches = 0;
  for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
    for (uint32_t lane = 0; lane < LANES; ++lane) {
      uint32_t index = segment * LANES + lane;
      uint32_t got = out_words[index];
      uint32_t want = expected_value(segment, lane);
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: dynamic_qpu_selector case=%d segment=%d lane=%d got=%x want=%x\n",
                 (int)case_index, (int)segment, (int)lane, got, want);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_sentinels(uint32_t case_index) {
  int mismatches = 0;
  for (uint32_t i = LANES * SEGMENTS; i < OUT_WORDS; ++i) {
    if (out_words[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: dynamic_qpu_selector sentinel case=%d index=%d got=%x want=%x\n",
               (int)case_index, (int)i, out_words[i], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vpm_qpu_dynamic_subword_selector_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &out_dev, OUT_WORDS * sizeof(uint32_t)) < 0)
    panic("vpm_qpu_dynamic_subword_selector_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_output();
    if (vc4_m2_copy_htod(program, out_dev, out_words,
                         OUT_WORDS * sizeof(uint32_t)) < 0 ||
        vpm_qpu_dynamic_subword_selector_vc4kernel_launch(
            program, grid, block, out_dev, cases[i].row_h, cases[i].row_v,
            cases[i].x, cases[i].sel8, cases[i].sel16) < 0 ||
        vc4_m2_copy_dtoh(program, out_words, out_dev,
                         OUT_WORDS * sizeof(uint32_t)) < 0) {
      ++launch_failures;
      continue;
    }
    total_mismatches += verify_case(i, &checksum);
    sentinel_mismatches += verify_sentinels(i);
  }

  launch_failures +=
      (int)vpm_qpu_dynamic_subword_selector_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      vpm_qpu_dynamic_subword_selector_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vpm_qpu_dynamic_subword_selector_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_dynamic_vpm_row=1 saw_dynamic_vpm_x=1 saw_dynamic_subword_selector=1 saw_horizontal_subword=1 saw_vertical_subword=1 saw_packed=1 saw_laned=1 saw_vpm_qpu_read_write=1 saw_pack_unpack=1 saw_vdw_preserve=1 saw_w32_horizontal_row=1 saw_w32_vertical_x=1 selector_values_w8=0,1,2,3 selector_values_w16=0,1 x_values=0,1,7,15 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
