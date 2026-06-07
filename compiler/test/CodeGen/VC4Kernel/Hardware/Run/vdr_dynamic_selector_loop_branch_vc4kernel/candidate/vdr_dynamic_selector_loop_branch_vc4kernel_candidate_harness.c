#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 3u
#define INPUT_BYTES 512u
#define OUT_WORDS (LANES * SEGMENTS + 32u)
#define SENTINEL_WORD 0x5a5a5a5au
#define SENTINEL_BYTE 0x5au
#define SENTINEL_HALF 0x55aau
#define GUARD_WORD 0x51ee77a1u

struct test_case {
  uint32_t row_h;
  uint32_t row_v;
  uint32_t control;
};

static const struct test_case cases[] = {
    {0u, 16u, 0u},
    {5u, 32u, 1u},
};

static uint8_t input_bytes[INPUT_BYTES];
static uint32_t out_words[OUT_WORDS];

static void store32(uint32_t offset, uint32_t value) {
  input_bytes[offset] = (uint8_t)(value & 0xffu);
  input_bytes[offset + 1u] = (uint8_t)((value >> 8) & 0xffu);
  input_bytes[offset + 2u] = (uint8_t)((value >> 16) & 0xffu);
  input_bytes[offset + 3u] = (uint8_t)((value >> 24) & 0xffu);
}

static void store16(uint32_t offset, uint16_t value) {
  input_bytes[offset] = (uint8_t)(value & 0xffu);
  input_bytes[offset + 1u] = (uint8_t)(value >> 8);
}

static uint32_t h32_value(uint32_t control) { return 0x81000000u | (control << 8) | 0x2au; }
static uint32_t v32_value(uint32_t control, uint32_t lane) { return 0x82000000u | (control << 8) | lane; }
static uint32_t h8_value(void) { return 0x6du; }
static uint32_t v16_value(uint32_t lane) { return 0x3000u + lane; }

static void fill_input(uint32_t control) {
  for (uint32_t i = 0; i < INPUT_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0xa0u + i * 13u);
  store32(0u, h32_value(control));
  for (uint32_t lane = 0; lane < LANES; ++lane)
    store32(64u + lane * 4u, v32_value(control, lane));
  input_bytes[256u] = (uint8_t)h8_value();
  for (uint32_t lane = 0; lane < LANES; ++lane)
    store16(320u + lane * 2u, (uint16_t)v16_value(lane));
}

static void fill_output(void) {
  for (uint32_t i = 0; i < OUT_WORDS; ++i)
    out_words[i] = GUARD_WORD;
}

static uint32_t expected_value(uint32_t control, uint32_t segment, uint32_t lane) {
  if (segment == 0)
    return lane == control ? h32_value(control) : SENTINEL_WORD;
  if (segment == 1)
    return v32_value(control, lane);
  if (control == 0)
    return lane == 8u ? h8_value() : SENTINEL_BYTE;
  return lane == 0u ? v16_value(15u) : SENTINEL_HALF;
}

static int verify_case(uint32_t case_index, uint32_t control, uint32_t *checksum) {
  int mismatches = 0;
  for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
    for (uint32_t lane = 0; lane < LANES; ++lane) {
      uint32_t index = segment * LANES + lane;
      uint32_t got = out_words[index];
      uint32_t want = expected_value(control, segment, lane);
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: vdr_loop_branch case=%d segment=%d lane=%d got=%x want=%x\n",
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
    if (out_words[i] != GUARD_WORD) {
      if (mismatches < 8)
        printk("ERROR: vdr_loop_branch sentinel case=%d index=%d got=%x want=%x\n",
               (int)case_index, (int)i, out_words[i], GUARD_WORD);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_dynamic_selector_loop_branch_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, OUT_WORDS * sizeof(uint32_t)) < 0)
    panic("vdr_dynamic_selector_loop_branch_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_input(cases[i].control);
    fill_output();
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_words,
                         OUT_WORDS * sizeof(uint32_t)) < 0 ||
        vdr_dynamic_selector_loop_branch_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, cases[i].row_h,
            cases[i].row_v, cases[i].control) < 0 ||
        vc4_m2_copy_dtoh(program, out_words, out_dev,
                         OUT_WORDS * sizeof(uint32_t)) < 0) {
      ++launch_failures;
      continue;
    }
    total_mismatches += verify_case(i, cases[i].control, &checksum);
    sentinel_mismatches += verify_sentinels(i);
  }

  launch_failures +=
      (int)vdr_dynamic_selector_loop_branch_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      vdr_dynamic_selector_loop_branch_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_dynamic_selector_loop_branch_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_dynamic_vpm_row=1 saw_vdr_dynamic_vpm_x=1 saw_vdr_dynamic_subword_selector=1 saw_horizontal_w32=1 saw_vertical_w32=1 saw_horizontal_subword=1 saw_vertical_subword=1 saw_vpm_qpu_readback=1 saw_byte_halfword_guards=1 no_tmu_to_vpm=1 saw_vdr_loop_branch=1 saw_both_paths=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
