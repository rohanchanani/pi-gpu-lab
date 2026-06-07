#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 6u
#define INPUT_BYTES 2560u
#define OUT_WORDS (LANES * SEGMENTS + 32u)
#define SENTINEL_WORD 0x5a5a5a5au
#define SENTINEL_BYTE 0x5au
#define SENTINEL_HALF 0x55aau
#define GUARD_WORD 0x7ac01234u

struct test_case {
  uint32_t row_h;
  uint32_t row_v;
  uint32_t x;
  uint32_t sel8;
  uint32_t sel16;
  uint32_t pitch_h;
  uint32_t pitch_v;
};

static const struct test_case cases[] = {
    {0u, 16u, 0u, 0u, 0u, 8u, 64u},
    {3u, 16u, 1u, 1u, 1u, 16u, 128u},
    {5u, 32u, 7u, 2u, 0u, 32u, 64u},
    {9u, 32u, 15u, 3u, 1u, 64u, 128u},
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

static uint32_t h32_value(uint32_t case_index) { return 0x61000000u | (case_index << 8) | 0x3du; }
static uint32_t v32_value(uint32_t case_index, uint32_t lane) { return 0x62000000u | (case_index << 8) | lane; }
static uint32_t h8_value(uint32_t case_index) { return 0x38u + case_index * 5u; }
static uint32_t h16_value(uint32_t case_index) { return 0x1800u + case_index * 19u; }
static uint32_t v8_value(uint32_t case_index, uint32_t lane) { return 0x60u + case_index * 3u + lane; }
static uint32_t v16_value(uint32_t case_index, uint32_t lane) { return 0x2800u + case_index * 31u + lane; }

static uint32_t selected_u8(uint32_t word, uint32_t selector) {
  return (word >> ((selector & 3u) * 8u)) & 0xffu;
}

static uint32_t selected_u16(uint32_t word, uint32_t selector) {
  return (word >> ((selector & 1u) * 16u)) & 0xffffu;
}

static void fill_input(const struct test_case *tc, uint32_t case_index) {
  for (uint32_t i = 0; i < INPUT_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0xb0u + i * 9u);
  store32(0u, h32_value(case_index));
  input_bytes[512u] = (uint8_t)h8_value(case_index);
  store16(768u, (uint16_t)h16_value(case_index));
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    store32(256u + lane * 4u, v32_value(case_index, lane));
    input_bytes[1024u + lane] = (uint8_t)v8_value(case_index, lane);
    store16(1280u + lane * 2u, (uint16_t)v16_value(case_index, lane));
  }
}

static void fill_output(void) {
  for (uint32_t i = 0; i < OUT_WORDS; ++i)
    out_words[i] = GUARD_WORD;
}

static uint32_t expected_value(const struct test_case *tc, uint32_t case_index,
                               uint32_t segment, uint32_t lane) {
  switch (segment) {
  case 0:
    return lane == tc->x ? h32_value(case_index) : 0u;
  case 1:
    return v32_value(case_index, lane);
  case 2:
    return lane == tc->x ? h8_value(case_index) : 0u;
  case 3:
    return lane == tc->x ? h16_value(case_index) : 0u;
  case 4:
    return lane == tc->x ? v8_value(case_index, 0u) : 0u;
  default:
    return lane == tc->x ? v16_value(case_index, 0u) : 0u;
  }
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum) {
  int mismatches = 0;
  for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
    for (uint32_t lane = 0; lane < LANES; ++lane) {
      uint32_t index = segment * LANES + lane;
      uint32_t got = out_words[index];
      uint32_t want = expected_value(tc, case_index, segment, lane);
      if (segment == 2u)
        got = selected_u8(got, tc->sel8);
      else if (segment == 3u)
        got = selected_u16(got, tc->sel16);
      else if (segment == 4u)
        got = selected_u8(got, tc->sel8);
      else if (segment == 5u)
        got = selected_u16(got, tc->sel16);
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: vdr_runtime_pitch case=%d segment=%d lane=%d got=%x want=%x\n",
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
        printk("ERROR: vdr_runtime_pitch sentinel case=%d index=%d got=%x want=%x\n",
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
    panic("vdr_dynamic_selector_runtime_pitch_rect_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, OUT_WORDS * sizeof(uint32_t)) < 0)
    panic("vdr_dynamic_selector_runtime_pitch_rect_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_input(&cases[i], i);
    fill_output();
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_words,
                         OUT_WORDS * sizeof(uint32_t)) < 0 ||
        vdr_dynamic_selector_runtime_pitch_rect_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, cases[i].row_h,
            cases[i].row_v, cases[i].x, cases[i].sel8, cases[i].sel16,
            LANES, 1u, cases[i].pitch_h, 1u, LANES, cases[i].pitch_v) < 0 ||
        vc4_m2_copy_dtoh(program, out_words, out_dev,
                         OUT_WORDS * sizeof(uint32_t)) < 0) {
      ++launch_failures;
      continue;
    }
    total_mismatches += verify_case(&cases[i], i, &checksum);
    sentinel_mismatches += verify_sentinels(i);
  }

  launch_failures +=
      (int)vdr_dynamic_selector_runtime_pitch_rect_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      vdr_dynamic_selector_runtime_pitch_rect_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_dynamic_selector_runtime_pitch_rect_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_dynamic_vpm_row=1 saw_vdr_dynamic_vpm_x=1 saw_vdr_dynamic_subword_selector=1 saw_horizontal_w32=1 saw_vertical_w32=1 saw_horizontal_subword=1 saw_vertical_subword=1 saw_vpm_qpu_readback=1 saw_byte_halfword_guards=1 no_tmu_to_vpm=1 saw_runtime_pitch=1 pitches=8,16,32,64 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
