#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define IN_ROWS 16u
#define IN_COLS 16u
#define OUT_ROWS 2u
#define OUT_COLS 16u
#define IN_WORDS (IN_ROWS * IN_COLS)
#define OUT_WORDS (OUT_ROWS * OUT_COLS)
#define GUARD 32u
#define SENTINEL 0x5a5a5a5au
#define BASE 0x43000000u

static uint32_t input_values[IN_WORDS];
static uint32_t output_values[OUT_WORDS + GUARD];

static uint32_t pattern(uint32_t row, uint32_t col) {
  return BASE | (row << 8) | col;
}

static void fill_buffers(void) {
  for (uint32_t r = 0; r < IN_ROWS; ++r)
    for (uint32_t c = 0; c < IN_COLS; ++c)
      input_values[r * IN_COLS + c] = pattern(r, c);
  for (uint32_t i = 0; i < OUT_WORDS + GUARD; ++i)
    output_values[i] = SENTINEL;
}

static uint32_t expected(uint32_t row, uint32_t lane) {
  uint32_t col = row == 0u ? 3u : 11u;
  return pattern(lane, col);
}

static int verify(uint32_t *checksum) {
  int mismatches = 0;
  *checksum = 0;
  for (uint32_t r = 0; r < OUT_ROWS; ++r) {
    for (uint32_t c = 0; c < OUT_COLS; ++c) {
      uint32_t got = output_values[r * OUT_COLS + c];
      uint32_t want = expected(r, c);
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: vdr_vdw_v row=%d lane=%d got=%x want=%x\n",
                 (int)r, (int)c, got, want);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_guard(void) {
  int mismatches = 0;
  for (uint32_t i = OUT_WORDS; i < OUT_WORDS + GUARD; ++i)
    if (output_values[i] != SENTINEL)
      ++mismatches;
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_vdw_vertical_32_roundtrip_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, IN_WORDS * sizeof(uint32_t)) < 0 ||
      vc4_m2_malloc(program, &out_dev, (OUT_WORDS + GUARD) * sizeof(uint32_t)) < 0)
    panic("vdr_vdw_vertical_32_roundtrip_vc4kernel allocation failed");
  fill_buffers();
  int launch_failures = 0, total_mismatches = 0, sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  if (vc4_m2_copy_htod(program, in_dev, input_values, IN_WORDS * sizeof(uint32_t)) < 0 ||
      vc4_m2_copy_htod(program, out_dev, output_values, (OUT_WORDS + GUARD) * sizeof(uint32_t)) < 0 ||
      vdr_vdw_vertical_32_roundtrip_vc4kernel_launch(program, grid, block, in_dev, out_dev) < 0 ||
      vc4_m2_copy_dtoh(program, output_values, out_dev, (OUT_WORDS + GUARD) * sizeof(uint32_t)) < 0) {
    ++launch_failures;
  } else {
    total_mismatches = verify(&checksum);
    sentinel_mismatches = verify_guard();
  }
  launch_failures += (int)vdr_vdw_vertical_32_roundtrip_vc4kernel_runtime_launch_failures();
  uint32_t launches = vdr_vdw_vertical_32_roundtrip_vc4kernel_runtime_launches();
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_vdw_vertical_32_roundtrip_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d rows=%d cols=%d checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)OUT_ROWS, (int)OUT_COLS, checksum, launches, timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
