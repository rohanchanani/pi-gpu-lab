#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ROWS 4u
#define COLS 16u
#define ACTIVE_COLS 8u
#define INPUT_STRIDE_WORDS 32u
#define INPUT_WORDS (ROWS * INPUT_STRIDE_WORDS)
#define OUTPUT_WORDS (ROWS * COLS)
#define GUARD 32u
#define SENTINEL 0x5a5a5a5au
#define BASE 0x44000000u

static uint32_t input_values[INPUT_WORDS];
static uint32_t output_values[OUTPUT_WORDS + GUARD];

static uint32_t pattern(uint32_t row, uint32_t col) {
  return BASE | (row << 8) | col;
}

static void fill_buffers(void) {
  for (uint32_t r = 0; r < ROWS; ++r)
    for (uint32_t c = 0; c < INPUT_STRIDE_WORDS; ++c)
      input_values[r * INPUT_STRIDE_WORDS + c] = pattern(r, c);
  for (uint32_t i = 0; i < OUTPUT_WORDS + GUARD; ++i)
    output_values[i] = SENTINEL;
}

static uint32_t expected(uint32_t row, uint32_t col) {
  return col < ACTIVE_COLS ? pattern(row, col) : SENTINEL;
}

static int verify(uint32_t *checksum) {
  int mismatches = 0;
  *checksum = 0;
  for (uint32_t r = 0; r < ROWS; ++r) {
    for (uint32_t c = 0; c < COLS; ++c) {
      uint32_t got = output_values[r * COLS + c];
      uint32_t want = expected(r, c);
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: vdr_vdw_stride row=%d col=%d got=%x want=%x\n",
                 (int)r, (int)c, got, want);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_guard(void) {
  int mismatches = 0;
  for (uint32_t i = OUTPUT_WORDS; i < OUTPUT_WORDS + GUARD; ++i)
    if (output_values[i] != SENTINEL)
      ++mismatches;
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_vdw_strided_pitch_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, INPUT_WORDS * sizeof(uint32_t)) < 0 ||
      vc4_m2_malloc(program, &out_dev, (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0)
    panic("vdr_vdw_strided_pitch_vc4kernel allocation failed");
  fill_buffers();
  int launch_failures = 0, total_mismatches = 0, sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  if (vc4_m2_copy_htod(program, in_dev, input_values, INPUT_WORDS * sizeof(uint32_t)) < 0 ||
      vc4_m2_copy_htod(program, out_dev, output_values, (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0 ||
      vdr_vdw_strided_pitch_vc4kernel_launch(program, grid, block, in_dev, out_dev) < 0 ||
      vc4_m2_copy_dtoh(program, output_values, out_dev, (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0) {
    ++launch_failures;
  } else {
    total_mismatches = verify(&checksum);
    sentinel_mismatches = verify_guard();
  }
  launch_failures += (int)vdr_vdw_strided_pitch_vc4kernel_runtime_launch_failures();
  uint32_t launches = vdr_vdw_strided_pitch_vc4kernel_runtime_launches();
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_vdw_strided_pitch_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d rows=%d cols=%d active_cols=%d checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)ROWS, (int)COLS, (int)ACTIVE_COLS, checksum, launches,
         timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
