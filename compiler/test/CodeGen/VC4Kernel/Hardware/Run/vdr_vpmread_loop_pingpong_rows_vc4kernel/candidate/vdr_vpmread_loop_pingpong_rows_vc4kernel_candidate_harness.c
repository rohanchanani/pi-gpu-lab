#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ROWS 4u
#define COLS 16u
#define INPUT_WORDS (ROWS * COLS)
#define OUTPUT_WORDS COLS
#define GUARD 32u
#define SENTINEL 0x5a5a5a5au
#define BASE 0x74000000u

static const uint32_t active_cols_cases[] = {16u, 1u};
static uint32_t input_values[INPUT_WORDS];
static uint32_t output_values[OUTPUT_WORDS + GUARD];

static uint32_t pattern(uint32_t row, uint32_t col) {
  return BASE | (row << 8) | col;
}

static void fill_input(void) {
  for (uint32_t r = 0; r < ROWS; ++r)
    for (uint32_t c = 0; c < COLS; ++c)
      input_values[r * COLS + c] = pattern(r, c);
}

static void fill_output(void) {
  for (uint32_t i = 0; i < OUTPUT_WORDS + GUARD; ++i)
    output_values[i] = SENTINEL;
}

static uint32_t expected(uint32_t active_cols, uint32_t col) {
  uint32_t sum = 0;
  for (uint32_t r = 0; r < ROWS; ++r)
    if ((r % 2u) == 0u || col < active_cols)
      sum += pattern(r, col);
  return sum;
}

static int verify_case(uint32_t active_cols, uint32_t *checksum) {
  int mismatches = 0;
  for (uint32_t c = 0; c < COLS; ++c) {
    uint32_t got = output_values[c];
    uint32_t want = expected(active_cols, c);
    *checksum += got;
    if (got != want) {
      if (mismatches < 8)
        printk("ERROR: pingpong_rows active_cols=%d col=%d got=%x want=%x\n",
               (int)active_cols, (int)c, got, want);
      ++mismatches;
    }
  }
  return mismatches;
}

static int verify_guard(uint32_t active_cols) {
  int mismatches = 0;
  for (uint32_t i = OUTPUT_WORDS; i < OUTPUT_WORDS + GUARD; ++i) {
    if (output_values[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: pingpong_rows active_cols=%d guard=%d got=%x want=%x\n",
               (int)active_cols, (int)i, output_values[i], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_vpmread_loop_pingpong_rows_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, INPUT_WORDS * sizeof(uint32_t)) < 0 ||
      vc4_m2_malloc(program, &out_dev, (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0)
    panic("vdr_vpmread_loop_pingpong_rows_vc4kernel allocation failed");

  fill_input();
  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  const uint32_t cases =
      sizeof(active_cols_cases) / sizeof(active_cols_cases[0]);

  for (uint32_t i = 0; i < cases; ++i) {
    uint32_t active_cols = active_cols_cases[i];
    fill_output();
    if (vc4_m2_copy_htod(program, in_dev, input_values, INPUT_WORDS * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_values,
                         (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0 ||
        vdr_vpmread_loop_pingpong_rows_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, active_cols,
            COLS * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_dtoh(program, output_values, out_dev,
                         (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0) {
      ++launch_failures;
      continue;
    }
    total_mismatches += verify_case(active_cols, &checksum);
    sentinel_mismatches += verify_guard(active_cols);
  }

  launch_failures +=
      (int)vdr_vpmread_loop_pingpong_rows_vc4kernel_runtime_launch_failures();
  uint32_t launches = vdr_vpmread_loop_pingpong_rows_vc4kernel_runtime_launches();
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == cases)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_vpmread_loop_pingpong_rows_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_cols_cases=16,1 pingpong_rows=0-1,2-3 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)cases, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
