#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 2u
#define COLS 16u
#define MAX_PITCH_WORDS 65u
#define OUT_WORDS (ROWS * COLS)
#define GUARD_WORDS 16u
#define SENTINEL 0x8accee31u

static uint32_t input_values[MAX_PITCH_WORDS + COLS + GUARD_WORDS];
static uint32_t output_values[OUT_WORDS + GUARD_WORDS];
static const uint32_t active_rows_cases[] = {0u, 1u, 2u, 3u};
static const uint32_t active_cols_cases[] = {0u, 1u, 8u, 19u};
static const uint32_t pitch_bytes_cases[] = {64u, 76u, 124u, 260u};

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x63000000u | (row << 8) | col;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t zero_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdr_runtime_pitch_rows_cols_zero_fill_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &output_dev, sizeof(output_values)) < 0)
    panic("dynamic_vdr_runtime_pitch_rows_cols_zero_fill_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t p = 0;
       p < sizeof(pitch_bytes_cases) / sizeof(pitch_bytes_cases[0]); ++p) {
    uint32_t pitch_bytes = pitch_bytes_cases[p];
    uint32_t pitch_words = pitch_bytes / 4u;
    for (uint32_t r = 0;
         r < sizeof(active_rows_cases) / sizeof(active_rows_cases[0]); ++r) {
      uint32_t active_rows = active_rows_cases[r];
      uint32_t expected_rows = active_rows < ROWS ? active_rows : ROWS;
      for (uint32_t c = 0;
           c < sizeof(active_cols_cases) / sizeof(active_cols_cases[0]); ++c) {
        uint32_t active_cols = active_cols_cases[c];
        uint32_t expected_cols = active_cols < COLS ? active_cols : COLS;
        for (uint32_t i = 0; i < MAX_PITCH_WORDS + COLS + GUARD_WORDS; ++i)
          input_values[i] = SENTINEL;
        for (uint32_t i = 0; i < OUT_WORDS + GUARD_WORDS; ++i)
          output_values[i] = SENTINEL;
        for (uint32_t row = 0; row < ROWS; ++row)
          for (uint32_t col = 0; col < COLS; ++col)
            input_values[row * pitch_words + col] = pattern(row, col);

        if (vc4MemcpyHtoD(program, input_dev, input_values,
                          sizeof(input_values)) < 0 ||
            vc4MemcpyHtoD(program, output_dev, output_values,
                          sizeof(output_values)) < 0 ||
            dynamic_vdr_runtime_pitch_rows_cols_zero_fill_ssavc4_launch(
                program, grid, block, input_dev, output_dev, active_rows,
                active_cols, pitch_bytes) < 0 ||
            vc4MemcpyDtoH(program, output_values, output_dev,
                          sizeof(output_values)) < 0) {
          printk("ERROR: dynamic_vdr_runtime_pitch_rows_cols launch/copy failed pitch_bytes=%d active_rows=%d active_cols=%d\n",
                 (int)pitch_bytes, (int)active_rows, (int)active_cols);
          ++launch_failures;
        }

        for (uint32_t row = 0; row < ROWS; ++row) {
          for (uint32_t col = 0; col < COLS; ++col) {
            uint32_t index = row * COLS + col;
            uint32_t actual = output_values[index];
            uint32_t expected =
                row < expected_rows && col < expected_cols
                    ? pattern(row, col)
                    : 0u;
            if (row < expected_rows && col < expected_cols)
              checksum += actual;
            if (actual != expected) {
              if (expected != 0u) {
                if (total_mismatches < 8)
                  printk("ERROR: dynamic_vdr_runtime_pitch_rows_cols pitch_bytes=%d active_rows=%d active_cols=%d row=%d col=%d actual=%x expected=%x\n",
                         (int)pitch_bytes, (int)active_rows,
                         (int)active_cols, (int)row, (int)col, actual,
                         expected);
                ++total_mismatches;
              } else {
                if (zero_mismatches < 8)
                  printk("ERROR: dynamic_vdr_runtime_pitch_rows_cols pitch_bytes=%d active_rows=%d active_cols=%d zero_row=%d zero_col=%d actual=%x expected=0\n",
                         (int)pitch_bytes, (int)active_rows,
                         (int)active_cols, (int)row, (int)col, actual);
                ++zero_mismatches;
              }
            }
          }
        }
        for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
          uint32_t actual = output_values[OUT_WORDS + i];
          if (actual != SENTINEL) {
            if (sentinel_mismatches < 8)
              printk("ERROR: dynamic_vdr_runtime_pitch_rows_cols pitch_bytes=%d active_rows=%d active_cols=%d guard=%d actual=%x expected=%x\n",
                     (int)pitch_bytes, (int)active_rows, (int)active_cols,
                     (int)i, actual, SENTINEL);
            ++sentinel_mismatches;
          }
        }
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_vdr_runtime_pitch_rows_cols_zero_fill_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdr_runtime_pitch_rows_cols_zero_fill_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && zero_mismatches == 0 &&
       sentinel_mismatches == 0 && launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdr_runtime_pitch_rows_cols_zero_fill_ssavc4 status=%s total_mismatches=%d zero_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)zero_mismatches,
         (int)sentinel_mismatches, (int)launch_failures,
         (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
