#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 16u
#define COLS 16u
#define STATIC_COLS 5u
#define MAX_PITCH_WORDS 65u
#define IN_WORDS ((ROWS - 1u) * MAX_PITCH_WORDS + COLS)
#define OUT_WORDS (ROWS * COLS)
#define GUARD_WORDS 32u
#define SENTINEL 0x9234cafeu

static uint32_t input_values[IN_WORDS + GUARD_WORDS];
static uint32_t output_dynamic[OUT_WORDS + GUARD_WORDS];
static uint32_t output_static5[OUT_WORDS + GUARD_WORDS];
static const uint32_t active_rows_cases[] = {0u, 1u, 7u, 16u};
static const uint32_t active_cols_cases[] = {0u, 1u, 5u, 16u};
static const uint32_t pitch_bytes_cases[] = {64u, 68u, 260u};

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x93000000u | (row << 8) | col;
}

static void fill_input(uint32_t pitch_words) {
  for (uint32_t i = 0; i < IN_WORDS + GUARD_WORDS; ++i)
    input_values[i] = SENTINEL;
  for (uint32_t row = 0; row < ROWS; ++row)
    for (uint32_t col = 0; col < COLS; ++col)
      input_values[row * pitch_words + col] = pattern(row, col);
}

static void fill_output(uint32_t *output) {
  for (uint32_t i = 0; i < OUT_WORDS + GUARD_WORDS; ++i)
    output[i] = SENTINEL;
}

static void verify_output(const char *label, const uint32_t *out_values,
                          uint32_t pitch_bytes, uint32_t active_rows,
                          uint32_t active_cols, uint32_t expected_cols,
                          uint32_t *total_mismatches,
                          uint32_t *zero_mismatches,
                          uint32_t *sentinel_mismatches,
                          uint32_t *checksum) {
  uint32_t expected_rows = active_rows < ROWS ? active_rows : ROWS;
  for (uint32_t row = 0; row < ROWS; ++row) {
    for (uint32_t col = 0; col < COLS; ++col) {
      uint32_t actual = out_values[row * COLS + col];
      uint32_t expected =
          row < expected_rows && col < expected_cols ? pattern(row, col) : 0u;
      if (expected != 0u)
        *checksum += actual;
      if (actual != expected) {
        if (expected != 0u) {
          if (*total_mismatches < 8)
            printk("ERROR: dynamic_vdr_true_rect %s pitch_bytes=%d active_rows=%d active_cols=%d row=%d col=%d actual=%x expected=%x\n",
                   label, (int)pitch_bytes, (int)active_rows,
                   (int)active_cols, (int)row, (int)col, actual, expected);
          ++*total_mismatches;
        } else {
          if (*zero_mismatches < 8)
            printk("ERROR: dynamic_vdr_true_rect %s pitch_bytes=%d active_rows=%d active_cols=%d zero_row=%d zero_col=%d actual=%x expected=0\n",
                   label, (int)pitch_bytes, (int)active_rows,
                   (int)active_cols, (int)row, (int)col, actual);
          ++*zero_mismatches;
        }
      }
    }
  }
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    uint32_t actual = out_values[OUT_WORDS + i];
    if (actual != SENTINEL) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: dynamic_vdr_true_rect %s pitch_bytes=%d active_rows=%d active_cols=%d guard=%d actual=%x expected=%x\n",
               label, (int)pitch_bytes, (int)active_rows, (int)active_cols,
               (int)i, actual, SENTINEL);
      ++*sentinel_mismatches;
    }
  }
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dynamic_dev = 0;
  vc4_deviceptr_t output_static5_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t zero_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdr_true_rect_dynamic_shape_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &output_dynamic_dev, sizeof(output_dynamic)) < 0 ||
      vc4Malloc(program, &output_static5_dev, sizeof(output_static5)) < 0)
    panic("dynamic_vdr_true_rect_dynamic_shape_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t p = 0;
       p < sizeof(pitch_bytes_cases) / sizeof(pitch_bytes_cases[0]); ++p) {
    uint32_t pitch_bytes = pitch_bytes_cases[p];
    uint32_t pitch_words = pitch_bytes / 4u;
    for (uint32_t r = 0;
         r < sizeof(active_rows_cases) / sizeof(active_rows_cases[0]); ++r) {
      uint32_t active_rows = active_rows_cases[r];
      for (uint32_t c = 0;
           c < sizeof(active_cols_cases) / sizeof(active_cols_cases[0]); ++c) {
        uint32_t active_cols = active_cols_cases[c];
        uint32_t expected_cols = active_cols < COLS ? active_cols : COLS;

        fill_input(pitch_words);
        fill_output(output_dynamic);
        fill_output(output_static5);
        if (vc4MemcpyHtoD(program, input_dev, input_values,
                          sizeof(input_values)) < 0 ||
            vc4MemcpyHtoD(program, output_dynamic_dev, output_dynamic,
                          sizeof(output_dynamic)) < 0 ||
            vc4MemcpyHtoD(program, output_static5_dev, output_static5,
                          sizeof(output_static5)) < 0 ||
            dynamic_vdr_true_rect_dynamic_shape_ssavc4_launch(
                program, grid, block, input_dev, output_dynamic_dev,
                output_static5_dev, active_rows, active_cols, pitch_bytes) < 0 ||
            vc4MemcpyDtoH(program, output_dynamic, output_dynamic_dev,
                          sizeof(output_dynamic)) < 0 ||
            vc4MemcpyDtoH(program, output_static5, output_static5_dev,
                          sizeof(output_static5)) < 0) {
          printk("ERROR: dynamic_vdr_true_rect launch/copy failed pitch_bytes=%d active_rows=%d active_cols=%d\n",
                 (int)pitch_bytes, (int)active_rows, (int)active_cols);
          ++launch_failures;
        }

        verify_output("dynamic", output_dynamic, pitch_bytes, active_rows,
                      active_cols, expected_cols, &total_mismatches,
                      &zero_mismatches, &sentinel_mismatches, &checksum);
        verify_output("static5", output_static5, pitch_bytes, active_rows,
                      active_cols, STATIC_COLS, &total_mismatches,
                      &zero_mismatches, &sentinel_mismatches, &checksum);
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_vdr_true_rect_dynamic_shape_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdr_true_rect_dynamic_shape_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && zero_mismatches == 0 &&
       sentinel_mismatches == 0 && launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdr_true_rect_dynamic_shape_ssavc4 status=%s total_mismatches=%d zero_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)zero_mismatches,
         (int)sentinel_mismatches, (int)launch_failures,
         (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, output_dynamic_dev);
  vc4Free(program, output_static5_dev);
}
