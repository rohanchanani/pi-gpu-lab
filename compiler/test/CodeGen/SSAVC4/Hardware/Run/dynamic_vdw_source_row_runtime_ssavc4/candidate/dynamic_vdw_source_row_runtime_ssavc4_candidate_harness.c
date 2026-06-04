#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 64u
#define BAND_ROWS 16u
#define COLS 16u
#define CASE_ROWS 8u
#define CASE_COLS 4u
#define GUARD_WORDS 16u
#define SENTINEL 0x6d0f00d5u

static uint32_t input0[BAND_ROWS * COLS];
static uint32_t input16[BAND_ROWS * COLS];
static uint32_t input32[BAND_ROWS * COLS];
static uint32_t input48[BAND_ROWS * COLS];
static uint32_t output_values[GUARD_WORDS + COLS + GUARD_WORDS];
static const uint32_t source_rows[CASE_ROWS] = {0u,  1u,  15u, 16u,
                                                31u, 32u, 48u, 63u};
static const uint32_t active_cols_cases[CASE_COLS] = {0u, 1u, 7u, 16u};

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x91000000u | (row << 8) | col;
}

static void fill_band(uint32_t *values, uint32_t base_row) {
  for (uint32_t row = 0; row < BAND_ROWS; ++row)
    for (uint32_t col = 0; col < COLS; ++col)
      values[row * COLS + col] = pattern(base_row + row, col);
}

static void fill_output(void) {
  for (uint32_t i = 0; i < sizeof(output_values) / sizeof(output_values[0]);
       ++i)
    output_values[i] = SENTINEL;
}

static void check_output(uint32_t source_row, uint32_t active_cols,
                         uint32_t *total_mismatches,
                         uint32_t *sentinel_mismatches, uint32_t *checksum) {
  uint32_t expected_cols = active_cols < COLS ? active_cols : COLS;
  for (uint32_t i = 0; i < sizeof(output_values) / sizeof(output_values[0]);
       ++i) {
    uint32_t expected = SENTINEL;
    if (i >= GUARD_WORDS && i < GUARD_WORDS + expected_cols)
      expected = pattern(source_row, i - GUARD_WORDS);
    uint32_t actual = output_values[i];
    if (expected != SENTINEL)
      *checksum += actual;
    if (actual != expected) {
      if (expected == SENTINEL) {
        if (*sentinel_mismatches < 8)
          printk("ERROR: dynamic_vdw_source_row_runtime source_row=%d active_cols=%d sentinel_index=%d actual=%x expected=%x\n",
                 (int)source_row, (int)active_cols, (int)i, actual,
                 SENTINEL);
        ++*sentinel_mismatches;
      } else {
        if (*total_mismatches < 8)
          printk("ERROR: dynamic_vdw_source_row_runtime source_row=%d active_cols=%d col=%d actual=%x expected=%x\n",
                 (int)source_row, (int)active_cols, (int)(i - GUARD_WORDS),
                 actual, expected);
        ++*total_mismatches;
      }
    }
  }
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input0_dev = 0;
  vc4_deviceptr_t input16_dev = 0;
  vc4_deviceptr_t input32_dev = 0;
  vc4_deviceptr_t input48_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdw_source_row_runtime_ssavc4 program create failed");
  if (vc4Malloc(program, &input0_dev, sizeof(input0)) < 0 ||
      vc4Malloc(program, &input16_dev, sizeof(input16)) < 0 ||
      vc4Malloc(program, &input32_dev, sizeof(input32)) < 0 ||
      vc4Malloc(program, &input48_dev, sizeof(input48)) < 0 ||
      vc4Malloc(program, &output_dev, sizeof(output_values)) < 0)
    panic("dynamic_vdw_source_row_runtime_ssavc4 allocation failed");

  fill_band(input0, 0u);
  fill_band(input16, 16u);
  fill_band(input32, 32u);
  fill_band(input48, 48u);

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t r = 0; r < CASE_ROWS; ++r) {
    uint32_t source_row = source_rows[r];
    for (uint32_t c = 0; c < CASE_COLS; ++c) {
      uint32_t active_cols = active_cols_cases[c];
      fill_output();
      if (vc4MemcpyHtoD(program, input0_dev, input0, sizeof(input0)) < 0 ||
          vc4MemcpyHtoD(program, input16_dev, input16, sizeof(input16)) < 0 ||
          vc4MemcpyHtoD(program, input32_dev, input32, sizeof(input32)) < 0 ||
          vc4MemcpyHtoD(program, input48_dev, input48, sizeof(input48)) < 0 ||
          vc4MemcpyHtoD(program, output_dev, output_values,
                        sizeof(output_values)) < 0 ||
          dynamic_vdw_source_row_runtime_ssavc4_launch(
              program, grid, block, input0_dev, input16_dev, input32_dev,
              input48_dev, output_dev + GUARD_WORDS * sizeof(uint32_t),
              source_row, active_cols) < 0 ||
          vc4MemcpyDtoH(program, output_values, output_dev,
                        sizeof(output_values)) < 0) {
        printk("ERROR: dynamic_vdw_source_row_runtime launch/copy failed source_row=%d active_cols=%d\n",
               (int)source_row, (int)active_cols);
        ++launch_failures;
      } else {
        check_output(source_row, active_cols, &total_mismatches,
                     &sentinel_mismatches, &checksum);
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_vdw_source_row_runtime_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdw_source_row_runtime_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdw_source_row_runtime_ssavc4 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)(CASE_ROWS * CASE_COLS), (int)total_mismatches,
         (int)sentinel_mismatches, (int)launch_failures,
         (int)runtime_launches, checksum);

  vc4Free(program, input0_dev);
  vc4Free(program, input16_dev);
  vc4Free(program, input32_dev);
  vc4Free(program, input48_dev);
  vc4Free(program, output_dev);
}
