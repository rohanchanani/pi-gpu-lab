#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 2u
#define COLS 16u
#define GUARD_WORDS 16u
#define SENTINEL 0x8accee31u

static uint32_t input_values[ROWS * COLS];
static uint32_t output_values[ROWS * COLS + GUARD_WORDS];

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x61000000u | (row << 8) | col;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  for (uint32_t row = 0; row < ROWS; ++row)
    for (uint32_t col = 0; col < COLS; ++col)
      input_values[row * COLS + col] = pattern(row, col);
  for (uint32_t i = 0; i < ROWS * COLS + GUARD_WORDS; ++i)
    output_values[i] = SENTINEL;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdr_vdw_two_row_staticshape_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &output_dev, sizeof(output_values)) < 0)
    panic("dynamic_vdr_vdw_two_row_staticshape_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  if (vc4MemcpyHtoD(program, input_dev, input_values, sizeof(input_values)) < 0 ||
      vc4MemcpyHtoD(program, output_dev, output_values, sizeof(output_values)) < 0 ||
      dynamic_vdr_vdw_two_row_staticshape_ssavc4_launch(program, grid, block,
                                                      input_dev, output_dev) < 0 ||
      vc4MemcpyDtoH(program, output_values, output_dev, sizeof(output_values)) < 0) {
    printk("ERROR: dynamic_vdr_vdw_two_row_staticshape launch/copy failed\n");
    ++launch_failures;
  }

  for (uint32_t row = 0; row < ROWS; ++row) {
    for (uint32_t col = 0; col < COLS; ++col) {
      uint32_t index = row * COLS + col;
      uint32_t actual = output_values[index];
      uint32_t expected = pattern(row, col);
      checksum += actual;
      if (actual != expected) {
        if (total_mismatches < 8)
          printk("ERROR: dynamic_vdr_vdw_two_row_staticshape row=%d col=%d actual=%x expected=%x\n",
                 (int)row, (int)col, actual, expected);
        ++total_mismatches;
      }
    }
  }
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    uint32_t actual = output_values[ROWS * COLS + i];
    if (actual != SENTINEL) {
      if (sentinel_mismatches < 8)
        printk("ERROR: dynamic_vdr_vdw_two_row_staticshape guard=%d actual=%x expected=%x\n",
               (int)i, actual, SENTINEL);
      ++sentinel_mismatches;
    }
  }

  uint32_t runtime_launches =
      dynamic_vdr_vdw_two_row_staticshape_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdr_vdw_two_row_staticshape_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdr_vdw_two_row_staticshape_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
