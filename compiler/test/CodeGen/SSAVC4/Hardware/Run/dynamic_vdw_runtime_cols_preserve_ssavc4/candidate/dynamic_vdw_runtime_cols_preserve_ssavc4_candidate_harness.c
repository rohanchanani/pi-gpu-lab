#include "rpi.h"
#include "kernel_launch.h"

#define COLS 16u
#define GUARD_WORDS 16u
#define SENTINEL 0x8accee31u

static uint32_t input_values[COLS];
static uint32_t output_values[COLS + GUARD_WORDS];
static const uint32_t active_cols_cases[] = {0u, 8u, 16u, 19u};

static uint32_t pattern(uint32_t col) { return 0x61000000u | col; }

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdw_runtime_cols_preserve_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &output_dev, sizeof(output_values)) < 0)
    panic("dynamic_vdw_runtime_cols_preserve_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  for (uint32_t i = 0; i < COLS; ++i)
    input_values[i] = pattern(i);

  for (uint32_t c = 0;
       c < sizeof(active_cols_cases) / sizeof(active_cols_cases[0]); ++c) {
    uint32_t active_cols = active_cols_cases[c];
    uint32_t expected_active = active_cols < COLS ? active_cols : COLS;
    for (uint32_t i = 0; i < COLS + GUARD_WORDS; ++i)
      output_values[i] = SENTINEL;

    if (vc4MemcpyHtoD(program, input_dev, input_values, sizeof(input_values)) < 0 ||
        vc4MemcpyHtoD(program, output_dev, output_values, sizeof(output_values)) < 0 ||
        dynamic_vdw_runtime_cols_preserve_ssavc4_launch(program, grid, block,
                                                        input_dev, output_dev,
                                                        active_cols) < 0 ||
        vc4MemcpyDtoH(program, output_values, output_dev, sizeof(output_values)) < 0) {
      printk("ERROR: dynamic_vdw_runtime_cols_preserve launch/copy failed active_cols=%d\n",
             (int)active_cols);
      ++launch_failures;
    }

    for (uint32_t i = 0; i < expected_active; ++i) {
      uint32_t actual = output_values[i];
      uint32_t expected = pattern(i);
      checksum += actual;
      if (actual != expected) {
        if (total_mismatches < 8)
          printk("ERROR: dynamic_vdw_runtime active_cols=%d active_col=%d actual=%x expected=%x\n",
                 (int)active_cols, (int)i, actual, expected);
        ++total_mismatches;
      }
    }
    for (uint32_t i = expected_active; i < COLS + GUARD_WORDS; ++i) {
      uint32_t actual = output_values[i];
      if (actual != SENTINEL) {
        if (sentinel_mismatches < 8)
          printk("ERROR: dynamic_vdw_runtime active_cols=%d preserved_col=%d actual=%x expected=%x\n",
                 (int)active_cols, (int)i, actual, SENTINEL);
        ++sentinel_mismatches;
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_vdw_runtime_cols_preserve_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdw_runtime_cols_preserve_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdw_runtime_cols_preserve_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
