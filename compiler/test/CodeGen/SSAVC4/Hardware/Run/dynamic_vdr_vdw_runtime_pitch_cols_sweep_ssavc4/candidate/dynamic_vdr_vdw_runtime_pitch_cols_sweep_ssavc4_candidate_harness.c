#include "rpi.h"
#include "kernel_launch.h"

#define COLS 16u
#define MAX_PITCH_WORDS 65u
#define GUARD_WORDS 16u
#define SENTINEL 0x8accee31u

static uint32_t input_values[MAX_PITCH_WORDS + GUARD_WORDS];
static uint32_t output_zero_values[MAX_PITCH_WORDS + GUARD_WORDS];
static uint32_t output_preserve_values[MAX_PITCH_WORDS + GUARD_WORDS];
static const uint32_t active_cols_cases[] = {0u, 1u, 8u, 19u};
static const uint32_t pitch_bytes_cases[] = {64u, 76u, 124u, 260u};

static uint32_t pattern(uint32_t col) { return 0x62000000u | col; }

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t zero_mismatches = 0;
  uint32_t preserve_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &output_dev,
                sizeof(output_zero_values) + sizeof(output_preserve_values)) < 0)
    panic("dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};

  for (uint32_t p = 0;
       p < sizeof(pitch_bytes_cases) / sizeof(pitch_bytes_cases[0]); ++p) {
    uint32_t pitch_bytes = pitch_bytes_cases[p];
    uint32_t pitch_words = pitch_bytes / 4u;
    for (uint32_t c = 0;
         c < sizeof(active_cols_cases) / sizeof(active_cols_cases[0]); ++c) {
      uint32_t active_cols = active_cols_cases[c];
      uint32_t expected_active = active_cols < COLS ? active_cols : COLS;
      for (uint32_t i = 0; i < MAX_PITCH_WORDS + GUARD_WORDS; ++i) {
        input_values[i] = SENTINEL;
        output_zero_values[i] = SENTINEL;
        output_preserve_values[i] = SENTINEL;
      }
      for (uint32_t col = 0; col < COLS; ++col)
        input_values[col] = pattern(col);

      if (vc4MemcpyHtoD(program, input_dev, input_values, sizeof(input_values)) < 0 ||
          vc4MemcpyHtoD(program, output_dev, output_zero_values,
                        sizeof(output_zero_values)) < 0 ||
          vc4MemcpyHtoD(program,
                        output_dev + sizeof(output_zero_values),
                        output_preserve_values,
                        sizeof(output_preserve_values)) < 0 ||
          dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4_launch(
              program, grid, block, input_dev, output_dev,
              output_dev + sizeof(output_zero_values), active_cols,
              pitch_bytes) < 0 ||
          vc4MemcpyDtoH(program, output_zero_values, output_dev,
                        sizeof(output_zero_values)) < 0 ||
          vc4MemcpyDtoH(program, output_preserve_values,
                        output_dev + sizeof(output_zero_values),
                        sizeof(output_preserve_values)) < 0) {
        printk("ERROR: dynamic_vdr_vdw_runtime_pitch_cols_sweep launch/copy failed pitch_bytes=%d active_cols=%d\n",
               (int)pitch_bytes, (int)active_cols);
        ++launch_failures;
      }

      for (uint32_t i = 0; i < MAX_PITCH_WORDS + GUARD_WORDS; ++i) {
        uint32_t expected_zero = SENTINEL;
        uint32_t expected_preserve = SENTINEL;
        if (i < expected_active)
          expected_zero = pattern(i);
        else if (i < COLS)
          expected_zero = 0u;
        if (i < expected_active)
          expected_preserve = pattern(i);
        uint32_t actual_zero = output_zero_values[i];
        uint32_t actual_preserve = output_preserve_values[i];
        if (i < expected_active)
          checksum += actual_zero + actual_preserve;
        if (actual_zero != expected_zero) {
          if (i < expected_active) {
            if (total_mismatches < 8)
              printk("ERROR: dynamic_vdr_vdw_runtime_pitch_cols_sweep zero_out pitch_bytes=%d active_cols=%d col=%d actual=%x expected=%x\n",
                     (int)pitch_bytes, (int)active_cols, (int)i, actual_zero,
                     expected_zero);
            ++total_mismatches;
          } else if (i < COLS) {
            if (zero_mismatches < 8)
              printk("ERROR: dynamic_vdr_vdw_runtime_pitch_cols_sweep zero_out pitch_bytes=%d active_cols=%d inactive_col=%d actual=%x expected=0\n",
                     (int)pitch_bytes, (int)active_cols, (int)i, actual_zero);
            ++zero_mismatches;
          } else {
            if (sentinel_mismatches < 8)
              printk("ERROR: dynamic_vdr_vdw_runtime_pitch_cols_sweep zero_out pitch_bytes=%d active_cols=%d sentinel_index=%d actual=%x expected=%x\n",
                     (int)pitch_bytes, (int)active_cols, (int)i, actual_zero,
                     SENTINEL);
            ++sentinel_mismatches;
          }
        }
        if (actual_preserve != expected_preserve) {
          if (i < expected_active) {
            if (total_mismatches < 8)
              printk("ERROR: dynamic_vdr_vdw_runtime_pitch_cols_sweep preserve_out pitch_bytes=%d active_cols=%d col=%d actual=%x expected=%x\n",
                     (int)pitch_bytes, (int)active_cols, (int)i,
                     actual_preserve, expected_preserve);
            ++total_mismatches;
          } else {
            if (preserve_mismatches < 8)
              printk("ERROR: dynamic_vdr_vdw_runtime_pitch_cols_sweep preserve_out pitch_bytes=%d active_cols=%d preserved_index=%d actual=%x expected=%x\n",
                     (int)pitch_bytes, (int)active_cols, (int)i,
                     actual_preserve, SENTINEL);
            ++preserve_mismatches;
          }
        }
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && zero_mismatches == 0 &&
       preserve_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4 status=%s total_mismatches=%d zero_mismatches=%d preserve_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)zero_mismatches,
         (int)preserve_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
