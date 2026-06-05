#include "rpi.h"
#include "kernel_launch.h"

#define ROWS 4u
#define COLS 16u
#define MAX_PITCH_WORDS 65u
#define OUT_WORDS (ROWS * COLS)
#define GUARD_WORDS 16u
#define SENTINEL 0x8accee31u

static uint32_t input_values[ROWS * MAX_PITCH_WORDS + COLS + GUARD_WORDS];
static uint32_t output_values[OUT_WORDS + GUARD_WORDS];

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x64000000u | (row << 8) | col;
}

static void fill_buffers(uint32_t pitch_words) {
  for (uint32_t i = 0; i < ROWS * MAX_PITCH_WORDS + COLS + GUARD_WORDS; ++i)
    input_values[i] = SENTINEL;
  for (uint32_t i = 0; i < OUT_WORDS + GUARD_WORDS; ++i)
    output_values[i] = SENTINEL;
  for (uint32_t row = 0; row < ROWS; ++row)
    for (uint32_t col = 0; col < COLS; ++col)
      input_values[row * pitch_words + col] = pattern(row, col);
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdr_branch_skip_runtime_pitch_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, sizeof(input_values)) < 0 ||
      vc4Malloc(program, &output_dev, sizeof(output_values)) < 0)
    panic("dynamic_vdr_branch_skip_runtime_pitch_ssavc4 allocation failed");

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  const uint32_t pitch_cases[] = {64u, 76u, 260u};
  const uint32_t mode_cases[] = {0u, 1u};

  for (uint32_t p = 0; p < sizeof(pitch_cases) / sizeof(pitch_cases[0]); ++p) {
    uint32_t pitch_bytes = pitch_cases[p];
    uint32_t pitch_words = pitch_bytes / 4u;
    for (uint32_t m = 0; m < sizeof(mode_cases) / sizeof(mode_cases[0]); ++m) {
      uint32_t mode = mode_cases[m];
      fill_buffers(pitch_words);
      if (vc4MemcpyHtoD(program, input_dev, input_values,
                        sizeof(input_values)) < 0 ||
          vc4MemcpyHtoD(program, output_dev, output_values,
                        sizeof(output_values)) < 0 ||
          dynamic_vdr_branch_skip_runtime_pitch_ssavc4_launch(
              program, grid, block, input_dev, output_dev, mode, ROWS, COLS,
              pitch_bytes) < 0 ||
          vc4MemcpyDtoH(program, output_values, output_dev,
                        sizeof(output_values)) < 0) {
        printk("ERROR: dynamic_vdr_branch_skip_runtime_pitch launch/copy failed mode=%d pitch_bytes=%d\n",
               (int)mode, (int)pitch_bytes);
        ++launch_failures;
      }

      if (mode == 0u) {
        for (uint32_t i = 0; i < OUT_WORDS; ++i) {
          uint32_t actual = output_values[i];
          if (actual != SENTINEL) {
            if (sentinel_mismatches < 8)
              printk("ERROR: dynamic_vdr_branch_skip_runtime_pitch skip pitch_bytes=%d index=%d actual=%x expected=%x\n",
                     (int)pitch_bytes, (int)i, actual, SENTINEL);
            ++sentinel_mismatches;
          }
        }
      } else {
        for (uint32_t row = 0; row < ROWS; ++row) {
          for (uint32_t col = 0; col < COLS; ++col) {
            uint32_t index = row * COLS + col;
            uint32_t actual = output_values[index];
            uint32_t expected = pattern(row, col);
            checksum += actual;
            if (actual != expected) {
              if (total_mismatches < 8)
                printk("ERROR: dynamic_vdr_branch_skip_runtime_pitch run pitch_bytes=%d row=%d col=%d actual=%x expected=%x\n",
                       (int)pitch_bytes, (int)row, (int)col, actual, expected);
              ++total_mismatches;
            }
          }
        }
      }

      for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
        uint32_t actual = output_values[OUT_WORDS + i];
        if (actual != SENTINEL) {
          if (sentinel_mismatches < 8)
            printk("ERROR: dynamic_vdr_branch_skip_runtime_pitch guard mode=%d pitch_bytes=%d guard=%d actual=%x expected=%x\n",
                   (int)mode, (int)pitch_bytes, (int)i, actual, SENTINEL);
          ++sentinel_mismatches;
        }
      }
    }
  }

  uint32_t runtime_launches =
      dynamic_vdr_branch_skip_runtime_pitch_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      dynamic_vdr_branch_skip_runtime_pitch_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdr_branch_skip_runtime_pitch_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
