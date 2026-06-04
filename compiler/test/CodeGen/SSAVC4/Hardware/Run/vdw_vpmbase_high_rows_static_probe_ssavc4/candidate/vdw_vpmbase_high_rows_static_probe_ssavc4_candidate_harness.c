#include "rpi.h"
#include "kernel_launch.h"

#define CASES 7u
#define GUARD_WORDS 16u
#define SENTINEL 0x5a5a5a5au

static uint32_t output_values[CASES + GUARD_WORDS];
static const uint32_t source_rows[CASES] = {0u, 1u, 2u, 12u, 13u, 14u, 15u};

static uint32_t pattern(uint32_t row, uint32_t col) {
  return 0x7d000000u | (row << 8) | col;
}

static void fill_output(void) {
  for (uint32_t i = 0; i < CASES + GUARD_WORDS; ++i)
    output_values[i] = SENTINEL;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdw_vpmbase_high_rows_static_probe_ssavc4 program create failed");
  if (vc4Malloc(program, &output_dev, sizeof(output_values)) < 0)
    panic("vdw_vpmbase_high_rows_static_probe_ssavc4 allocation failed");

  fill_output();
  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  if (vc4MemcpyHtoD(program, output_dev, output_values,
                    sizeof(output_values)) < 0 ||
      vdw_vpmbase_high_rows_static_probe_ssavc4_launch(
          program, grid, block, output_dev) < 0 ||
      vc4MemcpyDtoH(program, output_values, output_dev,
                    sizeof(output_values)) < 0) {
    printk("ERROR: vdw_vpmbase_high_rows_static_probe launch/copy failed\n");
    ++launch_failures;
  }

  for (uint32_t i = 0; i < CASES; ++i) {
    uint32_t expected = pattern(source_rows[i], 0u);
    uint32_t actual = output_values[i];
    checksum += actual;
    if (actual != expected) {
      if (total_mismatches < 8)
        printk("ERROR: vdw_vpmbase_high_rows_static_probe case=%d source_row=%d actual=%x expected=%x\n",
               (int)i, (int)source_rows[i], actual, expected);
      ++total_mismatches;
    }
  }
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    uint32_t actual = output_values[CASES + i];
    if (actual != SENTINEL) {
      if (sentinel_mismatches < 8)
        printk("ERROR: vdw_vpmbase_high_rows_static_probe guard=%d actual=%x expected=%x\n",
               (int)i, actual, SENTINEL);
      ++sentinel_mismatches;
    }
  }

  uint32_t runtime_launches =
      vdw_vpmbase_high_rows_static_probe_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      vdw_vpmbase_high_rows_static_probe_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_vpmbase_high_rows_static_probe_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, output_dev);
}
