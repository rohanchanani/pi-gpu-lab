#include "rpi.h"
#include "kernel_launch.h"

#define COLUMNS 2u
#define LANES 16u
#define WORDS (COLUMNS * LANES)
#define GUARD_WORDS 16u
#define SENTINEL 0x7a7a7a7au

static uint32_t output_values[WORDS + GUARD_WORDS];

static uint32_t expected_value(uint32_t column_case, uint32_t lane) {
  uint32_t x = (column_case == 0u) ? 3u : 11u;
  return 0x71000000u | (lane << 8) | x;
}

static void fill_output(void) {
  for (uint32_t i = 0; i < WORDS + GUARD_WORDS; ++i)
    output_values[i] = SENTINEL;
}

static int verify_output(uint32_t *checksum) {
  int mismatches = 0;
  *checksum = 0;
  for (uint32_t column_case = 0; column_case < COLUMNS; ++column_case) {
    for (uint32_t lane = 0; lane < LANES; ++lane) {
      uint32_t index = column_case * LANES + lane;
      uint32_t actual = output_values[index];
      uint32_t expected = expected_value(column_case, lane);
      *checksum += actual;
      if (actual != expected) {
        if (mismatches < 8)
          printk("ERROR: vpm_vertical_32_column_probe_ssavc4 column_case=%d lane=%d actual=%x expected=%x\n",
                 (int)column_case, (int)lane, actual, expected);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = 0; i < GUARD_WORDS; ++i) {
    uint32_t index = WORDS + i;
    if (output_values[index] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: vpm_vertical_32_column_probe_ssavc4 sentinel i=%d actual=%x expected=%x\n",
               (int)i, output_values[index], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t output_dev = 0;
  const uint32_t output_bytes = (WORDS + GUARD_WORDS) * sizeof(uint32_t);
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vpm_vertical_32_column_probe_ssavc4 program create failed");
  if (vc4Malloc(program, &output_dev, output_bytes) < 0)
    panic("vpm_vertical_32_column_probe_ssavc4 output allocation failed");

  fill_output();
  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  uint32_t checksum_accum = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int start = timer_get_usec();

  printk("Running VC4 vpm_vertical_32_column_probe_ssavc4 candidate bundle.\n");
  if (vc4MemcpyHtoD(program, output_dev, output_values, output_bytes) < 0 ||
      vpm_vertical_32_column_probe_ssavc4_launch(program, grid, block,
                                                 output_dev) < 0 ||
      vc4MemcpyDtoH(program, output_values, output_dev, output_bytes) < 0) {
    printk("ERROR: vpm_vertical_32_column_probe_ssavc4 launch/copy failed\n");
    ++launch_failures;
  } else {
    total_mismatches = verify_output(&checksum_accum);
    sentinel_mismatches = verify_sentinels();
  }

  uint32_t runtime_launches = vpm_vertical_32_column_probe_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      vpm_vertical_32_column_probe_ssavc4_runtime_launch_failures();
  if (runtime_launch_failures != 0)
    launch_failures += (int)runtime_launch_failures;
  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vpm_vertical_32_column_probe_ssavc4 status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d columns=%d lanes=%d checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         (int)COLUMNS, (int)LANES, checksum_accum, runtime_launches, elapsed);
  vc4Free(program, output_dev);
}
