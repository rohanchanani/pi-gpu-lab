#include "rpi.h"
#include "kernel_launch.h"

#define VDR_LOAD_ROUNDTRIP_SSAVC4_ROWS 16u
#define VDR_LOAD_ROUNDTRIP_SSAVC4_COLS 16u
#define VDR_LOAD_ROUNDTRIP_SSAVC4_WORDS \
  (VDR_LOAD_ROUNDTRIP_SSAVC4_ROWS * VDR_LOAD_ROUNDTRIP_SSAVC4_COLS)
#define VDR_LOAD_ROUNDTRIP_SSAVC4_GUARD_WORDS 16u
#define VDR_LOAD_ROUNDTRIP_SSAVC4_SENTINEL 0xdeadbeefu

static uint32_t input_values[VDR_LOAD_ROUNDTRIP_SSAVC4_WORDS];
static uint32_t output_values[VDR_LOAD_ROUNDTRIP_SSAVC4_WORDS +
                              VDR_LOAD_ROUNDTRIP_SSAVC4_GUARD_WORDS];

static uint32_t expected_value(uint32_t row, uint32_t col) {
  return 1000u + 37u * row + col;
}

static void fill_input_and_output(void) {
  for (uint32_t row = 0; row < VDR_LOAD_ROUNDTRIP_SSAVC4_ROWS; ++row) {
    for (uint32_t col = 0; col < VDR_LOAD_ROUNDTRIP_SSAVC4_COLS; ++col)
      input_values[row * VDR_LOAD_ROUNDTRIP_SSAVC4_COLS + col] =
          expected_value(row, col);
  }
  for (uint32_t i = 0;
       i < VDR_LOAD_ROUNDTRIP_SSAVC4_WORDS +
               VDR_LOAD_ROUNDTRIP_SSAVC4_GUARD_WORDS;
       ++i)
    output_values[i] = VDR_LOAD_ROUNDTRIP_SSAVC4_SENTINEL;
}

static int verify_output(uint32_t *checksum) {
  int mismatches = 0;
  *checksum = 0;
  for (uint32_t row = 0; row < VDR_LOAD_ROUNDTRIP_SSAVC4_ROWS; ++row) {
    for (uint32_t col = 0; col < VDR_LOAD_ROUNDTRIP_SSAVC4_COLS; ++col) {
      uint32_t index = row * VDR_LOAD_ROUNDTRIP_SSAVC4_COLS + col;
      uint32_t expected = expected_value(row, col);
      uint32_t actual = output_values[index];
      *checksum += actual;
      if (actual != expected) {
        if (mismatches < 8)
          printk("ERROR: vdr_load_roundtrip_ssavc4 row=%d col=%d gpu=%x expected=%x\n",
                 (int)row, (int)col, actual, expected);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_sentinels(void) {
  int mismatches = 0;
  for (uint32_t i = 0; i < VDR_LOAD_ROUNDTRIP_SSAVC4_GUARD_WORDS; ++i) {
    uint32_t index = VDR_LOAD_ROUNDTRIP_SSAVC4_WORDS + i;
    if (output_values[index] != VDR_LOAD_ROUNDTRIP_SSAVC4_SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: vdr_load_roundtrip_ssavc4 sentinel changed i=%d value=%x expected=%x\n",
               (int)i, output_values[index], VDR_LOAD_ROUNDTRIP_SSAVC4_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  const uint32_t input_bytes = VDR_LOAD_ROUNDTRIP_SSAVC4_WORDS * sizeof(uint32_t);
  const uint32_t output_bytes =
      (VDR_LOAD_ROUNDTRIP_SSAVC4_WORDS +
       VDR_LOAD_ROUNDTRIP_SSAVC4_GUARD_WORDS) * sizeof(uint32_t);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_load_roundtrip_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, input_bytes) < 0 ||
      vc4Malloc(program, &output_dev, output_bytes) < 0)
    panic("vdr_load_roundtrip_ssavc4 device allocation failed");

  fill_input_and_output();

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  uint32_t checksum_accum = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  int start = timer_get_usec();

  printk("Running VC4 vdr_load_roundtrip_ssavc4 candidate bundle.\n");
  if (vc4MemcpyHtoD(program, input_dev, input_values, input_bytes) < 0 ||
      vc4MemcpyHtoD(program, output_dev, output_values, output_bytes) < 0 ||
      vdr_load_roundtrip_ssavc4_launch(program, grid, block, input_dev,
                                       output_dev) < 0 ||
      vc4MemcpyDtoH(program, output_values, output_dev, output_bytes) < 0) {
    printk("ERROR: vdr_load_roundtrip_ssavc4 launch/copy failed\n");
    ++launch_failures;
  } else {
    total_mismatches = verify_output(&checksum_accum);
    sentinel_mismatches = verify_sentinels();
  }

  uint32_t runtime_allocations = vdr_load_roundtrip_ssavc4_runtime_allocations();
  uint32_t runtime_launches = vdr_load_roundtrip_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      vdr_load_roundtrip_ssavc4_runtime_launch_failures();
  if (runtime_launch_failures != 0)
    launch_failures += (int)runtime_launch_failures;

  int elapsed = timer_get_usec() - start;
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_load_roundtrip_ssavc4 status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=16 rows=16 cols=16 checksum_accum=%u runtime_allocations=%u runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         checksum_accum, runtime_allocations, runtime_launches, elapsed);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
