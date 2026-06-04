#include "rpi.h"
#include "kernel_launch.h"

#define OUTPUT_BYTES 1024u
#define SENTINEL_BYTE 0x5au

static uint8_t output_bytes[OUTPUT_BYTES];

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t output_dev = 0;
  uint32_t total_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t launch_failures = 0;
  uint32_t checksum = 0;

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdw_active_cols_zero_no_dma_probe_ssavc4 program create failed");
  if (vc4Malloc(program, &output_dev, sizeof(output_bytes)) < 0)
    panic("vdw_active_cols_zero_no_dma_probe_ssavc4 allocation failed");

  for (uint32_t i = 0; i < sizeof(output_bytes); ++i)
    output_bytes[i] = SENTINEL_BYTE;

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  if (vc4MemcpyHtoD(program, output_dev, output_bytes,
                    sizeof(output_bytes)) < 0 ||
      vdw_active_cols_zero_no_dma_probe_ssavc4_launch(
          program, grid, block, output_dev, 7u, 0u, 64u) < 0 ||
      vc4MemcpyDtoH(program, output_bytes, output_dev,
                    sizeof(output_bytes)) < 0) {
    printk("ERROR: vdw_active_cols_zero_no_dma_probe launch/copy failed\n");
    ++launch_failures;
  }

  for (uint32_t i = 0; i < sizeof(output_bytes); ++i) {
    uint8_t actual = output_bytes[i];
    checksum += actual;
    if (actual != SENTINEL_BYTE) {
      if (sentinel_mismatches < 8)
        printk("ERROR: vdw_active_cols_zero_no_dma_probe byte=%d actual=%x expected=%x\n",
               (int)i, actual, SENTINEL_BYTE);
      ++sentinel_mismatches;
    }
  }

  uint32_t runtime_launches =
      vdw_active_cols_zero_no_dma_probe_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      vdw_active_cols_zero_no_dma_probe_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;
  const char *status =
      (total_mismatches == 0 && sentinel_mismatches == 0 &&
       launch_failures == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_active_cols_zero_no_dma_probe_ssavc4 status=%s total_mismatches=%d sentinel_mismatches=%d launch_failures=%d runtime_launches=%d checksum_accum=%u\n",
         status, (int)total_mismatches, (int)sentinel_mismatches,
         (int)launch_failures, (int)runtime_launches, checksum);

  vc4Free(program, output_dev);
}
