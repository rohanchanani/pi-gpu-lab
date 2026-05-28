#include "rpi.h"
#include "kernel_launch.h"

#define CASES 5u
#define ROW_WORDS 4u
#define INPUT_ROWS 4u
#define INPUT_COLS 16u
#define GUARD_WORDS 16u
#define SENTINEL 0x5a5a5a5au

static uint32_t input_values[INPUT_ROWS * INPUT_COLS];
static uint32_t output_values[CASES * ROW_WORDS + GUARD_WORDS];
static const uint32_t active_lanes[CASES] = {0u, 1u, 2u, 3u, 4u};

static uint32_t input_pattern(uint32_t row, uint32_t col) {
  return 0x63000000u | (row << 8) | col;
}

static void fill_buffers(void) {
  for (uint32_t row = 0; row < INPUT_ROWS; ++row)
    for (uint32_t col = 0; col < INPUT_COLS; ++col)
      input_values[row * INPUT_COLS + col] = input_pattern(row, col);
  for (uint32_t i = 0; i < CASES * ROW_WORDS + GUARD_WORDS; ++i)
    output_values[i] = SENTINEL;
}

static void verify_case(uint32_t case_id, uint32_t *model_mismatches,
                        uint32_t *sentinel_mismatches) {
  uint32_t active = active_lanes[case_id];
  uint32_t *base = &output_values[case_id * ROW_WORDS];
  uint32_t local_model = 0;
  uint32_t local_sentinel = 0;
  uint32_t checksum = 0;
  for (uint32_t lane = 0; lane < ROW_WORDS; ++lane) {
    uint32_t actual = base[lane];
    checksum += actual;
    if (lane < active) {
      uint32_t expected = input_pattern(case_id - 1u, lane);
      if (actual != expected) {
        if (local_model < 6)
          printk("ERROR: ROW4_MISMATCH case=%d lane=%d actual=%x expected=%x\n",
                 (int)case_id, (int)lane, actual, expected);
        ++local_model;
      }
    } else if (actual != SENTINEL) {
      if (local_sentinel < 6)
        printk("ERROR: ROW4_SENTINEL case=%d lane=%d actual=%x expected=%x\n",
               (int)case_id, (int)lane, actual, SENTINEL);
      ++local_sentinel;
    }
  }
  printk("HASH: ROW4_CASE id=%d active_lanes=%d model_mismatches=%d sentinel_mismatches=%d checksum=%u sample0=%x sample_after=%x\n",
         (int)case_id, (int)active, (int)local_model, (int)local_sentinel,
         checksum, base[0], active < ROW_WORDS ? base[active] : 0u);
  *model_mismatches += local_model;
  *sentinel_mismatches += local_sentinel;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  const uint32_t input_bytes = INPUT_ROWS * INPUT_COLS * sizeof(uint32_t);
  const uint32_t output_bytes =
      (CASES * ROW_WORDS + GUARD_WORDS) * sizeof(uint32_t);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdw_store_vpm_dynamic_row4_probe_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, input_bytes) < 0 ||
      vc4Malloc(program, &output_dev, output_bytes) < 0)
    panic("vdw_store_vpm_dynamic_row4_probe_ssavc4 allocation failed");

  fill_buffers();

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  uint32_t launch_failures = 0;
  uint32_t model_mismatches = 0;
  uint32_t sentinel_mismatches = 0;

  printk("Running VC4 vdw_store_vpm_dynamic_row4_probe_ssavc4 candidate bundle.\n");
  if (vc4MemcpyHtoD(program, input_dev, input_values, input_bytes) < 0 ||
      vc4MemcpyHtoD(program, output_dev, output_values, output_bytes) < 0 ||
      vdw_store_vpm_dynamic_row4_probe_ssavc4_launch(program, grid, block,
                                                     input_dev, output_dev) < 0 ||
      vc4MemcpyDtoH(program, output_values, output_dev, output_bytes) < 0) {
    printk("ERROR: vdw_store_vpm_dynamic_row4_probe_ssavc4 launch/copy failed\n");
    ++launch_failures;
  } else {
    for (uint32_t i = 0; i < CASES; ++i)
      verify_case(i, &model_mismatches, &sentinel_mismatches);
    for (uint32_t i = CASES * ROW_WORDS; i < CASES * ROW_WORDS + GUARD_WORDS; ++i)
      if (output_values[i] != SENTINEL)
        ++sentinel_mismatches;
  }

  uint32_t runtime_launches =
      vdw_store_vpm_dynamic_row4_probe_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      vdw_store_vpm_dynamic_row4_probe_ssavc4_runtime_launch_failures();
  launch_failures += runtime_launch_failures;

  const char *status =
      (launch_failures == 0 && model_mismatches == 0 &&
       sentinel_mismatches == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_store_vpm_dynamic_row4_probe_ssavc4 status=%s cases=%d launch_failures=%d model_mismatches=%d sentinel_mismatches=%d runtime_launches=%d\n",
         status, (int)CASES, (int)launch_failures, (int)model_mismatches,
         (int)sentinel_mismatches, (int)runtime_launches);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
