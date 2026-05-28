#include "rpi.h"
#include "kernel_launch.h"

#define PROBE_ROWS 16u
#define PROBE_COLS 16u
#define PROBE_SEGMENT_WORDS (PROBE_ROWS * PROBE_COLS)
#define PROBE_CASES 8u
#define PROBE_GUARD_WORDS 32u
#define PROBE_SENTINEL 0x5a5a5a5au

static uint32_t input_values[PROBE_SEGMENT_WORDS];
static uint32_t output_values[PROBE_CASES * PROBE_SEGMENT_WORDS + PROBE_GUARD_WORDS];

struct probe_case {
  const char *name;
  uint32_t vertical;
  uint32_t y;
  uint32_t x;
  uint32_t nrows;
  uint32_t row_len;
};

static const struct probe_case cases[PROBE_CASES] = {
    {"h_n4_len16_y0_x0", 0, 0, 0, 4, 16},
    {"h_n16_len16_y0_x0", 0, 0, 0, 16, 16},
    {"v_n1_len16_y0_x0", 1, 0, 0, 1, 16},
    {"v_n4_len16_y0_x0", 1, 0, 0, 4, 16},
    {"v_n16_len16_y0_x0", 1, 0, 0, 16, 16},
    {"v_n4_len16_y0_x1", 1, 0, 1, 4, 16},
    {"v_n4_len8_y0_x0", 1, 0, 0, 4, 8},
    {"v_n3_len8_y4_x2", 1, 4, 2, 3, 8},
};

static uint32_t input_pattern(uint32_t row, uint32_t col) {
  return 0x51000000u | (row << 8) | col;
}

static uint32_t expected_model_value(const struct probe_case *spec,
                                     uint32_t row, uint32_t col) {
  if (spec->vertical)
    return input_pattern(spec->y + col, spec->x + row);
  return input_pattern(spec->y + row, spec->x + col);
}

static void fill_buffers(void) {
  for (uint32_t row = 0; row < PROBE_ROWS; ++row) {
    for (uint32_t col = 0; col < PROBE_COLS; ++col)
      input_values[row * PROBE_COLS + col] = input_pattern(row, col);
  }
  for (uint32_t i = 0; i < PROBE_CASES * PROBE_SEGMENT_WORDS + PROBE_GUARD_WORDS; ++i)
    output_values[i] = PROBE_SENTINEL;
}

static void verify_case(uint32_t case_id, uint32_t *total_model_mismatches,
                        uint32_t *total_sentinel_mismatches) {
  const struct probe_case *spec = &cases[case_id];
  uint32_t *base = &output_values[case_id * PROBE_SEGMENT_WORDS];
  uint32_t model_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t checksum = 0;

  for (uint32_t row = 0; row < PROBE_ROWS; ++row) {
    for (uint32_t col = 0; col < PROBE_COLS; ++col) {
      uint32_t actual = base[row * PROBE_COLS + col];
      checksum += actual;
      if (row < spec->nrows && col < spec->row_len) {
        uint32_t expected = expected_model_value(spec, row, col);
        if (actual != expected) {
          if (model_mismatches < 6)
            printk("ERROR: PROBE_MISMATCH case=%s row=%d col=%d actual=%x expected_model=%x\n",
                   spec->name, (int)row, (int)col, actual, expected);
          ++model_mismatches;
        }
      } else if (actual != PROBE_SENTINEL) {
        if (sentinel_mismatches < 6)
          printk("ERROR: PROBE_SENTINEL case=%s row=%d col=%d actual=%x expected=%x\n",
                 spec->name, (int)row, (int)col, actual, PROBE_SENTINEL);
        ++sentinel_mismatches;
      }
    }
  }

  printk("HASH: PROBE_CASE id=%d name=%s orientation=%s y=%d x=%d nrows=%d row_len=%d model_mismatches=%d sentinel_mismatches=%d checksum=%u sample00=%x sample01=%x sample10=%x sample15=%x sample_last=%x\n",
         (int)case_id, spec->name, spec->vertical ? "vertical" : "horizontal",
         (int)spec->y, (int)spec->x, (int)spec->nrows, (int)spec->row_len,
         (int)model_mismatches, (int)sentinel_mismatches, checksum,
         base[0], base[1], base[PROBE_COLS], base[15],
         base[(spec->nrows - 1u) * PROBE_COLS + (spec->row_len - 1u)]);

  *total_model_mismatches += model_mismatches;
  *total_sentinel_mismatches += sentinel_mismatches;
}

static uint32_t verify_guard(void) {
  uint32_t mismatches = 0;
  uint32_t start = PROBE_CASES * PROBE_SEGMENT_WORDS;
  for (uint32_t i = 0; i < PROBE_GUARD_WORDS; ++i) {
    if (output_values[start + i] != PROBE_SENTINEL) {
      if (mismatches < 6)
        printk("ERROR: PROBE_GUARD i=%d actual=%x expected=%x\n",
               (int)i, output_values[start + i], PROBE_SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t input_dev = 0;
  vc4_deviceptr_t output_dev = 0;
  const uint32_t input_bytes = PROBE_SEGMENT_WORDS * sizeof(uint32_t);
  const uint32_t output_bytes =
      (PROBE_CASES * PROBE_SEGMENT_WORDS + PROBE_GUARD_WORDS) * sizeof(uint32_t);

  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdw_vertical_multirow_probe_ssavc4 program create failed");
  if (vc4Malloc(program, &input_dev, input_bytes) < 0 ||
      vc4Malloc(program, &output_dev, output_bytes) < 0)
    panic("vdw_vertical_multirow_probe_ssavc4 device allocation failed");

  fill_buffers();

  vc4_dim3 grid = {1u, 1u, 1u};
  vc4_dim3 block = {16u, 1u, 1u};
  uint32_t model_mismatches = 0;
  uint32_t sentinel_mismatches = 0;
  uint32_t guard_mismatches = 0;
  uint32_t launch_failures = 0;

  printk("Running VC4 vdw_vertical_multirow_probe_ssavc4 candidate bundle.\n");
  if (vc4MemcpyHtoD(program, input_dev, input_values, input_bytes) < 0 ||
      vc4MemcpyHtoD(program, output_dev, output_values, output_bytes) < 0 ||
      vdw_vertical_multirow_probe_ssavc4_launch(program, grid, block, input_dev,
                                                output_dev) < 0 ||
      vc4MemcpyDtoH(program, output_values, output_dev, output_bytes) < 0) {
    printk("ERROR: vdw_vertical_multirow_probe_ssavc4 launch/copy failed\n");
    ++launch_failures;
  } else {
    for (uint32_t i = 0; i < PROBE_CASES; ++i)
      verify_case(i, &model_mismatches, &sentinel_mismatches);
    guard_mismatches = verify_guard();
  }

  uint32_t runtime_launches = vdw_vertical_multirow_probe_ssavc4_runtime_launches();
  uint32_t runtime_launch_failures =
      vdw_vertical_multirow_probe_ssavc4_runtime_launch_failures();
  if (runtime_launch_failures != 0)
    launch_failures += runtime_launch_failures;

  const char *status =
      (launch_failures == 0 && guard_mismatches == 0 &&
       model_mismatches == 0 && sentinel_mismatches == 0)
          ? "PASS"
          : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_vertical_multirow_probe_ssavc4 status=%s cases=%d launch_failures=%d guard_mismatches=%d model_mismatches=%d sentinel_mismatches=%d runtime_launches=%d\n",
         status, (int)PROBE_CASES, (int)launch_failures,
         (int)guard_mismatches, (int)model_mismatches,
         (int)sentinel_mismatches, (int)runtime_launches);

  vc4Free(program, input_dev);
  vc4Free(program, output_dev);
}
