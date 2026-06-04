#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ROWS 8u
#define COLS 16u
#define ACTIVE_ROWS 1u
#define ACTIVE_COLS 16u
#define INPUT_WORDS COLS
#define OUTPUT_WORDS (ROWS * COLS)
#define GUARD 32u
#define SENTINEL 0x5a5a5a5au
#define INIT_BASE 0x71000000u
#define LOAD_BASE 0x72000000u

static const uint32_t sweep_rows[] = {0u, 1u, 2u, 3u, 7u};
static uint32_t input_values[INPUT_WORDS];
static uint32_t output_values[OUTPUT_WORDS + GUARD];

static uint32_t init_pattern(uint32_t row, uint32_t col) {
  return INIT_BASE | (row << 8) | col;
}

static uint32_t load_pattern(uint32_t row, uint32_t col) {
  return LOAD_BASE | (row << 8) | col;
}

static void fill_input(uint32_t dst_row) {
  for (uint32_t c = 0; c < COLS; ++c)
    input_values[c] = load_pattern(dst_row, c);
}

static void fill_output(void) {
  for (uint32_t i = 0; i < OUTPUT_WORDS + GUARD; ++i)
    output_values[i] = SENTINEL;
}

static uint32_t expected(uint32_t dst_row, uint32_t row, uint32_t col) {
  if (row == dst_row)
    return load_pattern(dst_row, col);
  return init_pattern(row, col);
}

static int verify_case(uint32_t dst_row, uint32_t *checksum) {
  int mismatches = 0;
  for (uint32_t r = 0; r < ROWS; ++r) {
    for (uint32_t c = 0; c < COLS; ++c) {
      uint32_t got = output_values[r * COLS + c];
      uint32_t want = expected(dst_row, r, c);
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: dynamic_vdr_rect_runtime_dst_row row_arg=%d row=%d col=%d got=%x want=%x\n",
                 (int)dst_row, (int)r, (int)c, got, want);
        ++mismatches;
      }
    }
  }
  return mismatches;
}

static int verify_guard(uint32_t dst_row) {
  int mismatches = 0;
  for (uint32_t i = OUTPUT_WORDS; i < OUTPUT_WORDS + GUARD; ++i) {
    if (output_values[i] != SENTINEL) {
      if (mismatches < 8)
        printk("ERROR: dynamic_vdr_rect_runtime_dst_row row_arg=%d guard=%d got=%x want=%x\n",
               (int)dst_row, (int)i, output_values[i], SENTINEL);
      ++mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("dynamic_vdr_rect_runtime_dst_row_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, INPUT_WORDS * sizeof(uint32_t)) < 0 ||
      vc4_m2_malloc(program, &out_dev, (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0)
    panic("dynamic_vdr_rect_runtime_dst_row_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);

  for (uint32_t i = 0; i < sizeof(sweep_rows) / sizeof(sweep_rows[0]); ++i) {
    uint32_t dst_row = sweep_rows[i];
    fill_input(dst_row);
    fill_output();
    if (vc4_m2_copy_htod(program, in_dev, input_values, INPUT_WORDS * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_values,
                         (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0 ||
        dynamic_vdr_rect_runtime_dst_row_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, dst_row, ACTIVE_ROWS,
            ACTIVE_COLS, COLS * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_dtoh(program, output_values, out_dev,
                         (OUTPUT_WORDS + GUARD) * sizeof(uint32_t)) < 0) {
      ++launch_failures;
      continue;
    }
    total_mismatches += verify_case(dst_row, &checksum);
    sentinel_mismatches += verify_guard(dst_row);
  }

  launch_failures +=
      (int)dynamic_vdr_rect_runtime_dst_row_vc4kernel_runtime_launch_failures();
  uint32_t launches = dynamic_vdr_rect_runtime_dst_row_vc4kernel_runtime_launches();
  const uint32_t cases = sizeof(sweep_rows) / sizeof(sweep_rows[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == cases)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=dynamic_vdr_rect_runtime_dst_row_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d rows=%d cols=%d swept_rows=0,1,2,3,7 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)cases, total_mismatches, sentinel_mismatches,
         launch_failures, (int)ROWS, (int)COLS, checksum, launches,
         timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
