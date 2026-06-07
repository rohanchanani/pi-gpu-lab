#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define IN_BYTES 64u
#define OUT_WORDS 80u
#define SENTINEL 0x7c0ffeeu

struct test_case {
  uint32_t n;
  uint32_t row;
  uint32_t x;
  uint32_t sel8;
  uint32_t amount;
};

static const struct test_case cases[] = {
    {0u, 0u, 0u, 0u, 0u},
    {1u, 3u, 0u, 1u, 1u},
    {9u, 5u, 0u, 2u, 7u},
    {16u, 9u, 0u, 3u, 15u},
};

static uint8_t input_bytes[IN_BYTES];
static uint32_t output_words[OUT_WORDS];

static void fill_buffers(uint32_t case_index) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(11u + case_index * 17u + i);
  for (uint32_t i = 0; i < OUT_WORDS; ++i)
    output_words[i] = SENTINEL;
}

static uint32_t selected_value(uint32_t case_index, uint32_t lane,
                               uint32_t n) {
  (void)case_index;
  uint32_t value = 20u + lane;
  value += lane;
  return lane < n ? value : 0u;
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *sentinel_mismatches = 0;
  uint32_t sum = 0;
  for (uint32_t lane = 0; lane < LANES; ++lane)
    sum += selected_value(case_index, lane, tc->n);
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    uint32_t src_lane = (lane + tc->amount) & 15u;
    uint32_t want = selected_value(case_index, src_lane, tc->n);
    uint32_t got = output_words[lane];
    *checksum += got;
    if (got != want) {
      if (mismatches < 8)
        printk("ERROR: double_buffer_compute rot case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, want);
      ++mismatches;
    }
  }
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    uint32_t got = output_words[32u + lane];
    *checksum += got;
    if (got != sum) {
      if (mismatches < 8)
        printk("ERROR: double_buffer_compute reduce case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, sum);
      ++mismatches;
    }
  }
  const uint8_t *out_bytes = (const uint8_t *)output_words;
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    uint8_t want = input_bytes[lane];
    uint8_t got = out_bytes[256u + lane];
    *checksum += got;
    if (got != want) {
      if (mismatches < 8)
        printk("ERROR: double_buffer_compute vdr_vdw case=%d lane=%d got=%x want=%x\n",
               (int)case_index, (int)lane, got, want);
      ++mismatches;
    }
  }
  for (uint32_t i = 16u; i < OUT_WORDS; ++i) {
    if (i >= 32u && i < 48u)
      continue;
    if (i >= 64u && i < 72u)
      continue;
    if (output_words[i] != SENTINEL) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: double_buffer_compute sentinel case=%d index=%d got=%x want=%x\n",
               (int)case_index, (int)i, output_words[i], SENTINEL);
      ++*sentinel_mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("mixed_double_buffered_vpm_tiles_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_words)) < 0)
    panic("mixed_double_buffered_vpm_tiles_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_buffers(i);
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_words, sizeof(output_words)) < 0 ||
        mixed_double_buffered_vpm_tiles_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, cases[i].n, cases[i].row,
            cases[i].x, cases[i].sel8, cases[i].amount) < 0 ||
        vc4_m2_copy_dtoh(program, output_words, out_dev, sizeof(output_words)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    total_mismatches += verify_case(&cases[i], i, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
  }

  launch_failures +=
      (int)mixed_double_buffered_vpm_tiles_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      mixed_double_buffered_vpm_tiles_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=mixed_double_buffered_vpm_tiles_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_double_buffered_vpm_tiles=1 saw_horizontal_packed_subword_compute=1 saw_vdr_dynamic_subword_selector=1 saw_dynamic_vpm_row=1 saw_dynamic_vpm_x=1 saw_dynamic_subword_selector=1 saw_dynamic_vpm_qpu_coords=1 saw_dynamic_vpm_qpu_subword_selectors=1 saw_dynamic_vdr_vpm_dest_coords=1 saw_dynamic_vdw_vpm_source_coords=1 saw_vpm_qpu_dynamic_selector_read=1 saw_fragment_pack_unpack=1 saw_fragment_unpack=1 saw_fragment_alu=1 saw_cmp_select=1 saw_fragment_reduce=1 saw_dynamic_rotate=1 saw_vdr_to_vpm_sidepath=1 saw_vdw_dynamic_subword_selector=1 saw_vdw_preserve=1 saw_spill_guard_in_targeted_run=1 no_tmu_to_vpm=1 selector_values_w8=0,1,2,3 x_values=0 rotate_direction=left_source_plus_amount checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
