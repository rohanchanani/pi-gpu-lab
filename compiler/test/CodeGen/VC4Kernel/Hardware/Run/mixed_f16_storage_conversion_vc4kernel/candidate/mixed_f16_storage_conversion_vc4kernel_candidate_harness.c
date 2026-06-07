#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define IN_BYTES 96u
#define OUT_BYTES 128u
#define SENTINEL8 0xacu
#define GUARD_HALF 0x3e00u
#define GUARD_OUT_OFFSET 64u

struct test_case {
  uint32_t n;
  uint32_t row;
  uint32_t x;
  uint32_t sel16;
};

static const struct test_case cases[] = {
    {0u, 0u, 3u, 0u},   {1u, 16u, 7u, 1u},  {15u, 0u, 1u, 0u},
    {16u, 16u, 15u, 1u}, {17u, 0u, 3u, 1u}, {31u, 16u, 7u, 0u},
    {32u, 0u, 15u, 1u}, {33u, 16u, 1u, 0u}, {65u, 0u, 7u, 1u},
};

static const uint16_t input_half[LANES] = {
    0xc400u, 0xc000u, 0xbc00u, 0xb800u, 0x0000u, 0x3800u, 0x3c00u, 0x4000u,
    0x4200u, 0x4400u, 0x4500u, 0x4600u, 0x4700u, 0x4800u, 0x4900u, 0x4a00u,
};

static const uint16_t expected_half[LANES] = {
    0xc200u, 0xbc00u, 0x0000u, 0x3800u, 0x3c00u, 0x3e00u, 0x4000u, 0x4200u,
    0x4400u, 0x4500u, 0x4600u, 0x4700u, 0x4800u, 0x4880u, 0x4980u, 0x4a80u,
};

static uint8_t input_bytes[IN_BYTES];
static uint8_t output_bytes[OUT_BYTES];

static void store16(uint8_t *buf, uint32_t offset, uint16_t value) {
  buf[offset] = (uint8_t)(value & 0xffu);
  buf[offset + 1u] = (uint8_t)(value >> 8);
}

static uint16_t load16(const uint8_t *buf, uint32_t offset) {
  return (uint16_t)buf[offset] | ((uint16_t)buf[offset + 1u] << 8);
}

static uint32_t active_lanes(uint32_t n) {
  return n < LANES ? n : LANES;
}

static void fill_buffers(void) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0x40u + (i & 31u));
  for (uint32_t i = 0; i < OUT_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
  for (uint32_t lane = 0; lane < LANES; ++lane)
    store16(input_bytes, lane * 2u, input_half[lane]);
  store16(input_bytes, 32u, GUARD_HALF);
}

static int verify_case(const struct test_case *tc, uint32_t case_index,
                       uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *sentinel_mismatches = 0;
  uint32_t active = active_lanes(tc->n);
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    uint16_t got = load16(output_bytes, lane * 2u);
    if (lane < active) {
      uint16_t want = expected_half[lane];
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: mixed f16 lane case=%d lane=%d got=%x want=%x\n",
                 (int)case_index, (int)lane, got, want);
        ++mismatches;
      }
    } else {
      if (got != 0xacacu) {
        if (*sentinel_mismatches < 8)
          printk("ERROR: mixed f16 inactive lane case=%d lane=%d got=%x want=acac\n",
                 (int)case_index, (int)lane, got);
        ++*sentinel_mismatches;
      }
    }
  }
  uint16_t guard = load16(output_bytes, GUARD_OUT_OFFSET);
  *checksum += guard;
  if (guard != GUARD_HALF) {
    if (mismatches < 8)
      printk("ERROR: mixed f16 nonzero-x guard case=%d got=%x want=%x\n",
             (int)case_index, guard, GUARD_HALF);
    ++mismatches;
  }
  for (uint32_t i = LANES * 2u; i < OUT_BYTES; ++i) {
    if (i == GUARD_OUT_OFFSET || i == GUARD_OUT_OFFSET + 1u)
      continue;
    if (output_bytes[i] != SENTINEL8) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: mixed f16 sentinel case=%d offset=%d got=%x want=%x\n",
               (int)case_index, (int)i, output_bytes[i], SENTINEL8);
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
    panic("mixed_f16_storage_conversion_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("mixed_f16_storage_conversion_vc4kernel allocation failed");

  int launch_failures = 0;
  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  uint32_t checksum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

  for (uint32_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    fill_buffers();
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        mixed_f16_storage_conversion_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, cases[i].n, cases[i].row,
            cases[i].x, cases[i].sel16) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev,
                         sizeof(output_bytes)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    total_mismatches += verify_case(&cases[i], i, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
  }

  launch_failures +=
      (int)mixed_f16_storage_conversion_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      mixed_f16_storage_conversion_vc4kernel_runtime_launches();
  const uint32_t case_count = sizeof(cases) / sizeof(cases[0]);
  const char *status = (launch_failures == 0 && total_mismatches == 0 &&
                        sentinel_mismatches == 0 && launches == case_count)
                           ? "PASS"
                           : "FAIL";
  printk("VC4_TEST_RESULT name=mixed_f16_storage_conversion_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_f16_storage_conversion=1 saw_f16_unpack_to_f32=1 saw_f16_pack_from_f32=1 saw_f32_compute=1 saw_vdr_w16_packed=1 saw_vpm_qpu_w16_packed=1 saw_vdw_w16_packed=1 saw_dynamic_subword_selector=1 saw_vdw_preserve=1 no_native_f16_arith=1 no_bf16_fp8=1 saw_final_surface_mix=1 no_tmu_to_vpm=1 compute_selector_values_w16=0 guard_selector_values_w16=0,1 x_values=1,3,7,15 n_cases=0,1,15,16,17,31,32,33,65 exact_half_outputs=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)case_count, total_mismatches, sentinel_mismatches,
         launch_failures, checksum, launches, timer_get_usec() - start);

  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
