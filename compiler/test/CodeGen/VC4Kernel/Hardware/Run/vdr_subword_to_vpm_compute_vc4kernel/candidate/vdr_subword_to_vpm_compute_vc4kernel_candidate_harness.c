#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define IN_BYTES 256u
#define OUT_BYTES 320u
#define GUARD_BYTES 64u
#define OUT16_OFFSET 128u
#define SENTINEL8 0xacu

static uint8_t input_bytes[IN_BYTES];
static uint8_t output_bytes[OUT_BYTES + GUARD_BYTES];

static void put16(uint32_t offset, uint16_t value) {
  input_bytes[offset] = (uint8_t)(value & 0xffu);
  input_bytes[offset + 1u] = (uint8_t)(value >> 8);
}

static uint16_t out16(uint32_t offset) {
  return (uint16_t)output_bytes[offset] |
         ((uint16_t)output_bytes[offset + 1u] << 8);
}

static void fill_buffers(void) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0x40u + i);
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    input_bytes[lane] = (uint8_t)(11u + lane);
    put16(64u + lane * 2u, (uint16_t)(1000u + lane * 5u));
  }
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
}

static int verify_outputs(uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *checksum = 0;
  *sentinel_mismatches = 0;
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    uint8_t want8 = (uint8_t)(18u + lane);
    uint8_t got8 = output_bytes[lane];
    *checksum += got8;
    if (got8 != want8) {
      if (mismatches < 8)
        printk("ERROR: vdr_subword_compute u8 lane=%d got=%x want=%x\n",
               (int)lane, got8, want8);
      ++mismatches;
    }
    uint16_t want16 = (uint16_t)(1031u + lane * 5u);
    uint16_t got16 = out16(OUT16_OFFSET + lane * 2u);
    *checksum += got16;
    if (got16 != want16) {
      if (mismatches < 8)
        printk("ERROR: vdr_subword_compute u16 lane=%d got=%x want=%x\n",
               (int)lane, got16, want16);
      ++mismatches;
    }
  }
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i) {
    int active = (i < LANES) ||
                 (i >= OUT16_OFFSET && i < OUT16_OFFSET + LANES * 2u);
    if (!active && output_bytes[i] != SENTINEL8) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: vdr_subword_compute sentinel offset=%d got=%x\n",
               (int)i, output_bytes[i]);
      ++*sentinel_mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_subword_to_vpm_compute_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("vdr_subword_to_vpm_compute_vc4kernel allocation failed");
  fill_buffers();
  int launch_failures = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
      vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
      vdr_subword_to_vpm_compute_vc4kernel_launch(program, grid, block,
                                                  in_dev, out_dev) < 0 ||
      vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0)
    ++launch_failures;
  uint32_t checksum = 0;
  int sentinel_mismatches = 0;
  int total_mismatches = launch_failures ? 0 : verify_outputs(&checksum, &sentinel_mismatches);
  launch_failures +=
      (int)vdr_subword_to_vpm_compute_vc4kernel_runtime_launch_failures();
  uint32_t launches = vdr_subword_to_vpm_compute_vc4kernel_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_subword_to_vpm_compute_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_w8=1 saw_vdr_w16=1 saw_vdw_w8=1 saw_vdw_w16=1 saw_byte_halfword_guards=1 saw_pack_unpack=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         checksum, launches, timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
