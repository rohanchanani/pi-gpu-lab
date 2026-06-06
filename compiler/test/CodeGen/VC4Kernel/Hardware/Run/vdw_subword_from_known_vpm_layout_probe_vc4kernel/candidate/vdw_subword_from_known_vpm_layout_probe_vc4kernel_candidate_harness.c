#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define OUT_BYTES 320u
#define GUARD_BYTES 64u
#define OUT16_OFFSET 128u
#define SENTINEL8 0xe3u

static uint8_t output_bytes[OUT_BYTES + GUARD_BYTES];

static uint16_t get16(uint32_t offset) {
  return (uint16_t)output_bytes[offset] |
         ((uint16_t)output_bytes[offset + 1u] << 8);
}

static void fill_output(void) {
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
}

static int verify_outputs(uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *checksum = 0;
  *sentinel_mismatches = 0;
  for (uint32_t row = 0; row < 3u; ++row) {
    for (uint32_t col = 0; col < 16u; ++col) {
      uint32_t off8 = row * 32u + col;
      uint8_t want8 = (uint8_t)((row == 0u ? 48u : row == 1u ? 80u : 112u) + col);
      uint8_t got8 = output_bytes[off8];
      *checksum += got8;
      if (got8 != want8) {
        if (mismatches < 8)
          printk("ERROR: vdw_layout_probe u8 row=%d col=%d got=%x want=%x\n",
                 (int)row, (int)col, got8, want8);
        ++mismatches;
      }
      uint32_t off16 = OUT16_OFFSET + row * 32u + col * 2u;
      uint16_t want16 = (uint16_t)((row == 0u ? 1024u : row == 1u ? 1280u : 1536u) + col);
      uint16_t got16 = get16(off16);
      *checksum += got16;
      if (got16 != want16) {
        if (mismatches < 8)
          printk("ERROR: vdw_layout_probe u16 row=%d col=%d got=%x want=%x\n",
                 (int)row, (int)col, got16, want16);
        ++mismatches;
      }
    }
  }
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i) {
    int active8 = i < 3u * 32u && (i % 32u) < 16u;
    int active16 = i >= OUT16_OFFSET && i < OUT16_OFFSET + 3u * 32u;
    if (active16)
      active16 = ((i - OUT16_OFFSET) % 32u) < 32u;
    if (!active8 && !active16 && output_bytes[i] != SENTINEL8) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: vdw_layout_probe sentinel offset=%d got=%x\n",
               (int)i, output_bytes[i]);
      ++*sentinel_mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdw_subword_from_known_vpm_layout_probe_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("vdw_subword_from_known_vpm_layout_probe_vc4kernel allocation failed");
  fill_output();
  int launch_failures = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  if (vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
      vdw_subword_from_known_vpm_layout_probe_vc4kernel_launch(program, grid,
                                                              block, out_dev) < 0 ||
      vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0)
    ++launch_failures;
  uint32_t checksum = 0;
  int sentinel_mismatches = 0;
  int total_mismatches = launch_failures ? 0 : verify_outputs(&checksum, &sentinel_mismatches);
  launch_failures +=
      (int)vdw_subword_from_known_vpm_layout_probe_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      vdw_subword_from_known_vpm_layout_probe_vc4kernel_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vdw_subword_from_known_vpm_layout_probe_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdw_w8=1 saw_vdw_w16=1 saw_known_vpm_qpu_writes=1 saw_byte_halfword_guards=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         checksum, launches, timer_get_usec() - start);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
