#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 6u
#define OUT_WORDS (LANES * SEGMENTS + 16u)
#define IN_BYTES 1024u
#define SENTINEL 0xd17ad17au

static uint8_t input_bytes[IN_BYTES];
static uint32_t output_words[OUT_WORDS];

static uint8_t pattern8(uint32_t row, uint32_t col) {
  return (uint8_t)(0x30u + row * 17u + col);
}

static uint16_t pattern16(uint32_t row, uint32_t col) {
  return (uint16_t)(0x400u + row * 257u + col * 3u);
}

static void put16(uint32_t offset, uint16_t value) {
  input_bytes[offset] = (uint8_t)(value & 0xffu);
  input_bytes[offset + 1u] = (uint8_t)(value >> 8);
}

static void fill_buffers(void) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0x90u + i);
  for (uint32_t row = 0; row < 3u; ++row) {
    for (uint32_t col = 0; col < LANES; ++col) {
      input_bytes[row * 32u + col] = pattern8(row, col);
      put16(512u + row * 32u + col * 2u, pattern16(row, col));
    }
  }
  for (uint32_t i = 0; i < OUT_WORDS; ++i)
    output_words[i] = SENTINEL;
}

static uint32_t expected_value(uint32_t segment, uint32_t lane) {
  if (segment < 3u)
    return pattern8(segment, lane);
  return pattern16(segment - 3u, lane);
}

static int verify_outputs(uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *checksum = 0;
  *sentinel_mismatches = 0;
  for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
    for (uint32_t lane = 0; lane < LANES; ++lane) {
      uint32_t index = segment * LANES + lane;
      uint32_t want = expected_value(segment, lane);
      uint32_t got = output_words[index];
      *checksum += got;
      if (got != want) {
        if (mismatches < 8)
          printk("ERROR: vdr_layout_probe seg=%d lane=%d got=%x want=%x\n",
                 (int)segment, (int)lane, got, want);
        ++mismatches;
      }
    }
  }
  for (uint32_t i = SEGMENTS * LANES; i < OUT_WORDS; ++i) {
    if (output_words[i] != SENTINEL) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: vdr_layout_probe sentinel index=%d got=%x\n",
               (int)i, output_words[i]);
      ++*sentinel_mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("vdr_subword_vpm_layout_probe_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_words)) < 0)
    panic("vdr_subword_vpm_layout_probe_vc4kernel allocation failed");
  fill_buffers();
  int launch_failures = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
      vc4_m2_copy_htod(program, out_dev, output_words, sizeof(output_words)) < 0 ||
      vdr_subword_vpm_layout_probe_vc4kernel_launch(program, grid, block,
                                                    in_dev, out_dev) < 0 ||
      vc4_m2_copy_dtoh(program, output_words, out_dev, sizeof(output_words)) < 0)
    ++launch_failures;
  uint32_t checksum = 0;
  int sentinel_mismatches = 0;
  int total_mismatches = launch_failures ? 0 : verify_outputs(&checksum, &sentinel_mismatches);
  launch_failures +=
      (int)vdr_subword_vpm_layout_probe_vc4kernel_runtime_launch_failures();
  uint32_t launches = vdr_subword_vpm_layout_probe_vc4kernel_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=vdr_subword_vpm_layout_probe_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_w8=1 saw_vdr_w16=1 saw_vpm_qpu_readback=1 saw_pack_unpack=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, total_mismatches, sentinel_mismatches, launch_failures,
         checksum, launches, timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
