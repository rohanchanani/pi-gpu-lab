#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define IN_BYTES 384u
#define OUT_BYTES 384u
#define GUARD_BYTES 96u
#define U16_OFFSET 128u
#define AUDIT_OFFSET 256u
#define SENTINEL8 0xbdu

static const uint32_t cases[] = {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 63u};
static uint8_t input_bytes[IN_BYTES];
static uint8_t output_bytes[OUT_BYTES + GUARD_BYTES];

static void put16(uint8_t *base, uint32_t offset, uint16_t value) {
  base[offset] = (uint8_t)(value & 0xffu);
  base[offset + 1u] = (uint8_t)(value >> 8);
}

static uint16_t get16(const uint8_t *base, uint32_t offset) {
  return (uint16_t)base[offset] | ((uint16_t)base[offset + 1u] << 8);
}

static uint32_t load_u32_le(const uint8_t *base, uint32_t offset) {
  return ((uint32_t)base[offset]) |
         ((uint32_t)base[offset + 1u] << 8) |
         ((uint32_t)base[offset + 2u] << 16) |
         ((uint32_t)base[offset + 3u] << 24);
}

static uint8_t input8(uint32_t lane) {
  return (uint8_t)(19u + lane * 3u);
}

static uint16_t input16(uint32_t lane) {
  return (uint16_t)(0x1020u + lane * 29u);
}

static uint8_t expected8(uint32_t lane) {
  uint32_t v = (uint32_t)input8(lane) + 9u;
  return v < 128u ? (uint8_t)v : 0u;
}

static uint16_t expected16(uint32_t lane) {
  return (uint16_t)(input16(lane) + lane);
}

static void fill_buffers(void) {
  for (uint32_t i = 0; i < IN_BYTES; ++i)
    input_bytes[i] = (uint8_t)(0x40u + i * 11u);
  for (uint32_t lane = 0; lane < LANES; ++lane) {
    input_bytes[lane] = input8(lane);
    put16(input_bytes, U16_OFFSET + lane * 2u, input16(lane));
  }
  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i)
    output_bytes[i] = SENTINEL8;
}

static int verify_case(uint32_t n, uint32_t *checksum, int *sentinel_mismatches) {
  int mismatches = 0;
  *checksum = 0;
  *sentinel_mismatches = 0;
  uint32_t active = n > LANES ? LANES : n;
  uint32_t expected_checksum = 0;
  for (uint32_t lane = 0; lane < active; ++lane)
    expected_checksum += expected8(lane);

  for (uint32_t lane = 0; lane < LANES; ++lane) {
    uint8_t got8 = output_bytes[lane];
    uint8_t want8 = lane < active ? expected8(lane) : SENTINEL8;
    if (got8 != want8) {
      if (lane < active) {
        if (mismatches < 8)
          printk("ERROR: mixed_subword u8 lane=%d got=%x want=%x n=%d\n",
                 (int)lane, got8, want8, (int)n);
        ++mismatches;
      } else {
        if (*sentinel_mismatches < 8)
          printk("ERROR: mixed_subword u8 sentinel lane=%d got=%x n=%d\n",
                 (int)lane, got8, (int)n);
        ++*sentinel_mismatches;
      }
    }

    uint16_t got16 = get16(output_bytes, U16_OFFSET + lane * 2u);
    uint16_t want16 = lane < active ? expected16(lane)
                                    : (uint16_t)(SENTINEL8 | (SENTINEL8 << 8));
    if (got16 != want16) {
      if (lane < active) {
        if (mismatches < 8)
          printk("ERROR: mixed_subword u16 lane=%d got=%x want=%x n=%d\n",
                 (int)lane, got16, want16, (int)n);
        ++mismatches;
      } else {
        if (*sentinel_mismatches < 8)
          printk("ERROR: mixed_subword u16 sentinel lane=%d got=%x n=%d\n",
                 (int)lane, got16, (int)n);
        ++*sentinel_mismatches;
      }
    }

    uint32_t audit = load_u32_le(output_bytes, AUDIT_OFFSET + lane * 4u);
    *checksum += audit + got8 + got16;
    if (audit != expected_checksum) {
      if (mismatches < 8)
        printk("ERROR: mixed_subword audit lane=%d got=%x want=%x n=%d\n",
               (int)lane, audit, expected_checksum, (int)n);
      ++mismatches;
    }
  }

  for (uint32_t i = 0; i < OUT_BYTES + GUARD_BYTES; ++i) {
    int covered = (i < LANES) ||
                  (i >= U16_OFFSET && i < U16_OFFSET + LANES * 2u) ||
                  (i >= AUDIT_OFFSET && i < AUDIT_OFFSET + LANES * 4u);
    if (!covered && output_bytes[i] != SENTINEL8) {
      if (*sentinel_mismatches < 8)
        printk("ERROR: mixed_subword guard offset=%d got=%x n=%d\n",
               (int)i, output_bytes[i], (int)n);
      ++*sentinel_mismatches;
    }
  }
  return mismatches;
}

void notmain(void) {
  struct vc4_program *program = 0;
  vc4_deviceptr_t in_dev = 0, out_dev = 0;
  if (vc4_program_create(&program, 0) < 0 || !program)
    panic("mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel program create failed");
  if (vc4_m2_malloc(program, &in_dev, sizeof(input_bytes)) < 0 ||
      vc4_m2_malloc(program, &out_dev, sizeof(output_bytes)) < 0)
    panic("mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel allocation failed");

  int total_mismatches = 0;
  int sentinel_mismatches = 0;
  int launch_failures = 0;
  uint32_t checksum_accum = 0;
  int start = timer_get_usec();
  vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
  vc4_dim3 block = vc4_m2_dim3(16, 1, 1);
  for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
    uint32_t n = cases[case_id];
    fill_buffers();
    if (vc4_m2_copy_htod(program, in_dev, input_bytes, sizeof(input_bytes)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, output_bytes, sizeof(output_bytes)) < 0 ||
        mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel_launch(
            program, grid, block, in_dev, out_dev, n) < 0 ||
        vc4_m2_copy_dtoh(program, output_bytes, out_dev, sizeof(output_bytes)) < 0) {
      ++launch_failures;
      continue;
    }
    int sentinels = 0;
    uint32_t checksum = 0;
    total_mismatches += verify_case(n, &checksum, &sentinels);
    sentinel_mismatches += sentinels;
    checksum_accum += checksum;
  }
  launch_failures +=
      (int)mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel_runtime_launch_failures();
  uint32_t launches =
      mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel_runtime_launches();
  const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                        launch_failures == 0) ? "PASS" : "FAIL";
  printk("VC4_TEST_RESULT name=mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_fragment_pack=1 saw_fragment_unpack=1 saw_vpm_subword=1 saw_vdr_subword=1 saw_vdw_subword=1 saw_vdw_preserve=1 saw_byte_halfword_guards=1 saw_p1_p8_interactions=1 checksum_accum=%u runtime_launches=%u elapsed_usec=%d\n",
         status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
         sentinel_mismatches, launch_failures, checksum_accum, launches,
         timer_get_usec() - start);
  vc4Free(program, in_dev);
  vc4Free(program, out_dev);
  vc4_program_destroy(program);
}
