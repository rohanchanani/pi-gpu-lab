#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 2u
#define OUT_WORDS (LANES * SEGMENTS + 16u)
#define SENTINEL 0x6d2b79f5u

static const uint32_t input_words[LANES] = {
    0x00000000u, 0x00000001u, 0x0000007fu, 0x00000080u,
    0x000000ffu, 0x00000100u, 0x00000123u, 0x00007fffu,
    0x00008000u, 0x0000ffffu, 0x00010000u, 0x12345678u,
    0xffffffffu, 0xffffff80u, 0x80000001u, 0x7fffffff
};

static uint32_t out_words[OUT_WORDS];

static uint32_t expect_pack_u8(uint32_t value) {
    return value & 0xffu;
}

static uint32_t expect_pack_u16(uint32_t value) {
    return value & 0xffffu;
}

static void fill_outputs(void) {
    for (uint32_t i = 0; i < OUT_WORDS; ++i)
        out_words[i] = SENTINEL;
}

static int verify_outputs(uint32_t *checksum) {
    int mismatches = 0;
    *checksum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t expected_u8 = expect_pack_u8(input_words[lane]);
        uint32_t expected_u16 = expect_pack_u16(input_words[lane]);
        uint32_t got_u8 = out_words[lane];
        uint32_t got_u16 = out_words[LANES + lane];
        *checksum += got_u8 + got_u16;
        if (got_u8 != expected_u8 || got_u16 != expected_u16) {
            if (mismatches < 8) {
                printk("ERROR: fragment_pack lane=%d got_u8=%x exp_u8=%x got_u16=%x exp_u16=%x input=%x\n",
                       (int)lane, got_u8, expected_u8, got_u16, expected_u16,
                       input_words[lane]);
            }
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES * SEGMENTS; i < OUT_WORDS; ++i) {
        if (out_words[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_pack sentinel i=%d got=%x\n", (int)i, out_words[i]);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t input_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &out_dev, OUT_WORDS * sizeof(uint32_t)) < 0)
        panic("fragment_pack allocation failed");

    fill_outputs();
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, input_dev, input_words, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_words, OUT_WORDS * sizeof(uint32_t)) < 0 ||
        fragment_pack_subword_vc4kernel_launch(program, grid, block, input_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_words, out_dev, OUT_WORDS * sizeof(uint32_t)) < 0)
        launch_failures++;

    uint32_t checksum = 0;
    int total_mismatches = launch_failures ? 0 : verify_outputs(&checksum);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_pack_subword_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_pack=1 saw_unpack=0 saw_u8_or_s8=1 saw_u16_or_s16=1 saw_policy_edge_values=1 pack_modes=2 runtime_allocations=2 runtime_launches=1 checksum_accum=%u elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, checksum, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
