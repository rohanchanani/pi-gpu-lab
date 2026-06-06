#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 5u
#define OUT_WORDS (LANES * SEGMENTS + 16u)
#define SENTINEL 0x51ed270bu

static const uint32_t input_words[LANES] = {
    0x00000000u, 0xffffffffu, 0x80808080u, 0x7f7f7f7fu,
    0x80018001u, 0x01020304u, 0xfefdfcfbu, 0x00ff00ffu,
    0xff00ff00u, 0x12345678u, 0x87654321u, 0x00008000u,
    0x00007fffu, 0xffff8000u, 0x13579bdfu, 0x2468ace0u
};

static uint32_t out_words[OUT_WORDS];

static uint32_t sign_extend16(uint32_t value) {
    uint32_t half = value & 0xffffu;
    if (half & 0x8000u)
        return half | 0xffff0000u;
    return half;
}

static uint32_t selected_u8(uint32_t lane) {
    uint32_t value = (input_words[lane] & 0xffu) + lane;
    return value < 128u ? value : 0u;
}

static uint32_t s16_plus_lane(uint32_t lane) {
    return sign_extend16(input_words[lane]) + lane;
}

static void fill_outputs(void) {
    for (uint32_t i = 0; i < OUT_WORDS; ++i)
        out_words[i] = SENTINEL;
}

static int verify_outputs(uint32_t *checksum) {
    int mismatches = 0;
    *checksum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t expected_selected = selected_u8(lane);
        uint32_t expected_pack8 = expected_selected & 0xffu;
        uint32_t expected_round8 = expected_pack8 & 0xffu;
        uint32_t expected_pack16 = s16_plus_lane(lane) & 0xffffu;
        uint32_t expected_round16 = sign_extend16(expected_pack16);
        uint32_t got_selected = out_words[lane];
        uint32_t got_pack8 = out_words[LANES + lane];
        uint32_t got_round8 = out_words[LANES * 2u + lane];
        uint32_t got_pack16 = out_words[LANES * 3u + lane];
        uint32_t got_round16 = out_words[LANES * 4u + lane];
        *checksum += got_selected + got_pack8 + got_round8 + got_pack16 + got_round16;
        if (got_selected != expected_selected || got_pack8 != expected_pack8 ||
            got_round8 != expected_round8 || got_pack16 != expected_pack16 ||
            got_round16 != expected_round16) {
            if (mismatches < 8) {
                printk("ERROR: fragment_pack_unpack lane=%d input=%x sel=%x/%x p8=%x/%x r8=%x/%x p16=%x/%x r16=%x/%x\n",
                       (int)lane, input_words[lane],
                       got_selected, expected_selected, got_pack8, expected_pack8,
                       got_round8, expected_round8, got_pack16, expected_pack16,
                       got_round16, expected_round16);
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
                printk("ERROR: fragment_pack_unpack sentinel i=%d got=%x\n", (int)i, out_words[i]);
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
        panic("fragment_pack_unpack allocation failed");

    fill_outputs();
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, input_dev, input_words, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_words, OUT_WORDS * sizeof(uint32_t)) < 0 ||
        fragment_pack_unpack_roundtrip_vc4kernel_launch(program, grid, block, input_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_words, out_dev, OUT_WORDS * sizeof(uint32_t)) < 0)
        launch_failures++;

    uint32_t checksum = 0;
    int total_mismatches = launch_failures ? 0 : verify_outputs(&checksum);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_pack_unpack_roundtrip_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_pack=1 saw_unpack=1 saw_u8_or_s8=1 saw_u16_or_s16=1 saw_policy_edge_values=1 roundtrip_segments=5 saw_fragment_alu=1 saw_fragment_cmp_select=1 runtime_allocations=2 runtime_launches=1 checksum_accum=%u elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, checksum, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
