#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 7u
#define BUFFER_N (LANES * SEGMENTS + 16u)
#define SENTINEL 0xdeadbeefu

static const uint32_t a_values[LANES] = {
    0x00000000u, 0xffffffffu, 0x01020304u, 0x10203040u,
    0x7f8081feu, 0x0080ff01u, 0x55aa33ccu, 0xaa55cc33u,
    0x0f1e2d3cu, 0xf0e1d2c3u, 0x12345678u, 0x87654321u,
    0x00010203u, 0xfcfdfeffu, 0x7f7f7f7fu, 0x80808080u
};
static const uint32_t b_values[LANES] = {
    0x00000000u, 0xffffffffu, 0x04030201u, 0x40302010u,
    0x02017f80u, 0xff800100u, 0xcc33aa55u, 0x33cc55aau,
    0x3c2d1e0fu, 0xc3d2e1f0u, 0x11111111u, 0x22222222u,
    0xfffefdfcu, 0x03020100u, 0x01010101u, 0x80808080u
};
static uint32_t out_values[BUFFER_N];

static uint32_t byte_at(uint32_t v, uint32_t index) {
    return (v >> (index * 8u)) & 0xffu;
}

static uint32_t pack_byte_result(uint32_t a, uint32_t b, uint32_t op) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t av = byte_at(a, i);
        uint32_t bv = byte_at(b, i);
        uint32_t ov = 0;
        if (op == 0 || op == 5) ov = av + bv > 255u ? 255u : av + bv;
        else if (op == 1 || op == 6) ov = av < bv ? 0u : av - bv;
        else if (op == 2) ov = (av * bv + 127u) / 255u;
        else if (op == 3) ov = av < bv ? av : bv;
        else ov = av > bv ? av : bv;
        result |= (ov & 0xffu) << (i * 8u);
    }
    return result;
}

static uint32_t expected_value(uint32_t segment, uint32_t lane) {
    return pack_byte_result(a_values[lane], b_values[lane], segment);
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t index = segment * LANES + lane;
            uint32_t expected = expected_value(segment, lane);
            if (out_values[index] != expected) {
                if (mismatches < 12)
                    printk("ERROR: fragment_alu_v8_ops seg=%d lane=%d gpu=%x expected=%x\n",
                           (int)segment, (int)lane, out_values[index], expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = SEGMENTS * LANES; i < BUFFER_N; ++i) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_alu_v8_ops sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t a_dev = 0, b_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &b_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        panic("fragment_alu_v8_ops allocation failed");

    fill_output();
    int start = timer_get_usec();
    int launch_failures = 0;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, a_dev, a_values, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, b_dev, b_values, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, BUFFER_N * sizeof(uint32_t)) < 0 ||
        fragment_alu_v8_ops_vc4kernel_launch(program, grid, block, a_dev, b_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        launch_failures++;

    int total_mismatches = launch_failures ? 0 : verify_active();
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_alu_v8_ops_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d op_segments=%d unsigned_byte_oracle=1 checksum_accum=0 runtime_allocations=3 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)SEGMENTS, elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
