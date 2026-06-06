#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 27u
#define BUFFER_N (LANES * SEGMENTS + 16u)
#define SENTINEL 0xdeadbeefu

static const uint32_t a_values[LANES] = {
    0u, 1u, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0x55555555u,
    0xaaaaaaaau, 0x00ff00ffu, 0xff00ff00u, 0x12345678u, 0x87654321u,
    0x00010000u, 0x00000080u, 0x01000000u, 0x0f0f0f0fu, 0xf0f0f0f0u
};
static const uint32_t b_values[LANES] = {
    0u, 7u, 1u, 0xffffffffu, 0x80000000u, 0x11111111u,
    0x22222222u, 0xff00ff00u, 0x00ff00ffu, 0x01020304u, 0x10203040u,
    31u, 15u, 0xaaaaaaaa, 0x13579bdfu, 0x2468ace0u
};
static uint32_t out_values[BUFFER_N];

static uint32_t clz32(uint32_t v) {
    if (v == 0)
        return 32u;
    uint32_t n = 0;
    for (int bit = 31; bit >= 0; --bit) {
        if ((v >> bit) & 1u)
            break;
        n++;
    }
    return n;
}

static uint32_t ror32(uint32_t v, uint32_t s) {
    s &= 31u;
    return s == 0 ? v : ((v >> s) | (v << (32u - s)));
}

static uint32_t asr32(uint32_t v, uint32_t s) {
    s &= 31u;
    return (uint32_t)(((int32_t)v) >> s);
}

static uint32_t expected_value(uint32_t segment, uint32_t lane) {
    uint32_t a = a_values[lane];
    uint32_t b = b_values[lane];
    static const uint32_t shifts[5] = {0u, 1u, 7u, 15u, 31u};
    if (segment == 0) return a + b;
    if (segment == 1) return a - b;
    if (segment == 2) return a & b;
    if (segment == 3) return a | b;
    if (segment == 4) return a ^ b;
    if (segment == 5) return ~a;
    if (segment == 6) return clz32(a);
    if (segment >= 7 && segment <= 11) return a << shifts[segment - 7];
    if (segment >= 12 && segment <= 16) return a >> shifts[segment - 12];
    if (segment >= 17 && segment <= 21) return asr32(a, shifts[segment - 17]);
    return ror32(a, shifts[segment - 22]);
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
                if (mismatches < 16)
                    printk("ERROR: fragment_alu_add_i32_logic_shift seg=%d lane=%d gpu=%x expected=%x\n",
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
                printk("ERROR: fragment_alu_add_i32_logic_shift sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t checksum_low16(void) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < SEGMENTS * LANES; ++i)
        checksum += out_values[i] & 0xffffu;
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t a_dev = 0, b_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &b_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        panic("fragment_alu_add_i32_logic_shift allocation failed");

    fill_output();
    int start = timer_get_usec();
    int launch_failures = 0;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, a_dev, a_values, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, b_dev, b_values, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, BUFFER_N * sizeof(uint32_t)) < 0 ||
        fragment_alu_add_i32_logic_shift_vc4kernel_launch(program, grid, block, a_dev, b_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        launch_failures++;

    int total_mismatches = launch_failures ? 0 : verify_active();
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int checksum = launch_failures ? 0 : (int)checksum_low16();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_alu_add_i32_logic_shift_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d op_segments=%d checksum_accum=%d runtime_allocations=3 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)SEGMENTS, checksum, elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
