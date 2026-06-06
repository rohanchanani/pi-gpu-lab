#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define RAW_SEGMENTS 4u
#define CONSUMER_SEGMENTS 2u
#define SEGMENTS (RAW_SEGMENTS + CONSUMER_SEGMENTS)
#define ACTIVE_N (LANES * SEGMENTS)
#define BUFFER_N (ACTIVE_N + 16u)
#define SENTINEL 0xdeadbeefu

static const uint32_t input_values[LANES] = {
    0x80000000u, 0xffffffffu, 0u, 5u, 0x7fffffffu, 0xaaaaaaaau,
    17u, 0x55555555u, 3u, 0xfffffffeu, 0x12345678u, 0x87654321u,
    0x00010001u, 0xffff0000u, 9u, 0x40000000u
};

static uint32_t out_values[BUFFER_N];

static int sparse(uint32_t lane) {
    return lane == 0u || lane == 2u || lane == 6u || lane == 11u;
}

static int32_t as_i32(uint32_t value) { return (int32_t)value; }

static uint32_t reduce_add_sparse(void) {
    uint32_t acc = 0u;
    for (uint32_t lane = 0; lane < LANES; lane++)
        if (sparse(lane))
            acc += input_values[lane];
    return acc;
}

static uint32_t reduce_min_s_full(void) {
    uint32_t acc = 0x7fffffffu;
    for (uint32_t lane = 0; lane < LANES; lane++)
        if (as_i32(input_values[lane]) < as_i32(acc))
            acc = input_values[lane];
    return acc;
}

static uint32_t reduce_max_u_sparse(void) {
    uint32_t acc = 0u;
    for (uint32_t lane = 0; lane < LANES; lane++)
        if (sparse(lane) && input_values[lane] > acc)
            acc = input_values[lane];
    return acc;
}

static uint32_t reduce_xor_full(void) {
    uint32_t acc = 0u;
    for (uint32_t lane = 0; lane < LANES; lane++)
        acc ^= input_values[lane];
    return acc;
}

static uint32_t expected_raw(uint32_t segment) {
    if (segment == 0u)
        return reduce_add_sparse();
    if (segment == 1u)
        return reduce_min_s_full();
    if (segment == 2u)
        return reduce_max_u_sparse();
    return reduce_xor_full();
}

static uint32_t expected_value(uint32_t segment, uint32_t lane) {
    if (segment < RAW_SEGMENTS)
        return expected_raw(segment);
    if (segment == 4u)
        return reduce_add_sparse() + lane;
    return lane < 8u ? reduce_min_s_full() : reduce_max_u_sparse();
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; i++)
        out_values[i] = SENTINEL;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t segment = 0; segment < SEGMENTS; segment++) {
        for (uint32_t lane = 0; lane < LANES; lane++) {
            uint32_t index = segment * LANES + lane;
            uint32_t expected = expected_value(segment, lane);
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: reduce_broadcast_consumers segment=%d lane=%d gpu=%x expected=%x\n",
                           (int)segment, (int)lane, out_values[index],
                           expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = ACTIVE_N; i < BUFFER_N; i++) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: reduce_broadcast_consumers sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < ACTIVE_N; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    uint32_t input_bytes = LANES * sizeof(uint32_t);
    uint32_t out_bytes = BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &input_dev, input_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("fragment_reduce_i32_broadcast_consumers allocation failed");

    fill_output();
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, input_dev, input_values, input_bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
        fragment_reduce_i32_broadcast_consumers_vc4kernel_launch(program, grid, block,
                                                                 input_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0)
        launch_failures++;

    int total_mismatches = launch_failures ? 0 : verify_active();
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int checksum_accum = launch_failures ? 0 : checksum_low16();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_reduce_i32_broadcast_consumers_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d raw_broadcast_segments=%d consumer_segments=%d checksum_accum=%d runtime_allocations=2 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)RAW_SEGMENTS, (int)CONSUMER_SEGMENTS,
           checksum_accum, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
