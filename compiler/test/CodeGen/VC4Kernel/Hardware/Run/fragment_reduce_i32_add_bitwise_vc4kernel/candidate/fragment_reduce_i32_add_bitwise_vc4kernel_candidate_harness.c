#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define KINDS 4u
#define PREDICATES 5u
#define SEGMENTS (KINDS * PREDICATES)
#define ACTIVE_N (LANES * SEGMENTS)
#define BUFFER_N (ACTIVE_N + 16u)
#define SENTINEL 0xdeadbeefu

static const uint32_t input_values[LANES] = {
    0u, 1u, 0xffffffffu, 0xaaaaaaaau, 0x55555555u, 16u, 31u,
    0x80000000u, 0x7fffffffu, 0x13579bdfu, 0x2468ace0u,
    0xf0f0f0f0u, 0x0f0f0f0fu, 0x00010001u, 0xffff0000u, 3u
};

static uint32_t out_values[BUFFER_N];

static int active(uint32_t predicate, uint32_t lane) {
    if (predicate == 0u)
        return 1;
    if (predicate == 1u)
        return 0;
    if (predicate == 2u)
        return lane < 5u;
    if (predicate == 3u)
        return lane < 12u;
    return lane == 0u || lane == 3u || lane == 5u || lane == 9u ||
           lane == 14u;
}

static uint32_t identity(uint32_t kind) {
    if (kind == 1u)
        return 0xffffffffu;
    return 0u;
}

static uint32_t combine(uint32_t kind, uint32_t acc, uint32_t value) {
    if (kind == 0u)
        return acc + value;
    if (kind == 1u)
        return acc & value;
    if (kind == 2u)
        return acc | value;
    return acc ^ value;
}

static uint32_t expected_value(uint32_t kind, uint32_t predicate) {
    uint32_t acc = identity(kind);
    for (uint32_t lane = 0; lane < LANES; lane++) {
        if (active(predicate, lane))
            acc = combine(kind, acc, input_values[lane]);
    }
    return acc;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; i++)
        out_values[i] = SENTINEL;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t kind = 0; kind < KINDS; kind++) {
        for (uint32_t predicate = 0; predicate < PREDICATES; predicate++) {
            uint32_t segment = kind * PREDICATES + predicate;
            uint32_t expected = expected_value(kind, predicate);
            for (uint32_t lane = 0; lane < LANES; lane++) {
                uint32_t index = segment * LANES + lane;
                if (out_values[index] != expected) {
                    if (mismatches < 8)
                        printk("ERROR: reduce_add_bitwise kind=%d pred=%d lane=%d gpu=%x expected=%x\n",
                               (int)kind, (int)predicate, (int)lane,
                               out_values[index], expected);
                    mismatches++;
                }
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
                printk("ERROR: reduce_add_bitwise sentinel i=%d gpu=%x expected=%x\n",
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
        panic("fragment_reduce_i32_add_bitwise allocation failed");

    fill_output();
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, input_dev, input_values, input_bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
        fragment_reduce_i32_add_bitwise_vc4kernel_launch(program, grid, block,
                                                         input_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0)
        launch_failures++;

    int total_mismatches = launch_failures ? 0 : verify_active();
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int checksum_accum = launch_failures ? 0 : checksum_low16();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_reduce_i32_add_bitwise_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d kinds=%d predicates=%d checksum_accum=%d runtime_allocations=2 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)KINDS, (int)PREDICATES, checksum_accum, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
