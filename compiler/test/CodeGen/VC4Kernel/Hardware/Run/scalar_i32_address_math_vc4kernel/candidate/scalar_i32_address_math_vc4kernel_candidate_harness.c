#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define CASES 6u
#define BUFFER_N 288u
#define SENTINEL 0xdeadbeefu

struct address_case {
    uint32_t selector;
    uint32_t stride;
};

static const struct address_case cases[CASES] = {
    {0u, 17u},
    {1u, 17u},
    {7u, 17u},
    {8u, 17u},
    {15u, 17u},
    {15u, 33u},
};

static uint32_t out_values[BUFFER_N];

static uint32_t expected_offset(const struct address_case *tc) {
    uint32_t masked = tc->selector & 7u;
    uint32_t group = tc->selector >> 3;
    return masked * tc->stride + (group << 1);
}

static uint32_t expected_value(const struct address_case *tc) {
    uint32_t addr = expected_offset(tc);
    uint32_t tag = tc->selector ^ 85u;
    return addr + tag;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_active(const struct address_case *tc, uint32_t case_id) {
    int mismatches = 0;
    uint32_t offset = expected_offset(tc);
    uint32_t expected = expected_value(tc);
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t index = offset + lane;
        if (index >= BUFFER_N || out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: scalar_i32_address case=%d lane=%d index=%d gpu=%x expected=%x selector=%x stride=%x\n",
                       (int)case_id, (int)lane, (int)index,
                       index < BUFFER_N ? out_values[index] : 0xffffffffu,
                       expected, tc->selector, tc->stride);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct address_case *tc) {
    int mismatches = 0;
    uint32_t offset = expected_offset(tc);
    uint32_t end = offset + LANES;
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        if (i >= offset && i < end)
            continue;
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: scalar_i32_address sentinel i=%d gpu=%x expected=%x offset=%d\n",
                       (int)i, out_values[i], SENTINEL, (int)offset);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("scalar_i32_address_math allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    for (uint32_t case_id = 0; case_id < CASES; ++case_id) {
        fill_output();
        const struct address_case *tc = &cases[case_id];
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            scalar_i32_address_math_vc4kernel_launch(program, grid, block,
                                                     out_dev, tc->selector,
                                                     tc->stride) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            launch_failures++;
            continue;
        }
        int mismatches = verify_active(tc, case_id);
        int sentinels = verify_sentinels(tc);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += (int)(expected_value(tc) & 0xffffu);
        printk("SCALAR_I32_ADDRESS_MATH_CASE case=%d selector=%x stride=%x offset=%d value=%x mismatches=%d sentinel_mismatches=%d\n",
               (int)case_id, tc->selector, tc->stride,
               (int)expected_offset(tc), expected_value(tc), mismatches,
               sentinels);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=scalar_i32_address_math_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d address_math_cases=%d checksum_accum=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)CASES, total_mismatches, sentinel_mismatches,
           launch_failures, (int)LANES, (int)CASES, checksum_accum,
           (int)CASES, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
