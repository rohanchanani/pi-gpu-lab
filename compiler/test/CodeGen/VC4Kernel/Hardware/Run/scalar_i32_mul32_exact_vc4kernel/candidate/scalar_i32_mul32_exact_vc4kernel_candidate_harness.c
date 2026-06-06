#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define CASES 8u
#define BUFFER_N (LANES * CASES + 16u)
#define SENTINEL 0xdeadbeefu

struct mul32_case {
    uint32_t a;
    uint32_t b;
};

static const struct mul32_case cases[CASES] = {
    {3u, 7u},
    {0x00ffffffu, 5u},
    {0x01000001u, 0x00010003u},
    {0xffffffffu, 2u},
    {0xfffffffeu, 0xfffffffdu},
    {0x80000000u, 2u},
    {0xffffffffu, 0xffffffffu},
    {0x12345678u, 0x87654321u},
};

static uint32_t out_values[BUFFER_N];

static uint32_t expected_product(const struct mul32_case *tc) {
    return (uint32_t)((uint64_t)tc->a * (uint64_t)tc->b);
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_active(const struct mul32_case *tc, uint32_t case_id) {
    int mismatches = 0;
    uint32_t expected = expected_product(tc);
    uint32_t base = case_id * LANES;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t index = base + lane;
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: scalar_i32_mul32 case=%d lane=%d gpu=%x expected=%x a=%x b=%x\n",
                       (int)case_id, (int)lane, out_values[index], expected,
                       tc->a, tc->b);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t case_id) {
    int mismatches = 0;
    uint32_t active_begin = case_id * LANES;
    uint32_t active_end = active_begin + LANES;
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        if (i >= active_begin && i < active_end)
            continue;
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: scalar_i32_mul32 sentinel i=%d gpu=%x expected=%x\n",
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

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("scalar_i32_mul32_exact allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    for (uint32_t case_id = 0; case_id < CASES; ++case_id) {
        fill_output();
        const struct mul32_case *tc = &cases[case_id];
        uint32_t offset_elems = case_id * LANES;
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            scalar_i32_mul32_exact_vc4kernel_launch(program, grid, block,
                                                    out_dev, tc->a, tc->b,
                                                    offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            launch_failures++;
            continue;
        }
        int mismatches = verify_active(tc, case_id);
        int sentinels = verify_sentinels(case_id);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += (int)(expected_product(tc) & 0xffffu);
        printk("SCALAR_I32_MUL32_EXACT_CASE case=%d a=%x b=%x expected=%x mismatches=%d sentinel_mismatches=%d\n",
               (int)case_id, tc->a, tc->b, expected_product(tc), mismatches,
               sentinels);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=scalar_i32_mul32_exact_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d mul32_edge_cases=%d checksum_accum=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)CASES, total_mismatches, sentinel_mismatches,
           launch_failures, (int)LANES, (int)CASES, checksum_accum,
           (int)CASES, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
