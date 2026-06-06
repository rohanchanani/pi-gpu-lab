#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define PREDICATE_SEGMENTS 10u
#define CONTROL_SEGMENTS 1u
#define SEGMENTS (PREDICATE_SEGMENTS + CONTROL_SEGMENTS)
#define CASES 6u
#define BUFFER_N (LANES * SEGMENTS * CASES + 16u)
#define SENTINEL 0xdeadbeefu

struct cmpi_case {
    uint32_t a;
    uint32_t b;
};

static const struct cmpi_case cases[CASES] = {
    {0u, 0u},
    {0x80000000u, 0u},
    {0x80000000u, 0x7fffffffu},
    {0xffffffffu, 1u},
    {0xffffffffu, 0u},
    {0x7fffffffu, 0x80000000u},
};

static uint32_t out_values[BUFFER_N];

static int32_t s32(uint32_t v) { return (int32_t)v; }

static int predicate_value(const struct cmpi_case *tc, uint32_t seg) {
    if (seg == 0) return tc->a == tc->b;
    if (seg == 1) return tc->a != tc->b;
    if (seg == 2) return s32(tc->a) < s32(tc->b);
    if (seg == 3) return s32(tc->a) <= s32(tc->b);
    if (seg == 4) return s32(tc->a) > s32(tc->b);
    if (seg == 5) return s32(tc->a) >= s32(tc->b);
    if (seg == 6) return tc->a < tc->b;
    if (seg == 7) return tc->a <= tc->b;
    if (seg == 8) return tc->a > tc->b;
    return tc->a >= tc->b;
}

static uint32_t expected_value(const struct cmpi_case *tc, uint32_t seg) {
    if (seg < PREDICATE_SEGMENTS)
        return predicate_value(tc, seg) ? (0x1000u + seg) : (0x2000u + seg);
    return predicate_value(tc, 2) ? 0x3000u : 0x4000u;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_active(const struct cmpi_case *tc, uint32_t case_id) {
    int mismatches = 0;
    uint32_t base = case_id * SEGMENTS * LANES;
    for (uint32_t seg = 0; seg < SEGMENTS; ++seg) {
        uint32_t expected = expected_value(tc, seg);
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t index = base + seg * LANES + lane;
            if (out_values[index] != expected) {
                if (mismatches < 16)
                    printk("ERROR: scalar_i32_cmpi case=%d seg=%d lane=%d gpu=%x expected=%x a=%x b=%x\n",
                           (int)case_id, (int)seg, (int)lane, out_values[index],
                           expected, tc->a, tc->b);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t case_id) {
    int mismatches = 0;
    uint32_t active_begin = case_id * SEGMENTS * LANES;
    uint32_t active_end = active_begin + SEGMENTS * LANES;
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        if (i >= active_begin && i < active_end)
            continue;
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: scalar_i32_cmpi sentinel i=%d gpu=%x expected=%x\n",
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
        panic("scalar_i32_cmpi_control allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    for (uint32_t case_id = 0; case_id < CASES; ++case_id) {
        fill_output();
        const struct cmpi_case *tc = &cases[case_id];
        uint32_t offset_elems = case_id * SEGMENTS * LANES;
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            scalar_i32_cmpi_control_vc4kernel_launch(program, grid, block,
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
        for (uint32_t seg = 0; seg < SEGMENTS; ++seg)
            checksum_accum += (int)expected_value(tc, seg);
        printk("SCALAR_I32_CMPI_CONTROL_CASE case=%d a=%x b=%x mismatches=%d sentinel_mismatches=%d\n",
               (int)case_id, tc->a, tc->b, mismatches, sentinels);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=scalar_i32_cmpi_control_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d predicate_segments=%d control_segments=%d checksum_accum=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)CASES, total_mismatches, sentinel_mismatches,
           launch_failures, (int)LANES, (int)PREDICATE_SEGMENTS,
           (int)CONTROL_SEGMENTS, checksum_accum, (int)CASES, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
