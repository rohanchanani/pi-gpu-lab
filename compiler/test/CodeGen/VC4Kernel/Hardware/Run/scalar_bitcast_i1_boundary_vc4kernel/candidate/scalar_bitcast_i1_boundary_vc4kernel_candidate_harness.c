#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 8u
#define CASES 6u
#define BUFFER_N (LANES * SEGMENTS * CASES + 16u)
#define SENTINEL 0xdeadbeefu

struct scalar_boundary_case {
    uint32_t bits;
    uint32_t x;
    uint32_t y;
};

static const struct scalar_boundary_case cases[CASES] = {
    {0x00000000u, 3u, 1u},
    {0x80000000u, 2u, 1u},
    {0x3f800000u, 1u, 3u},
    {0xbf800000u, 0u, 3u},
    {0x7fc00000u, 0xffffffffu, 0u},
    {0xff800001u, 0x80000000u, 0x7fffffffu},
};

static uint32_t out_values[BUFFER_N];

static int32_t s32(uint32_t v) { return (int32_t)v; }

static uint32_t expected_segment(const struct scalar_boundary_case *tc,
                                 uint32_t segment) {
    uint32_t gt = s32(tc->x) > s32(tc->y) ? 1u : 0u;
    uint32_t low = tc->x & 1u;
    uint32_t and_v = gt & low;
    uint32_t or_v = gt | low;
    uint32_t xor_v = gt ^ low;
    if (segment == 0) return tc->bits;
    if (segment == 1) return 0x3f800000u;
    if (segment == 2) return and_v;
    if (segment == 3) return or_v;
    if (segment == 4) return xor_v;
    if (segment == 5) return low;
    if (segment == 6) return xor_v ? 0x11111111u : 0x22222222u;
    return and_v ? 0x33333333u : 0x44444444u;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_active(const struct scalar_boundary_case *tc,
                         uint32_t case_id) {
    int mismatches = 0;
    uint32_t base = case_id * SEGMENTS * LANES;
    for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
        uint32_t expected = expected_segment(tc, segment);
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t index = base + segment * LANES + lane;
            if (out_values[index] != expected) {
                if (mismatches < 16)
                    printk("ERROR: scalar_bitcast_i1 case=%d segment=%d lane=%d gpu=%x expected=%x bits=%x x=%x y=%x\n",
                           (int)case_id, (int)segment, (int)lane,
                           out_values[index], expected, tc->bits, tc->x, tc->y);
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
                printk("ERROR: scalar_bitcast_i1 sentinel i=%d gpu=%x expected=%x\n",
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
        panic("scalar_bitcast_i1_boundary allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    for (uint32_t case_id = 0; case_id < CASES; ++case_id) {
        fill_output();
        const struct scalar_boundary_case *tc = &cases[case_id];
        uint32_t offset_elems = case_id * SEGMENTS * LANES;
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            scalar_bitcast_i1_boundary_vc4kernel_launch(
                program, grid, block, out_dev, tc->bits, tc->x, tc->y,
                offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            launch_failures++;
            continue;
        }
        int mismatches = verify_active(tc, case_id);
        int sentinels = verify_sentinels(case_id);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        for (uint32_t segment = 0; segment < SEGMENTS; ++segment)
            checksum_accum += (int)(expected_segment(tc, segment) & 0xffffu);
        printk("SCALAR_BITCAST_I1_BOUNDARY_CASE case=%d bits=%x x=%x y=%x mismatches=%d sentinel_mismatches=%d\n",
               (int)case_id, tc->bits, tc->x, tc->y, mismatches, sentinels);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=scalar_bitcast_i1_boundary_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d bitcast_roundtrip_cases=%d i1_boundary_segments=%d checksum_accum=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)CASES, total_mismatches, sentinel_mismatches,
           launch_failures, (int)LANES, (int)CASES, (int)(SEGMENTS - 2u),
           checksum_accum, (int)CASES, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
