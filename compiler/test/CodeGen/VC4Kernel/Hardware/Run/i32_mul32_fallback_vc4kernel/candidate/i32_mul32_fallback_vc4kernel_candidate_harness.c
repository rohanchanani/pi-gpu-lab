#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define I32_MUL32_FALLBACK_VC4KERNEL_LANES 16u
#define I32_MUL32_FALLBACK_VC4KERNEL_ACTIVE_QPUS 1u
#define I32_MUL32_FALLBACK_VC4KERNEL_MAX_N 48u
#define I32_MUL32_FALLBACK_VC4KERNEL_GUARD 16u
#define I32_MUL32_FALLBACK_VC4KERNEL_BUFFER_N (I32_MUL32_FALLBACK_VC4KERNEL_MAX_N + I32_MUL32_FALLBACK_VC4KERNEL_GUARD)
#define I32_MUL32_FALLBACK_VC4KERNEL_SENTINEL 0xdeadbeefu

struct i32_mul32_fallback_case {
    uint32_t lhs_base;
    uint32_t rhs_base;
    uint32_t offset_elems;
};

static const struct i32_mul32_fallback_case cases[] = {
    {0x01000001u, 0x00010003u, 0u},
    {0xfffffff0u, 0x00000021u, 16u},
    {0x80000005u, 0x00010007u, 32u},
};

static uint32_t out_values[I32_MUL32_FALLBACK_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < I32_MUL32_FALLBACK_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = I32_MUL32_FALLBACK_VC4KERNEL_SENTINEL;
}

static uint32_t expected_value(uint32_t lhs_base, uint32_t rhs_base,
                               uint32_t lane) {
    uint32_t lhs = lhs_base + lane;
    uint32_t rhs = rhs_base + lane;
    return (uint32_t)((uint64_t)lhs * (uint64_t)rhs);
}

static int verify_active(const struct i32_mul32_fallback_case *tc) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < I32_MUL32_FALLBACK_VC4KERNEL_LANES; lane++) {
        uint32_t index = tc->offset_elems + lane;
        uint32_t expected = expected_value(tc->lhs_base, tc->rhs_base, lane);
        if (out_values[index] != expected) {
            if (mismatches < 8)
                printk("ERROR: i32_mul32_fallback index=%d lhs=%x rhs=%x gpu=%x expected=%x\n",
                       (int)index, tc->lhs_base + lane, tc->rhs_base + lane,
                       out_values[index], expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t offset_elems) {
    int mismatches = 0;
    uint32_t active_end = offset_elems + I32_MUL32_FALLBACK_VC4KERNEL_LANES;
    for (uint32_t i = 0; i < I32_MUL32_FALLBACK_VC4KERNEL_BUFFER_N; i++) {
        if (i >= offset_elems && i < active_end)
            continue;
        if (out_values[i] != I32_MUL32_FALLBACK_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: i32_mul32_fallback sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i],
                       I32_MUL32_FALLBACK_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t offset_elems) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < I32_MUL32_FALLBACK_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[offset_elems + lane] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = I32_MUL32_FALLBACK_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("i32_mul32_fallback_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(I32_MUL32_FALLBACK_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 i32_mul32_fallback_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct i32_mul32_fallback_case *tc = &cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            i32_mul32_fallback_vc4kernel_launch(program, grid, block, out_dev,
                                                tc->lhs_base, tc->rhs_base,
                                                tc->offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: i32_mul32_fallback launch/copy failed case=%d lhs=%x rhs=%x offset=%d\n",
                   (int)case_id, tc->lhs_base, tc->rhs_base,
                   (int)tc->offset_elems);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(tc);
        int sentinels = verify_sentinels(tc->offset_elems);
        int checksum = checksum_low16(tc->offset_elems);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("I32_MUL32_FALLBACK_VC4KERNEL_CASE case=%d lhs_base=%x rhs_base=%x offset=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, tc->lhs_base, tc->rhs_base,
               (int)tc->offset_elems, mismatches, sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=i32_mul32_fallback_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           I32_MUL32_FALLBACK_VC4KERNEL_ACTIVE_QPUS,
           I32_MUL32_FALLBACK_VC4KERNEL_LANES,
           I32_MUL32_FALLBACK_VC4KERNEL_MAX_N, checksum_accum, 1,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
