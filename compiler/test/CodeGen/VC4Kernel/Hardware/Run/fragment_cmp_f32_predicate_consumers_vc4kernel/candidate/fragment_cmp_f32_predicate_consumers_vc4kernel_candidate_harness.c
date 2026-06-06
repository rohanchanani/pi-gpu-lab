#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ACTIVE_QPUS 1u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES 16u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ROWS 6u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ACTIVE_N (FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES * FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ROWS)
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_GUARD 16u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_BUFFER_N (FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ACTIVE_N + FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_GUARD)
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_SENTINEL 0xdeadbeefu
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_TRUE_BASE 0x65000000u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_FALSE_BASE 0x66000000u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ROW_STEP 0x1000u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ANY_TRUE_BASE 0x77001000u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ANY_FALSE_BASE 0x77002000u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ALL_TRUE_BASE 0x78001000u
#define FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ALL_FALSE_BASE 0x78002000u

static const float lhs_values[FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES] = {
    -3.0f, 0.0f, 1.0f, 2.0f, -8.0f, 5.0f, -1.0f, 7.0f,
    8.0f, 15.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f
};

static const float rhs_values[FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES] = {
    0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 4.0f, 0.0f, 8.0f,
    7.0f, -8.0f, 10.0f, 9.0f, 13.0f, 12.0f, 15.0f, 14.0f
};

static uint32_t out_values[FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_SENTINEL;
}

static int lt(uint32_t lane) {
    return lhs_values[lane] < rhs_values[lane];
}

static int gt(uint32_t lane) {
    return lhs_values[lane] > rhs_values[lane];
}

static int row_predicate_true(uint32_t row, uint32_t lane) {
    int l = lt(lane);
    int g = gt(lane);
    if (row == 0u)
        return l;
    if (row == 1u)
        return g;
    if (row == 2u)
        return l || g;
    return !l;
}

static uint32_t expected_row_value(uint32_t row, uint32_t lane) {
    if (row < 4u) {
        uint32_t base = row_predicate_true(row, lane)
                            ? FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_TRUE_BASE
                            : FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_FALSE_BASE;
        return base + row * FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ROW_STEP + lane;
    }
    if (row == 4u) {
        uint32_t any_lt = 0u;
        for (uint32_t i = 0; i < FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES; i++)
            any_lt |= lt(i) ? 1u : 0u;
        return (any_lt ? FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ANY_TRUE_BASE
                       : FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ANY_FALSE_BASE) + lane;
    }
    return FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ALL_TRUE_BASE + lane;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t row = 0; row < FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ROWS; row++) {
        for (uint32_t lane = 0; lane < FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES; lane++) {
            uint32_t index = row * FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES + lane;
            uint32_t expected = expected_row_value(row, lane);
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: fragment_cmp_f32_consumers row=%d lane=%d lhs=%f rhs=%f gpu=%x expected=%x\n",
                           (int)row, (int)lane, lhs_values[lane],
                           rhs_values[lane], out_values[index], expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ACTIVE_N;
         i < FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_cmp_f32_consumers sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i],
                       FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ACTIVE_N; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t lhs_dev = 0;
    vc4_deviceptr_t rhs_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    uint32_t input_bytes = FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES * sizeof(float);
    uint32_t out_bytes = FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &lhs_dev, input_bytes) < 0 ||
        vc4_m2_malloc(program, &rhs_dev, input_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("fragment_cmp_f32_predicate_consumers_vc4kernel device allocation failed");

    fill_output();
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 fragment_cmp_f32_predicate_consumers_vc4kernel candidate bundle...\n");
    if (vc4_m2_copy_htod(program, lhs_dev, lhs_values, input_bytes) < 0 ||
        vc4_m2_copy_htod(program, rhs_dev, rhs_values, input_bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
        fragment_cmp_f32_predicate_consumers_vc4kernel_launch(program, grid, block, lhs_dev, rhs_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
        printk("ERROR: fragment_cmp_f32_predicate_consumers_vc4kernel launch/copy failed\n");
        launch_failures++;
    } else {
        total_mismatches = verify_active();
        sentinel_mismatches = verify_sentinels();
        checksum_accum = checksum_low16();
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_cmp_f32_predicate_consumers_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d consumer_rows=%d finite_only=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, 1, total_mismatches, sentinel_mismatches, launch_failures,
           FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ACTIVE_QPUS,
           FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_LANES,
           FRAGMENT_CMP_F32_PREDICATE_CONSUMERS_VC4KERNEL_ROWS,
           1, checksum_accum, 3, 1, elapsed);

    vc4Free(program, lhs_dev);
    vc4Free(program, rhs_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
