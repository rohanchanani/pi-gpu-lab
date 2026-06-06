#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_ACTIVE_QPUS 1u
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES 16u
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_PREDICATES 4u
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_ACTIVE_N (FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES * FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_PREDICATES)
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_GUARD 16u
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_BUFFER_N (FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_ACTIVE_N + FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_GUARD)
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_SENTINEL 0xdeadbeefu
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_TRUE_BASE 0x51000000u
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_FALSE_BASE 0x52000000u
#define FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_ROW_STEP 0x1000u

static const int32_t lhs_values[FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES] = {
    (int32_t)0x80000000u, (int32_t)0x80000000u, (int32_t)0x80000000u,
    -1, -1, 0, 1, 0x7fffffff, 0x7fffffff, 0,
    (int32_t)0x80000000u, 0x7fffffff, -42, 42, -1, 1
};

static const int32_t rhs_values[FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES] = {
    -1, 0, 0x7fffffff, 0, 1, 0, -1, (int32_t)0x80000000u,
    -1, 1, (int32_t)0x80000000u, 0x7fffffff, 42, -42, -1, 1
};

static uint32_t out_values[FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_SENTINEL;
}

static int predicate_true(uint32_t pred, int32_t lhs, int32_t rhs) {
    if (pred == 0u)
        return lhs < rhs;
    if (pred == 1u)
        return lhs <= rhs;
    if (pred == 2u)
        return lhs > rhs;
    return lhs >= rhs;
}

static uint32_t expected_value(uint32_t pred, uint32_t lane) {
    uint32_t base = predicate_true(pred, lhs_values[lane], rhs_values[lane])
                        ? FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_TRUE_BASE
                        : FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_FALSE_BASE;
    return base + pred * FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_ROW_STEP + lane;
}

static int verify_active(void) {
    int mismatches = 0;
    for (uint32_t pred = 0; pred < FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_PREDICATES; pred++) {
        for (uint32_t lane = 0; lane < FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES; lane++) {
            uint32_t index = pred * FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES + lane;
            uint32_t expected = expected_value(pred, lane);
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: fragment_cmp_i32_signed pred=%d lane=%d lhs=%d rhs=%d gpu=%x expected=%x\n",
                           (int)pred, (int)lane, lhs_values[lane],
                           rhs_values[lane], out_values[index], expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_ACTIVE_N;
         i < FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_cmp_i32_signed sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i],
                       FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_ACTIVE_N; i++)
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
    uint32_t input_bytes = FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES * sizeof(uint32_t);
    uint32_t out_bytes = FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &lhs_dev, input_bytes) < 0 ||
        vc4_m2_malloc(program, &rhs_dev, input_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("fragment_cmp_i32_signed_vc4kernel device allocation failed");

    fill_output();
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 fragment_cmp_i32_signed_vc4kernel candidate bundle...\n");
    if (vc4_m2_copy_htod(program, lhs_dev, lhs_values, input_bytes) < 0 ||
        vc4_m2_copy_htod(program, rhs_dev, rhs_values, input_bytes) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
        fragment_cmp_i32_signed_vc4kernel_launch(program, grid, block, lhs_dev, rhs_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
        printk("ERROR: fragment_cmp_i32_signed_vc4kernel launch/copy failed\n");
        launch_failures++;
    } else {
        total_mismatches = verify_active();
        sentinel_mismatches = verify_sentinels();
        checksum_accum = checksum_low16();
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_cmp_i32_signed_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d predicates=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, 1, total_mismatches, sentinel_mismatches, launch_failures,
           FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_ACTIVE_QPUS,
           FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_LANES,
           FRAGMENT_CMP_I32_SIGNED_VC4KERNEL_PREDICATES, checksum_accum, 3, 1,
           elapsed);

    vc4Free(program, lhs_dev);
    vc4Free(program, rhs_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
