#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define FRAGMENT_CMP_SELECT_VC4KERNEL_ACTIVE_QPUS 1u
#define FRAGMENT_CMP_SELECT_VC4KERNEL_LANES 16u
#define FRAGMENT_CMP_SELECT_VC4KERNEL_PREDICATES 6u
#define FRAGMENT_CMP_SELECT_VC4KERNEL_ACTIVE_N (FRAGMENT_CMP_SELECT_VC4KERNEL_LANES * FRAGMENT_CMP_SELECT_VC4KERNEL_PREDICATES)
#define FRAGMENT_CMP_SELECT_VC4KERNEL_GUARD 16u
#define FRAGMENT_CMP_SELECT_VC4KERNEL_BUFFER_N (FRAGMENT_CMP_SELECT_VC4KERNEL_ACTIVE_N + FRAGMENT_CMP_SELECT_VC4KERNEL_GUARD)
#define FRAGMENT_CMP_SELECT_VC4KERNEL_SENTINEL 0xdeadbeefu
#define FRAGMENT_CMP_SELECT_VC4KERNEL_TRUE_BASE 0x30000000u
#define FRAGMENT_CMP_SELECT_VC4KERNEL_FALSE_BASE 0x40000000u
#define FRAGMENT_CMP_SELECT_VC4KERNEL_ROW_STEP 0x1000u

static const uint32_t cases[] = {0u, 1u, 7u, 15u, 16u};
static uint32_t out_values[FRAGMENT_CMP_SELECT_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < FRAGMENT_CMP_SELECT_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = FRAGMENT_CMP_SELECT_VC4KERNEL_SENTINEL;
}

static int predicate_true(uint32_t pred, uint32_t lane, uint32_t threshold) {
    if (pred == 0u)
        return lane == threshold;
    if (pred == 1u)
        return lane != threshold;
    if (pred == 2u)
        return lane < threshold;
    if (pred == 3u)
        return lane <= threshold;
    if (pred == 4u)
        return lane > threshold;
    return lane >= threshold;
}

static uint32_t expected_value(uint32_t pred, uint32_t lane, uint32_t threshold) {
    uint32_t base = predicate_true(pred, lane, threshold) ? FRAGMENT_CMP_SELECT_VC4KERNEL_TRUE_BASE
                                                         : FRAGMENT_CMP_SELECT_VC4KERNEL_FALSE_BASE;
    return base + pred * FRAGMENT_CMP_SELECT_VC4KERNEL_ROW_STEP + lane;
}

static int verify_active(uint32_t threshold) {
    int mismatches = 0;
    for (uint32_t pred = 0; pred < FRAGMENT_CMP_SELECT_VC4KERNEL_PREDICATES; pred++) {
        for (uint32_t lane = 0; lane < FRAGMENT_CMP_SELECT_VC4KERNEL_LANES; lane++) {
            uint32_t index = pred * FRAGMENT_CMP_SELECT_VC4KERNEL_LANES + lane;
            uint32_t expected = expected_value(pred, lane, threshold);
            if (out_values[index] != expected) {
                if (mismatches < 8)
                    printk("ERROR: fragment_cmp_select threshold=%d pred=%d lane=%d gpu=%x expected=%x\n",
                           (int)threshold, (int)pred, (int)lane, out_values[index], expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = FRAGMENT_CMP_SELECT_VC4KERNEL_ACTIVE_N;
         i < FRAGMENT_CMP_SELECT_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != FRAGMENT_CMP_SELECT_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_cmp_select sentinel i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], FRAGMENT_CMP_SELECT_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < FRAGMENT_CMP_SELECT_VC4KERNEL_ACTIVE_N; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = FRAGMENT_CMP_SELECT_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("fragment_cmp_select_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(FRAGMENT_CMP_SELECT_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 fragment_cmp_select_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t threshold = cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            fragment_cmp_select_vc4kernel_launch(program, grid, block, out_dev, threshold) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: fragment_cmp_select_vc4kernel launch/copy failed case=%d threshold=%d\n",
                   (int)case_id, (int)threshold);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(threshold);
        int sentinels = verify_sentinels();
        int checksum = checksum_low16();
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("FRAGMENT_CMP_SELECT_VC4KERNEL_CASE case=%d threshold=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)threshold, mismatches, sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_cmp_select_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d predicates=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures,
           FRAGMENT_CMP_SELECT_VC4KERNEL_ACTIVE_QPUS,
           FRAGMENT_CMP_SELECT_VC4KERNEL_LANES,
           FRAGMENT_CMP_SELECT_VC4KERNEL_PREDICATES, checksum_accum, 1,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
