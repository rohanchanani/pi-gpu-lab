#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_MAX_N 257u
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_GUARD 32u
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_ACTIVE_QPUS 12u
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_LANES 16u
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_ELEMENTS_PER_WAVE (MIXED_PREDICATE_RESOURCE_VC4KERNEL_ACTIVE_QPUS * MIXED_PREDICATE_RESOURCE_VC4KERNEL_LANES)
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_MAX_WAVES ((MIXED_PREDICATE_RESOURCE_VC4KERNEL_MAX_N + MIXED_PREDICATE_RESOURCE_VC4KERNEL_ELEMENTS_PER_WAVE - 1u) / MIXED_PREDICATE_RESOURCE_VC4KERNEL_ELEMENTS_PER_WAVE)
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_MAX_COVERAGE_N (MIXED_PREDICATE_RESOURCE_VC4KERNEL_MAX_WAVES * MIXED_PREDICATE_RESOURCE_VC4KERNEL_ELEMENTS_PER_WAVE)
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_BUFFER_N (MIXED_PREDICATE_RESOURCE_VC4KERNEL_MAX_COVERAGE_N + MIXED_PREDICATE_RESOURCE_VC4KERNEL_GUARD)
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_SENTINEL 0xdeadbeefu
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_TAG_A 0x60000000u
#define MIXED_PREDICATE_RESOURCE_VC4KERNEL_TAG_B 0x61000000u

struct mixed_predicate_resource_case {
    uint32_t n;
    uint32_t threshold;
    uint32_t control;
};

static const struct mixed_predicate_resource_case cases[] = {
    {0u, 7u, 1u},   {1u, 7u, 0u},   {15u, 5u, 1u},  {16u, 16u, 0u},
    {17u, 9u, 1u},  {193u, 3u, 0u}, {255u, 11u, 1u}, {257u, 13u, 0u},
};

static uint32_t out_values[MIXED_PREDICATE_RESOURCE_VC4KERNEL_BUFFER_N];
static uint32_t expected_values[MIXED_PREDICATE_RESOURCE_VC4KERNEL_BUFFER_N];

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + MIXED_PREDICATE_RESOURCE_VC4KERNEL_ELEMENTS_PER_WAVE - 1u) /
                     MIXED_PREDICATE_RESOURCE_VC4KERNEL_ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static void fill_expected(uint32_t n, uint32_t threshold, uint32_t control) {
    uint32_t tag = control != 0u ? MIXED_PREDICATE_RESOURCE_VC4KERNEL_TAG_A
                                 : MIXED_PREDICATE_RESOURCE_VC4KERNEL_TAG_B;
    for (uint32_t i = 0; i < MIXED_PREDICATE_RESOURCE_VC4KERNEL_BUFFER_N; i++) {
        out_values[i] = MIXED_PREDICATE_RESOURCE_VC4KERNEL_SENTINEL;
        if (i < n) {
            uint32_t lane = i & 15u;
            expected_values[i] = lane < threshold ? tag + i : 0u;
        } else {
            expected_values[i] = MIXED_PREDICATE_RESOURCE_VC4KERNEL_SENTINEL;
        }
    }
}

static int verify_active(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (out_values[i] != expected_values[i]) {
            if (mismatches < 8)
                printk("ERROR: mixed_predicate_resource active i=%d gpu=%x expected=%x\n",
                       (int)i, out_values[i], expected_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < MIXED_PREDICATE_RESOURCE_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != MIXED_PREDICATE_RESOURCE_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: mixed_predicate_resource sentinel n=%d i=%d gpu=%x expected=%x\n",
                       (int)n, (int)i, out_values[i],
                       MIXED_PREDICATE_RESOURCE_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(out_values[i] & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = MIXED_PREDICATE_RESOURCE_VC4KERNEL_BUFFER_N * sizeof(uint32_t);
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("mixed_predicate_resource_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(MIXED_PREDICATE_RESOURCE_VC4KERNEL_ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 mixed_predicate_resource_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t n = cases[case_id].n;
        uint32_t threshold = cases[case_id].threshold;
        uint32_t control = cases[case_id].control;
        uint32_t waves = rounded_waves(n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_expected(n, threshold, control);
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            mixed_predicate_resource_vc4kernel_launch(program, grid, block, out_dev, n, threshold, control) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: mixed_predicate_resource_vc4kernel launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = verify_active(n);
        int sentinels = verify_sentinels(n);
        int checksum = checksum_low16(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("MIXED_PREDICATE_RESOURCE_VC4KERNEL_CASE case=%d n=%d threshold=%d control=%d waves=%d coverage=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)n, (int)threshold, (int)control, (int)waves,
               (int)(waves * MIXED_PREDICATE_RESOURCE_VC4KERNEL_ELEMENTS_PER_WAVE),
               mismatches, sentinels, checksum);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_predicate_resource_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           MIXED_PREDICATE_RESOURCE_VC4KERNEL_ACTIVE_QPUS,
           MIXED_PREDICATE_RESOURCE_VC4KERNEL_LANES,
           MIXED_PREDICATE_RESOURCE_VC4KERNEL_MAX_N,
           MIXED_PREDICATE_RESOURCE_VC4KERNEL_MAX_COVERAGE_N,
           MIXED_PREDICATE_RESOURCE_VC4KERNEL_BUFFER_N, checksum_accum, 1,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
