#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define CASES 5u
#define GUARD_WORDS 16u
#define SENTINEL 0x7fc0aa55u
#define TAG 1515870810u

static uint32_t out_words[LANES + GUARD_WORDS];

struct mask_case {
    uint32_t lo;
    uint32_t hi;
};

static const struct mask_case cases[CASES] = {
    {0u, 16u},
    {3u, 11u},
    {8u, 8u},
    {0u, 1u},
    {15u, 16u},
};

static void fill_output(void) {
    for (uint32_t i = 0; i < LANES + GUARD_WORDS; i++)
        out_words[i] = SENTINEL;
}

static uint32_t checksum_words(const uint32_t *values, uint32_t count) {
    uint32_t hash = 2166136261u;
    for (uint32_t i = 0; i < count; i++) {
        hash ^= values[i];
        hash *= 16777619u;
    }
    return hash;
}

static int verify_case(uint32_t case_id, const struct mask_case *tc) {
    int mismatches = 0;
    int sentinel_mismatches = 0;
    for (uint32_t lane = 0; lane < LANES; lane++) {
        uint32_t expected = SENTINEL;
        if (lane >= tc->lo && lane < tc->hi)
            expected = TAG + lane;
        if (out_words[lane] != expected) {
            if (mismatches < 8)
                printk("ERROR: materialized_mask case=%d lane=%d lo=%d hi=%d got=%x expected=%x\n",
                       (int)case_id, (int)lane, (int)tc->lo, (int)tc->hi,
                       out_words[lane], expected);
            mismatches++;
        }
    }
    for (uint32_t i = 0; i < GUARD_WORDS; i++) {
        uint32_t got = out_words[LANES + i];
        if (got != SENTINEL) {
            if (sentinel_mismatches < 8)
                printk("ERROR: materialized_mask guard=%d got=%x expected=%x\n",
                       (int)i, got, SENTINEL);
            sentinel_mismatches++;
        }
    }
    return mismatches + sentinel_mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t out_dev = 0;
    uint32_t out_bytes = (LANES + GUARD_WORDS) * sizeof(uint32_t);

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4Malloc(program, &out_dev, out_bytes) < 0)
        panic("general_mask_materialized_combinators_vc4kernel allocation failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    printk("Running VC4 general_mask_materialized_combinators_vc4kernel candidate bundle...\n");
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checksum_accum = 0u;
    int start = timer_get_usec();

    for (uint32_t case_id = 0; case_id < CASES; case_id++) {
        const struct mask_case *tc = &cases[case_id];
        fill_output();
        if (vc4MemcpyHtoD(program, out_dev, out_words, out_bytes) < 0 ||
            general_mask_materialized_combinators_vc4kernel_launch(program, grid, block, out_dev, tc->lo, tc->hi) < 0 ||
            vc4MemcpyDtoH(program, out_words, out_dev, out_bytes) < 0) {
            printk("ERROR: general_mask_materialized_combinators_vc4kernel launch/copy failed case=%d lo=%d hi=%d\n",
                   (int)case_id, (int)tc->lo, (int)tc->hi);
            launch_failures++;
            continue;
        }

        int combined = verify_case(case_id, tc);
        int case_sentinel = 0;
        for (uint32_t i = 0; i < GUARD_WORDS; i++)
            if (out_words[LANES + i] != SENTINEL)
                case_sentinel++;
        total_mismatches += combined - case_sentinel;
        sentinel_mismatches += case_sentinel;
        uint32_t checksum = checksum_words(out_words, LANES);
        checksum_accum ^= checksum + case_id * 0x9e3779b9u;
        printk("GENERAL_MASK_MATERIALIZED_CASE case=%d lo=%d hi=%d mismatches=%d sentinel_mismatches=%d checksum=%x\n",
               (int)case_id, (int)tc->lo, (int)tc->hi,
               combined - case_sentinel, case_sentinel, checksum);
    }

    launch_failures += (int)general_mask_materialized_combinators_vc4kernel_runtime_launch_failures();
    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=general_mask_materialized_combinators_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d materialized_general_mask=%d runtime_allocations=%d runtime_launches=%d checksum_accum=%x elapsed_usec=%d\n",
           status, CASES, total_mismatches, sentinel_mismatches, launch_failures,
           1, LANES, 1, 1, (int)general_mask_materialized_combinators_vc4kernel_runtime_launches(),
           checksum_accum, elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
