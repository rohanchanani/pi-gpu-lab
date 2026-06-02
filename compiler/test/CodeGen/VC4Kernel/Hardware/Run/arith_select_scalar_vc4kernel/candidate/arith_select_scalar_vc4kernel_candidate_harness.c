#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ARITH_SELECT_SCALAR_VC4KERNEL_LANES 16u
#define ARITH_SELECT_SCALAR_VC4KERNEL_ACTIVE_QPUS 1u
#define ARITH_SELECT_SCALAR_VC4KERNEL_MAX_N 48u
#define ARITH_SELECT_SCALAR_VC4KERNEL_GUARD 16u
#define ARITH_SELECT_SCALAR_VC4KERNEL_BUFFER_N (ARITH_SELECT_SCALAR_VC4KERNEL_MAX_N + ARITH_SELECT_SCALAR_VC4KERNEL_GUARD)
#define ARITH_SELECT_SCALAR_VC4KERNEL_SENTINEL (-12345.0f)
#define ARITH_SELECT_SCALAR_VC4KERNEL_CHECKSUM_SCALE 1024.0f

struct arith_select_scalar_case {
    uint32_t control;
    float alpha;
    float beta;
    uint32_t offset_elems;
};

static const struct arith_select_scalar_case cases[] = {
    {0u, 1.5f, 2.0f, 0u},
    {1u, 1.5f, 2.0f, 16u},
    {7u, -0.5f, 3.0f, 32u},
};

static float out_values[ARITH_SELECT_SCALAR_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < ARITH_SELECT_SCALAR_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = ARITH_SELECT_SCALAR_VC4KERNEL_SENTINEL;
}

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float expected_value(const struct arith_select_scalar_case *tc) {
    return tc->control ? tc->alpha : tc->beta;
}

static int verify_active(const struct arith_select_scalar_case *tc,
                         float *max_abs_diff) {
    int mismatches = 0;
    float expected = expected_value(tc);
    *max_abs_diff = 0.0f;
    for (uint32_t lane = 0; lane < ARITH_SELECT_SCALAR_VC4KERNEL_LANES; lane++) {
        uint32_t index = tc->offset_elems + lane;
        float diff = out_values[index] - expected;
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad != 0.0f) {
            if (mismatches < 8)
                printk("ERROR: arith_select_scalar index=%d gpu=%f expected=%f diff=%f\n",
                       (int)index, out_values[index], expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t offset_elems) {
    int mismatches = 0;
    uint32_t active_end = offset_elems + ARITH_SELECT_SCALAR_VC4KERNEL_LANES;
    for (uint32_t i = 0; i < ARITH_SELECT_SCALAR_VC4KERNEL_BUFFER_N; i++) {
        if (i >= offset_elems && i < active_end)
            continue;
        if (out_values[i] != ARITH_SELECT_SCALAR_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: arith_select_scalar sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i],
                       ARITH_SELECT_SCALAR_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int scaled_checksum(uint32_t offset_elems) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < ARITH_SELECT_SCALAR_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[offset_elems + lane] *
                          ARITH_SELECT_SCALAR_VC4KERNEL_CHECKSUM_SCALE);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = ARITH_SELECT_SCALAR_VC4KERNEL_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("arith_select_scalar_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(ARITH_SELECT_SCALAR_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 arith_select_scalar_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct arith_select_scalar_case *tc = &cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            arith_select_scalar_vc4kernel_launch(program, grid, block, out_dev,
                                                 tc->control, tc->alpha,
                                                 tc->beta, tc->offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: arith_select_scalar_vc4kernel launch/copy failed case=%d control=%x\n",
                   (int)case_id, tc->control);
            launch_failures++;
            continue;
        }

        float max_abs_diff = 0.0f;
        int mismatches = verify_active(tc, &max_abs_diff);
        int sentinels = verify_sentinels(tc->offset_elems);
        int checksum = scaled_checksum(tc->offset_elems);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("ARITH_SELECT_SCALAR_VC4KERNEL_CASE case=%d control=%x offset=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, tc->control, (int)tc->offset_elems, mismatches,
               sentinels, checksum, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=arith_select_scalar_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures,
           ARITH_SELECT_SCALAR_VC4KERNEL_ACTIVE_QPUS,
           ARITH_SELECT_SCALAR_VC4KERNEL_LANES,
           ARITH_SELECT_SCALAR_VC4KERNEL_MAX_N, checksum_accum,
           max_abs_diff_overall, 1, (int)(sizeof(cases) / sizeof(cases[0])),
           elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
