#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES 16u
#define TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_MAX_K 33u
#define TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_B_WORDS \
    (TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_MAX_K * TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES)
#define TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_OUT_GUARD 16u
#define TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_OUT_WORDS \
    (TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES + TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_OUT_GUARD)
#define TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_EPSILON 0.0001f
#define TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_SENTINEL (-731.25f)

struct case_spec {
    uint32_t active_cols;
    uint32_t k;
};

static const struct case_spec cases[] = {
    {1u, 0u},  {1u, 1u},  {1u, 2u},  {1u, 17u},  {1u, 33u},
    {15u, 0u}, {15u, 1u}, {15u, 2u}, {15u, 17u}, {15u, 33u},
    {16u, 0u}, {16u, 1u}, {16u, 2u}, {16u, 17u}, {16u, 33u},
};

static float b_values[TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_B_WORDS];
static float out_values[TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_OUT_WORDS];
static float expected_values[TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float b_value(uint32_t j, uint32_t lane) {
    return ((float)((j * 13u + lane * 7u + 5u) % 31u) - 15.0f) * 0.03125f;
}

static void fill_inputs(uint32_t active_cols, uint32_t k) {
    for (uint32_t j = 0; j < TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_MAX_K; j++)
        for (uint32_t lane = 0; lane < TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES; lane++)
            b_values[j * TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES + lane] =
                b_value(j, lane);
    for (uint32_t lane = 0; lane < TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES; lane++) {
        float sum = 0.0f;
        if (lane < active_cols)
            for (uint32_t j = 0; j < k; j++)
                sum += b_value(j, lane);
        expected_values[lane] = sum;
    }
    for (uint32_t i = 0; i < TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_OUT_WORDS; i++)
        out_values[i] = TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_SENTINEL;
}

static int verify_values(float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t lane = 0; lane < TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES; lane++) {
        float diff = out_values[lane] - expected_values[lane];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_EPSILON) {
            if (mismatches < 8)
                printk("ERROR: tmu_tail_loop lane=%d gpu=%f cpu=%f diff=%f\n",
                       (int)lane, out_values[lane], expected_values[lane], diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES;
         i < TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_OUT_WORDS; i++) {
        if (out_values[i] != TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: tmu_tail_loop sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int scaled_checksum(void) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[lane] * 1024.0f);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t b_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    uint32_t b_bytes = TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_B_WORDS * sizeof(float);
    uint32_t out_bytes = TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_OUT_WORDS * sizeof(float);
    if (vc4_m2_malloc(program, &b_dev, b_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0)
        panic("tmu_tail_loop_no_spill_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_k0 = 0;
    int saw_k33 = 0;
    int saw_active_cols1 = 0;
    int saw_active_cols15 = 0;
    int saw_active_cols16 = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 tmu_tail_loop_no_spill_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t active_cols = cases[case_id].active_cols;
        uint32_t k = cases[case_id].k;
        fill_inputs(active_cols, k);
        if (vc4_m2_copy_htod(program, b_dev, b_values, b_bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            tmu_tail_loop_no_spill_vc4kernel_launch(program, grid, block, b_dev, out_dev, k, active_cols) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            printk("ERROR: tmu_tail_loop_no_spill_vc4kernel launch/copy failed case=%d active_cols=%d k=%d\n",
                   (int)case_id, (int)active_cols, (int)k);
            launch_failures++;
            continue;
        }

        float max_abs_diff = 0.0f;
        int mismatches = verify_values(&max_abs_diff);
        int sentinels = verify_sentinels();
        int checksum = scaled_checksum();
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (k == 0u)
            saw_k0 = 1;
        if (k == 33u)
            saw_k33 = 1;
        if (active_cols == 1u)
            saw_active_cols1 = 1;
        if (active_cols == 15u)
            saw_active_cols15 = 1;
        if (active_cols == 16u)
            saw_active_cols16 = 1;
        printk("TMU_TAIL_LOOP_NO_SPILL_VC4KERNEL_CASE case=%d active_cols=%d k=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, (int)active_cols, (int)k, mismatches, sentinels,
               checksum, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_k0 && saw_k33 &&
                          saw_active_cols1 && saw_active_cols15 &&
                          saw_active_cols16) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=tmu_tail_loop_no_spill_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_k0=%d saw_k33=%d saw_active_cols1=%d saw_active_cols15=%d saw_active_cols16=%d spill_frame_bytes=0 checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, saw_k0, saw_k33,
           saw_active_cols1, saw_active_cols15, saw_active_cols16,
           checksum_accum, max_abs_diff_overall,
           (int)tmu_tail_loop_no_spill_vc4kernel_runtime_allocations(),
           (int)tmu_tail_loop_no_spill_vc4kernel_runtime_launches(), elapsed);

    vc4Free(program, b_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
