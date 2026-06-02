#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define F32_FRAGMENT_ARITH_VC4KERNEL_LANES 16u
#define F32_FRAGMENT_ARITH_VC4KERNEL_ACTIVE_QPUS 1u
#define F32_FRAGMENT_ARITH_VC4KERNEL_MAX_N 48u
#define F32_FRAGMENT_ARITH_VC4KERNEL_GUARD 16u
#define F32_FRAGMENT_ARITH_VC4KERNEL_BUFFER_N (F32_FRAGMENT_ARITH_VC4KERNEL_MAX_N + F32_FRAGMENT_ARITH_VC4KERNEL_GUARD)
#define F32_FRAGMENT_ARITH_VC4KERNEL_SENTINEL (-12345.0f)
#define F32_FRAGMENT_ARITH_VC4KERNEL_CHECKSUM_SCALE 1024.0f

struct f32_fragment_arith_case {
    float alpha;
    float beta;
    uint32_t offset_elems;
};

static const struct f32_fragment_arith_case cases[] = {
    {1.5f, 2.0f, 0u},
    {-2.0f, 0.5f, 16u},
    {0.75f, -1.25f, 32u},
};

static float out_values[F32_FRAGMENT_ARITH_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < F32_FRAGMENT_ARITH_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = F32_FRAGMENT_ARITH_VC4KERNEL_SENTINEL;
}

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float expected_value(float alpha, float beta) {
    float sum = alpha + beta;
    float diff = sum - beta;
    return diff * sum;
}

static int verify_active(uint32_t offset_elems, float alpha, float beta,
                         float *max_abs_diff) {
    int mismatches = 0;
    float expected = expected_value(alpha, beta);
    *max_abs_diff = 0.0f;
    for (uint32_t lane = 0; lane < F32_FRAGMENT_ARITH_VC4KERNEL_LANES; lane++) {
        uint32_t index = offset_elems + lane;
        float diff = out_values[index] - expected;
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad != 0.0f) {
            if (mismatches < 8)
                printk("ERROR: f32_fragment_arith index=%d gpu=%f expected=%f diff=%f\n",
                       (int)index, out_values[index], expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t offset_elems) {
    int mismatches = 0;
    uint32_t active_end = offset_elems + F32_FRAGMENT_ARITH_VC4KERNEL_LANES;
    for (uint32_t i = 0; i < F32_FRAGMENT_ARITH_VC4KERNEL_BUFFER_N; i++) {
        if (i >= offset_elems && i < active_end)
            continue;
        if (out_values[i] != F32_FRAGMENT_ARITH_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: f32_fragment_arith sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], F32_FRAGMENT_ARITH_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int scaled_checksum(uint32_t offset_elems) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < F32_FRAGMENT_ARITH_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[offset_elems + lane] *
                          F32_FRAGMENT_ARITH_VC4KERNEL_CHECKSUM_SCALE);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = F32_FRAGMENT_ARITH_VC4KERNEL_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("f32_fragment_arith_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(F32_FRAGMENT_ARITH_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 f32_fragment_arith_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        float alpha = cases[case_id].alpha;
        float beta = cases[case_id].beta;
        uint32_t offset_elems = cases[case_id].offset_elems;
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            f32_fragment_arith_vc4kernel_launch(program, grid, block, out_dev, alpha, beta, offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: f32_fragment_arith_vc4kernel launch/copy failed case=%d\n",
                   (int)case_id);
            launch_failures++;
            continue;
        }

        float max_abs_diff = 0.0f;
        int mismatches = verify_active(offset_elems, alpha, beta, &max_abs_diff);
        int sentinels = verify_sentinels(offset_elems);
        int checksum = scaled_checksum(offset_elems);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("F32_FRAGMENT_ARITH_VC4KERNEL_CASE case=%d offset=%d alpha=%f beta=%f mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, (int)offset_elems, alpha, beta, mismatches,
               sentinels, checksum, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=f32_fragment_arith_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures,
           F32_FRAGMENT_ARITH_VC4KERNEL_ACTIVE_QPUS,
           F32_FRAGMENT_ARITH_VC4KERNEL_LANES,
           F32_FRAGMENT_ARITH_VC4KERNEL_MAX_N, checksum_accum,
           max_abs_diff_overall, 1, (int)(sizeof(cases) / sizeof(cases[0])),
           elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
