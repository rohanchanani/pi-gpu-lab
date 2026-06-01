#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define FORMAL_F32_SPLAT_VC4KERNEL_LANES 16u
#define FORMAL_F32_SPLAT_VC4KERNEL_ACTIVE_QPUS 1u
#define FORMAL_F32_SPLAT_VC4KERNEL_MAX_N 32u
#define FORMAL_F32_SPLAT_VC4KERNEL_GUARD 16u
#define FORMAL_F32_SPLAT_VC4KERNEL_BUFFER_N (FORMAL_F32_SPLAT_VC4KERNEL_MAX_N + FORMAL_F32_SPLAT_VC4KERNEL_GUARD)
#define FORMAL_F32_SPLAT_VC4KERNEL_SENTINEL (-12345.0f)
#define FORMAL_F32_SPLAT_VC4KERNEL_CHECKSUM_SCALE 1024.0f

struct formal_f32_splat_vc4kernel_case {
    float alpha;
    uint32_t offset_elems;
};

static const struct formal_f32_splat_vc4kernel_case cases[] = {
    {1.25f, 0u},
    {-2.5f, 16u},
};

static float out_values[FORMAL_F32_SPLAT_VC4KERNEL_BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < FORMAL_F32_SPLAT_VC4KERNEL_BUFFER_N; i++)
        out_values[i] = FORMAL_F32_SPLAT_VC4KERNEL_SENTINEL;
}

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static int verify_active(uint32_t offset_elems, float alpha, float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t lane = 0; lane < FORMAL_F32_SPLAT_VC4KERNEL_LANES; lane++) {
        uint32_t index = offset_elems + lane;
        float diff = out_values[index] - alpha;
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad != 0.0f) {
            if (mismatches < 8)
                printk("ERROR: formal_f32_splat active index=%d gpu=%f expected=%f diff=%f\n", (int)index, out_values[index], alpha, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t offset_elems) {
    int mismatches = 0;
    uint32_t active_end = offset_elems + FORMAL_F32_SPLAT_VC4KERNEL_LANES;
    for (uint32_t i = 0; i < FORMAL_F32_SPLAT_VC4KERNEL_BUFFER_N; i++) {
        if (i >= offset_elems && i < active_end)
            continue;
        if (out_values[i] != FORMAL_F32_SPLAT_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: formal_f32_splat sentinel index=%d gpu=%f expected=%f\n", (int)i, out_values[i], FORMAL_F32_SPLAT_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int scaled_checksum(uint32_t offset_elems) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < FORMAL_F32_SPLAT_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[offset_elems + lane] * FORMAL_F32_SPLAT_VC4KERNEL_CHECKSUM_SCALE);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = FORMAL_F32_SPLAT_VC4KERNEL_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("formal_f32_splat_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(FORMAL_F32_SPLAT_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 formal_f32_splat_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        float alpha = cases[case_id].alpha;
        uint32_t offset_elems = cases[case_id].offset_elems;
        fill_output();
        if (vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            formal_f32_splat_vc4kernel_launch(program, grid, block, out_dev, alpha, offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: formal_f32_splat_vc4kernel launch/copy failed case=%d offset_elems=%d\n", (int)case_id, (int)offset_elems);
            launch_failures++;
            continue;
        }

        float max_abs_diff = 0.0f;
        int mismatches = verify_active(offset_elems, alpha, &max_abs_diff);
        int sentinels = verify_sentinels(offset_elems);
        int checksum = scaled_checksum(offset_elems);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("FORMAL_F32_SPLAT_VC4KERNEL_CASE case=%d offset_elems=%d alpha=%f mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, (int)offset_elems, alpha, mismatches, sentinels, checksum, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=formal_f32_splat_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches, sentinel_mismatches, launch_failures,
           FORMAL_F32_SPLAT_VC4KERNEL_ACTIVE_QPUS, FORMAL_F32_SPLAT_VC4KERNEL_LANES,
           FORMAL_F32_SPLAT_VC4KERNEL_MAX_N, checksum_accum, max_abs_diff_overall, 1,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
