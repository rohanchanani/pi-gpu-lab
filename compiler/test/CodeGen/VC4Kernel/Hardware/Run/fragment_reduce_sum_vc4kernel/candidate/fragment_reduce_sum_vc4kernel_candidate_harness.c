#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define FRAGMENT_REDUCE_SUM_VC4KERNEL_LANES 16u
#define FRAGMENT_REDUCE_SUM_VC4KERNEL_ACTIVE_QPUS 1u
#define FRAGMENT_REDUCE_SUM_VC4KERNEL_GUARD 16u
#define FRAGMENT_REDUCE_SUM_VC4KERNEL_BUFFER_N (FRAGMENT_REDUCE_SUM_VC4KERNEL_LANES + FRAGMENT_REDUCE_SUM_VC4KERNEL_GUARD)
#define FRAGMENT_REDUCE_SUM_VC4KERNEL_SENTINEL (-12345.0f)
#define FRAGMENT_REDUCE_SUM_VC4KERNEL_CHECKSUM_SCALE 1024.0f

static const uint32_t cases[] = {0u, 1u, 5u, 8u, 13u, 16u};
static float input_values[FRAGMENT_REDUCE_SUM_VC4KERNEL_BUFFER_N];
static float out_values[FRAGMENT_REDUCE_SUM_VC4KERNEL_BUFFER_N];

static void fill_buffers(uint32_t case_id) {
    for (uint32_t i = 0; i < FRAGMENT_REDUCE_SUM_VC4KERNEL_BUFFER_N; i++) {
        input_values[i] = 0.0f;
        out_values[i] = FRAGMENT_REDUCE_SUM_VC4KERNEL_SENTINEL;
    }
    for (uint32_t lane = 0; lane < FRAGMENT_REDUCE_SUM_VC4KERNEL_LANES; lane++) {
        int raw = (int)((lane * 3u + case_id * 5u + 1u) % 13u) - 6;
        input_values[lane] = (float)raw * 0.25f;
    }
}

static float expected_sum(uint32_t threshold) {
    float sum = 0.0f;
    for (uint32_t lane = 0; lane < threshold; lane++)
        sum += input_values[lane];
    return sum;
}

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static int verify_values(uint32_t threshold, float *max_abs_diff) {
    int mismatches = 0;
    float expected = expected_sum(threshold);
    *max_abs_diff = 0.0f;
    for (uint32_t lane = 0; lane < FRAGMENT_REDUCE_SUM_VC4KERNEL_LANES; lane++) {
        float diff = out_values[lane] - expected;
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad != 0.0f) {
            if (mismatches < 8)
                printk("ERROR: fragment_reduce_sum threshold=%d lane=%d gpu=%f expected=%f diff=%f\n",
                       (int)threshold, (int)lane, out_values[lane], expected,
                       diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = FRAGMENT_REDUCE_SUM_VC4KERNEL_LANES;
         i < FRAGMENT_REDUCE_SUM_VC4KERNEL_BUFFER_N; i++) {
        if (out_values[i] != FRAGMENT_REDUCE_SUM_VC4KERNEL_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_reduce_sum sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], FRAGMENT_REDUCE_SUM_VC4KERNEL_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int scaled_checksum(void) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < FRAGMENT_REDUCE_SUM_VC4KERNEL_LANES; lane++)
        checksum += (int)(out_values[lane] *
                          FRAGMENT_REDUCE_SUM_VC4KERNEL_CHECKSUM_SCALE);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = FRAGMENT_REDUCE_SUM_VC4KERNEL_BUFFER_N * sizeof(float);
    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("fragment_reduce_sum_vc4kernel device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_threshold0 = 0;
    int saw_threshold13 = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(FRAGMENT_REDUCE_SUM_VC4KERNEL_LANES, 1, 1);

    printk("Running VC4 fragment_reduce_sum_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t threshold = cases[case_id];
        fill_buffers(case_id);
        if (vc4_m2_copy_htod(program, input_dev, input_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            fragment_reduce_sum_vc4kernel_launch(program, grid, block, input_dev, out_dev, threshold) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: fragment_reduce_sum_vc4kernel launch/copy failed case=%d threshold=%d\n",
                   (int)case_id, (int)threshold);
            launch_failures++;
            continue;
        }

        float max_abs_diff = 0.0f;
        int mismatches = verify_values(threshold, &max_abs_diff);
        int sentinels = verify_sentinels();
        int checksum = scaled_checksum();
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        if (threshold == 0u)
            saw_threshold0 = 1;
        if (threshold == 13u)
            saw_threshold13 = 1;
        printk("FRAGMENT_REDUCE_SUM_VC4KERNEL_CASE case=%d threshold=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, (int)threshold, mismatches, sentinels, checksum,
               max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_threshold0 &&
                          saw_threshold13) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_reduce_sum_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d saw_threshold0=%d saw_threshold13=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures,
           FRAGMENT_REDUCE_SUM_VC4KERNEL_ACTIVE_QPUS,
           FRAGMENT_REDUCE_SUM_VC4KERNEL_LANES, saw_threshold0,
           saw_threshold13, checksum_accum, max_abs_diff_overall, 2,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
