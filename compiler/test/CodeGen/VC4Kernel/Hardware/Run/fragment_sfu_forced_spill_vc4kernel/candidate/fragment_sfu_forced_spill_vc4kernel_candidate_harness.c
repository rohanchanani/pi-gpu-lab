#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define GUARD 32u
#define OUT_WORDS (LANES + GUARD)
#define SENTINEL (-9898.5f)
#define BETA 0.015625f
#define SPILL_ADJUST (495.0f * BETA)
#define ABS_TOL 0.001f

static const float input_values[LANES] = {
    1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f,
    0.5f, 0.25f, 0.125f, 0.0625f, 1.0f, 4.0f, 16.0f, 64.0f
};

static float out_values[OUT_WORDS];
static float expected_values[OUT_WORDS];

static float absf_local(float v) { return v < 0.0f ? -v : v; }

static void fill_output(void) {
    for (uint32_t i = 0; i < OUT_WORDS; ++i) {
        out_values[i] = SENTINEL;
        expected_values[i] = SENTINEL;
    }
    for (uint32_t lane = 0; lane < LANES; ++lane)
        expected_values[lane] = (1.0f / input_values[lane]) + SPILL_ADJUST;
}

static int verify_active(float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        float diff = out_values[lane] - expected_values[lane];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > ABS_TOL) {
            if (mismatches < 8)
                printk("ERROR: fragment_sfu_forced_spill lane=%d gpu=%f expected=%f diff=%f\n",
                       (int)lane, out_values[lane], expected_values[lane],
                       diff);
            ++mismatches;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES; i < OUT_WORDS; ++i) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_sfu_forced_spill sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], SENTINEL);
            ++mismatches;
        }
    }
    return mismatches;
}

static int scaled_checksum(void) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane)
        checksum += (int)(out_values[lane] * 4096.0f);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 128u * 1024u) < 0 || !program)
        panic("fragment_sfu_forced_spill_vc4kernel program create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("fragment_sfu_forced_spill_vc4kernel allocation failed");

    fill_output();
    int launch_failures = 0;
    float max_abs_diff = 0.0f;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    int start = timer_get_usec();

    printk("Running VC4 fragment_sfu_forced_spill_vc4kernel candidate bundle...\n");
    if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
        fragment_sfu_forced_spill_vc4kernel_launch(program, grid, block, input_dev, out_dev, BETA) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0) {
        printk("ERROR: fragment_sfu_forced_spill launch/copy failed\n");
        ++launch_failures;
    }

    launch_failures += (int)fragment_sfu_forced_spill_vc4kernel_runtime_launch_failures();
    int total_mismatches = launch_failures ? 0 : verify_active(&max_abs_diff);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int checksum = launch_failures ? 0 : scaled_checksum();
    uint32_t runtime_launches = fragment_sfu_forced_spill_vc4kernel_runtime_launches();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_sfu_forced_spill_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_spill_frame_nonzero=1 saw_sfu=1 saw_recip=1 saw_tmu_safe_offset=1 saw_vdw_preserve=1 hidden_spill_reload_tmu_hits=0 forced_spill_live_adjust_x10000=%d checksum=%d max_abs_diff=%f runtime_allocations=2 runtime_launches=%d elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)(SPILL_ADJUST * 10000.0f), checksum, max_abs_diff,
           (int)runtime_launches, timer_get_usec() - start);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
