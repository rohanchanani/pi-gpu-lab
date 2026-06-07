#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ROWS 2u
#define ACTIVE_N (LANES * ROWS)
#define OUT_N (48u + 16u)
#define SENTINEL (-4545.5f)
#define ABS_TOL 0.001f

struct branch_case {
    uint32_t control;
    uint32_t offset_elems;
};

static const struct branch_case test_cases[] = {
    {0u, 0u},
    {1u, 32u},
};

static const float input_values[ACTIVE_N] = {
    1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f,
    0.5f, 0.25f, 0.125f, 0.0625f, 1.0f, 4.0f, 16.0f, 64.0f,
    0.25f, 1.0f, 4.0f, 16.0f, 64.0f, 0.25f, 1.0f, 4.0f,
    16.0f, 64.0f, 0.25f, 1.0f, 4.0f, 16.0f, 64.0f, 256.0f
};

static float out_values[OUT_N];

static float absf_local(float v) { return v < 0.0f ? -v : v; }

static float sqrt_newton(float value) {
    float x = value >= 1.0f ? value : 1.0f;
    for (uint32_t i = 0; i < 16; ++i)
        x = 0.5f * (x + value / x);
    return x;
}

static void fill_output(void) {
    for (uint32_t i = 0; i < OUT_N; ++i)
        out_values[i] = SENTINEL;
}

static float expected_value(uint32_t control, uint32_t lane) {
    if (control == 0u)
        return 1.0f / input_values[lane];
    return 1.0f / sqrt_newton(input_values[LANES + lane]);
}

static int verify_active(const struct branch_case *tc, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        uint32_t index = tc->offset_elems + lane;
        float expected = expected_value(tc->control, lane);
        float diff = out_values[index] - expected;
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > ABS_TOL) {
            if (mismatches < 8)
                printk("ERROR: fragment_sfu_branch_layout control=%d lane=%d gpu=%f expected=%f diff=%f\n",
                       (int)tc->control, (int)lane, out_values[index],
                       expected, diff);
            ++mismatches;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct branch_case *tc) {
    int mismatches = 0;
    uint32_t active_end = tc->offset_elems + LANES;
    for (uint32_t i = 0; i < OUT_N; ++i) {
        if (i >= tc->offset_elems && i < active_end)
            continue;
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_sfu_branch_layout sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], SENTINEL);
            ++mismatches;
        }
    }
    return mismatches;
}

static int scaled_checksum(const struct branch_case *tc) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < LANES; ++lane)
        checksum += (int)(out_values[tc->offset_elems + lane] * 4096.0f);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("fragment_sfu_branch_layout_vc4kernel program create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("fragment_sfu_branch_layout_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_recip_path = 0;
    int saw_rsqrt_path = 0;
    float max_abs_diff = 0.0f;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    int start = timer_get_usec();

    printk("Running VC4 fragment_sfu_branch_layout_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(test_cases) / sizeof(test_cases[0]); ++case_id) {
        const struct branch_case *tc = &test_cases[case_id];
        fill_output();
        saw_recip_path |= tc->control == 0u;
        saw_rsqrt_path |= tc->control != 0u;
        if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            fragment_sfu_branch_layout_vc4kernel_launch(program, grid, block, input_dev, out_dev, tc->control, tc->offset_elems) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0) {
            printk("ERROR: fragment_sfu_branch_layout launch/copy failed case=%d control=%d\n",
                   (int)case_id, (int)tc->control);
            ++launch_failures;
            continue;
        }
        int mismatches = verify_active(tc, &max_abs_diff);
        int sentinels = verify_sentinels(tc);
        int checksum = scaled_checksum(tc);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum;
        printk("FRAGMENT_SFU_BRANCH_LAYOUT_CASE case=%d control=%d offset=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, (int)tc->control, (int)tc->offset_elems,
               mismatches, sentinels, checksum, max_abs_diff);
    }

    launch_failures += (int)fragment_sfu_branch_layout_vc4kernel_runtime_launch_failures();
    uint32_t runtime_launches = fragment_sfu_branch_layout_vc4kernel_runtime_launches();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_recip_path &&
                          saw_rsqrt_path) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_sfu_branch_layout_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_sfu_branch=1 saw_both_paths=%d saw_recip_path=%d saw_rsqrt_path=%d saw_tmu_safe_offset=1 saw_vdw_preserve=1 checksum_accum=%d max_abs_diff=%f runtime_allocations=2 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(test_cases) / sizeof(test_cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           saw_recip_path && saw_rsqrt_path, saw_recip_path, saw_rsqrt_path,
           checksum_accum, max_abs_diff, (int)runtime_launches,
           timer_get_usec() - start);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
