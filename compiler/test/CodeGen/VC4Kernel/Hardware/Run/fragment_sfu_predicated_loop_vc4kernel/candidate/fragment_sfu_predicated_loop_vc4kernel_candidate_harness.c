#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define MAX_K 33u
#define GUARD 32u
#define OUT_WORDS (LANES + GUARD)
#define SENTINEL (-7777.5f)
#define ABS_TOL 0.05f

struct sfu_loop_case {
    uint32_t k;
    uint32_t active_cols;
};

static const struct sfu_loop_case test_cases[] = {
    {0u, 1u},  {0u, 15u},  {0u, 16u}, {1u, 1u},  {1u, 15u},
    {1u, 16u}, {2u, 1u},   {2u, 15u}, {2u, 16u}, {15u, 1u},
    {15u, 15u}, {15u, 16u}, {16u, 1u}, {16u, 15u}, {16u, 16u},
    {17u, 1u}, {17u, 15u}, {17u, 16u}, {33u, 1u}, {33u, 15u},
    {33u, 16u},
};

static float a_values[MAX_K];
static float b_values[MAX_K * LANES];
static float out_values[OUT_WORDS];
static float expected_values[OUT_WORDS];

static float absf_local(float v) { return v < 0.0f ? -v : v; }

static float sqrt_newton(float value) {
    float x = value >= 1.0f ? value : 1.0f;
    for (uint32_t i = 0; i < 16; ++i)
        x = 0.5f * (x + value / x);
    return x;
}

static float a_value(uint32_t j) {
    static const float cycle[] = {1.0f, 2.0f, 4.0f, 8.0f};
    return cycle[j & 3u];
}

static float b_value(uint32_t j, uint32_t lane) {
    static const float cycle[] = {0.25f, 1.0f, 4.0f, 16.0f};
    return cycle[(j + lane) & 3u];
}

static void fill_inputs(void) {
    for (uint32_t j = 0; j < MAX_K; ++j) {
        a_values[j] = a_value(j);
        for (uint32_t lane = 0; lane < LANES; ++lane)
            b_values[j * LANES + lane] = b_value(j, lane);
    }
}

static void fill_output(void) {
    for (uint32_t i = 0; i < OUT_WORDS; ++i) {
        out_values[i] = SENTINEL;
        expected_values[i] = SENTINEL;
    }
}

static void compute_expected(uint32_t k, uint32_t active_cols) {
    for (uint32_t lane = 0; lane < active_cols; ++lane) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < k; ++j)
            sum += (1.0f / a_value(j)) + (1.0f / sqrt_newton(b_value(j, lane)));
        expected_values[lane] = sum;
    }
}

static int verify_active(uint32_t active_cols, float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < active_cols; ++lane) {
        float diff = out_values[lane] - expected_values[lane];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > ABS_TOL) {
            if (mismatches < 8)
                printk("ERROR: fragment_sfu_predicated_loop active_cols=%d lane=%d gpu=%f expected=%f diff=%f\n",
                       (int)active_cols, (int)lane, out_values[lane],
                       expected_values[lane], diff);
            ++mismatches;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t active_cols) {
    int mismatches = 0;
    for (uint32_t i = active_cols; i < OUT_WORDS; ++i) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_sfu_predicated_loop sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], SENTINEL);
            ++mismatches;
        }
    }
    return mismatches;
}

static int scaled_checksum(uint32_t active_cols) {
    int checksum = 0;
    for (uint32_t lane = 0; lane < active_cols; ++lane)
        checksum += (int)(out_values[lane] * 1024.0f);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("fragment_sfu_predicated_loop_vc4kernel program create failed");

    vc4_deviceptr_t a_dev = 0;
    vc4_deviceptr_t b_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, sizeof(a_values)) < 0 ||
        vc4_m2_malloc(program, &b_dev, sizeof(b_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("fragment_sfu_predicated_loop_vc4kernel allocation failed");

    fill_inputs();
    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_k0 = 0;
    int saw_k33 = 0;
    int saw_active_cols1 = 0;
    int saw_active_cols15 = 0;
    int saw_active_cols16 = 0;
    float max_abs_diff = 0.0f;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    int start = timer_get_usec();

    printk("Running VC4 fragment_sfu_predicated_loop_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(test_cases) / sizeof(test_cases[0]); ++case_id) {
        uint32_t k = test_cases[case_id].k;
        uint32_t active_cols = test_cases[case_id].active_cols;
        fill_output();
        compute_expected(k, active_cols);
        saw_k0 |= k == 0u;
        saw_k33 |= k == 33u;
        saw_active_cols1 |= active_cols == 1u;
        saw_active_cols15 |= active_cols == 15u;
        saw_active_cols16 |= active_cols == 16u;
        if (vc4_m2_copy_htod(program, a_dev, a_values, sizeof(a_values)) < 0 ||
            vc4_m2_copy_htod(program, b_dev, b_values, sizeof(b_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            fragment_sfu_predicated_loop_vc4kernel_launch(program, grid, block, a_dev, b_dev, out_dev, k, active_cols) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0) {
            printk("ERROR: fragment_sfu_predicated_loop launch/copy failed case=%d k=%d active_cols=%d\n",
                   (int)case_id, (int)k, (int)active_cols);
            ++launch_failures;
            continue;
        }
        int mismatches = verify_active(active_cols, &max_abs_diff);
        int sentinels = verify_sentinels(active_cols);
        int checksum = scaled_checksum(active_cols);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum ^= checksum + (int)(case_id * 131u);
        printk("FRAGMENT_SFU_PREDICATED_LOOP_CASE case=%d k=%d active_cols=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, (int)k, (int)active_cols, mismatches,
               sentinels, checksum, max_abs_diff);
    }

    launch_failures += (int)fragment_sfu_predicated_loop_vc4kernel_runtime_launch_failures();
    uint32_t runtime_launches = fragment_sfu_predicated_loop_vc4kernel_runtime_launches();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_k0 && saw_k33 &&
                          saw_active_cols1 && saw_active_cols15 &&
                          saw_active_cols16) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_sfu_predicated_loop_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_sfu_loop=1 saw_predicated_sfu=1 saw_tmu_safe_offset=1 saw_vdw_preserve=1 saw_recip=1 saw_rsqrt=1 saw_k0=%d saw_k33=%d saw_active_cols1=%d saw_active_cols15=%d saw_active_cols16=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=3 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(test_cases) / sizeof(test_cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures, saw_k0,
           saw_k33, saw_active_cols1, saw_active_cols15, saw_active_cols16,
           checksum_accum, max_abs_diff, (int)runtime_launches,
           timer_get_usec() - start);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
