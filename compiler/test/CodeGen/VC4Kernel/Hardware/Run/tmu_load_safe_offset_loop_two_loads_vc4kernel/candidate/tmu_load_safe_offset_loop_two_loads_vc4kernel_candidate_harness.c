#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ACTIVE_CASES 3u
#define MAX_K 33u
#define A_WORDS MAX_K
#define B_WORDS (MAX_K * LANES)
#define OUT_WORDS (ACTIVE_CASES * LANES + 16u)
#define EPSILON 0.0001f
#define SENTINEL (-913.5f)

static const uint32_t active_cols_cases[ACTIVE_CASES] = {1u, 15u, 16u};
static const uint32_t k_cases[] = {0u, 1u, 2u, 17u, 33u};

static float a_values[A_WORDS];
static float b_values[B_WORDS];
static float out_values[OUT_WORDS];
static float expected_values[ACTIVE_CASES][LANES];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float a_value(uint32_t j) {
    return ((float)((j * 5u + 3u) % 19u) - 9.0f) * 0.0625f;
}

static float b_value(uint32_t j, uint32_t lane) {
    return ((float)((j * 11u + lane * 7u + 1u) % 29u) - 14.0f) * 0.03125f;
}

static void fill_inputs(uint32_t k) {
    for (uint32_t j = 0; j < MAX_K; j++) {
        a_values[j] = a_value(j);
        for (uint32_t lane = 0; lane < LANES; lane++)
            b_values[j * LANES + lane] = b_value(j, lane);
    }
    for (uint32_t ac = 0; ac < ACTIVE_CASES; ac++) {
        uint32_t active_cols = active_cols_cases[ac];
        for (uint32_t lane = 0; lane < LANES; lane++) {
            float sum = 0.0f;
            if (lane < active_cols) {
                for (uint32_t j = 0; j < k; j++)
                    sum += a_value(j) * b_value(j, lane);
            }
            expected_values[ac][lane] = sum;
        }
    }
    for (uint32_t i = 0; i < OUT_WORDS; i++)
        out_values[i] = SENTINEL;
}

static int verify_values(float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t ac = 0; ac < ACTIVE_CASES; ac++) {
        for (uint32_t lane = 0; lane < LANES; lane++) {
            float got = out_values[ac * LANES + lane];
            float diff = got - expected_values[ac][lane];
            float ad = absf_local(diff);
            if (ad > *max_abs_diff)
                *max_abs_diff = ad;
            if (ad > EPSILON) {
                if (mismatches < 8)
                    printk("ERROR: safe_two_load_loop active_cols=%d lane=%d gpu=%f cpu=%f diff=%f\n",
                           (int)active_cols_cases[ac], (int)lane, got,
                           expected_values[ac][lane], diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = ACTIVE_CASES * LANES; i < OUT_WORDS; i++) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: safe_two_load_loop sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int scaled_checksum(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < ACTIVE_CASES * LANES; i++)
        checksum += (int)(out_values[i] * 1024.0f);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t a_dev = 0;
    vc4_deviceptr_t b_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, sizeof(a_values)) < 0 ||
        vc4_m2_malloc(program, &b_dev, sizeof(b_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("tmu_load_safe_offset_loop_two_loads_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_k0 = 0;
    int saw_k33 = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 tmu_load_safe_offset_loop_two_loads_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(k_cases) / sizeof(k_cases[0]); case_id++) {
        uint32_t k = k_cases[case_id];
        fill_inputs(k);
        if (vc4_m2_copy_htod(program, a_dev, a_values, sizeof(a_values)) < 0 ||
            vc4_m2_copy_htod(program, b_dev, b_values, sizeof(b_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            tmu_load_safe_offset_loop_two_loads_vc4kernel_launch(program, grid, block, a_dev, b_dev, out_dev, k) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0) {
            printk("ERROR: safe_two_load_loop launch/copy failed case=%d k=%d\n",
                   (int)case_id, (int)k);
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
        saw_k0 |= k == 0u;
        saw_k33 |= k == 33u;
        printk("TMU_LOAD_SAFE_OFFSET_LOOP_TWO_LOADS_CASE case=%d k=%d logical_active_cols=3 mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f\n",
               (int)case_id, (int)k, mismatches, sentinels, checksum,
               max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_k0 && saw_k33) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=tmu_load_safe_offset_loop_two_loads_vc4kernel status=%s cases=15 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_k0=%d saw_k33=%d saw_active_cols1=1 saw_active_cols15=1 saw_active_cols16=1 poison_inactive_offsets=1 checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           saw_k0, saw_k33, checksum_accum, max_abs_diff_overall,
           (int)tmu_load_safe_offset_loop_two_loads_vc4kernel_runtime_allocations(),
           (int)tmu_load_safe_offset_loop_two_loads_vc4kernel_runtime_launches(), elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
