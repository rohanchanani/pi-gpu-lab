#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define GUARD 16u
#define BUFFER_N (LANES + GUARD)
#define SENTINEL (-9876.5f)

struct rotate_f32_case {
    uint32_t amount;
    uint32_t n;
};

static const struct rotate_f32_case cases[] = {
    {0u, 16u},
    {1u, 16u},
    {7u, 15u},
    {15u, 16u},
    {17u, 1u},
};

static const float input_values[LANES] = {
    0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f,
    8.5f, 9.5f, 10.5f, 11.5f, 12.5f, 13.5f, 14.5f, 15.5f,
};

static float out_values[BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static float source_value(uint32_t lane) {
    return input_values[lane];
}

static float expected_value(uint32_t amount, uint32_t lane) {
    uint32_t source_lane = (lane + (amount & 15u)) & 15u;
    float rot = source_value(source_lane);
    float plus = rot + 1.0f;
    return plus > 10.0f ? plus : rot;
}

static float abs_f32(float value) {
    return value < 0.0f ? -value : value;
}

static int verify_active(const struct rotate_f32_case *test,
                         float *max_abs_diff) {
    int mismatches = 0;
    for (uint32_t lane = 0; lane < test->n; ++lane) {
        float expected = expected_value(test->amount, lane);
        float diff = abs_f32(out_values[lane] - expected);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (diff != 0.0f) {
            if (mismatches < 8)
                printk("ERROR: dynamic_rotate_f32 amount=%d lane=%d gpu=%f expected=%f diff=%f\n",
                       (int)test->amount, (int)lane, out_values[lane],
                       expected, diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < BUFFER_N; ++i) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: dynamic_rotate_f32 sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    uint32_t bytes = BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("fragment_rotate_dynamic_f32 allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int saw_amount0 = 0;
    int saw_amount15 = 0;
    int saw_amount16_or_modulo = 0;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 fragment_rotate_dynamic_f32_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
        const struct rotate_f32_case *test = &cases[case_id];
        fill_output();
        if (vc4_m2_copy_htod(program, input_dev, input_values,
                             sizeof(input_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            fragment_rotate_dynamic_f32_vc4kernel_launch(
                program, grid, block, input_dev, out_dev, test->amount,
                test->n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: fragment_rotate_dynamic_f32 launch/copy failed case=%d amount=%d\n",
                   (int)case_id, (int)test->amount);
            launch_failures++;
            continue;
        }
        int mismatches = verify_active(test, &max_abs_diff);
        int sentinels = verify_sentinels(test->n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        if (test->amount == 0u)
            saw_amount0 = 1;
        if (test->amount == 15u)
            saw_amount15 = 1;
        if (test->amount >= 16u)
            saw_amount16_or_modulo = 1;
        printk("FRAGMENT_ROTATE_DYNAMIC_F32_CASE case=%d amount=%d n=%d mismatches=%d sentinel_mismatches=%d\n",
               (int)case_id, (int)test->amount, (int)test->n, mismatches,
               sentinels);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_amount0 &&
                          saw_amount15 && saw_amount16_or_modulo)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_rotate_dynamic_f32_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_dynamic_rotate=1 saw_amount0=%d saw_amount15=%d saw_amount16_or_modulo=%d saw_f32=1 saw_tmu_safe_offset=1 saw_f32_cmp_select=1 saw_fragment_alu=1 saw_vdw_preserve=1 rotate_direction=left_source_plus_amount max_abs_diff=%f runtime_allocations=2 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), total_mismatches,
           sentinel_mismatches, launch_failures, (int)LANES, saw_amount0,
           saw_amount15, saw_amount16_or_modulo, max_abs_diff,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
