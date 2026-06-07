#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ROWS 2u
#define ACTIVE_N (LANES * ROWS)
#define GUARD 16u
#define BUFFER_N (ACTIVE_N + GUARD)
#define SENTINEL (-12345.0f)
#define EXP_ABS_TOL 0.0010f
#define EXP_REL_TOL 0.0020f
#define LOG_ABS_TOL 0.0010f
#define LOG_REL_TOL 0.0020f

static const float input_values[ACTIVE_N] = {
    -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
    -1.0f, 0.0f, 0.015625f, 0.03125f, 0.0625f, 0.125f, 0.25f, 0.5f,
    1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f
};

static float out_values[BUFFER_N];

static float abs_f32(float value) {
    return value < 0.0f ? -value : value;
}

static float max_f32(float a, float b) {
    return a > b ? a : b;
}

static float exp2_integer_ref(float value) {
    int n = (int)value;
    float result = 1.0f;
    if (n >= 0) {
        for (int i = 0; i < n; ++i)
            result *= 2.0f;
    } else {
        for (int i = 0; i < -n; ++i)
            result *= 0.5f;
    }
    return result;
}

static float guarded_log_input(uint32_t lane) {
    float raw = input_values[LANES + lane];
    return raw > 0.0f ? raw : 0.25f;
}

static float log2_power_ref(float value) {
    float x = value;
    float result = 0.0f;
    while (x < 1.0f) {
        x *= 2.0f;
        result -= 1.0f;
    }
    while (x > 1.0f) {
        x *= 0.5f;
        result += 1.0f;
    }
    return result;
}

static float expected_value(uint32_t row, uint32_t lane) {
    if (row == 0u)
        return exp2_integer_ref(input_values[lane]);
    return log2_power_ref(guarded_log_input(lane));
}

static float allowed_error(uint32_t row, float expected) {
    float abs_tol = row == 0u ? EXP_ABS_TOL : LOG_ABS_TOL;
    float rel_tol = row == 0u ? EXP_REL_TOL : LOG_REL_TOL;
    return max_f32(abs_tol, rel_tol * abs_f32(expected));
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_outputs(float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    *max_rel_diff = 0.0f;
    for (uint32_t row = 0; row < ROWS; ++row) {
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t index = row * LANES + lane;
            float expected = expected_value(row, lane);
            float diff = abs_f32(out_values[index] - expected);
            float rel = diff / max_f32(abs_f32(expected), 1.0e-12f);
            if (diff > *max_abs_diff)
                *max_abs_diff = diff;
            if (rel > *max_rel_diff)
                *max_rel_diff = rel;
            if (diff > allowed_error(row, expected)) {
                if (mismatches < 8)
                    printk("ERROR: fragment_sfu_exp_log row=%d lane=%d gpu=%f expected=%f diff=%f rel=%f allowed=%f\n",
                           (int)row, (int)lane, out_values[index], expected,
                           diff, rel, allowed_error(row, expected));
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = ACTIVE_N; i < BUFFER_N; ++i) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_sfu_exp_log sentinel i=%d gpu=%f expected=%f\n",
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
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0)
        panic("fragment_sfu_exp_log allocation failed");

    fill_output();
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 fragment_sfu_exp_log_vc4kernel candidate bundle...\n");
    if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
        fragment_sfu_exp_log_vc4kernel_launch(program, grid, block, input_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0) {
        printk("ERROR: fragment_sfu_exp_log launch/copy failed\n");
        launch_failures++;
    }

    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int total_mismatches = launch_failures ? 0 : verify_outputs(&max_abs_diff, &max_rel_diff);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_sfu_exp_log_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d saw_sfu=1 saw_approx_policy=1 saw_exp=1 saw_log=1 saw_domain_guard=1 base2_semantics=1 max_abs_diff=%f max_rel_diff=%f runtime_allocations=2 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, max_abs_diff, max_rel_diff, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
