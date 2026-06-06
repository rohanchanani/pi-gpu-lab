#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 6u
#define BUFFER_N (LANES * SEGMENTS + 16u)
#define SENTINEL (-12345.0f)

static const float a_values[LANES] = {
    -8.0f, -4.0f, -2.5f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f,
    1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, -12.0f, 16.0f
};
static const float b_values[LANES] = {
    2.0f, -1.0f, 1.5f, -3.0f, 0.25f, -0.75f, 0.5f, -1.0f,
    2.5f, -2.0f, 1.0f, -8.0f, 6.0f, -4.0f, 3.0f, -16.0f
};
static float out_values[BUFFER_N];

static float absf_local(float v) { return v < 0.0f ? -v : v; }

static float expected_value(uint32_t segment, uint32_t lane) {
    float a = a_values[lane];
    float b = b_values[lane];
    if (segment == 0) return a + b;
    if (segment == 1) return a - b;
    if (segment == 2) return a < b ? a : b;
    if (segment == 3) return a > b ? a : b;
    if (segment == 4) return absf_local(a) < absf_local(b) ? absf_local(a) : absf_local(b);
    return absf_local(a) > absf_local(b) ? absf_local(a) : absf_local(b);
}

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i)
        out_values[i] = SENTINEL;
}

static int verify_active(float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            uint32_t index = segment * LANES + lane;
            float expected = expected_value(segment, lane);
            float diff = absf_local(out_values[index] - expected);
            if (diff > *max_abs_diff)
                *max_abs_diff = diff;
            if (diff != 0.0f) {
                if (mismatches < 8)
                    printk("ERROR: fragment_alu_add_f32_basic seg=%d lane=%d gpu=%f expected=%f diff=%f\n",
                           (int)segment, (int)lane, out_values[index], expected, diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = SEGMENTS * LANES; i < BUFFER_N; ++i) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_alu_add_f32_basic sentinel i=%d gpu=%f expected=%f\n",
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

    vc4_deviceptr_t a_dev = 0, b_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, LANES * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &b_dev, LANES * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &out_dev, BUFFER_N * sizeof(float)) < 0)
        panic("fragment_alu_add_f32_basic allocation failed");

    fill_output();
    int start = timer_get_usec();
    int launch_failures = 0;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, a_dev, a_values, LANES * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, b_dev, b_values, LANES * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values, BUFFER_N * sizeof(float)) < 0 ||
        fragment_alu_add_f32_basic_vc4kernel_launch(program, grid, block, a_dev, b_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev, BUFFER_N * sizeof(float)) < 0)
        launch_failures++;

    float max_abs_diff = 0.0f;
    int total_mismatches = launch_failures ? 0 : verify_active(&max_abs_diff);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_alu_add_f32_basic_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d op_segments=%d finite_only=1 checksum_accum=0 max_abs_diff=%f runtime_allocations=3 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)SEGMENTS, max_abs_diff, elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
