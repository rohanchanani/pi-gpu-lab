#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define SEGMENTS 3u
#define ACTIVE_N (LANES * SEGMENTS)
#define BUFFER_N (ACTIVE_N + 16u)
#define SENTINEL (-12345.0f)
#define CHECKSUM_SCALE 1024.0f

static const float input_values[LANES] = {
    -4.0f, 1.25f, 0.5f, -2.75f, 8.0f, -0.25f, 3.5f, 6.0f,
    -1.5f, 2.25f, -3.0f, 4.75f, 0.125f, 5.5f, -6.25f, 7.0f
};
static const uint32_t broadcast_lanes[SEGMENTS] = {2u, 9u, 14u};

static float out_values[BUFFER_N];

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; i++)
        out_values[i] = SENTINEL;
}

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static int verify_active(float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t segment = 0; segment < SEGMENTS; segment++) {
        float expected = input_values[broadcast_lanes[segment]];
        for (uint32_t lane = 0; lane < LANES; lane++) {
            uint32_t index = segment * LANES + lane;
            float diff = out_values[index] - expected;
            float ad = absf_local(diff);
            if (ad > *max_abs_diff)
                *max_abs_diff = ad;
            if (ad != 0.0f) {
                if (mismatches < 8)
                    printk("ERROR: reduce_f32_broadcast segment=%d lane=%d gpu=%f expected=%f diff=%f\n",
                           (int)segment, (int)lane, out_values[index],
                           expected, diff);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = ACTIVE_N; i < BUFFER_N; i++) {
        if (out_values[i] != SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: reduce_f32_broadcast sentinel i=%d gpu=%f expected=%f\n",
                       (int)i, out_values[i], SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int scaled_checksum(void) {
    int checksum = 0;
    for (uint32_t i = 0; i < ACTIVE_N; i++)
        checksum += (int)(out_values[i] * CHECKSUM_SCALE);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, LANES * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &out_dev, BUFFER_N * sizeof(float)) < 0)
        panic("fragment_reduce_f32_broadcast allocation failed");

    fill_output();
    int launch_failures = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, input_dev, input_values,
                         LANES * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, out_dev, out_values,
                         BUFFER_N * sizeof(float)) < 0 ||
        fragment_reduce_f32_broadcast_gemm_style_vc4kernel_launch(
            program, grid, block, input_dev, out_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_values, out_dev,
                         BUFFER_N * sizeof(float)) < 0)
        launch_failures++;

    float max_abs_diff = 0.0f;
    int total_mismatches = launch_failures ? 0 : verify_active(&max_abs_diff);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int checksum_accum = launch_failures ? 0 : scaled_checksum();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_reduce_f32_broadcast_gemm_style_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d one_hot_broadcasts=%d finite_tree=1 checksum_accum=%d max_abs_diff=%f runtime_allocations=2 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, (int)SEGMENTS, checksum_accum, max_abs_diff, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
