#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define BUFFER_N (LANES + 16u)
#define I_SENTINEL 0xdeadbeefu
#define F_SENTINEL (-12345.0f)

static const float fa_values[LANES] = {
    -8.0f, -4.0f, -2.0f, -1.0f, -0.5f, 0.25f, 0.5f, 1.0f,
    1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, -12.0f, 16.0f
};
static const float fb_values[LANES] = {
    0.5f, -0.5f, 1.5f, -2.0f, 4.0f, -8.0f, 6.0f, -1.0f,
    2.0f, -3.0f, 4.0f, -0.25f, 0.5f, -0.5f, 0.25f, -0.125f
};
static const uint32_t ia_values[LANES] = {
    0u, 1u, 2u, 3u, 7u, 15u, 255u, 256u,
    1024u, 4096u, 65535u, 0x00ffffu, 0x0fffffu, 0x7fffffu, 0x800000u, 0xffffffu
};
static const uint32_t ib_values[LANES] = {
    0u, 3u, 5u, 7u, 11u, 13u, 17u, 19u,
    23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u
};
static float out_f32[BUFFER_N];
static uint32_t out_i32[BUFFER_N];

static float absf_local(float v) { return v < 0.0f ? -v : v; }

static void fill_output(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        out_f32[i] = F_SENTINEL;
        out_i32[i] = I_SENTINEL;
    }
}

static int verify_outputs(float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        float expected_f = fa_values[lane] * fb_values[lane];
        uint32_t expected_i = (ia_values[lane] & 0xffffffu) * (ib_values[lane] & 0xffffffu);
        float diff = absf_local(out_f32[lane] - expected_f);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (out_i32[lane] != expected_i || diff != 0.0f) {
            if (mismatches < 8)
                printk("ERROR: fragment_alu_mul_f32_mul24 lane=%d got_i=%x exp_i=%x got_f=%f exp_f=%f\n",
                       (int)lane, out_i32[lane], expected_i, out_f32[lane], expected_f);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(void) {
    int mismatches = 0;
    for (uint32_t i = LANES; i < BUFFER_N; ++i) {
        if (out_i32[i] != I_SENTINEL || out_f32[i] != F_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: fragment_alu_mul_f32_mul24 sentinel i=%d got_i=%x got_f=%f\n",
                       (int)i, out_i32[i], out_f32[i]);
            mismatches++;
        }
    }
    return mismatches;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t fa_dev = 0, fb_dev = 0, ia_dev = 0, ib_dev = 0, out_f_dev = 0, out_i_dev = 0;
    if (vc4_m2_malloc(program, &fa_dev, LANES * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &fb_dev, LANES * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &ia_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &ib_dev, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_malloc(program, &out_f_dev, BUFFER_N * sizeof(float)) < 0 ||
        vc4_m2_malloc(program, &out_i_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        panic("fragment_alu_mul_f32_mul24 allocation failed");

    fill_output();
    int start = timer_get_usec();
    int launch_failures = 0;
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    if (vc4_m2_copy_htod(program, fa_dev, fa_values, LANES * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, fb_dev, fb_values, LANES * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, ia_dev, ia_values, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, ib_dev, ib_values, LANES * sizeof(uint32_t)) < 0 ||
        vc4_m2_copy_htod(program, out_f_dev, out_f32, BUFFER_N * sizeof(float)) < 0 ||
        vc4_m2_copy_htod(program, out_i_dev, out_i32, BUFFER_N * sizeof(uint32_t)) < 0 ||
        fragment_alu_mul_f32_mul24_vc4kernel_launch(program, grid, block, fa_dev, fb_dev, ia_dev, ib_dev, out_f_dev, out_i_dev) < 0 ||
        vc4_m2_copy_dtoh(program, out_f32, out_f_dev, BUFFER_N * sizeof(float)) < 0 ||
        vc4_m2_copy_dtoh(program, out_i32, out_i_dev, BUFFER_N * sizeof(uint32_t)) < 0)
        launch_failures++;

    float max_abs_diff = 0.0f;
    int total_mismatches = launch_failures ? 0 : verify_outputs(&max_abs_diff);
    int sentinel_mismatches = launch_failures ? 0 : verify_sentinels();
    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=fragment_alu_mul_f32_mul24_vc4kernel status=%s cases=1 total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=1 lanes=%d mul24_24bit_operands=1 checksum_accum=0 max_abs_diff=%f runtime_allocations=6 runtime_launches=1 elapsed_usec=%d\n",
           status, total_mismatches, sentinel_mismatches, launch_failures,
           (int)LANES, max_abs_diff, elapsed);

    vc4Free(program, fa_dev);
    vc4Free(program, fb_dev);
    vc4Free(program, ia_dev);
    vc4Free(program, ib_dev);
    vc4Free(program, out_f_dev);
    vc4Free(program, out_i_dev);
    vc4_program_destroy(program);
}
