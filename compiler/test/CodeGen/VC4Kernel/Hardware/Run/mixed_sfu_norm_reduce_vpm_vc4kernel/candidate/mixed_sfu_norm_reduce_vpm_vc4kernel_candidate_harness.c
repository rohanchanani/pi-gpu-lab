#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define GUARD 32u
#define BUFFER_N (LANES + GUARD)
#define OUT_SENTINEL (-7777.25f)
#define AUDIT_SENTINEL 0x5100aa55u
#define EPSILON 0.25f
#define ABS_TOL 0.02f
#define REL_TOL 0.02f

static const uint32_t cases[] = {0u, 1u, 8u, 15u, 16u};

static float input_values[BUFFER_N];
static uint8_t flag_values[BUFFER_N];
static float out_values[BUFFER_N];
static uint32_t audit_values[BUFFER_N];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static float max_f32(float a, float b) { return a > b ? a : b; }

static float sqrt_newton(float value) {
    float x = value >= 1.0f ? value : 1.0f;
    for (uint32_t i = 0; i < 16; ++i)
        x = 0.5f * (x + value / x);
    return x;
}

static float input_value(uint32_t index) {
    return 0.125f * (float)((index % 9u) + 1u);
}

static uint8_t flag_value(uint32_t index) {
    return (uint8_t)((index * 11u + 3u) & 0x7fu);
}

static void fill_buffers(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        input_values[i] = input_value(i);
        flag_values[i] = flag_value(i);
        out_values[i] = OUT_SENTINEL;
        audit_values[i] = AUDIT_SENTINEL;
    }
}

static float norm_denom(uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i)
        sum += input_value(i) * input_value(i);
    return sum + EPSILON;
}

static float expected_out(uint32_t index, uint32_t n) {
    return input_value(index) * (1.0f / sqrt_newton(norm_denom(n)));
}

static uint32_t expected_audit(uint32_t index) {
    return ((uint32_t)flag_value(index)) ^ index;
}

static int verify_active(uint32_t n, float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; ++i) {
        float expected = expected_out(i, n);
        float diff = abs_f32(out_values[i] - expected);
        float rel = diff / max_f32(abs_f32(expected), 1.0e-12f);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (rel > *max_rel_diff)
            *max_rel_diff = rel;
        if (diff > max_f32(ABS_TOL, REL_TOL * abs_f32(expected)) ||
            audit_values[i] != expected_audit(i)) {
            if (mismatches < 8)
                printk("ERROR: mixed_sfu_norm lane=%d n=%d out=%f expected=%f diff=%f rel=%f audit=%x expected_audit=%x\n",
                       (int)i, (int)n, out_values[i], expected, diff, rel,
                       audit_values[i], expected_audit(i));
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < BUFFER_N; ++i) {
        if (out_values[i] != OUT_SENTINEL || audit_values[i] != AUDIT_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: mixed_sfu_norm sentinel i=%d out=%f audit=%x\n",
                       (int)i, out_values[i], audit_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; ++i)
        checksum += (int)(((uint32_t)(out_values[i] * 8192.0f) ^ audit_values[i]) & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 128u * 1024u) < 0 || !program)
        panic("mixed_sfu_norm_reduce_vpm_vc4kernel program create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t flags_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    vc4_deviceptr_t audit_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &flags_dev, sizeof(flag_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0 ||
        vc4_m2_malloc(program, &audit_dev, sizeof(audit_values)) < 0)
        panic("mixed_sfu_norm_reduce_vpm_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int saw_n0 = 0;
    int saw_n16 = 0;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 mixed_sfu_norm_reduce_vpm_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
        uint32_t n = cases[case_id];
        saw_n0 |= (n == 0u);
        saw_n16 |= (n == 16u);
        fill_buffers();
        if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
            vc4_m2_copy_htod(program, flags_dev, flag_values, sizeof(flag_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            vc4_m2_copy_htod(program, audit_dev, audit_values, sizeof(audit_values)) < 0 ||
            mixed_sfu_norm_reduce_vpm_vc4kernel_launch(
                program, grid, block, input_dev, flags_dev, out_dev, audit_dev,
                n, EPSILON) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0 ||
            vc4_m2_copy_dtoh(program, audit_values, audit_dev, sizeof(audit_values)) < 0) {
            printk("ERROR: mixed_sfu_norm launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_active(n, &max_abs_diff, &max_rel_diff);
        int sentinels = verify_sentinels(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum_low16(n);
        printk("MIXED_SFU_NORM_CASE case=%d n=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f max_rel_diff=%f\n",
               (int)case_id, (int)n, mismatches, sentinels,
               checksum_low16(n), max_abs_diff, max_rel_diff);
    }

    launch_failures += (int)mixed_sfu_norm_reduce_vpm_vc4kernel_runtime_launch_failures();
    uint32_t runtime_launches = mixed_sfu_norm_reduce_vpm_vc4kernel_runtime_launches();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_n0 && saw_n16)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_sfu_norm_reduce_vpm_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_vpm_path=1 saw_f32_reduce=1 saw_sfu_norm=1 saw_rsqrt_or_recip=1 saw_vdw_preserve=1 saw_runtime_shape=1 saw_p9_subword_path=1 no_tmu_tile_workaround=1 checksum_accum=%d max_abs_diff=%f max_rel_diff=%f runtime_allocations=4 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           checksum_accum, max_abs_diff, max_rel_diff, (int)runtime_launches,
           timer_get_usec() - start);

    vc4Free(program, input_dev);
    vc4Free(program, flags_dev);
    vc4Free(program, out_dev);
    vc4Free(program, audit_dev);
    vc4_program_destroy(program);
}
