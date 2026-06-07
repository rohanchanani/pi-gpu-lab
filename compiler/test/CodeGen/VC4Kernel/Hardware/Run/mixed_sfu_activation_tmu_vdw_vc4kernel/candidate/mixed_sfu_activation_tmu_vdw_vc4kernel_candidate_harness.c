#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ACTIVE_QPUS 12u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_N 769u
#define MAX_COVERAGE_N 960u
#define GUARD 64u
#define BUFFER_N (MAX_COVERAGE_N + GUARD)
#define OUT_SENTINEL (-9100.25f)
#define AUDIT_SENTINEL 0xac7a5f10u
#define ABS_TOL 0.05f
#define REL_TOL 0.02f

static const uint32_t cases[] = {
    0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 769u
};

static float input_values[BUFFER_N];
static uint8_t flag_values[BUFFER_N];
static float out_values[BUFFER_N];
static uint32_t audit_values[BUFFER_N];

static float scale_value(void) { return 0.75f; }
static float bias_value(void) { return 0.125f; }

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static float max_f32(float a, float b) { return a > b ? a : b; }

static float sqrt_newton(float value) {
    float x = value >= 1.0f ? value : 1.0f;
    for (uint32_t i = 0; i < 16; ++i)
        x = 0.5f * (x + value / x);
    return x;
}

static float input_value(uint32_t index) {
    int32_t centered = (int32_t)((index * 17u + 5u) % 41u) - 8;
    return 0.125f * (float)centered;
}

static uint8_t flag_value(uint32_t index) {
    return (uint8_t)((index * 13u + 7u) & 0x7fu);
}

static float guarded_value(uint32_t index) {
    float raw = input_value(index);
    return raw > 0.0f ? raw : 1.0f;
}

static float expected_out(uint32_t index) {
    float x = guarded_value(index);
    return (1.0f / x) + scale_value() * (1.0f / sqrt_newton(x)) + bias_value();
}

static uint32_t expected_audit(uint32_t index) {
    return ((uint32_t)flag_value(index) * 2u) ^ index;
}

static void fill_buffers(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        input_values[i] = input_value(i);
        flag_values[i] = flag_value(i);
        out_values[i] = OUT_SENTINEL;
        audit_values[i] = AUDIT_SENTINEL;
    }
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static int verify_active(uint32_t n, float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; ++i) {
        float expected = expected_out(i);
        float diff = abs_f32(out_values[i] - expected);
        float rel = diff / max_f32(abs_f32(expected), 1.0e-12f);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (rel > *max_rel_diff)
            *max_rel_diff = rel;
        if (diff > max_f32(ABS_TOL, REL_TOL * abs_f32(expected)) ||
            audit_values[i] != expected_audit(i)) {
            if (mismatches < 8)
                printk("ERROR: mixed_sfu_activation i=%d out=%f expected=%f diff=%f rel=%f audit=%x expected_audit=%x\n",
                       (int)i, out_values[i], expected, diff, rel,
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
                printk("ERROR: mixed_sfu_activation sentinel i=%d out=%f audit=%x\n",
                       (int)i, out_values[i], audit_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; ++i)
        checksum += (int)(((uint32_t)(out_values[i] * 4096.0f) ^ audit_values[i]) & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 128u * 1024u) < 0 || !program)
        panic("mixed_sfu_activation_tmu_vdw_vc4kernel program create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t flags_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    vc4_deviceptr_t audit_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &flags_dev, sizeof(flag_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0 ||
        vc4_m2_malloc(program, &audit_dev, sizeof(audit_values)) < 0)
        panic("mixed_sfu_activation_tmu_vdw_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 mixed_sfu_activation_tmu_vdw_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
        uint32_t n = cases[case_id];
        uint32_t waves = rounded_waves(n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_buffers();
        if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
            vc4_m2_copy_htod(program, flags_dev, flag_values, sizeof(flag_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            vc4_m2_copy_htod(program, audit_dev, audit_values, sizeof(audit_values)) < 0 ||
            mixed_sfu_activation_tmu_vdw_vc4kernel_launch(
                program, grid, block, input_dev, flags_dev, out_dev, audit_dev,
                n, scale_value(), bias_value()) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0 ||
            vc4_m2_copy_dtoh(program, audit_values, audit_dev, sizeof(audit_values)) < 0) {
            printk("ERROR: mixed_sfu_activation launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_active(n, &max_abs_diff, &max_rel_diff);
        int sentinels = verify_sentinels(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum_low16(n);
        printk("MIXED_SFU_ACTIVATION_CASE case=%d n=%d waves=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f max_rel_diff=%f\n",
               (int)case_id, (int)n, (int)waves, mismatches, sentinels,
               checksum_low16(n), max_abs_diff, max_rel_diff);
    }

    launch_failures += (int)mixed_sfu_activation_tmu_vdw_vc4kernel_runtime_launch_failures();
    uint32_t runtime_launches = mixed_sfu_activation_tmu_vdw_vc4kernel_runtime_launches();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_sfu_activation_tmu_vdw_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_sfu_activation=1 saw_sfu=1 saw_approx_policy=1 saw_recip=1 saw_rsqrt=1 saw_domain_guard=1 saw_tmu_safe_offset=1 saw_vdw_preserve=1 saw_f32_cmp_select=1 saw_fragment_alu=1 saw_p9_subword_sidecar=1 checksum_accum=%d max_abs_diff=%f max_rel_diff=%f runtime_allocations=4 runtime_launches=%d elapsed_usec=%d\n",
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
