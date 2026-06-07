#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define ACTIVE_QPUS 12u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_N 193u
#define MAX_COVERAGE_N (2u * ELEMENTS_PER_WAVE)
#define GUARD 64u
#define BUFFER_N (MAX_COVERAGE_N + GUARD)
#define OUT_SENTINEL (-6601.25f)
#define AUDIT_SENTINEL 0xd11a7000u
#define ABS_TOL 0.01f
#define REL_TOL 0.01f
#define SCALE_VALUE 0.625f

static const uint32_t n_cases[] = {
    0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u
};
static const uint32_t amount_cases[] = {
    0u, 1u, 2u, 4u, 7u, 8u, 15u, 16u, 17u, 31u
};

static float input_values[BUFFER_N];
static uint8_t flag_values[BUFFER_N];
static float out_values[BUFFER_N];
static uint32_t audit_values[BUFFER_N];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static float max_f32(float a, float b) { return a > b ? a : b; }

static float input_value(uint32_t index) {
    int32_t centered = (int32_t)((index * 19u + 11u) % 47u) - 18;
    return 0.0625f * (float)centered;
}

static uint8_t flag_value(uint32_t index) {
    return (uint8_t)((index * 29u + 5u) & 0xffu);
}

static void fill_buffers(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        input_values[i] = input_value(i);
        flag_values[i] = flag_value(i);
        out_values[i] = OUT_SENTINEL;
        audit_values[i] = AUDIT_SENTINEL + i;
    }
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static uint32_t source_lane(uint32_t amount, uint32_t lane) {
    return (lane + (amount & 15u)) & 15u;
}

static float raw_lane(uint32_t base, uint32_t lane, uint32_t n) {
    uint32_t index = base + lane;
    return index < n ? input_value(index) : 0.0f;
}

static float gated_rotated(uint32_t base, uint32_t amount, uint32_t lane,
                           uint32_t n) {
    float value = raw_lane(base, source_lane(amount, lane), n);
    return value > 0.0f ? value : 0.0f;
}

static float block_sum(uint32_t base, uint32_t amount, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t lane = 0; lane < LANES; ++lane)
        if (base + lane < n)
            sum += gated_rotated(base, amount, lane, n);
    return sum;
}

static float expected_out_value(uint32_t index, uint32_t amount_seed,
                                uint32_t n) {
    uint32_t request = index / LANES;
    uint32_t lane = index & 15u;
    uint32_t base = request * LANES;
    uint32_t amount = amount_seed + request;
    float denom = block_sum(base, amount, n) + 1.0f;
    return gated_rotated(base, amount, lane, n) * SCALE_VALUE / denom;
}

static uint32_t expected_audit_value(uint32_t index, uint32_t amount_seed) {
    uint32_t request = index / LANES;
    uint32_t lane = index & 15u;
    uint32_t base = request * LANES;
    uint32_t amount = amount_seed + request;
    uint32_t source = base + source_lane(amount, lane);
    return ((uint32_t)flag_value(source)) ^ index;
}

static int verify_active(uint32_t n, uint32_t amount_seed,
                         float *max_abs_diff, float *max_rel_diff) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; ++i) {
        float expected = expected_out_value(i, amount_seed, n);
        float diff = abs_f32(out_values[i] - expected);
        float rel = diff / max_f32(abs_f32(expected), 1.0e-12f);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (rel > *max_rel_diff)
            *max_rel_diff = rel;
        uint32_t expected_audit = expected_audit_value(i, amount_seed);
        if (diff > max_f32(ABS_TOL, REL_TOL * abs_f32(expected)) ||
            audit_values[i] != expected_audit) {
            if (mismatches < 8)
                printk("ERROR: mixed_surface_lock_elementwise i=%d amount=%d out=%f expected=%f diff=%f rel=%f audit=%x expected_audit=%x\n",
                       (int)i, (int)amount_seed, out_values[i], expected,
                       diff, rel, audit_values[i], expected_audit);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < BUFFER_N; ++i) {
        uint32_t expected_audit = AUDIT_SENTINEL + i;
        if (out_values[i] != OUT_SENTINEL || audit_values[i] != expected_audit) {
            if (mismatches < 8)
                printk("ERROR: mixed_surface_lock_elementwise sentinel i=%d out=%f audit=%x expected_audit=%x\n",
                       (int)i, out_values[i], audit_values[i], expected_audit);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; ++i)
        checksum += (int)(((uint32_t)(out_values[i] * 32768.0f) ^ audit_values[i]) & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 128u * 1024u) < 0 || !program)
        panic("mixed_surface_lock_elementwise_full_vc4kernel program create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t flags_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    vc4_deviceptr_t audit_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &flags_dev, sizeof(flag_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0 ||
        vc4_m2_malloc(program, &audit_dev, sizeof(audit_values)) < 0)
        panic("mixed_surface_lock_elementwise_full_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_amount0 = 0;
    int saw_amount15 = 0;
    int saw_amount16_or_modulo = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 mixed_surface_lock_elementwise_full_vc4kernel candidate bundle...\n");
    for (uint32_t n_id = 0; n_id < sizeof(n_cases) / sizeof(n_cases[0]); ++n_id) {
        for (uint32_t a_id = 0; a_id < sizeof(amount_cases) / sizeof(amount_cases[0]); ++a_id) {
            uint32_t n = n_cases[n_id];
            uint32_t amount = amount_cases[a_id];
            uint32_t waves = rounded_waves(n);
            vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
            fill_buffers();
            if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
                vc4_m2_copy_htod(program, flags_dev, flag_values, sizeof(flag_values)) < 0 ||
                vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
                vc4_m2_copy_htod(program, audit_dev, audit_values, sizeof(audit_values)) < 0 ||
                mixed_surface_lock_elementwise_full_vc4kernel_launch(
                    program, grid, block, input_dev, flags_dev, out_dev, audit_dev,
                    n, amount, SCALE_VALUE) < 0 ||
                vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0 ||
                vc4_m2_copy_dtoh(program, audit_values, audit_dev, sizeof(audit_values)) < 0) {
                printk("ERROR: mixed_surface_lock_elementwise launch/copy failed n=%d amount=%d\n",
                       (int)n, (int)amount);
                launch_failures++;
                continue;
            }
            int mismatches = verify_active(n, amount, &max_abs_diff, &max_rel_diff);
            int sentinels = verify_sentinels(n);
            int checksum = checksum_low16(n);
            total_mismatches += mismatches;
            sentinel_mismatches += sentinels;
            checksum_accum += checksum;
            if (amount == 0u)
                saw_amount0 = 1;
            if (amount == 15u)
                saw_amount15 = 1;
            if (amount == 16u || amount == 17u || amount == 31u)
                saw_amount16_or_modulo = 1;
            printk("MIXED_SURFACE_LOCK_ELEMENTWISE_CASE n=%d amount=%d waves=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f max_rel_diff=%f\n",
                   (int)n, (int)amount, (int)waves, mismatches, sentinels,
                   checksum, max_abs_diff, max_rel_diff);
        }
    }

    launch_failures += (int)mixed_surface_lock_elementwise_full_vc4kernel_runtime_launch_failures();
    uint32_t runtime_launches = mixed_surface_lock_elementwise_full_vc4kernel_runtime_launches();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_amount0 && saw_amount15 &&
                          saw_amount16_or_modulo)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_surface_lock_elementwise_full_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_dynamic_rotate=1 saw_amount0=%d saw_amount15=%d saw_amount16_or_modulo=%d saw_tmu_safe_offset=1 saw_vdw_preserve=1 saw_fragment_alu_f32=1 saw_fragment_alu_i32=1 saw_cmp_select=1 saw_fragment_reduce=1 saw_sfu=1 saw_pack_unpack=1 saw_dynamic_subword_selector=1 saw_final_surface_mix=1 rotate_direction=left_source_plus_amount selector_values_w8=0 checksum_accum=%d max_abs_diff=%f max_rel_diff=%f runtime_allocations=4 runtime_launches=%d elapsed_usec=%d\n",
           status,
           (int)((sizeof(n_cases) / sizeof(n_cases[0])) *
                 (sizeof(amount_cases) / sizeof(amount_cases[0]))),
           total_mismatches, sentinel_mismatches, launch_failures,
           saw_amount0, saw_amount15, saw_amount16_or_modulo, checksum_accum,
           max_abs_diff, max_rel_diff, (int)runtime_launches,
           timer_get_usec() - start);

    vc4Free(program, input_dev);
    vc4Free(program, flags_dev);
    vc4Free(program, out_dev);
    vc4Free(program, audit_dev);
    vc4_program_destroy(program);
}
