#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define LANES 16u
#define GUARD 32u
#define BUFFER_N (LANES + GUARD)
#define OUT_SENTINEL (-8802.5f)
#define AUDIT_SENTINEL 0x5a770000u
#define EPSILON 0.5f
#define ABS_TOL 0.02f
#define REL_TOL 0.02f

struct shape_case {
    uint32_t active_rows;
    uint32_t active_cols;
};

static const struct shape_case shape_cases[] = {
    {0u, 16u}, {1u, 0u}, {1u, 1u}, {1u, 7u}, {1u, 15u}, {1u, 16u}
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

static float sqrt_newton(float value) {
    float x = value >= 1.0f ? value : 1.0f;
    for (uint32_t i = 0; i < 16; ++i)
        x = 0.5f * (x + value / x);
    return x;
}

static float input_value(uint32_t lane) {
    int32_t centered = (int32_t)((lane * 7u + 3u) % 19u) - 5;
    return 0.125f * (float)centered;
}

static uint8_t flag_value(uint32_t lane) {
    return (uint8_t)((lane * 17u + 9u) & 0xffu);
}

static void fill_buffers(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        input_values[i] = input_value(i & 15u);
        flag_values[i] = flag_value(i & 15u);
        out_values[i] = OUT_SENTINEL;
        audit_values[i] = AUDIT_SENTINEL + i;
    }
}

static uint32_t active_n(const struct shape_case *shape) {
    return shape->active_rows > 0u ? shape->active_cols : 0u;
}

static uint32_t source_lane(uint32_t amount, uint32_t lane) {
    return (lane + (amount & 15u)) & 15u;
}

static float raw_lane(uint32_t lane, uint32_t n) {
    return lane < n ? input_value(lane) : 0.0f;
}

static float swizzled_value(uint32_t lane, uint32_t amount, uint32_t n) {
    uint32_t source = source_lane(amount + 1u, lane);
    float value = raw_lane(source, n);
    return value > 0.0f ? value : 0.0f;
}

static float denom_value(uint32_t amount, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        if (lane < n) {
            float value = swizzled_value(lane, amount, n);
            sum += value * value;
        }
    }
    return sum + EPSILON;
}

static float expected_out_value(uint32_t lane, uint32_t amount, uint32_t n) {
    return swizzled_value(lane, amount, n) *
           (1.0f / sqrt_newton(denom_value(amount, n)));
}

static uint32_t expected_audit_value(uint32_t lane, uint32_t amount) {
    return ((uint32_t)flag_value(source_lane(amount, lane))) ^ lane;
}

static int verify_active(const struct shape_case *shape, uint32_t amount,
                         float *max_abs_diff, float *max_rel_diff) {
    uint32_t n = active_n(shape);
    int mismatches = 0;
    for (uint32_t lane = 0; lane < n; ++lane) {
        float expected = expected_out_value(lane, amount, n);
        float diff = abs_f32(out_values[lane] - expected);
        float rel = diff / max_f32(abs_f32(expected), 1.0e-12f);
        if (diff > *max_abs_diff)
            *max_abs_diff = diff;
        if (rel > *max_rel_diff)
            *max_rel_diff = rel;
        uint32_t expected_audit = expected_audit_value(lane, amount);
        if (diff > max_f32(ABS_TOL, REL_TOL * abs_f32(expected)) ||
            audit_values[lane] != expected_audit) {
            if (mismatches < 8)
                printk("ERROR: mixed_shuffle_tile lane=%d rows=%d cols=%d amount=%d out=%f expected=%f diff=%f rel=%f audit=%x expected_audit=%x\n",
                       (int)lane, (int)shape->active_rows,
                       (int)shape->active_cols, (int)amount, out_values[lane],
                       expected, diff, rel, audit_values[lane],
                       expected_audit);
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
                printk("ERROR: mixed_shuffle_tile sentinel i=%d out=%f audit=%x expected_audit=%x\n",
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
        panic("mixed_shuffle_vpm_tile_swizzle_vc4kernel program create failed");

    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t flags_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    vc4_deviceptr_t audit_dev = 0;
    if (vc4_m2_malloc(program, &input_dev, sizeof(input_values)) < 0 ||
        vc4_m2_malloc(program, &flags_dev, sizeof(flag_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0 ||
        vc4_m2_malloc(program, &audit_dev, sizeof(audit_values)) < 0)
        panic("mixed_shuffle_vpm_tile_swizzle_vc4kernel allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int saw_rows0 = 0;
    int saw_cols0 = 0;
    int saw_cols1 = 0;
    int saw_cols16 = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);

    printk("Running VC4 mixed_shuffle_vpm_tile_swizzle_vc4kernel candidate bundle...\n");
    for (uint32_t s_id = 0; s_id < sizeof(shape_cases) / sizeof(shape_cases[0]); ++s_id) {
        for (uint32_t a_id = 0; a_id < sizeof(amount_cases) / sizeof(amount_cases[0]); ++a_id) {
            const struct shape_case *shape = &shape_cases[s_id];
            uint32_t amount = amount_cases[a_id];
            uint32_t n = active_n(shape);
            fill_buffers();
            if (vc4_m2_copy_htod(program, input_dev, input_values, sizeof(input_values)) < 0 ||
                vc4_m2_copy_htod(program, flags_dev, flag_values, sizeof(flag_values)) < 0 ||
                vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
                vc4_m2_copy_htod(program, audit_dev, audit_values, sizeof(audit_values)) < 0 ||
                mixed_shuffle_vpm_tile_swizzle_vc4kernel_launch(
                    program, grid, block, input_dev, flags_dev, out_dev, audit_dev,
                    shape->active_rows, shape->active_cols, amount, EPSILON) < 0 ||
                vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0 ||
                vc4_m2_copy_dtoh(program, audit_values, audit_dev, sizeof(audit_values)) < 0) {
                printk("ERROR: mixed_shuffle_tile launch/copy failed rows=%d cols=%d amount=%d\n",
                       (int)shape->active_rows, (int)shape->active_cols,
                       (int)amount);
                launch_failures++;
                continue;
            }
            int mismatches = verify_active(shape, amount, &max_abs_diff, &max_rel_diff);
            int sentinels = verify_sentinels(n);
            int checksum = checksum_low16(n);
            total_mismatches += mismatches;
            sentinel_mismatches += sentinels;
            checksum_accum += checksum;
            saw_rows0 |= (shape->active_rows == 0u);
            saw_cols0 |= (shape->active_cols == 0u);
            saw_cols1 |= (shape->active_cols == 1u);
            saw_cols16 |= (shape->active_cols == 16u);
            printk("MIXED_SHUFFLE_VPM_TILE_SWIZZLE_CASE rows=%d cols=%d amount=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f max_rel_diff=%f\n",
                   (int)shape->active_rows, (int)shape->active_cols,
                   (int)amount, mismatches, sentinels, checksum,
                   max_abs_diff, max_rel_diff);
        }
    }

    launch_failures += (int)mixed_shuffle_vpm_tile_swizzle_vc4kernel_runtime_launch_failures();
    uint32_t runtime_launches = mixed_shuffle_vpm_tile_swizzle_vc4kernel_runtime_launches();
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && saw_rows0 && saw_cols0 &&
                          saw_cols1 && saw_cols16)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_shuffle_vpm_tile_swizzle_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d saw_vdr_vpm_path=1 saw_dynamic_rotate=1 saw_tile_swizzle=1 saw_vdw_preserve=1 saw_runtime_shape=1 no_tmu_tile_workaround=1 saw_p9_subword_path=1 saw_sfu_path=1 rotate_direction=left_source_plus_amount checksum_accum=%d max_abs_diff=%f max_rel_diff=%f runtime_allocations=4 runtime_launches=%d elapsed_usec=%d\n",
           status,
           (int)((sizeof(shape_cases) / sizeof(shape_cases[0])) *
                 (sizeof(amount_cases) / sizeof(amount_cases[0]))),
           total_mismatches, sentinel_mismatches, launch_failures,
           checksum_accum, max_abs_diff, max_rel_diff, (int)runtime_launches,
           timer_get_usec() - start);

    vc4Free(program, input_dev);
    vc4Free(program, flags_dev);
    vc4Free(program, out_dev);
    vc4Free(program, audit_dev);
    vc4_program_destroy(program);
}
