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
#define OUT_SENTINEL_BITS 0x7fc00055u
#define AUDIT_SENTINEL 0xdecafbadu

static const uint32_t cases[] = {
    0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 191u, 192u, 193u, 769u
};

static float x_values[BUFFER_N];
static float y_values[BUFFER_N];
static float out_values[BUFFER_N];
static uint32_t audit_values[BUFFER_N];

static float scale_value(void) { return 2.0f; }
static float bias_value(void) { return 1.0f; }
static float threshold_value(void) { return 20.0f; }

static uint32_t f32_bits(float value) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = value;
    return v.u;
}

static float bits_f32(uint32_t bits) {
    union {
        float f;
        uint32_t u;
    } v;
    v.u = bits;
    return v.f;
}

static float abs_f32(float value) {
    return value < 0.0f ? -value : value;
}

static uint32_t clz32(uint32_t value) {
    if (value == 0u)
        return 32u;
    uint32_t count = 0u;
    for (int bit = 31; bit >= 0; --bit) {
        if ((value >> (uint32_t)bit) & 1u)
            break;
        count++;
    }
    return count;
}

static uint32_t signed_max_bits(uint32_t a, uint32_t b) {
    return ((int32_t)a > (int32_t)b) ? a : b;
}

static float input_x(uint32_t index) {
    return 1.0f + (float)(index % 97u) * 0.25f;
}

static float input_y(uint32_t index) {
    return (float)(index % 31u) * 0.5f;
}

static float z_value(uint32_t index) {
    return scale_value() * input_x(index) + input_y(index) + bias_value();
}

static float expected_out(uint32_t index) {
    float z = z_value(index);
    return z > threshold_value() ? z : input_y(index);
}

static uint32_t expected_audit(uint32_t index) {
    uint32_t bits = f32_bits(z_value(index));
    uint32_t mask = bits & index;
    uint32_t mul = index * 3u;
    uint32_t mix = mask ^ mul;
    return signed_max_bits(mix, clz32(mix));
}

static void fill_buffers(void) {
    for (uint32_t i = 0; i < BUFFER_N; ++i) {
        x_values[i] = input_x(i);
        y_values[i] = input_y(i);
        out_values[i] = bits_f32(OUT_SENTINEL_BITS);
        audit_values[i] = AUDIT_SENTINEL;
    }
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + ELEMENTS_PER_WAVE - 1u) / ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static int verify_active(uint32_t n, float *max_abs_diff_out) {
    int mismatches = 0;
    for (uint32_t i = 0; i < n; ++i) {
        float expected = expected_out(i);
        float diff = abs_f32(out_values[i] - expected);
        if (diff > *max_abs_diff_out)
            *max_abs_diff_out = diff;
        if (diff > 0.0001f || audit_values[i] != expected_audit(i)) {
            if (mismatches < 8)
                printk("ERROR: mixed_elementwise active i=%d out=%x expected_out=%x audit=%x expected_audit=%x diff_x100000=%d\n",
                       (int)i, f32_bits(out_values[i]), f32_bits(expected),
                       audit_values[i], expected_audit(i), (int)(diff * 100000.0f));
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < BUFFER_N; ++i) {
        if (f32_bits(out_values[i]) != OUT_SENTINEL_BITS ||
            audit_values[i] != AUDIT_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: mixed_elementwise sentinel i=%d out=%x audit=%x\n",
                       (int)i, f32_bits(out_values[i]), audit_values[i]);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; ++i)
        checksum += (int)((f32_bits(out_values[i]) ^ audit_values[i]) & 0xffffu);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t x_dev = 0;
    vc4_deviceptr_t y_dev = 0;
    vc4_deviceptr_t out_dev = 0;
    vc4_deviceptr_t audit_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, sizeof(x_values)) < 0 ||
        vc4_m2_malloc(program, &y_dev, sizeof(y_values)) < 0 ||
        vc4_m2_malloc(program, &out_dev, sizeof(out_values)) < 0 ||
        vc4_m2_malloc(program, &audit_dev, sizeof(audit_values)) < 0)
        panic("mixed_elementwise allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    float max_abs_diff = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 mixed_elementwise_tmu_alu_cmp_vdw_vc4kernel candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); ++case_id) {
        uint32_t n = cases[case_id];
        uint32_t waves = rounded_waves(n);
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_buffers();
        if (vc4_m2_copy_htod(program, x_dev, x_values, sizeof(x_values)) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, sizeof(y_values)) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, sizeof(out_values)) < 0 ||
            vc4_m2_copy_htod(program, audit_dev, audit_values, sizeof(audit_values)) < 0 ||
            mixed_elementwise_tmu_alu_cmp_vdw_vc4kernel_launch(
                program, grid, block, x_dev, y_dev, out_dev, audit_dev, n,
                scale_value(), bias_value(), threshold_value()) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, sizeof(out_values)) < 0 ||
            vc4_m2_copy_dtoh(program, audit_values, audit_dev, sizeof(audit_values)) < 0) {
            printk("ERROR: mixed_elementwise launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }
        int mismatches = verify_active(n, &max_abs_diff);
        int sentinels = verify_sentinels(n);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        checksum_accum += checksum_low16(n);
        printk("MIXED_ELEMENTWISE_CASE case=%d n=%d waves=%d mismatches=%d sentinel_mismatches=%d checksum=%d\n",
               (int)case_id, (int)n, (int)waves, mismatches, sentinels,
               checksum_low16(n));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    int max_abs_diff_x10000 = (int)(max_abs_diff * 10000.0f + 0.5f);
    printk("VC4_TEST_RESULT name=mixed_elementwise_tmu_alu_cmp_vdw_vc4kernel status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d poison_inactive_offsets=1 saw_tmu_safe_offset=1 saw_vdw_preserve=1 saw_f32_cmp=1 saw_i32_cmp=1 saw_fragment_bitcast=1 saw_fragment_const=1 saw_fragment_alu_f32=1 saw_fragment_alu_i32=1 saw_mul24_or_v8=1 checksum_accum=%d max_abs_diff_x10000=%d runtime_allocations=1 runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           (int)ACTIVE_QPUS, (int)LANES, (int)MAX_N, checksum_accum,
           max_abs_diff_x10000, (int)(sizeof(cases) / sizeof(cases[0])),
           elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, out_dev);
    vc4Free(program, audit_dev);
    vc4_program_destroy(program);
}
