#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define VALUE_COPY_F32_TAIL_VC4VALUE_MAX_N 1000u
#define VALUE_COPY_F32_TAIL_VC4VALUE_GUARD 32u
#define VALUE_COPY_F32_TAIL_VC4VALUE_ACTIVE_QPUS 1u
#define VALUE_COPY_F32_TAIL_VC4VALUE_LANES 16u
#define VALUE_COPY_F32_TAIL_VC4VALUE_ELEMENTS_PER_WAVE \
    (VALUE_COPY_F32_TAIL_VC4VALUE_ACTIVE_QPUS * VALUE_COPY_F32_TAIL_VC4VALUE_LANES)
#define VALUE_COPY_F32_TAIL_VC4VALUE_MAX_WAVES \
    ((VALUE_COPY_F32_TAIL_VC4VALUE_MAX_N + VALUE_COPY_F32_TAIL_VC4VALUE_ELEMENTS_PER_WAVE - 1u) / \
     VALUE_COPY_F32_TAIL_VC4VALUE_ELEMENTS_PER_WAVE)
#define VALUE_COPY_F32_TAIL_VC4VALUE_MAX_COVERAGE_N \
    (VALUE_COPY_F32_TAIL_VC4VALUE_MAX_WAVES * VALUE_COPY_F32_TAIL_VC4VALUE_ELEMENTS_PER_WAVE)
#define VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N \
    (VALUE_COPY_F32_TAIL_VC4VALUE_MAX_COVERAGE_N + 2u * VALUE_COPY_F32_TAIL_VC4VALUE_GUARD)
#define VALUE_COPY_F32_TAIL_VC4VALUE_SENTINEL_BITS 0xc56a4000u

static const uint32_t test_sizes[] = {
    0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u,
    63u, 64u, 65u, 127u, 128u, 129u, 193u, 1000u
};

static float x_values[VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N];
static float y_values[VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N];
static uint32_t expected_bits[VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N];

static float bits_to_float(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } value;
    value.u = bits;
    return value.f;
}

static uint32_t float_to_bits(float value) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.f = value;
    return bits.u;
}

static uint32_t rounded_waves(uint32_t n) {
    uint32_t waves = (n + VALUE_COPY_F32_TAIL_VC4VALUE_ELEMENTS_PER_WAVE - 1u) /
                     VALUE_COPY_F32_TAIL_VC4VALUE_ELEMENTS_PER_WAVE;
    return waves == 0u ? 1u : waves;
}

static uint32_t input_bits_for_index(uint32_t i) {
    int32_t whole = (int32_t)((i * 17u + 5u) % 97u) - 48;
    uint32_t quarter = (i * 3u + 1u) & 3u;
    float value = (float)whole + (float)quarter * 0.25f;
    if ((i & 1u) != 0u)
        value = -value;
    if ((i % 29u) == 0u)
        value = 0.0f;
    return float_to_bits(value);
}

static void fill_buffers(uint32_t n) {
    const float sentinel = bits_to_float(VALUE_COPY_F32_TAIL_VC4VALUE_SENTINEL_BITS);
    for (uint32_t i = 0; i < VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N; i++) {
        uint32_t active_index = i - VALUE_COPY_F32_TAIL_VC4VALUE_GUARD;
        uint32_t bits = input_bits_for_index(active_index);
        x_values[i] = bits_to_float(bits);
        y_values[i] = sentinel;
        expected_bits[i] = VALUE_COPY_F32_TAIL_VC4VALUE_SENTINEL_BITS;
    }
    for (uint32_t i = 0; i < n; i++)
        expected_bits[VALUE_COPY_F32_TAIL_VC4VALUE_GUARD + i] = input_bits_for_index(i);
}

static void verify_results(uint32_t n, int *mismatch_count, float *max_abs_diff) {
    *mismatch_count = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = VALUE_COPY_F32_TAIL_VC4VALUE_GUARD + i;
        uint32_t got = float_to_bits(y_values[index]);
        uint32_t expected = expected_bits[index];
        if (got != expected) {
            float diff = y_values[index] - bits_to_float(expected);
            float abs_diff = diff < 0.0f ? -diff : diff;
            if (abs_diff > *max_abs_diff)
                *max_abs_diff = abs_diff;
            if (*mismatch_count < 8)
                printk("ERROR: value_copy_f32_tail_vc4value n=%d i=%d gpu_bits=%x expected_bits=%x gpu=%f expected=%f\n",
                       (int)n, (int)i, got, expected, y_values[index],
                       bits_to_float(expected));
            (*mismatch_count)++;
        }
    }
}

static int verify_sentinel_region(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N; i++) {
        uint32_t active_begin = VALUE_COPY_F32_TAIL_VC4VALUE_GUARD;
        uint32_t active_end = VALUE_COPY_F32_TAIL_VC4VALUE_GUARD + n;
        if (i >= active_begin && i < active_end)
            continue;
        uint32_t got = float_to_bits(y_values[i]);
        if (got != VALUE_COPY_F32_TAIL_VC4VALUE_SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: value_copy_f32_tail_vc4value sentinel changed n=%d i=%d bits=%x expected=%x\n",
                       (int)n, (int)i, got,
                       VALUE_COPY_F32_TAIL_VC4VALUE_SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static int checksum_low16(uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = VALUE_COPY_F32_TAIL_VC4VALUE_GUARD + i;
        checksum += (int)(float_to_bits(y_values[index]) & 0xffffu);
    }
    return checksum;
}

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t hash_output_bits(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = VALUE_COPY_F32_TAIL_VC4VALUE_GUARD + i;
        hash ^= float_to_bits(y_values[index]) + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N * sizeof(float);
    vc4_deviceptr_t x_dev = 0;
    vc4_deviceptr_t y_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0)
        panic("value_copy_f32_tail_vc4value device allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    uint32_t output_hash = 2166136261u;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(VALUE_COPY_F32_TAIL_VC4VALUE_ELEMENTS_PER_WAVE, 1, 1);

    printk("Running VC4 value_copy_f32_tail_vc4value candidate bundle...\n");
    for (uint32_t case_id = 0; case_id < sizeof(test_sizes) / sizeof(test_sizes[0]); case_id++) {
        uint32_t n = test_sizes[case_id];
        uint32_t waves = rounded_waves(n);
        uint32_t rounded_coverage = waves * VALUE_COPY_F32_TAIL_VC4VALUE_ELEMENTS_PER_WAVE;
        vc4_dim3 grid = vc4_m2_dim3(waves, 1, 1);
        fill_buffers(n);

        vc4_deviceptr_t x_active_dev =
            x_dev + VALUE_COPY_F32_TAIL_VC4VALUE_GUARD * sizeof(float);
        vc4_deviceptr_t y_active_dev =
            y_dev + VALUE_COPY_F32_TAIL_VC4VALUE_GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
            value_copy_f32_tail_vc4value_launch(program, grid, block, x_active_dev,
                                                y_active_dev, n) < 0 ||
            vc4_m2_copy_dtoh(program, y_values, y_dev, bytes) < 0) {
            printk("ERROR: value_copy_f32_tail_vc4value launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)n);
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(n, &mismatches, &max_abs_diff);
        int case_sentinels = verify_sentinel_region(n);
        int checksum = checksum_low16(n);
        output_hash ^= hash_output_bits(n) + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinels;
        checksum_accum += checksum;
        printk("VALUE_COPY_F32_TAIL_VC4VALUE_CASE case=%d n=%d waves=%d coverage=%d buffer_n=%d mismatches=%d sentinel_mismatches=%d checksum=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)n, (int)waves, (int)rounded_coverage,
               VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N, mismatches, case_sentinels,
               checksum, hash_output_bits(n), max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=value_copy_f32_tail_vc4value status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_coverage_n=%d buffer_n=%d checksum_accum=%d output_hash=%u max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(test_sizes) / sizeof(test_sizes[0])),
           total_mismatches, sentinel_mismatches, launch_failures,
           VALUE_COPY_F32_TAIL_VC4VALUE_ACTIVE_QPUS,
           VALUE_COPY_F32_TAIL_VC4VALUE_LANES,
           VALUE_COPY_F32_TAIL_VC4VALUE_MAX_N,
           VALUE_COPY_F32_TAIL_VC4VALUE_MAX_COVERAGE_N,
           VALUE_COPY_F32_TAIL_VC4VALUE_BUFFER_N, checksum_accum,
           output_hash, max_abs_diff_overall, 2,
           (int)(sizeof(test_sizes) / sizeof(test_sizes[0])), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4_program_destroy(program);
}
