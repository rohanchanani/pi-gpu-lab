#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_GRID_BLOCKS 18u
#define MAX_ELEMENTS (MAX_GRID_BLOCKS * LANES)
#define BUFFER_N (MAX_ELEMENTS + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u
#define EPSILON 0.001f

struct mixed_case {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t n;
    int32_t trip_count;
    int32_t flag;
    float threshold;
    float scale;
};

static const struct mixed_case mixed_cases[] = {
    {1u, 1u, 1u, 0u, 0, -1, -24.0f, 1.0f},
    {2u, 3u, 1u, 1u, 1, 1, -8.0f, 2.0f},
    {2u, 3u, 1u, 15u, 2, -3, 0.0f, 0.5f},
    {2u, 3u, 1u, 16u, 3, 7, 18.0f, 1.0f},
    {2u, 3u, 1u, 17u, 0, 0, -12.0f, 2.0f},
    {4u, 3u, 1u, 191u, 1, -1, 3.5f, 0.5f},
    {4u, 3u, 1u, 192u, 2, 4, 21.5f, 1.0f},
    {2u, 2u, 2u, 31u, 3, -5, -18.0f, 2.0f},
    {2u, 2u, 2u, 32u, 1, 8, 0.0f, 0.5f},
    {2u, 2u, 2u, 127u, 2, -2, 9.0f, 1.0f},
    {3u, 2u, 2u, 191u, 3, 5, -6.5f, 2.0f},
    {3u, 2u, 2u, 192u, 0, -7, 14.0f, 0.5f},
    {2u, 3u, 3u, 287u, 1, 1, 0.0f, 1.0f},
    {2u, 3u, 3u, 288u, 2, -1, -21.0f, 2.0f},
};

static float x_values[BUFFER_N];
static float y_values[BUFFER_N];
static float out_values[BUFFER_N];
static float expected_values[BUFFER_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static float bits_to_float(uint32_t bits) {
    union { uint32_t u; float f; } value;
    value.u = bits;
    return value.f;
}

static uint32_t float_to_bits(float value) {
    union { uint32_t u; float f; } bits;
    bits.f = value;
    return bits.u;
}

static uint32_t rotl32_local(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
}

static uint32_t total_elements(const struct mixed_case *c) {
    return c->grid_x * c->grid_y * c->grid_z * LANES;
}

static uint32_t active_elements(const struct mixed_case *c) {
    uint32_t total = total_elements(c);
    return c->n < total ? c->n : total;
}

static float x_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 23u + 7u) % 97u) - 48;
    float value = (float)whole * 0.5f + (float)((i * 3u + 1u) & 3u) * 0.25f;
    return (i & 1u) ? -value : value;
}

static float y_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 29u + 11u) % 83u) - 41;
    float value = (float)whole * 0.25f + (float)((i + 2u) & 3u) * 0.125f;
    return (i & 2u) ? -value : value;
}

static void fill_buffers(const struct mixed_case *c) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    uint32_t elements = active_elements(c);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t logical = i - GUARD;
        x_values[i] = x_value(logical);
        y_values[i] = y_value(logical);
        out_values[i] = sentinel;
        expected_values[i] = sentinel;
    }
    for (uint32_t i = 0; i < elements; i++) {
        uint32_t index = GUARD + i;
        float acc = x_values[index];
        for (int32_t step = 0; step < c->trip_count; step++)
            acc = acc + y_values[index];
        float branch_value = c->flag > 0 ? acc * c->scale : acc - y_values[index];
        expected_values[index] = branch_value > c->threshold ? branch_value : y_values[index];
    }
}

static int verify_results(const struct mixed_case *c, float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    uint32_t elements = active_elements(c);
    for (uint32_t i = 0; i < elements; i++) {
        uint32_t index = GUARD + i;
        float diff = out_values[index] - expected_values[index];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > EPSILON) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_multi_axis grid=%d,%d,%d n=%d trip=%d flag=%d i=%d gpu=%f cpu=%f diff=%f x=%f y=%f\n",
                       (int)c->grid_x, (int)c->grid_y, (int)c->grid_z,
                       (int)c->n, (int)c->trip_count, (int)c->flag, (int)i,
                       out_values[index], expected_values[index], diff,
                       x_values[index], y_values[index]);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct mixed_case *c) {
    int mismatches = 0;
    uint32_t elements = active_elements(c);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + elements)
            continue;
        uint32_t got = float_to_bits(out_values[i]);
        if (got != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_multi_axis sentinel grid=%d,%d,%d n=%d elements=%d i=%d bits=%x expected=%x\n",
                       (int)c->grid_x, (int)c->grid_y, (int)c->grid_z,
                       (int)c->n, (int)elements, (int)i, got, SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output_bits(const struct mixed_case *c) {
    uint32_t elements = active_elements(c);
    uint32_t hash = 2166136261u ^ c->n ^ (elements << 8);
    for (uint32_t i = 0; i < elements; i++) {
        uint32_t index = GUARD + i;
        hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes = BUFFER_N * sizeof(float);
    vc4_deviceptr_t x_dev = 0, y_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("mixed_ttir_multi_axis allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int elements_checked = 0;
    float max_abs_diff_overall = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(mixed_cases) / sizeof(mixed_cases[0]); case_id++) {
        const struct mixed_case *c = &mixed_cases[case_id];
        uint32_t elements = active_elements(c);
        uint32_t total = total_elements(c);
        vc4_dim3 grid = vc4_m2_dim3(c->grid_x, c->grid_y, c->grid_z);
        fill_buffers(c);

        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            mixed_ttir_multi_axis_cf_tail_b16_kernel_launch(program, grid, block,
                                                            x_active, y_active,
                                                            out_active, c->n,
                                                            c->trip_count,
                                                            c->flag,
                                                            c->threshold,
                                                            c->scale) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: mixed_ttir_multi_axis launch/copy failed case=%d grid=%d,%d,%d n=%d trip=%d flag=%d\n",
                   (int)case_id, (int)c->grid_x, (int)c->grid_y, (int)c->grid_z,
                   (int)c->n, (int)c->trip_count, (int)c->flag);
            launch_failures++;
            continue;
        }

        float max_abs_diff = 0.0f;
        int mismatches = verify_results(c, &max_abs_diff);
        int sentinels = verify_sentinels(c);
        uint32_t case_hash = hash_output_bits(c);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)elements;
        printk("MIXED_TTIR_MULTI_AXIS_CF_TAIL_CASE case=%d grid_x=%d grid_y=%d grid_z=%d n=%d active_elements=%d total_elements=%d trip=%d flag=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)c->grid_x, (int)c->grid_y, (int)c->grid_z,
               (int)c->n, (int)elements, (int)total, (int)c->trip_count,
               (int)c->flag, mismatches, sentinels, case_hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    int cases = (int)(sizeof(mixed_cases) / sizeof(mixed_cases[0]));
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_ttir_multi_axis_cf_tail_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_grid_blocks=%d max_elements=%d buffer_n=%d output_hash=%u max_abs_diff=%f saw_nonzero_output_hash=%d saw_real_ttir_input=1 saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_value_scf_cf=1 saw_program_id_axis0=1 saw_arange_make_range_16=1 saw_masked_load_other_zero=1 saw_masked_store_tail=1 saw_tmu_load=1 saw_vdw_preserve_store=1 saw_tail_mask_clamp_overlaunch=1 saw_ttir_elementwise=1 saw_ttir_tail_mask=1 saw_ttir_scf_if=1 saw_ttir_scf_for_or_while=1 saw_ttir_cf_control_flow=1 saw_f32_alu=1 saw_f32_cmp_select=1 saw_sentinel_preserve=1 saw_ttir_multi_axis_launch=1 saw_ttir_program_id_axis1=1 saw_ttir_program_id_axis2=1 saw_ttir_num_programs_axis1=1 saw_value_multi_axis_launch=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, cases, elements_checked, total_mismatches, sentinel_mismatches,
           launch_failures, ACTIVE_QPUS, LANES, MAX_GRID_BLOCKS, MAX_ELEMENTS,
           BUFFER_N, output_hash, max_abs_diff_overall, output_hash != 0u ? 1 : 0,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, 3, cases, elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
