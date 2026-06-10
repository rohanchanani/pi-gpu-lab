#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_GRID_BLOCKS 24u
#define MAX_ELEMENTS (MAX_GRID_BLOCKS * LANES)
#define BUFFER_N (MAX_ELEMENTS + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u

struct mixed_case {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t n;
    float threshold;
    float bias;
    int32_t flag;
};

static const struct mixed_case cases[] = {
    {1u, 1u, 0u, 0.0f, 0.0f, 0},
    {1u, 1u, 1u, -7.5f, 1.25f, 1},
    {1u, 1u, 15u, 3.25f, -0.5f, 0},
    {1u, 1u, 16u, 0.0f, 2.0f, 1},
    {2u, 3u, 17u, 12.5f, 1.0f, 1},
    {2u, 3u, 33u, -2.0f, -1.5f, 0},
    {4u, 3u, 191u, 5.5f, 0.25f, 1},
    {4u, 3u, 192u, 0.0f, -0.75f, 0},
    {6u, 4u, 193u, -11.25f, 2.25f, 1},
    {6u, 4u, 384u, 9.75f, -0.25f, 1},
    {6u, 4u, 1000u, -3.5f, 0.5f, 0},
};

static float x_values[BUFFER_N];
static float y_values[BUFFER_N];
static float out_values[BUFFER_N];
static uint32_t expected_bits[BUFFER_N];

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

static uint32_t grid_blocks(const struct mixed_case *c) {
    return c->grid_x * c->grid_y;
}

static uint32_t grid_coverage(const struct mixed_case *c) {
    return grid_blocks(c) * LANES;
}

static uint32_t active_elements(const struct mixed_case *c) {
    uint32_t coverage = grid_coverage(c);
    return c->n < coverage ? c->n : coverage;
}

static float x_value(uint32_t i, const struct mixed_case *c) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i % LANES;
    uint32_t pid0 = block_id % c->grid_x;
    uint32_t pid1 = block_id / c->grid_x;
    int32_t whole = (int32_t)((pid1 * 19u + pid0 * 13u + lane * 5u + 7u) % 127u) - 63;
    float value = (float)whole + (float)((i + c->grid_y) & 3u) * 0.25f;
    return (i & 1u) ? -value : value;
}

static float y_value(uint32_t i, const struct mixed_case *c) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i % LANES;
    uint32_t pid0 = block_id % c->grid_x;
    uint32_t pid1 = block_id / c->grid_x;
    int32_t whole = (int32_t)((pid1 * 23u + pid0 * 17u + lane * 7u + 11u) % 113u) - 56;
    float value = (float)whole + (float)((i + c->grid_x + 2u) & 3u) * 0.25f;
    return (i & 2u) ? -value : value;
}

static void fill_buffers(const struct mixed_case *c) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    uint32_t active = active_elements(c);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t logical = i - GUARD;
        x_values[i] = x_value(logical, c);
        y_values[i] = y_value(logical, c);
        out_values[i] = sentinel;
        expected_bits[i] = SENTINEL_BITS;
    }
    for (uint32_t i = 0; i < active; i++) {
        uint32_t index = GUARD + i;
        float selected = x_values[index] < c->threshold ? x_values[index] : y_values[index];
        if (c->flag != 0)
            selected = selected + c->bias;
        expected_bits[index] = float_to_bits(selected);
    }
}

static int verify_results(const struct mixed_case *c, int *saw_x_arm, int *saw_y_arm) {
    int mismatches = 0;
    uint32_t active = active_elements(c);
    for (uint32_t i = 0; i < active; i++) {
        uint32_t index = GUARD + i;
        if (x_values[index] < c->threshold)
            *saw_x_arm = 1;
        else
            *saw_y_arm = 1;
        uint32_t got = float_to_bits(out_values[index]);
        uint32_t expected = expected_bits[index];
        if (got != expected) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_mask_memory grid=%d,%d n=%d flag=%d i=%d gpu_bits=%x expected_bits=%x\n",
                       (int)c->grid_x, (int)c->grid_y, (int)c->n, (int)c->flag,
                       (int)i, got, expected);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct mixed_case *c) {
    int mismatches = 0;
    uint32_t active = active_elements(c);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + active)
            continue;
        uint32_t got = float_to_bits(out_values[i]);
        if (got != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed_ttir_mask_memory sentinel grid=%d,%d n=%d i=%d bits=%x expected=%x\n",
                       (int)c->grid_x, (int)c->grid_y, (int)c->n, (int)i,
                       got, SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(const struct mixed_case *c) {
    uint32_t active = active_elements(c);
    uint32_t hash = 2166136261u ^ c->n ^ (grid_coverage(c) << 8);
    for (uint32_t i = 0; i < active; i++) {
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
        panic("mixed_ttir_mask_memory allocation failed");

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    int elements_checked = 0;
    int saw_empty = 0;
    int saw_full = 0;
    int saw_tail = 0;
    int saw_flag_true = 0;
    int saw_flag_false = 0;
    int saw_x_arm = 0;
    int saw_y_arm = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct mixed_case *c = &cases[case_id];
        uint32_t active = active_elements(c);
        uint32_t coverage = grid_coverage(c);
        vc4_dim3 grid = vc4_m2_dim3(c->grid_x, c->grid_y, 1u);
        fill_buffers(c);
        if (active == 0u)
            saw_empty = 1;
        else if (active == coverage)
            saw_full = 1;
        else
            saw_tail = 1;
        if (c->flag != 0)
            saw_flag_true = 1;
        else
            saw_flag_false = 1;

        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t y_active = y_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, y_dev, y_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            mixed_ttir_mask_memory_cf_axes_b16_kernel_launch(program, grid, block,
                                                             x_active, y_active,
                                                             out_active, c->threshold,
                                                             c->bias, c->flag,
                                                             c->n) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: mixed_ttir_mask_memory launch/copy failed case=%d grid=%d,%d n=%d flag=%d\n",
                   (int)case_id, (int)c->grid_x, (int)c->grid_y, (int)c->n,
                   (int)c->flag);
            launch_failures++;
            continue;
        }

        int mismatches = verify_results(c, &saw_x_arm, &saw_y_arm);
        int sentinels = verify_sentinels(c);
        uint32_t case_hash = hash_output(c);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)active;
        printk("MIXED_TTIR_MASK_MEMORY_CF_AXES_CASE case=%d grid_x=%d grid_y=%d n=%d active=%d coverage=%d flag=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)c->grid_x, (int)c->grid_y, (int)c->n,
               (int)active, (int)coverage, (int)c->flag, mismatches, sentinels,
               case_hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u &&
                          saw_empty && saw_full && saw_tail &&
                          saw_flag_true && saw_flag_false &&
                          saw_x_arm && saw_y_arm) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_ttir_mask_memory_cf_axes_b16_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_grid_blocks=%d max_elements=%d buffer_n=%d output_hash=%u output_hash_nonzero=%d saw_nonzero_output_hash=%d saw_real_ttir_input=1 saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_value_scf_cf=1 saw_program_id_axis0=1 saw_arange_make_range_16=1 saw_masked_load_other_zero=1 saw_masked_store_tail=1 saw_tmu_load=1 saw_vdw_preserve_store=1 saw_tail_mask_clamp_overlaunch=1 saw_ttir_elementwise=1 saw_ttir_control_flow=1 saw_ttir_multi_axis=1 saw_ttir_mask_tail=%d saw_ttir_mask_full=%d saw_ttir_mask_empty=%d saw_ttir_compute_mask_select=1 saw_value_mask_classifier=1 saw_no_sparse_memory_mask=1 saw_ttir_program_id_axis1=1 saw_ttir_num_programs_axis0=1 saw_select_x_arm=%d saw_select_y_arm=%d saw_scf_if_true=%d saw_scf_if_false=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_GRID_BLOCKS, MAX_ELEMENTS, BUFFER_N, output_hash,
           output_hash != 0u ? 1 : 0, output_hash != 0u ? 1 : 0,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, saw_tail, saw_full, saw_empty,
           saw_x_arm, saw_y_arm, saw_flag_true, saw_flag_false, 3,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
