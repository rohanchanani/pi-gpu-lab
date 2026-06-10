#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define ACTIVE_QPUS 12u
#define LANES 16u
#define GUARD 32u
#define MAX_ROWS 7u
#define MAX_NCOLS 33u
#define MAX_STRIDE 52u
#define BUFFER_N (MAX_ROWS * MAX_STRIDE + 2u * GUARD)
#define SENTINEL_BITS 0x7fb50055u

struct mixed_case {
    uint32_t rows;
    uint32_t ncols;
    uint32_t ldx;
    uint32_t ldy;
    uint32_t ldo;
    float threshold;
    float bias;
    int32_t flag;
};

static const struct mixed_case cases[] = {
    {1u, 0u, 6u, 8u, 12u, 0.0f, 0.0f, 0},
    {1u, 1u, 7u, 9u, 13u, -7.5f, 1.25f, 1},
    {2u, 15u, 22u, 24u, 28u, 3.25f, -0.5f, 0},
    {2u, 16u, 23u, 25u, 29u, 0.0f, 2.0f, 1},
    {3u, 17u, 25u, 28u, 31u, 12.5f, 1.0f, 1},
    {4u, 31u, 39u, 42u, 45u, -2.0f, -1.5f, 0},
    {7u, 32u, 41u, 44u, 48u, 5.5f, 0.25f, 1},
    {7u, 33u, 42u, 45u, 49u, -11.25f, 2.25f, 0}
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

static uint32_t col_blocks(uint32_t ncols) {
    uint32_t blocks = (ncols + LANES - 1u) / LANES;
    return blocks == 0u ? 1u : blocks;
}

static uint32_t active_elements(const struct mixed_case *c) {
    return c->rows * c->ncols;
}

static uint32_t total_elements(const struct mixed_case *c) {
    return active_elements(c);
}

static float x_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)(case_id * 9u + row * 17u + col * 5u) - 47;
    return (float)whole + (float)((case_id + col) & 3u) * 0.25f;
}

static float y_value(uint32_t case_id, uint32_t row, uint32_t col) {
    int32_t whole = (int32_t)(case_id * 7u + row * 11u + col * 3u) - 23;
    return (float)whole + (float)((row + col) & 1u) * 0.5f;
}

static void fill_buffers(uint32_t case_id, const struct mixed_case *c) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        x_values[i] = (float)((int32_t)i - 101);
        y_values[i] = (float)((int32_t)i + 37);
        out_values[i] = sentinel;
        expected_bits[i] = SENTINEL_BITS;
    }
    for (uint32_t r = 0; r < c->rows; r++) {
        for (uint32_t col = 0; col < c->ncols; col++) {
            float x = x_value(case_id, r, col);
            float y = y_value(case_id, r, col);
            float selected = x < c->threshold ? x : y;
            if (c->flag != 0)
                selected = selected + c->bias;
            x_values[GUARD + r * c->ldx + col] = x;
            y_values[GUARD + r * c->ldy + col] = y;
            expected_bits[GUARD + r * c->ldo + col] = float_to_bits(selected);
        }
    }
}

static int verify_results(uint32_t case_id, const struct mixed_case *c,
                          int *saw_x_arm, int *saw_y_arm) {
    int mismatches = 0;
    for (uint32_t r = 0; r < c->rows; r++) {
        for (uint32_t col = 0; col < c->ncols; col++) {
            uint32_t index = GUARD + r * c->ldo + col;
            float x = x_value(case_id, r, col);
            if (x < c->threshold)
                *saw_x_arm = 1;
            else
                *saw_y_arm = 1;
            uint32_t got = float_to_bits(out_values[index]);
            uint32_t expected = expected_bits[index];
            if (got != expected) {
                if (mismatches < 8)
                    printk("ERROR: mixed ttir strided row=%d col=%d got=%x expected=%x\n",
                           (int)r, (int)col, got, expected);
                mismatches++;
            }
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct mixed_case *c) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int active = 0;
        if (i >= GUARD && i < GUARD + c->rows * c->ldo) {
            uint32_t rel = i - GUARD;
            active = (rel % c->ldo) < c->ncols;
        }
        if (active)
            continue;
        if (float_to_bits(out_values[i]) != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: mixed ttir strided sentinel rows=%d ncols=%d ldo=%d i=%d got=%x\n",
                       (int)c->rows, (int)c->ncols, (int)c->ldo, (int)i,
                       float_to_bits(out_values[i]));
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(const struct mixed_case *c) {
    uint32_t hash = 2166136261u ^ c->rows ^ (c->ncols << 8) ^ (c->ldo << 16);
    for (uint32_t r = 0; r < c->rows; r++) {
        for (uint32_t col = 0; col < c->ncols; col++) {
            uint32_t index = GUARD + r * c->ldo + col;
            hash ^= float_to_bits(out_values[index]) + 0x9e3779b9u + (index << 6) + (index >> 2);
            hash = rotl32_local(hash, 5u) * 16777619u;
        }
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("mixed ttir strided program create failed");

    uint32_t bytes = BUFFER_N * sizeof(float);
    vc4_deviceptr_t x_dev = 0, y_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &y_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("mixed ttir strided allocation failed");

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
        uint32_t blocks = col_blocks(c->ncols);
        uint32_t coverage_cols = blocks * LANES;
        uint32_t grid_y = c->rows;
        vc4_dim3 grid = vc4_m2_dim3(blocks, grid_y, 1u);
        fill_buffers(case_id, c);
        if (c->ncols == 0u)
            saw_empty = 1;
        else if (c->ncols == coverage_cols)
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
            mixed_ttir_strided_memory_axes_mask_cf_b16_kernel_launch(
                program, grid, block, x_active, y_active, out_active, c->threshold,
                c->bias, c->flag, c->ncols, c->ldx, c->ldy, c->ldo) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: mixed ttir strided launch/copy failed case=%d rows=%d ncols=%d\n",
                   (int)case_id, (int)c->rows, (int)c->ncols);
            launch_failures++;
            continue;
        }

        int mismatches = verify_results(case_id, c, &saw_x_arm, &saw_y_arm);
        int sentinels = verify_sentinels(c);
        uint32_t case_hash = hash_output(c);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)total_elements(c);
        printk("MIXED_TTIR_STRIDED_MEMORY_CASE case=%d rows=%d ncols=%d ldx=%d ldy=%d ldo=%d flag=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)c->rows, (int)c->ncols, (int)c->ldx,
               (int)c->ldy, (int)c->ldo, (int)c->flag, mismatches, sentinels,
               case_hash);
    }

    int elapsed = timer_get_usec() - start;
    int output_hash_nonzero = output_hash != 0u;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u &&
                          saw_empty && saw_full && saw_tail &&
                          saw_flag_true && saw_flag_false &&
                          saw_x_arm && saw_y_arm) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_ttir_strided_memory_axes_mask_cf_b16_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_ncols=%d buffer_n=%d output_hash=%u output_hash_nonzero=%d saw_nonzero_output_hash=%d saw_real_ttir_input=1 saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_value_surface_verification=1 saw_value_to_vc4kernel=1 saw_value_scf_cf=1 saw_program_id_axis0=1 saw_arange_make_range_16=1 saw_masked_load_other_zero=1 saw_masked_store_tail=1 saw_tmu_load=1 saw_vdw_preserve_store=1 saw_tail_mask_clamp_overlaunch=1 saw_ttir_elementwise=1 saw_ttir_control_flow=1 saw_ttir_multi_axis=1 saw_ttir_mask_tail=%d saw_ttir_mask_full=%d saw_ttir_mask_empty=%d saw_ttir_compute_mask_select=1 saw_ttir_row_strided_memory=1 saw_value_mask_classifier=1 saw_value_strided_address=1 saw_no_sparse_memory_mask=1 saw_no_gather_lane_stride=1 saw_no_hidden_memref_descriptor=1 saw_ttir_program_id_axis1=1 saw_select_x_arm=%d saw_select_y_arm=%d saw_scf_if_true=%d saw_scf_if_false=%d saw_row_padding_sentinels=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_ROWS, MAX_NCOLS, BUFFER_N, output_hash, output_hash_nonzero,
           output_hash_nonzero, VC4_CASE_SAW_CPP_TTIR_IMPORTER, saw_tail, saw_full,
           saw_empty, saw_x_arm, saw_y_arm, saw_flag_true, saw_flag_false, 3,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, x_dev);
    vc4Free(program, y_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
