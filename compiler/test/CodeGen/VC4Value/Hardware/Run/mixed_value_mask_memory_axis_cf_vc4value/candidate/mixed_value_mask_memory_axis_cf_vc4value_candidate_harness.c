#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define ELEMENTS_PER_WAVE (ACTIVE_QPUS * LANES)
#define MAX_GRID_BLOCKS 24u
#define MAX_ELEMENTS (MAX_GRID_BLOCKS * LANES)
#define BUFFER_N (MAX_ELEMENTS + 2u * GUARD)
#define SENTINEL_F_BITS 0xc57a0000u
#define SENTINEL_I 0x53a9b731
#define THRESHOLD_I 800
#define THRESHOLD_F 2.25f
#define EPSILON 0.001f

struct case_config {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t n;
    float a;
    uint32_t trip;
    int32_t use_i32_path;
};

static const struct case_config cases[] = {
    {2u, 2u, 0u, 1.0f, 0u, 0},
    {2u, 2u, 33u, -0.5f, 1u, 1},
    {12u, 1u, 192u, 0.25f, 2u, 0},
    {12u, 2u, 193u, 1.5f, 3u, 1},
    {12u, 2u, 257u, -1.0f, 1u, 0},
    {3u, 2u, 64u, 0.75f, 4u, 1},
    {1u, 1u, 1u, 2.0f, 0u, 0},
    {4u, 3u, 189u, -0.25f, 5u, 1},
};

static int32_t xi_values[BUFFER_N];
static float xf_values[BUFFER_N];
static float yf_values[BUFFER_N];
static float tail_out_f[BUFFER_N];
static int32_t tail_out_i[BUFFER_N];
static int32_t full_out_i[BUFFER_N];
static int32_t policy_out_i[BUFFER_N];
static int32_t empty_out_i[BUFFER_N];
static float expected_tail_f[BUFFER_N];
static int32_t expected_tail_i[BUFFER_N];

static float absf_local(float v) { return v < 0.0f ? -v : v; }

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

static uint32_t grid_blocks(const struct case_config *cfg) {
    return cfg->grid_x * cfg->grid_y;
}

static uint32_t grid_coverage(const struct case_config *cfg) {
    return grid_blocks(cfg) * LANES;
}

static int32_t xi_value(uint32_t i, const struct case_config *cfg) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i % LANES;
    uint32_t pid0 = block_id % cfg->grid_x;
    uint32_t pid1 = block_id / cfg->grid_x;
    int32_t value = (int32_t)(pid1 * 1700u + pid0 * 211u + lane * 37u);
    if (((i + cfg->grid_y) & 3u) == 0u)
        value = -value - 23;
    return value;
}

static float xf_value(uint32_t i, const struct case_config *cfg) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i % LANES;
    int32_t whole = (int32_t)((block_id * 17u + lane * 5u + cfg->grid_x * 3u) % 67u) - 33;
    return (float)whole * 0.25f;
}

static float yf_value(uint32_t i, const struct case_config *cfg) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i % LANES;
    int32_t whole = (int32_t)((block_id * 23u + lane * 7u + cfg->grid_y * 5u) % 71u) - 35;
    return (float)whole * 0.125f;
}

static float loop_factor(float a, uint32_t trip) {
    float factor = a;
    for (uint32_t i = 0; i < trip; i++)
        factor += 1.0f;
    return factor;
}

static void fill_buffers(const struct case_config *cfg) {
    float sentinel_f = bits_to_float(SENTINEL_F_BITS);
    float factor = loop_factor(cfg->a, cfg->trip);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active = i - GUARD;
        xi_values[i] = xi_value(active, cfg);
        xf_values[i] = xf_value(active, cfg);
        yf_values[i] = yf_value(active, cfg);
        tail_out_f[i] = sentinel_f;
        tail_out_i[i] = SENTINEL_I;
        full_out_i[i] = SENTINEL_I;
        policy_out_i[i] = SENTINEL_I;
        empty_out_i[i] = SENTINEL_I;
        expected_tail_f[i] = sentinel_f;
        expected_tail_i[i] = SENTINEL_I;
    }
    for (uint32_t i = 0; i < cfg->n; i++) {
        uint32_t index = GUARD + i;
        if (cfg->use_i32_path) {
            int cond = xi_values[index] > THRESHOLD_I;
            float candidate = factor * xf_values[index] + yf_values[index];
            expected_tail_f[index] = cond ? candidate : yf_values[index];
            expected_tail_i[index] = cond ? xi_values[index] + (int32_t)cfg->grid_x
                                          : xi_values[index];
        } else {
            int cond = xf_values[index] < THRESHOLD_F;
            float candidate = factor * yf_values[index] + xf_values[index];
            expected_tail_f[index] = cond ? candidate : xf_values[index];
            expected_tail_i[index] = xi_values[index] + (int32_t)cfg->grid_x;
        }
    }
}

static int verify_tail_results(const struct case_config *cfg, float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < cfg->n; i++) {
        uint32_t index = GUARD + i;
        float diff = tail_out_f[index] - expected_tail_f[index];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > EPSILON || tail_out_i[index] != expected_tail_i[index]) {
            if (mismatches < 8)
                printk("ERROR: mixed_mask_memory active n=%d i=%d gpu_f=%f cpu_f=%f gpu_i=%d cpu_i=%d diff=%f\n",
                       (int)cfg->n, (int)i, tail_out_f[index], expected_tail_f[index],
                       tail_out_i[index], expected_tail_i[index], diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_full_and_policy(const struct case_config *cfg) {
    int mismatches = 0;
    uint32_t coverage = grid_coverage(cfg);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active_begin = GUARD;
        uint32_t active_end = GUARD + coverage;
        int32_t expected_full = SENTINEL_I;
        int32_t expected_policy = SENTINEL_I;
        if (i >= active_begin && i < active_end) {
            uint32_t logical = i - GUARD;
            expected_full = xi_value(logical, cfg) + (int32_t)cfg->grid_x;
            expected_policy = logical < cfg->n ? xi_value(logical, cfg) : 0;
        }
        if (full_out_i[i] != expected_full || policy_out_i[i] != expected_policy) {
            if (mismatches < 8)
                printk("ERROR: mixed_mask_memory full/policy n=%d i=%d full=%x exp_full=%x policy=%x exp_policy=%x\n",
                       (int)cfg->n, (int)i, (uint32_t)full_out_i[i],
                       (uint32_t)expected_full, (uint32_t)policy_out_i[i],
                       (uint32_t)expected_policy);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(const struct case_config *cfg) {
    int mismatches = 0;
    float sentinel_f = bits_to_float(SENTINEL_F_BITS);
    uint32_t coverage = grid_coverage(cfg);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        int tail_active = i >= GUARD && i < GUARD + cfg->n;
        int full_active = i >= GUARD && i < GUARD + coverage;
        if (!tail_active &&
            (float_to_bits(tail_out_f[i]) != SENTINEL_F_BITS || tail_out_i[i] != SENTINEL_I)) {
            if (mismatches < 8)
                printk("ERROR: mixed_mask_memory tail sentinel n=%d i=%d got_f=%x got_i=%x\n",
                       (int)cfg->n, (int)i, float_to_bits(tail_out_f[i]),
                       (uint32_t)tail_out_i[i]);
            mismatches++;
        }
        if (!full_active && (full_out_i[i] != SENTINEL_I || policy_out_i[i] != SENTINEL_I)) {
            if (mismatches < 8)
                printk("ERROR: mixed_mask_memory full sentinel n=%d i=%d full=%x policy=%x\n",
                       (int)cfg->n, (int)i, (uint32_t)full_out_i[i],
                       (uint32_t)policy_out_i[i]);
            mismatches++;
        }
        if (empty_out_i[i] != SENTINEL_I) {
            if (mismatches < 8)
                printk("ERROR: mixed_mask_memory empty sentinel n=%d i=%d got=%x expected=%x\n",
                       (int)cfg->n, (int)i, (uint32_t)empty_out_i[i], SENTINEL_I);
            mismatches++;
        }
        (void)sentinel_f;
    }
    return mismatches;
}

static uint32_t hash_output(const struct case_config *cfg) {
    uint32_t hash = 2166136261u ^ cfg->n;
    uint32_t coverage = grid_coverage(cfg);
    for (uint32_t i = 0; i < cfg->n; i++) {
        uint32_t index = GUARD + i;
        hash ^= float_to_bits(tail_out_f[index]) + (uint32_t)tail_out_i[index] +
                0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    for (uint32_t i = 0; i < coverage; i++) {
        uint32_t index = GUARD + i;
        hash ^= (uint32_t)full_out_i[index] + (uint32_t)policy_out_i[index] +
                0x85ebca6bu + (i << 5);
        hash = rotl32_local(hash, 7u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes_f = BUFFER_N * sizeof(float);
    uint32_t bytes_i = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t xi_dev = 0, xf_dev = 0, yf_dev = 0, tail_f_dev = 0;
    vc4_deviceptr_t tail_i_dev = 0, full_i_dev = 0, policy_i_dev = 0, empty_i_dev = 0;
    if (vc4_m2_malloc(program, &xi_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &xf_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &yf_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &tail_f_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &tail_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &full_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &policy_i_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &empty_i_dev, bytes_i) < 0)
        panic("mixed_value_mask_memory_axis_cf allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    float max_abs_diff_overall = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(ELEMENTS_PER_WAVE, 1, 1);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        const struct case_config *cfg = &cases[case_id];
        vc4_dim3 grid = vc4_m2_dim3(cfg->grid_x, cfg->grid_y, 1);
        fill_buffers(cfg);
        vc4_deviceptr_t xi_active = xi_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t xf_active = xf_dev + GUARD * sizeof(float);
        vc4_deviceptr_t yf_active = yf_dev + GUARD * sizeof(float);
        vc4_deviceptr_t tail_f_active = tail_f_dev + GUARD * sizeof(float);
        vc4_deviceptr_t tail_i_active = tail_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t full_i_active = full_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t policy_i_active = policy_i_dev + GUARD * sizeof(int32_t);
        vc4_deviceptr_t empty_i_active = empty_i_dev + GUARD * sizeof(int32_t);
        if (vc4_m2_copy_htod(program, xi_dev, xi_values, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, xf_dev, xf_values, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, yf_dev, yf_values, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, tail_f_dev, tail_out_f, bytes_f) < 0 ||
            vc4_m2_copy_htod(program, tail_i_dev, tail_out_i, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, full_i_dev, full_out_i, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, policy_i_dev, policy_out_i, bytes_i) < 0 ||
            vc4_m2_copy_htod(program, empty_i_dev, empty_out_i, bytes_i) < 0 ||
            mixed_value_mask_memory_axis_cf_vc4value_launch(
                program, grid, block, xi_active, xf_active, yf_active,
                tail_f_active, tail_i_active, full_i_active, policy_i_active,
                empty_i_active, THRESHOLD_I, THRESHOLD_F, cfg->a, cfg->trip,
                cfg->use_i32_path, cfg->n) < 0 ||
            vc4_m2_copy_dtoh(program, tail_out_f, tail_f_dev, bytes_f) < 0 ||
            vc4_m2_copy_dtoh(program, tail_out_i, tail_i_dev, bytes_i) < 0 ||
            vc4_m2_copy_dtoh(program, full_out_i, full_i_dev, bytes_i) < 0 ||
            vc4_m2_copy_dtoh(program, policy_out_i, policy_i_dev, bytes_i) < 0 ||
            vc4_m2_copy_dtoh(program, empty_out_i, empty_i_dev, bytes_i) < 0) {
            printk("ERROR: mixed_value_mask_memory_axis_cf launch/copy failed case=%d n=%d\n",
                   (int)case_id, (int)cfg->n);
            launch_failures++;
            continue;
        }
        float max_abs_diff = 0.0f;
        int result_mismatches = verify_tail_results(cfg, &max_abs_diff);
        int full_mismatches = verify_full_and_policy(cfg);
        int sentinels = verify_sentinels(cfg);
        uint32_t case_hash = hash_output(cfg);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += result_mismatches + full_mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)cfg->n;
        printk("MIXED_VALUE_MASK_MEMORY_AXIS_CF_CASE case=%d grid=%dx%d n=%d trip=%d flag=%d result_mismatches=%d full_policy_mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
               (int)case_id, (int)cfg->grid_x, (int)cfg->grid_y, (int)cfg->n,
               (int)cfg->trip, (int)cfg->use_i32_path, result_mismatches,
               full_mismatches, sentinels, case_hash, max_abs_diff);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_value_mask_memory_axis_cf_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_grid_blocks=%d max_elements=%d buffer_n=%d output_hash=%u output_hash_nonzero=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d saw_value_multi_axis=1 saw_value_control_flow=1 saw_value_mask_full=1 saw_value_mask_empty=1 saw_value_mask_tail=1 saw_value_compute_mask_select=1 saw_value_load_inactive_zero=1 saw_value_store_inactive_preserve=1 saw_no_sparse_memory_mask=1 elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_GRID_BLOCKS, MAX_ELEMENTS, BUFFER_N, output_hash,
           output_hash != 0u ? 1 : 0, max_abs_diff_overall, 8,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);

    vc4Free(program, xi_dev);
    vc4Free(program, xf_dev);
    vc4Free(program, yf_dev);
    vc4Free(program, tail_f_dev);
    vc4Free(program, tail_i_dev);
    vc4Free(program, full_i_dev);
    vc4Free(program, policy_i_dev);
    vc4Free(program, empty_i_dev);
    vc4_program_destroy(program);
}
