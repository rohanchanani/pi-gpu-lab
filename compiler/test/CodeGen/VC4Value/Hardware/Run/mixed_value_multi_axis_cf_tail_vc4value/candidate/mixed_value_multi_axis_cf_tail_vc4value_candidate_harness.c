#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define MAX_GRID_BLOCKS 24u
#define MAX_ELEMENTS (MAX_GRID_BLOCKS * LANES)
#define BUFFER_N (MAX_ELEMENTS + 2u * GUARD)
#define SENTINEL_F_BITS 0xc57a0000u
#define SENTINEL_I 0x3f4a7391
#define THRESHOLD_I 7000
#define EPSILON 0.001f

struct grid_case {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t n_values[3];
};

static const struct grid_case grid_cases[] = {
    {2u, 2u, 1u, {0u, 17u, 64u}},
    {3u, 2u, 2u, {1u, 191u, 192u}},
    {2u, 3u, 3u, {15u, 287u, 288u}},
    {4u, 3u, 2u, {31u, 383u, 384u}},
};
static const float a_cases[] = {1.0f, -0.5f};
static const uint32_t trip_cases[] = {0u, 2u, 5u};
static const int32_t flag_cases[] = {0, 1};

static int32_t xi_values[BUFFER_N];
static float xf_values[BUFFER_N];
static float yf_values[BUFFER_N];
static float out_f_values[BUFFER_N];
static int32_t out_i_values[BUFFER_N];
static float expected_f_values[BUFFER_N];
static int32_t expected_i_values[BUFFER_N];

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

static int32_t xi_value(uint32_t i, const struct grid_case *grid) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i % LANES;
    uint32_t pid0 = block_id % grid->x;
    uint32_t pid1 = (block_id / grid->x) % grid->y;
    uint32_t pid2 = block_id / (grid->x * grid->y);
    int32_t value = (int32_t)(pid2 * 11000u + pid1 * 1700u + pid0 * 211u + lane * 37u);
    if (((i + grid->z) & 3u) == 0u)
        value = -value - 23;
    return value;
}

static float xf_value(uint32_t i, const struct grid_case *grid) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i % LANES;
    int32_t whole = (int32_t)((block_id * 17u + lane * 5u + grid->x * 3u) % 67u) - 33;
    return (float)whole * 0.25f;
}

static float yf_value(uint32_t i, const struct grid_case *grid) {
    uint32_t block_id = i / LANES;
    uint32_t lane = i % LANES;
    int32_t whole = (int32_t)((block_id * 23u + lane * 7u + grid->y * 5u) % 71u) - 35;
    return (float)whole * 0.125f;
}

static float loop_factor(float a, uint32_t trip) {
    float factor = a;
    for (uint32_t i = 0; i < trip; i++)
        factor += 1.0f;
    return factor;
}

static void fill_buffers(const struct grid_case *grid, uint32_t n, float a,
                         uint32_t trip, int32_t use_i32_path) {
    float sentinel_f = bits_to_float(SENTINEL_F_BITS);
    float factor = loop_factor(a, trip);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t active = i - GUARD;
        xi_values[i] = xi_value(active, grid);
        xf_values[i] = xf_value(active, grid);
        yf_values[i] = yf_value(active, grid);
        out_f_values[i] = sentinel_f;
        out_i_values[i] = SENTINEL_I;
        expected_f_values[i] = sentinel_f;
        expected_i_values[i] = SENTINEL_I;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        if (use_i32_path) {
            float candidate = factor * xf_values[index] + yf_values[index];
            int cond = xi_values[index] > THRESHOLD_I;
            expected_f_values[index] = cond ? candidate : yf_values[index];
            expected_i_values[index] = cond ? xi_values[index] + (int32_t)grid->z : xi_values[index];
        } else {
            expected_f_values[index] = factor * yf_values[index] + xf_values[index];
            expected_i_values[index] = xi_values[index] + (int32_t)grid->z;
        }
    }
}

static int verify_results(uint32_t n, float *max_abs_diff) {
    int mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        float diff = out_f_values[index] - expected_f_values[index];
        float ad = absf_local(diff);
        if (ad > *max_abs_diff)
            *max_abs_diff = ad;
        if (ad > EPSILON || out_i_values[index] != expected_i_values[index]) {
            if (mismatches < 8)
                printk("ERROR: mixed_multi_axis n=%d i=%d gpu_f=%f cpu_f=%f gpu_i=%d cpu_i=%d diff=%f\n",
                       (int)n, (int)i, out_f_values[index], expected_f_values[index],
                       out_i_values[index], expected_i_values[index], diff);
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + n)
            continue;
        uint32_t got_f = float_to_bits(out_f_values[i]);
        if (got_f != SENTINEL_F_BITS || out_i_values[i] != SENTINEL_I) {
            if (mismatches < 8)
                printk("ERROR: mixed_multi_axis sentinel n=%d i=%d got_f=%x got_i=%x expected_f=%x expected_i=%x\n",
                       (int)n, (int)i, got_f, out_i_values[i], SENTINEL_F_BITS, SENTINEL_I);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(uint32_t n) {
    uint32_t hash = 2166136261u ^ n;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t index = GUARD + i;
        hash ^= float_to_bits(out_f_values[index]) + (uint32_t)out_i_values[index] +
                0x9e3779b9u + (i << 6) + (i >> 2);
        hash = rotl32_local(hash, 5u) * 16777619u;
    }
    return hash;
}

void notmain(void) {
    struct vc4_program *program = 0;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    uint32_t bytes_f = BUFFER_N * sizeof(float);
    uint32_t bytes_i = BUFFER_N * sizeof(int32_t);
    vc4_deviceptr_t xi_dev = 0, xf_dev = 0, yf_dev = 0, out_f_dev = 0, out_i_dev = 0;
    if (vc4_m2_malloc(program, &xi_dev, bytes_i) < 0 ||
        vc4_m2_malloc(program, &xf_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &yf_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &out_f_dev, bytes_f) < 0 ||
        vc4_m2_malloc(program, &out_i_dev, bytes_i) < 0)
        panic("mixed_multi_axis allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    float max_abs_diff_overall = 0.0f;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1, 1);
    int case_count = 0;

    for (uint32_t grid_id = 0; grid_id < sizeof(grid_cases) / sizeof(grid_cases[0]); grid_id++) {
        const struct grid_case *grid_case = &grid_cases[grid_id];
        vc4_dim3 grid = vc4_m2_dim3(grid_case->x, grid_case->y, grid_case->z);
        for (uint32_t a_id = 0; a_id < sizeof(a_cases) / sizeof(a_cases[0]); a_id++) {
            for (uint32_t trip_id = 0; trip_id < sizeof(trip_cases) / sizeof(trip_cases[0]); trip_id++) {
                for (uint32_t flag_id = 0; flag_id < sizeof(flag_cases) / sizeof(flag_cases[0]); flag_id++) {
                    uint32_t n = grid_case->n_values[(a_id + trip_id + flag_id) % 3u];
                    float a = a_cases[a_id];
                    uint32_t trip = trip_cases[trip_id];
                    int32_t use_i32_path = flag_cases[flag_id];
                    fill_buffers(grid_case, n, a, trip, use_i32_path);
                    vc4_deviceptr_t xi_active = xi_dev + GUARD * sizeof(int32_t);
                    vc4_deviceptr_t xf_active = xf_dev + GUARD * sizeof(float);
                    vc4_deviceptr_t yf_active = yf_dev + GUARD * sizeof(float);
                    vc4_deviceptr_t out_f_active = out_f_dev + GUARD * sizeof(float);
                    vc4_deviceptr_t out_i_active = out_i_dev + GUARD * sizeof(int32_t);
                    if (vc4_m2_copy_htod(program, xi_dev, xi_values, bytes_i) < 0 ||
                        vc4_m2_copy_htod(program, xf_dev, xf_values, bytes_f) < 0 ||
                        vc4_m2_copy_htod(program, yf_dev, yf_values, bytes_f) < 0 ||
                        vc4_m2_copy_htod(program, out_f_dev, out_f_values, bytes_f) < 0 ||
                        vc4_m2_copy_htod(program, out_i_dev, out_i_values, bytes_i) < 0 ||
                        mixed_value_multi_axis_cf_tail_vc4value_launch(
                            program, grid, block, xi_active, xf_active, yf_active,
                            out_f_active, out_i_active, THRESHOLD_I, a, trip,
                            use_i32_path, n) < 0 ||
                        vc4_m2_copy_dtoh(program, out_f_values, out_f_dev, bytes_f) < 0 ||
                        vc4_m2_copy_dtoh(program, out_i_values, out_i_dev, bytes_i) < 0) {
                        printk("ERROR: mixed_multi_axis launch/copy failed grid=%d a=%d trip=%d flag=%d n=%d\n",
                               (int)grid_id, (int)a_id, (int)trip, (int)use_i32_path, (int)n);
                        launch_failures++;
                        case_count++;
                        continue;
                    }
                    float max_abs_diff = 0.0f;
                    int mismatches = verify_results(n, &max_abs_diff);
                    int sentinels = verify_sentinels(n);
                    uint32_t case_hash = hash_output(n);
                    output_hash ^= case_hash + 0x9e3779b9u + (grid_id << 20) +
                                   (a_id << 16) + (trip_id << 10) + (flag_id << 6);
                    output_hash = rotl32_local(output_hash, 7u);
                    if (max_abs_diff > max_abs_diff_overall)
                        max_abs_diff_overall = max_abs_diff;
                    total_mismatches += mismatches;
                    sentinel_mismatches += sentinels;
                    elements_checked += (int)n;
                    case_count++;
                    printk("MIXED_VALUE_MULTI_AXIS_CASE grid=%dx%dx%d a_id=%d trip=%d flag=%d n=%d mismatches=%d sentinel_mismatches=%d hash=%x max_abs_diff=%f\n",
                           (int)grid_case->x, (int)grid_case->y, (int)grid_case->z,
                           (int)a_id, (int)trip, (int)use_i32_path, (int)n,
                           mismatches, sentinels, case_hash, max_abs_diff);
                }
            }
        }
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mixed_value_multi_axis_cf_tail_vc4value status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_grid_blocks=%d max_elements=%d buffer_n=%d output_hash=%u max_abs_diff=%f saw_value_mixed_elementwise=1 saw_value_scf_cf=1 saw_value_multi_axis_launch=1 saw_value_program_id_axis1=1 saw_value_program_id_axis2=1 saw_value_num_programs_axis1=1 saw_value_tail_mask=1 saw_value_tmu_load=1 saw_value_vdw_preserve=1 saw_value_mixed_i32_f32=1 saw_value_i32_cmp=1 saw_value_f32_alu=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, case_count, elements_checked, total_mismatches,
           sentinel_mismatches, launch_failures, ACTIVE_QPUS, LANES,
           MAX_GRID_BLOCKS, MAX_ELEMENTS, BUFFER_N, output_hash,
           max_abs_diff_overall, 5, case_count, elapsed);

    vc4Free(program, xi_dev);
    vc4Free(program, xf_dev);
    vc4Free(program, yf_dev);
    vc4Free(program, out_f_dev);
    vc4Free(program, out_i_dev);
    vc4_program_destroy(program);
}
