#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MATMUL_BLOCKED_ACTIVE_QPUS 12u
#define MATMUL_BLOCKED_LANES 16u
#define MATMUL_BLOCKED_TILE_M 12u
#define MATMUL_BLOCKED_TILE_N 16u
#define MATMUL_BLOCKED_CASES 10u
#define MATMUL_BLOCKED_GUARD_WORDS 16u
#define MATMUL_BLOCKED_PROGRAM_HEAP_SLOP_BYTES (64u * 1024u)
#define MATMUL_BLOCKED_GUARD_BASE 0x6d6d4000u
#define CHECKSUM_SCALE 4096.0f

struct matmul_case {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

static const struct matmul_case cases[MATMUL_BLOCKED_CASES] = {
    {0u, 7u, 5u},
    {4u, 0u, 5u},
    {5u, 6u, 0u},
    {1u, 1u, 1u},
    {4u, 7u, 5u},
    {12u, 16u, 12u},
    {13u, 17u, 7u},
    {25u, 31u, 25u},
    {128u, 160u, 96u},
    {193u, 217u, 97u},
};

static float *a_values;
static float *b_values;
static float *c_values;
static float *expected_values;
static float *a_scratch;
static float *b_scratch;
static float *c_scratch;
static uint32_t compact_a_capacity;
static uint32_t compact_b_capacity;
static uint32_t compact_c_capacity;
static uint32_t padded_m_capacity;
static uint32_t padded_n_capacity;
static uint32_t padded_k_capacity;
static uint32_t padded_a_capacity;
static uint32_t padded_b_capacity;
static uint32_t padded_c_capacity;

static float absf_local(float value) { return value < 0.0f ? -value : value; }

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_from_bits(uint32_t bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t guard_word(uint32_t case_index, uint32_t word) {
    return MATMUL_BLOCKED_GUARD_BASE ^ (case_index << 8) ^ word;
}

static uint32_t round_up_u32(uint32_t value, uint32_t align) {
    if (value == 0u)
        return 0u;
    return ((value + align - 1u) / align) * align;
}

static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    if (value == 0u)
        return 0u;
    return 1u + (value - 1u) / divisor;
}

static int checked_product(uint32_t lhs, uint32_t rhs, uint32_t *out) {
    uint64_t product = (uint64_t)lhs * (uint64_t)rhs;
    if (product > (uint64_t)(0xffffffffu / sizeof(float)))
        return -1;
    *out = (uint32_t)product;
    return 0;
}

static int bytes_for_floats(uint32_t count, uint32_t extra, uint32_t *out) {
    if (count > 0xffffffffu - extra)
        return -1;
    uint32_t total = count + extra;
    if (total > 0xffffffffu / sizeof(float))
        return -1;
    *out = total * sizeof(float);
    return 0;
}

static int case_counts(const struct matmul_case *tc,
                       uint32_t *a_count,
                       uint32_t *b_count,
                       uint32_t *c_count) {
    if (checked_product(tc->m, tc->k, a_count) < 0)
        return -1;
    if (checked_product(tc->k, tc->n, b_count) < 0)
        return -1;
    if (checked_product(tc->m, tc->n, c_count) < 0)
        return -1;
    return 0;
}

static void compute_sweep_caps(uint32_t *sweep_max_m,
                               uint32_t *sweep_max_n,
                               uint32_t *sweep_max_k,
                               uint32_t *expected_tile_waves) {
    compact_a_capacity = 1u;
    compact_b_capacity = 1u;
    compact_c_capacity = 1u;
    *sweep_max_m = 0u;
    *sweep_max_n = 0u;
    *sweep_max_k = 0u;
    *expected_tile_waves = 0u;

    for (uint32_t i = 0; i < MATMUL_BLOCKED_CASES; i++) {
        uint32_t a_count = 0;
        uint32_t b_count = 0;
        uint32_t c_count = 0;
        const struct matmul_case *tc = &cases[i];
        if (case_counts(tc, &a_count, &b_count, &c_count) < 0)
            panic("matmul_blocked case shape overflows addressable bytes");
        if (a_count > compact_a_capacity)
            compact_a_capacity = a_count;
        if (b_count > compact_b_capacity)
            compact_b_capacity = b_count;
        if (c_count > compact_c_capacity)
            compact_c_capacity = c_count;
        if (tc->m > *sweep_max_m)
            *sweep_max_m = tc->m;
        if (tc->n > *sweep_max_n)
            *sweep_max_n = tc->n;
        if (tc->k > *sweep_max_k)
            *sweep_max_k = tc->k;
        *expected_tile_waves += ceil_div_u32(tc->m, MATMUL_BLOCKED_TILE_M) *
                                ceil_div_u32(tc->n, MATMUL_BLOCKED_TILE_N);
    }

    padded_m_capacity = round_up_u32(*sweep_max_m, MATMUL_BLOCKED_TILE_M);
    padded_n_capacity = round_up_u32(*sweep_max_n, MATMUL_BLOCKED_TILE_N);
    padded_k_capacity = round_up_u32(*sweep_max_k, MATMUL_BLOCKED_TILE_M);
    if (padded_m_capacity == 0u)
        padded_m_capacity = MATMUL_BLOCKED_TILE_M;
    if (padded_n_capacity == 0u)
        padded_n_capacity = MATMUL_BLOCKED_TILE_N;
    if (padded_k_capacity == 0u)
        padded_k_capacity = MATMUL_BLOCKED_TILE_M;
    if (checked_product(padded_m_capacity, padded_k_capacity, &padded_a_capacity) < 0 ||
        checked_product(padded_k_capacity, padded_n_capacity, &padded_b_capacity) < 0 ||
        checked_product(padded_m_capacity, padded_n_capacity, &padded_c_capacity) < 0)
        panic("matmul_blocked padded shape overflows addressable bytes");
}

static void allocate_host_buffers(void) {
    uint32_t a_bytes = 0;
    uint32_t b_bytes = 0;
    uint32_t c_bytes = 0;
    uint32_t expected_bytes = 0;
    uint32_t a_scratch_bytes = 0;
    uint32_t b_scratch_bytes = 0;
    uint32_t c_scratch_bytes = 0;
    if (bytes_for_floats(compact_a_capacity, 0u, &a_bytes) < 0 ||
        bytes_for_floats(compact_b_capacity, 0u, &b_bytes) < 0 ||
        bytes_for_floats(compact_c_capacity, MATMUL_BLOCKED_GUARD_WORDS, &c_bytes) < 0 ||
        bytes_for_floats(compact_c_capacity, 0u, &expected_bytes) < 0 ||
        bytes_for_floats(padded_a_capacity, 0u, &a_scratch_bytes) < 0 ||
        bytes_for_floats(padded_b_capacity, 0u, &b_scratch_bytes) < 0 ||
        bytes_for_floats(padded_c_capacity, MATMUL_BLOCKED_GUARD_WORDS, &c_scratch_bytes) < 0)
        panic("matmul_blocked host buffer byte size overflow");

    uint32_t total = a_bytes + b_bytes + c_bytes + expected_bytes +
                     a_scratch_bytes + b_scratch_bytes + c_scratch_bytes + 4096u;
    kmalloc_init_set_start((void *)(1024u * 1024u), total);
    a_values = (float *)kmalloc(a_bytes);
    b_values = (float *)kmalloc(b_bytes);
    c_values = (float *)kmalloc(c_bytes);
    expected_values = (float *)kmalloc(expected_bytes);
    a_scratch = (float *)kmalloc(a_scratch_bytes);
    b_scratch = (float *)kmalloc(b_scratch_bytes);
    c_scratch = (float *)kmalloc(c_scratch_bytes);
    if (!a_values || !b_values || !c_values || !expected_values ||
        !a_scratch || !b_scratch || !c_scratch)
        panic("matmul_blocked host buffer allocation failed");
}

static float make_a_value(uint32_t case_index, uint32_t row, uint32_t kk) {
    int centered = (int)((row * 7u + kk * 5u + case_index * 3u) % 23u) - 11;
    return ((float)centered) * 0.0625f;
}

static float make_b_value(uint32_t case_index, uint32_t kk, uint32_t col) {
    int centered = (int)((kk * 11u + col * 3u + case_index * 5u) % 29u) - 14;
    return ((float)centered) * 0.03125f;
}

static void fill_inputs(uint32_t case_index, const struct matmul_case *tc) {
    for (uint32_t i = 0; i < compact_a_capacity; i++)
        a_values[i] = 0.0f;
    for (uint32_t i = 0; i < compact_b_capacity; i++)
        b_values[i] = 0.0f;
    for (uint32_t i = 0; i < compact_c_capacity + MATMUL_BLOCKED_GUARD_WORDS; i++)
        c_values[i] = float_from_bits(guard_word(case_index, i));
    for (uint32_t i = 0; i < compact_c_capacity; i++)
        expected_values[i] = float_from_bits(guard_word(case_index, i));

    for (uint32_t row = 0; row < tc->m; row++)
        for (uint32_t kk = 0; kk < tc->k; kk++)
            a_values[row * tc->k + kk] = make_a_value(case_index, row, kk);

    for (uint32_t kk = 0; kk < tc->k; kk++)
        for (uint32_t col = 0; col < tc->n; col++)
            b_values[kk * tc->n + col] = make_b_value(case_index, kk, col);
}

static void pack_scratch(uint32_t case_index, const struct matmul_case *tc) {
    for (uint32_t i = 0; i < padded_a_capacity; i++)
        a_scratch[i] = 0.0f;
    for (uint32_t i = 0; i < padded_b_capacity; i++)
        b_scratch[i] = 0.0f;
    for (uint32_t i = 0; i < padded_c_capacity + MATMUL_BLOCKED_GUARD_WORDS; i++)
        c_scratch[i] = float_from_bits(guard_word(case_index, i));

    for (uint32_t row = 0; row < tc->m; row++)
        for (uint32_t kk = 0; kk < tc->k; kk++)
            a_scratch[row * padded_k_capacity + kk] = a_values[row * tc->k + kk];

    for (uint32_t kk = 0; kk < tc->k; kk++)
        for (uint32_t col = 0; col < tc->n; col++)
            b_scratch[kk * padded_n_capacity + col] = b_values[kk * tc->n + col];
}

static void unpack_c(const struct matmul_case *tc) {
    for (uint32_t row = 0; row < tc->m; row++)
        for (uint32_t col = 0; col < tc->n; col++)
            c_values[row * tc->n + col] = c_scratch[row * padded_n_capacity + col];
}

static void run_cpu_reference(const struct matmul_case *tc) {
    for (uint32_t row = 0; row < tc->m; row++) {
        for (uint32_t col = 0; col < tc->n; col++) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < tc->k; kk++)
                acc += a_values[row * tc->k + kk] * b_values[kk * tc->n + col];
            expected_values[row * tc->n + col] = acc;
        }
    }
}

static int scaled_checksum(const float *values, uint32_t count) {
    int checksum = 0;
    for (uint32_t i = 0; i < count; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_results(uint32_t case_index,
                           const struct matmul_case *tc,
                           int *mismatch_count,
                           int *sentinel_mismatches,
                           float *max_abs_diff) {
    uint32_t live_count = tc->m * tc->n;
    *mismatch_count = 0;
    *sentinel_mismatches = 0;
    *max_abs_diff = 0.0f;

    for (uint32_t i = 0; i < live_count; i++) {
        float diff = c_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;
        if (float_bits(c_values[i]) != float_bits(expected_values[i])) {
            if (*mismatch_count < 8) {
                uint32_t row = tc->n ? (i / tc->n) : 0u;
                uint32_t col = tc->n ? (i % tc->n) : 0u;
                printk("ERROR: case=%d row=%d col=%d gpu=%f cpu=%f gpu_bits=%x cpu_bits=%x diff=%f\n",
                       (int)case_index, (int)row, (int)col, c_values[i],
                       expected_values[i], float_bits(c_values[i]),
                       float_bits(expected_values[i]), diff);
            }
            (*mismatch_count)++;
        }
    }

    for (uint32_t i = live_count; i < live_count + MATMUL_BLOCKED_GUARD_WORDS; i++) {
        uint32_t expected = guard_word(case_index, i);
        if (float_bits(c_values[i]) != expected) {
            if (*sentinel_mismatches < 8)
                printk("ERROR: compact guard changed case=%d index=%d value_bits=%x expected_bits=%x\n",
                       (int)case_index, (int)i, float_bits(c_values[i]), expected);
            (*sentinel_mismatches)++;
        }
    }

    for (uint32_t row = 0; row < tc->m; row++) {
        for (uint32_t col = tc->n; col < padded_n_capacity; col++) {
            uint32_t index = row * padded_n_capacity + col;
            uint32_t expected = guard_word(case_index, index);
            if (float_bits(c_scratch[index]) != expected) {
                if (*sentinel_mismatches < 8)
                    printk("ERROR: padded C row tail changed case=%d row=%d col=%d value_bits=%x expected_bits=%x\n",
                           (int)case_index, (int)row, (int)col,
                           float_bits(c_scratch[index]), expected);
                (*sentinel_mismatches)++;
            }
        }
    }

    for (uint32_t i = padded_c_capacity;
         i < padded_c_capacity + MATMUL_BLOCKED_GUARD_WORDS; i++) {
        uint32_t expected = guard_word(case_index, i);
        if (float_bits(c_scratch[i]) != expected) {
            if (*sentinel_mismatches < 8)
                printk("ERROR: device guard changed case=%d index=%d value_bits=%x expected_bits=%x\n",
                       (int)case_index, (int)i, float_bits(c_scratch[i]), expected);
            (*sentinel_mismatches)++;
        }
    }
}

void notmain(void) {
    uint32_t sweep_max_m = 0;
    uint32_t sweep_max_n = 0;
    uint32_t sweep_max_k = 0;
    uint32_t expected_tile_waves = 0;
    compute_sweep_caps(&sweep_max_m, &sweep_max_n, &sweep_max_k, &expected_tile_waves);
    allocate_host_buffers();

    uint32_t a_bytes = 0;
    uint32_t b_bytes = 0;
    uint32_t c_bytes = 0;
    if (bytes_for_floats(padded_a_capacity, 0u, &a_bytes) < 0 ||
        bytes_for_floats(padded_b_capacity, 0u, &b_bytes) < 0 ||
        bytes_for_floats(padded_c_capacity, MATMUL_BLOCKED_GUARD_WORDS, &c_bytes) < 0)
        panic("matmul_blocked device buffer byte size overflow");

    struct vc4_program *program = 0;
    if (vc4_program_create(&program, a_bytes + b_bytes + c_bytes +
                                        MATMUL_BLOCKED_PROGRAM_HEAP_SLOP_BYTES) < 0)
        panic("matmul_blocked vc4 program create failed");

    vc4_deviceptr_t a_dev = 0;
    vc4_deviceptr_t b_dev = 0;
    vc4_deviceptr_t c_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, a_bytes) < 0 ||
        vc4_m2_malloc(program, &b_dev, b_bytes) < 0 ||
        vc4_m2_malloc(program, &c_dev, c_bytes) < 0)
        panic("matmul_blocked device allocation failed");

    printk("Running VC4 matmul_blocked M2 candidate bundle...\n");

    vc4_dim3 grid = {1u, 1u, 1u};
    vc4_dim3 block = {MATMUL_BLOCKED_TILE_M * MATMUL_BLOCKED_LANES, 1u, 1u};
    int start = timer_get_usec();
    int total_mismatches = 0;
    int total_sentinel_mismatches = 0;
    int launch_failures = 0;
    int checksum_accum = 0;
    int expected_checksum_accum = 0;
    uint32_t checked_elements = 0;
    uint32_t launched_tile_waves = 0;
    float max_abs_diff_overall = 0.0f;

    for (uint32_t case_index = 0; case_index < MATMUL_BLOCKED_CASES; case_index++) {
        const struct matmul_case *tc = &cases[case_index];
        uint32_t live_count = tc->m * tc->n;
        fill_inputs(case_index, tc);
        pack_scratch(case_index, tc);
        run_cpu_reference(tc);

        if (vc4_m2_copy_htod(program, a_dev, a_scratch, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, b_dev, b_scratch, b_bytes) < 0 ||
            vc4_m2_copy_htod(program, c_dev, c_scratch, c_bytes) < 0) {
            launch_failures++;
            continue;
        }

        uint32_t k_padded_limit = round_up_u32(tc->k, MATMUL_BLOCKED_TILE_M);
        for (uint32_t tile_row = 0; tile_row < tc->m; tile_row += MATMUL_BLOCKED_TILE_M) {
            for (uint32_t tile_col = 0; tile_col < tc->n; tile_col += MATMUL_BLOCKED_TILE_N) {
                if (matmul_blocked_launch(program, grid, block, a_dev, b_dev, c_dev,
                                          tc->m, tc->n, tc->k,
                                          padded_k_capacity, padded_n_capacity,
                                          tile_row, tile_col, k_padded_limit) < 0) {
                    launch_failures++;
                    printk("ERROR: launch failed case=%d tile_row=%d tile_col=%d\n",
                           (int)case_index, (int)tile_row, (int)tile_col);
                    break;
                }
                launched_tile_waves++;
            }
        }

        if (vc4_m2_copy_dtoh(program, c_scratch, c_dev, c_bytes) < 0) {
            launch_failures++;
            continue;
        }
        unpack_c(tc);

        int mismatches = 0;
        int sentinel_mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(case_index, tc, &mismatches, &sentinel_mismatches, &max_abs_diff);
        int checksum = scaled_checksum(c_values, live_count);
        int expected_checksum = scaled_checksum(expected_values, live_count);
        if (checksum != expected_checksum) {
            printk("ERROR: case=%d checksum mismatch gpu=%d cpu=%d\n",
                   (int)case_index, checksum, expected_checksum);
            mismatches++;
        }
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        total_sentinel_mismatches += sentinel_mismatches;
        checksum_accum += checksum;
        expected_checksum_accum += expected_checksum;
        checked_elements += live_count;
        printk("MATMUL_BLOCKED_CASE case=%d m=%d n=%d k=%d elements=%d mismatches=%d sentinel_mismatches=%d checksum=%d expected_checksum=%d max_abs_diff=%f launched_tile_waves=%d\n",
               (int)case_index, (int)tc->m, (int)tc->n, (int)tc->k,
               (int)live_count, mismatches, sentinel_mismatches, checksum,
               expected_checksum, max_abs_diff, (int)launched_tile_waves);
    }
    int elapsed = timer_get_usec() - start;

    uint32_t runtime_allocations = matmul_blocked_runtime_allocations();
    uint32_t runtime_launches = matmul_blocked_runtime_launches();
    uint32_t runtime_capacity = matmul_blocked_runtime_capacity();
    uint32_t code_uploads = matmul_blocked_runtime_code_uploads();
    uint32_t recorded_launch_failures = matmul_blocked_runtime_launch_failures();

    const char *status = (total_mismatches == 0 &&
                          total_sentinel_mismatches == 0 &&
                          checksum_accum == expected_checksum_accum &&
                          launch_failures == 0 &&
                          recorded_launch_failures == 0 &&
                          runtime_allocations == 1u &&
                          runtime_launches == expected_tile_waves &&
                          launched_tile_waves == expected_tile_waves &&
                          code_uploads == 1u) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=matmul_blocked status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d sweep_max_m=%d sweep_max_n=%d sweep_max_k=%d padded_m=%d padded_n=%d padded_k=%d backing_a_floats=%d backing_b_floats=%d backing_c_floats=%d checksum_accum=%d expected_checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_tile_waves=%d runtime_capacity=%d code_uploads=%d elapsed_usec=%d\n",
           status,
           (int)MATMUL_BLOCKED_CASES,
           (int)checked_elements,
           total_mismatches,
           total_sentinel_mismatches,
           launch_failures,
           (int)recorded_launch_failures,
           (int)MATMUL_BLOCKED_ACTIVE_QPUS,
           (int)MATMUL_BLOCKED_LANES,
           (int)sweep_max_m,
           (int)sweep_max_n,
           (int)sweep_max_k,
           (int)padded_m_capacity,
           (int)padded_n_capacity,
           (int)padded_k_capacity,
           (int)padded_a_capacity,
           (int)padded_b_capacity,
           (int)padded_c_capacity,
           checksum_accum,
           expected_checksum_accum,
           max_abs_diff_overall,
           (int)runtime_allocations,
           (int)runtime_launches,
           (int)launched_tile_waves,
           (int)runtime_capacity,
           (int)code_uploads,
           elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, c_dev);
    vc4_program_destroy(program);
}
