#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MATMUL_NAIVE_ACTIVE_QPUS 12u
#define MATMUL_NAIVE_LANES 16u
#define MATMUL_NAIVE_CASES 8u
#define MATMUL_NAIVE_B_GUARD_WORDS 16u
#define MATMUL_NAIVE_SCRATCH_WORDS 0u
#define MATMUL_NAIVE_GUARD_WORDS 16u
#define MATMUL_NAIVE_PROGRAM_HEAP_SLOP_BYTES (64u * 1024u)
#define MATMUL_NAIVE_GUARD_BASE 0x6d6d0000u
#define CHECKSUM_SCALE 4096.0f

struct matmul_case {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

static const struct matmul_case cases[MATMUL_NAIVE_CASES] = {
    {12u, 16u, 4u},
    {128u, 224u, 128u},
    {137u, 197u, 96u},
    {192u, 65u, 192u},
    {144u, 223u, 0u},
    {101u, 1u, 160u},
    {180u, 129u, 64u},
    {17u, 215u, 191u},
};

static float *a_values;
static float *b_values;
static float *c_values;
static float *expected_values;
static uint32_t a_capacity;
static uint32_t b_capacity;
static uint32_t c_capacity;

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
    return MATMUL_NAIVE_GUARD_BASE ^ (case_index << 8) ^ word;
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
    if (tc->m == 0u || tc->n == 0u)
        return -1;
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
                               uint32_t *sweep_max_k) {
    a_capacity = 1u;
    b_capacity = 1u;
    c_capacity = 1u;
    *sweep_max_m = 0u;
    *sweep_max_n = 0u;
    *sweep_max_k = 0u;

    for (uint32_t i = 0; i < MATMUL_NAIVE_CASES; i++) {
        uint32_t a_count = 0;
        uint32_t b_count = 0;
        uint32_t c_count = 0;
        const struct matmul_case *tc = &cases[i];
        if (case_counts(tc, &a_count, &b_count, &c_count) < 0)
            panic("matmul_naive case shape overflows addressable bytes");
        if (a_count > a_capacity)
            a_capacity = a_count;
        if (b_count > b_capacity)
            b_capacity = b_count;
        if (c_count > c_capacity)
            c_capacity = c_count;
        if (tc->m > *sweep_max_m)
            *sweep_max_m = tc->m;
        if (tc->n > *sweep_max_n)
            *sweep_max_n = tc->n;
        if (tc->k > *sweep_max_k)
            *sweep_max_k = tc->k;
    }
}

static void allocate_host_buffers(void) {
    uint32_t a_bytes = 0;
    uint32_t b_bytes = 0;
    uint32_t c_bytes = 0;
    uint32_t expected_bytes = 0;
    if (bytes_for_floats(a_capacity, 0u, &a_bytes) < 0 ||
        bytes_for_floats(b_capacity, MATMUL_NAIVE_B_GUARD_WORDS, &b_bytes) < 0 ||
        bytes_for_floats(c_capacity, MATMUL_NAIVE_GUARD_WORDS, &c_bytes) < 0 ||
        bytes_for_floats(c_capacity, 0u, &expected_bytes) < 0)
        panic("matmul_naive host buffer byte size overflow");
    if (a_bytes > 0xffffffffu - b_bytes ||
        a_bytes + b_bytes > 0xffffffffu - c_bytes ||
        a_bytes + b_bytes + c_bytes > 0xffffffffu - expected_bytes ||
        a_bytes + b_bytes + c_bytes + expected_bytes > 0xffffffffu - 4096u)
        panic("matmul_naive host heap byte size overflow");

    kmalloc_init_set_start((void *)(1024u * 1024u),
                           a_bytes + b_bytes + c_bytes + expected_bytes + 4096u);
    a_values = (float *)kmalloc(a_bytes);
    b_values = (float *)kmalloc(b_bytes);
    c_values = (float *)kmalloc(c_bytes);
    expected_values = (float *)kmalloc(expected_bytes);
    if (!a_values || !b_values || !c_values || !expected_values)
        panic("matmul_naive host buffer allocation failed");
}

static float make_a_value(uint32_t case_index, uint32_t row, uint32_t kk) {
    int centered = (int)((row * 7u + kk * 3u + case_index * 5u) % 17u) - 8;
    return ((float)centered) * 0.125f;
}

static float make_b_value(uint32_t case_index, uint32_t kk, uint32_t col) {
    int centered = (int)((kk * 11u + col * 5u + case_index * 7u) % 19u) - 9;
    return ((float)centered) * 0.0625f;
}

static void fill_inputs(uint32_t case_index, const struct matmul_case *tc) {
    for (uint32_t i = 0; i < a_capacity; i++)
        a_values[i] = 0.0f;
    for (uint32_t i = 0; i < b_capacity + MATMUL_NAIVE_B_GUARD_WORDS; i++)
        b_values[i] = 0.0f;

    for (uint32_t row = 0; row < tc->m; row++)
        for (uint32_t kk = 0; kk < tc->k; kk++)
            a_values[row * tc->k + kk] = make_a_value(case_index, row, kk);

    for (uint32_t kk = 0; kk < tc->k; kk++)
        for (uint32_t col = 0; col < tc->n; col++)
            b_values[kk * tc->n + col] = make_b_value(case_index, kk, col);

    for (uint32_t i = 0; i < c_capacity + MATMUL_NAIVE_GUARD_WORDS; i++)
        c_values[i] = float_from_bits(guard_word(case_index, i));
    for (uint32_t i = 0; i < c_capacity; i++)
        expected_values[i] = float_from_bits(guard_word(case_index, i));
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
                           int *guard_mismatches,
                           float *max_abs_diff) {
    uint32_t live_count = tc->m * tc->n;
    *mismatch_count = 0;
    *guard_mismatches = 0;
    *max_abs_diff = 0.0f;

    for (uint32_t i = 0; i < live_count; i++) {
        float diff = c_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;
        if (float_bits(c_values[i]) != float_bits(expected_values[i])) {
            if (*mismatch_count < 8) {
                uint32_t row = i / tc->n;
                uint32_t col = i % tc->n;
                printk("ERROR: case=%d row=%d col=%d gpu=%f cpu=%f gpu_bits=%x cpu_bits=%x diff=%f\n",
                       (int)case_index, (int)row, (int)col, c_values[i],
                       expected_values[i], float_bits(c_values[i]),
                       float_bits(expected_values[i]), diff);
            }
            (*mismatch_count)++;
        }
    }

    for (uint32_t i = live_count; i < c_capacity; i++) {
        uint32_t actual = float_bits(c_values[i]);
        uint32_t expected = guard_word(case_index, i);
        if (actual != expected) {
            if (*guard_mismatches < 8)
                printk("ERROR: case=%d untouched_word=%d actual=0x%x expected=0x%x\n",
                       (int)case_index, (int)i, actual, expected);
            (*guard_mismatches)++;
        }
    }

    for (uint32_t i = c_capacity + MATMUL_NAIVE_SCRATCH_WORDS;
         i < c_capacity + MATMUL_NAIVE_GUARD_WORDS; i++) {
        uint32_t actual = float_bits(c_values[i]);
        uint32_t expected = guard_word(case_index, i);
        if (actual != expected) {
            if (*guard_mismatches < 8)
                printk("ERROR: case=%d guard=%d actual=0x%x expected=0x%x\n",
                       (int)case_index, (int)i, actual, expected);
            (*guard_mismatches)++;
        }
    }
}

void notmain(void) {
    uint32_t sweep_max_m = 0;
    uint32_t sweep_max_n = 0;
    uint32_t sweep_max_k = 0;
    compute_sweep_caps(&sweep_max_m, &sweep_max_n, &sweep_max_k);
    allocate_host_buffers();

    uint32_t a_capacity_bytes = 0;
    uint32_t b_capacity_bytes = 0;
    uint32_t c_bytes = 0;
    if (bytes_for_floats(a_capacity, 0u, &a_capacity_bytes) < 0 ||
        bytes_for_floats(b_capacity, MATMUL_NAIVE_B_GUARD_WORDS, &b_capacity_bytes) < 0 ||
        bytes_for_floats(c_capacity, MATMUL_NAIVE_GUARD_WORDS, &c_bytes) < 0)
        panic("matmul_naive device buffer byte size overflow");
    if (a_capacity_bytes > 0xffffffffu - b_capacity_bytes ||
        a_capacity_bytes + b_capacity_bytes > 0xffffffffu - c_bytes ||
        a_capacity_bytes + b_capacity_bytes + c_bytes >
            0xffffffffu - MATMUL_NAIVE_PROGRAM_HEAP_SLOP_BYTES)
        panic("matmul_naive program heap byte size overflow");
    uint32_t program_heap_bytes = a_capacity_bytes + b_capacity_bytes + c_bytes +
                                  MATMUL_NAIVE_PROGRAM_HEAP_SLOP_BYTES;

    struct vc4_program *program = 0;
    if (vc4_program_create(&program, program_heap_bytes) < 0 || !program)
        panic("vc4_program_create failed");

    vc4_deviceptr_t a_dev = 0, b_dev = 0, c_dev = 0;
    if (vc4_m2_malloc(program, &a_dev, a_capacity_bytes) < 0 ||
        vc4_m2_malloc(program, &b_dev, b_capacity_bytes) < 0 ||
        vc4_m2_malloc(program, &c_dev, c_bytes) < 0)
        panic("matmul_naive device setup failed");

    vc4_dim3 grid = vc4_m2_dim3(1, 1, 1);
    vc4_dim3 block = vc4_m2_dim3(MATMUL_NAIVE_ACTIVE_QPUS * MATMUL_NAIVE_LANES, 1, 1);

    printk("Running VC4 matmul_naive M2 candidate bundle...\n");
    int start = timer_get_usec();
    int launch_failures = 0;
    int total_mismatches = 0;
    int total_guard_mismatches = 0;
    int checksum_accum = 0;
    int expected_checksum_accum = 0;
    uint32_t checked_elements = 0;
    float max_abs_diff_overall = 0.0f;

    for (uint32_t case_index = 0; case_index < MATMUL_NAIVE_CASES; case_index++) {
        const struct matmul_case *tc = &cases[case_index];
        uint32_t a_count = 0;
        uint32_t b_count = 0;
        uint32_t live_count = 0;
        if (case_counts(tc, &a_count, &b_count, &live_count) < 0)
            panic("matmul_naive case exceeds backing buffer capacity");
        uint32_t a_bytes = a_count * sizeof(float);
        uint32_t b_bytes = (b_count + MATMUL_NAIVE_B_GUARD_WORDS) * sizeof(float);
        fill_inputs(case_index, tc);
        run_cpu_reference(tc);

        if (vc4_m2_copy_htod(program, a_dev, a_values, a_bytes) < 0 ||
            vc4_m2_copy_htod(program, b_dev, b_values, b_bytes) < 0 ||
            vc4_m2_copy_htod(program, c_dev, c_values, c_bytes) < 0 ||
            matmul_naive_launch(program, grid, block, a_dev, b_dev, c_dev,
                                tc->m, tc->n, tc->k) < 0 ||
            vc4_m2_copy_dtoh(program, c_values, c_dev, c_bytes) < 0) {
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        int guard_mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(case_index, tc, &mismatches, &guard_mismatches, &max_abs_diff);
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
        total_guard_mismatches += guard_mismatches;
        checksum_accum += checksum;
        expected_checksum_accum += expected_checksum;
        checked_elements += live_count;
        printk("MATMUL_NAIVE_CASE case=%d m=%d n=%d k=%d elements=%d mismatches=%d guard_mismatches=%d checksum=%d expected_checksum=%d max_abs_diff=%f\n",
               (int)case_index, (int)tc->m, (int)tc->n, (int)tc->k,
               (int)live_count, mismatches, guard_mismatches, checksum,
               expected_checksum, max_abs_diff);
    }
    int elapsed = timer_get_usec() - start;

    uint32_t runtime_allocations = matmul_naive_runtime_allocations();
    uint32_t runtime_launches = matmul_naive_runtime_launches();
    uint32_t runtime_capacity = matmul_naive_runtime_capacity();
    uint32_t code_uploads = matmul_naive_runtime_code_uploads();
    uint32_t recorded_launch_failures = matmul_naive_runtime_launch_failures();

    const char *status = (total_mismatches == 0 &&
                          total_guard_mismatches == 0 &&
                          checksum_accum == expected_checksum_accum &&
                          launch_failures == 0 &&
                          recorded_launch_failures == 0 &&
                          runtime_allocations == 1u &&
                          runtime_launches == MATMUL_NAIVE_CASES &&
                          code_uploads == 1u) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=matmul_naive status=%s cases=%d checked_elements=%d total_mismatches=%d guard_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d sweep_max_m=%d sweep_max_n=%d sweep_max_k=%d backing_a_floats=%d backing_b_floats=%d backing_c_floats=%d b_guard_words=%d scratch_words=%d checksum_accum=%d expected_checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_capacity=%d code_uploads=%d elapsed_usec=%d\n",
           status,
           (int)MATMUL_NAIVE_CASES,
           (int)checked_elements,
           total_mismatches,
           total_guard_mismatches,
           launch_failures,
           (int)recorded_launch_failures,
           (int)MATMUL_NAIVE_ACTIVE_QPUS,
           (int)MATMUL_NAIVE_LANES,
           (int)sweep_max_m,
           (int)sweep_max_n,
           (int)sweep_max_k,
           (int)a_capacity,
           (int)b_capacity,
           (int)c_capacity,
           (int)MATMUL_NAIVE_B_GUARD_WORDS,
           (int)MATMUL_NAIVE_SCRATCH_WORDS,
           checksum_accum,
           expected_checksum_accum,
           max_abs_diff_overall,
           (int)runtime_allocations,
           (int)runtime_launches,
           (int)runtime_capacity,
           (int)code_uploads,
           elapsed);

    vc4Free(program, a_dev);
    vc4Free(program, b_dev);
    vc4Free(program, c_dev);
    vc4_program_destroy(program);
}
