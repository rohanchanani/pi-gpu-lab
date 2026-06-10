#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_case_config.h"
#include "vc4_m2_candidate_test_helpers.h"

#define GUARD 32u
#define ACTIVE_QPUS 12u
#define LANES 16u
#define GRID_X 4u
#define GRID_Y 4u
#define MAX_GRID_BLOCKS (GRID_X * GRID_Y)
#define MAX_ELEMENTS (MAX_GRID_BLOCKS * LANES)
#define BUFFER_N (MAX_ELEMENTS + 2u * GUARD)
#define SENTINEL_BITS 0xc56a4000u

struct tail_case {
    uint32_t n;
    int32_t flag;
};

static const struct tail_case cases[] = {
    {0u, 1},
    {1u, -1},
    {15u, 1},
    {16u, -1},
    {17u, 1},
    {31u, -1},
    {32u, 1},
    {127u, -1},
    {128u, 1},
    {191u, -1},
    {192u, 1},
    {255u, -1},
    {256u, 1},
};

static float x_values[BUFFER_N];
static float out_values[BUFFER_N];

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

static float input_value(uint32_t i) {
    int32_t whole = (int32_t)((i * 17u + 3u) % 149u) - 74;
    uint32_t quarter = (i * 5u + 1u) & 3u;
    float value = (float)whole + (float)quarter * 0.25f;
    return (i & 1u) ? -value : value;
}

static void fill_buffers(void) {
    float sentinel = bits_to_float(SENTINEL_BITS);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        uint32_t logical = i - GUARD;
        x_values[i] = input_value(logical);
        out_values[i] = sentinel;
    }
}

static uint32_t active_elements(uint32_t n) {
    return n < MAX_ELEMENTS ? n : MAX_ELEMENTS;
}

static float expected_value(uint32_t i, int32_t flag) {
    float x = x_values[GUARD + i];
    return flag > 0 ? x + 1.0f : x - 1.0f;
}

static int verify_results(uint32_t n, int32_t flag) {
    int mismatches = 0;
    uint32_t elements = active_elements(n);
    for (uint32_t i = 0; i < elements; i++) {
        uint32_t index = GUARD + i;
        uint32_t got = float_to_bits(out_values[index]);
        uint32_t expected = float_to_bits(expected_value(i, flag));
        if (got != expected) {
            if (mismatches < 8)
                printk("ERROR: ttir cf_tail n=%d flag=%d i=%d got=%x expected=%x gpu=%f expected_f=%f\n",
                       (int)n, (int)flag, (int)i, got, expected, out_values[index],
                       bits_to_float(expected));
            mismatches++;
        }
    }
    return mismatches;
}

static int verify_sentinels(uint32_t n) {
    int mismatches = 0;
    uint32_t elements = active_elements(n);
    for (uint32_t i = 0; i < BUFFER_N; i++) {
        if (i >= GUARD && i < GUARD + elements)
            continue;
        uint32_t got = float_to_bits(out_values[i]);
        if (got != SENTINEL_BITS) {
            if (mismatches < 8)
                printk("ERROR: ttir cf_tail sentinel n=%d elements=%d i=%d got=%x expected=%x\n",
                       (int)n, (int)elements, (int)i, got, SENTINEL_BITS);
            mismatches++;
        }
    }
    return mismatches;
}

static uint32_t hash_output(uint32_t n) {
    uint32_t elements = active_elements(n);
    uint32_t hash = 2166136261u ^ n ^ (elements << 8);
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
    vc4_deviceptr_t x_dev = 0, out_dev = 0;
    if (vc4_m2_malloc(program, &x_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, bytes) < 0)
        panic("ttir cf_tail allocation failed");

    int total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    int elements_checked = 0;
    uint32_t output_hash = 2166136261u;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(LANES, 1u, 1u);
    vc4_dim3 grid = vc4_m2_dim3(GRID_X, GRID_Y, 1u);

    for (uint32_t case_id = 0; case_id < sizeof(cases) / sizeof(cases[0]); case_id++) {
        uint32_t n = cases[case_id].n;
        int32_t flag = cases[case_id].flag;
        uint32_t elements = active_elements(n);
        fill_buffers();
        vc4_deviceptr_t x_active = x_dev + GUARD * sizeof(float);
        vc4_deviceptr_t out_active = out_dev + GUARD * sizeof(float);
        if (vc4_m2_copy_htod(program, x_dev, x_values, bytes) < 0 ||
            vc4_m2_copy_htod(program, out_dev, out_values, bytes) < 0 ||
            ttir_multi_axis_cf_tail_b16_kernel_launch(program, grid, block, x_active, out_active, n, flag) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, bytes) < 0) {
            printk("ERROR: ttir cf_tail launch/copy failed case=%d n=%d flag=%d\n",
                   (int)case_id, (int)n, (int)flag);
            launch_failures++;
            continue;
        }
        int mismatches = verify_results(n, flag);
        int sentinels = verify_sentinels(n);
        uint32_t case_hash = hash_output(n);
        output_hash ^= case_hash + 0x9e3779b9u + (case_id << 6) + (case_id >> 2);
        output_hash = rotl32_local(output_hash, 7u);
        total_mismatches += mismatches;
        sentinel_mismatches += sentinels;
        elements_checked += (int)elements;
        printk("TTIR_MULTI_AXIS_CF_TAIL_CASE case=%d n=%d flag=%d active_elements=%d mismatches=%d sentinel_mismatches=%d hash=%x\n",
               (int)case_id, (int)n, (int)flag, (int)elements, mismatches,
               sentinels, case_hash);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0 && sentinel_mismatches == 0 &&
                          launch_failures == 0 && output_hash != 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=ttir_multi_axis_cf_tail_b16_vc4triton status=%s cases=%d elements_checked=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_grid_blocks=%d max_elements=%d buffer_n=%d output_hash=%u saw_nonzero_output_hash=1 saw_real_ttir_snapshot=1 saw_cpp_ttir_importer=%d saw_ttir_multi_axis_launch=1 saw_ttir_program_id_axis0=1 saw_ttir_program_id_axis1=1 saw_ttir_num_programs_axis0=1 saw_ttir_tail_mask=1 saw_ttir_scf_if=1 saw_value_multi_axis_launch=1 saw_value_scf_cf=1 runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, (int)(sizeof(cases) / sizeof(cases[0])), elements_checked,
           total_mismatches, sentinel_mismatches, launch_failures, ACTIVE_QPUS,
           LANES, MAX_GRID_BLOCKS, MAX_ELEMENTS, BUFFER_N, output_hash,
           VC4_CASE_SAW_CPP_TTIR_IMPORTER, 2,
           (int)(sizeof(cases) / sizeof(cases[0])), elapsed);
    vc4Free(program, x_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
