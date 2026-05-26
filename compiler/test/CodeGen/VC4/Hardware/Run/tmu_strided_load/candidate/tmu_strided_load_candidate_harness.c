#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define CHECKSUM_SCALE 4096.0f
#define TMU_STRIDED_LOAD_CASES 11u
#define TMU_STRIDED_LOAD_MAX_N 257u
#define TMU_STRIDED_LOAD_MAX_STRIDE 4u
#define TMU_STRIDED_LOAD_MAX_INPUT_WORDS 1104u
#define TMU_STRIDED_LOAD_GUARD_WORDS 32u
#define TMU_STRIDED_LOAD_SENTINEL (-12345.0f)
#define TMU_STRIDED_LOAD_BUFFER_N (TMU_STRIDED_LOAD_MAX_N + TMU_STRIDED_LOAD_GUARD_WORDS)
#define TMU_STRIDED_LOAD_ACTIVE_QPUS 12u
#define TMU_STRIDED_LOAD_LANES 16u

struct tmu_strided_load_case {
    uint32_t n;
    uint32_t offset;
    uint32_t stride;
};

static const struct tmu_strided_load_case cases[TMU_STRIDED_LOAD_CASES] = {
    {0u, 0u, 1u}, {1u, 0u, 1u}, {15u, 2u, 1u}, {16u, 3u, 1u},
    {17u, 1u, 1u}, {31u, 4u, 2u}, {32u, 5u, 2u}, {33u, 7u, 3u},
    {65u, 11u, 4u}, {191u, 13u, 2u}, {257u, 17u, 3u},
};

static float input_values[TMU_STRIDED_LOAD_MAX_INPUT_WORDS];
static float out_values[TMU_STRIDED_LOAD_BUFFER_N];
static float expected_values[TMU_STRIDED_LOAD_BUFFER_N];

static float absf_local(float value) { return value < 0.0f ? -value : value; }
static int invalid_f32(float value) { return !(value == value) || value > 3.4e38f || value < -3.4e38f; }

static float make_input_value(uint32_t j) {
    return ((float)((j * 17u + 5u) % 113u) * 0.125f) - 6.0f;
}

static void fill_inputs(void) {
    for (uint32_t i = 0; i < TMU_STRIDED_LOAD_MAX_INPUT_WORDS; i++)
        input_values[i] = make_input_value(i);
}

static void fill_output_buffers(void) {
    for (uint32_t i = 0; i < TMU_STRIDED_LOAD_BUFFER_N; i++) {
        out_values[i] = TMU_STRIDED_LOAD_SENTINEL;
        expected_values[i] = TMU_STRIDED_LOAD_SENTINEL;
    }
}

static void run_cpu_reference(uint32_t n, uint32_t offset, uint32_t stride, float scale, float bias) {
    for (uint32_t i = 0; i < n; i++)
        expected_values[i] = scale * input_values[offset + i * stride] + bias;
}

static void verify_results(uint32_t case_id, uint32_t n, int *mismatches, float *max_abs_diff) {
    *mismatches = 0;
    *max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float diff = out_values[i] - expected_values[i];
        float abs_diff = absf_local(diff);
        if (abs_diff > *max_abs_diff)
            *max_abs_diff = abs_diff;
        if (invalid_f32(out_values[i]) || out_values[i] != expected_values[i]) {
            if (*mismatches < 8)
                printk("ERROR: case=%d i=%d gpu=%f cpu=%f diff=%f\n",
                       (int)case_id, (int)i, out_values[i], expected_values[i], diff);
            (*mismatches)++;
        }
    }
}

static int verify_sentinel_tail(uint32_t n) {
    int mismatches = 0;
    for (uint32_t i = n; i < n + TMU_STRIDED_LOAD_GUARD_WORDS && i < TMU_STRIDED_LOAD_BUFFER_N; i++) {
        if (out_values[i] != TMU_STRIDED_LOAD_SENTINEL) {
            if (mismatches < 8)
                printk("ERROR: sentinel changed i=%d value=%f expected=%f\n",
                       (int)i, out_values[i], TMU_STRIDED_LOAD_SENTINEL);
            mismatches++;
        }
    }
    return mismatches;
}

static int scaled_checksum(const float *values, uint32_t n) {
    int checksum = 0;
    for (uint32_t i = 0; i < n; i++)
        checksum += (int)(values[i] * CHECKSUM_SCALE);
    return checksum;
}

void notmain(void) {
    struct vc4_program *program = 0;
    const float scale = -1.75f;
    const float bias = 0.5f;
    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");

    fill_inputs();
    vc4_deviceptr_t input_dev = 0, out_dev = 0;
    uint32_t input_bytes = TMU_STRIDED_LOAD_MAX_INPUT_WORDS * sizeof(float);
    uint32_t out_bytes = TMU_STRIDED_LOAD_BUFFER_N * sizeof(float);
    if (vc4_m2_malloc(program, &input_dev, input_bytes) < 0 ||
        vc4_m2_malloc(program, &out_dev, out_bytes) < 0 ||
        vc4_m2_copy_htod(program, input_dev, input_values, input_bytes) < 0)
        panic("tmu_strided_load device setup failed");

    printk("TMU_STRIDED_LOAD_RUNTIME_SETUP max_n=%d max_input_words=%d allocations=%d capacity=%d code_uploads=%d\n",
           TMU_STRIDED_LOAD_MAX_N,
           TMU_STRIDED_LOAD_MAX_INPUT_WORDS,
           (int)tmu_strided_load_runtime_allocations(),
           (int)tmu_strided_load_runtime_capacity(),
           (int)tmu_strided_load_runtime_code_uploads());

    int total_mismatches = 0;
    int sentinel_mismatches = 0;
    int launch_failures = 0;
    uint32_t checked_elements = 0;
    int checksum_accum = 0;
    int expected_checksum_accum = 0;
    float max_abs_diff_overall = 0.0f;
    int start = timer_get_usec();
    vc4_dim3 block = vc4_m2_dim3(TMU_STRIDED_LOAD_LANES, 1, 1);

    for (uint32_t case_index = 0; case_index < TMU_STRIDED_LOAD_CASES; case_index++) {
        uint32_t n = cases[case_index].n;
        uint32_t offset = cases[case_index].offset;
        uint32_t stride = cases[case_index].stride;
        fill_output_buffers();
        run_cpu_reference(n, offset, stride, scale, bias);

        vc4_dim3 grid = vc4_m2_dim3(n == 0u ? 0u : ((n + TMU_STRIDED_LOAD_LANES - 1u) / TMU_STRIDED_LOAD_LANES), 1, 1);
        if (vc4_m2_copy_htod(program, out_dev, out_values, out_bytes) < 0 ||
            tmu_strided_load_launch(program, grid, block, input_dev, out_dev, n, offset, stride, scale, bias) < 0 ||
            vc4_m2_copy_dtoh(program, out_values, out_dev, out_bytes) < 0) {
            printk("ERROR: tmu_strided_load launch/copy failed case=%d n=%d offset=%d stride=%d\n",
                   (int)case_index, (int)n, (int)offset, (int)stride);
            launch_failures++;
            continue;
        }

        int mismatches = 0;
        float max_abs_diff = 0.0f;
        verify_results(case_index, n, &mismatches, &max_abs_diff);
        int case_sentinel_mismatches = verify_sentinel_tail(n);
        int checksum = scaled_checksum(out_values, n);
        int expected_checksum = scaled_checksum(expected_values, n);
        if (checksum != expected_checksum) {
            printk("ERROR: checksum mismatch case=%d gpu=%d cpu=%d\n", (int)case_index, checksum, expected_checksum);
            mismatches++;
        }
        if (max_abs_diff > max_abs_diff_overall)
            max_abs_diff_overall = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel_mismatches;
        checked_elements += n;
        checksum_accum += checksum;
        expected_checksum_accum += expected_checksum;

        printk("TMU_STRIDED_LOAD_CASE case=%d n=%d offset=%d stride=%d mismatches=%d sentinel_mismatches=%d checksum=%d expected_checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
               (int)case_index, (int)n, (int)offset, (int)stride, mismatches,
               case_sentinel_mismatches, checksum, expected_checksum, max_abs_diff,
               (int)tmu_strided_load_runtime_launches(), (int)tmu_strided_load_runtime_allocations());
    }

    int elapsed = timer_get_usec() - start;
    uint32_t runtime_allocations = tmu_strided_load_runtime_allocations();
    uint32_t runtime_launches = tmu_strided_load_runtime_launches();
    uint32_t runtime_capacity = tmu_strided_load_runtime_capacity();
    uint32_t code_uploads = tmu_strided_load_runtime_code_uploads();
    uint32_t recorded_launch_failures = tmu_strided_load_runtime_launch_failures();
    const char *status =
        (total_mismatches == 0 && sentinel_mismatches == 0 && launch_failures == 0 &&
         recorded_launch_failures == 0u &&
         checked_elements == 658u &&
         checksum_accum == expected_checksum_accum &&
         runtime_allocations == 1u && runtime_launches == TMU_STRIDED_LOAD_CASES &&
         runtime_capacity == 65536u &&
         code_uploads == 1u &&
         max_abs_diff_overall == 0.0f) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=tmu_strided_load status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_stride=%d max_abs_diff=%f checksum_accum=%d expected_checksum_accum=%d runtime_allocations=%d runtime_launches=%d runtime_capacity=%d code_uploads=%d elapsed_usec=%d\n",
           status, TMU_STRIDED_LOAD_CASES, (int)checked_elements, total_mismatches, sentinel_mismatches,
           launch_failures, (int)recorded_launch_failures, TMU_STRIDED_LOAD_ACTIVE_QPUS, TMU_STRIDED_LOAD_LANES,
           TMU_STRIDED_LOAD_MAX_N, TMU_STRIDED_LOAD_MAX_STRIDE, max_abs_diff_overall,
           checksum_accum, expected_checksum_accum, (int)runtime_allocations, (int)runtime_launches,
           (int)runtime_capacity, (int)code_uploads, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, out_dev);
    vc4_program_destroy(program);
}
