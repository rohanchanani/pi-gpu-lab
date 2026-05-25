#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define SOFTMAX_ROW_CASES 4u
#define SOFTMAX_ROW_MAX_ROWS 13u
#define SOFTMAX_ROW_MAX_WIDTH 16u
#define SOFTMAX_ROW_MAX_WORDS (SOFTMAX_ROW_MAX_ROWS * SOFTMAX_ROW_MAX_WIDTH)
#define SOFTMAX_ROW_GUARD_WORDS 64u
#define SOFTMAX_ROW_SENTINEL (-12345.25f)
#define SOFTMAX_ROW_EPSILON 0.02f
#define SOFTMAX_ROW_ACTIVE_QPUS 12u
#define SOFTMAX_ROW_LANES 16u

struct softmax_row_case { uint32_t rows, width, constant_rows; };

static const struct softmax_row_case softmax_row_cases[SOFTMAX_ROW_CASES] = {
    {3u, 8u, 0u}, {3u, 8u, 0u}, {3u, 8u, 0u}, {3u, 8u, 0u},
};

static float input_values[SOFTMAX_ROW_MAX_WORDS];
static float output_values[SOFTMAX_ROW_MAX_WORDS + SOFTMAX_ROW_GUARD_WORDS];
static float expected_values[SOFTMAX_ROW_MAX_WORDS];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static int invalid_f32(float value) { return !(value == value) || value > 3.4e38f || value < -3.4e38f; }

static float sqrt_approx(float x) {
    if (x <= 0.0f)
        return 0.0f;
    float y = x > 1.0f ? x : 1.0f;
    for (uint32_t i = 0; i < 12u; i++)
        y = 0.5f * (y + x / y);
    return y;
}

static float exp2_approx(float x) {
    const float ln2 = 0.6931471805599453f;
    x *= ln2;
    float term = 1.0f, sum = 1.0f;
    for (uint32_t i = 1u; i <= 16u; i++) {
        term *= x / (float)i;
        sum += term;
    }
    return sum;
}

static float make_input_value(uint32_t row, uint32_t lane, uint32_t case_id) {
    int raw = (int)((row * 17u + lane * 7u + case_id * 5u + 11u) % 23u) - 11;
    return (float)raw * 0.125f;
}

static void fill_case(uint32_t case_id, uint32_t rows, uint32_t width, uint32_t constant_rows,
                      float epsilon, float gamma, float beta) {
    if (case_id == 1u)
        printk("SOFTMAX_ROW_FILL case=1 enter\n");
    for (uint32_t i = 0; i < SOFTMAX_ROW_MAX_WORDS; i++) {
        input_values[i] = 0.0f;
        expected_values[i] = 0.0f;
    }
    for (uint32_t i = 0; i < SOFTMAX_ROW_MAX_WORDS + SOFTMAX_ROW_GUARD_WORDS; i++)
        output_values[i] = SOFTMAX_ROW_SENTINEL;
    if (case_id == 1u)
        printk("SOFTMAX_ROW_FILL case=1 zeroed\n");

    for (uint32_t row = 0; row < rows; row++) {
        float mean = 0.0f;
        for (uint32_t lane = 0; lane < width; lane++)
            mean += (constant_rows != 0u && row == 0u) ? 0.0f : make_input_value(row, lane, case_id);
        if (width != 0u)
            mean /= (float)width;
        for (uint32_t lane = 0; lane < width; lane++) {
            float value = (constant_rows != 0u && row == 0u) ? 0.0f : make_input_value(row, lane, case_id);
            input_values[row * width + lane] = value - mean;
        }
    }
    if (case_id == 1u)
        printk("SOFTMAX_ROW_FILL case=1 inputs\n");

    for (uint32_t row = 0; row < rows; row++) {
        uint32_t base = row * width;
        if (width == 0u)
            continue;
        float sum = 0.0f;
        if (case_id == 1u)
            printk("SOFTMAX_ROW_FILL case=1 row=%d before_exp\n", (int)row);
        for (uint32_t lane = 0; lane < width; lane++) {
            float e = exp2_approx(input_values[base + lane]);
            expected_values[base + lane] = e;
            sum += e;
        }
        if (case_id == 1u)
            printk("SOFTMAX_ROW_FILL case=1 row=%d after_exp sum=%f\n", (int)row, sum);
        for (uint32_t lane = 0; lane < width; lane++)
            expected_values[base + lane] /= sum;
    }
    if (case_id == 1u)
        printk("SOFTMAX_ROW_FILL case=1 done\n");
}

void notmain(void) {
    const float epsilon = 1.0e-3f, gamma = 1.0f, beta = -0.5f;
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0, output_dev = 0;
    uint32_t input_bytes = SOFTMAX_ROW_MAX_WORDS * sizeof(float);
    uint32_t output_bytes = (SOFTMAX_ROW_MAX_WORDS + SOFTMAX_ROW_GUARD_WORDS) * sizeof(float);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, row_sum_mismatches = 0, launch_failures = 0, checked_elements = 0;
    uint32_t runtime_launches = 0;
    float global_max_abs_diff = 0.0f;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &input_dev, input_bytes) < 0 || vc4_m2_malloc(program, &output_dev, output_bytes) < 0)
        panic("softmax_row device allocation failed");

    vc4_dim3 block = vc4_m2_dim3(SOFTMAX_ROW_LANES, 1u, 1u);
    printk("SOFTMAX_ROW_RUNTIME_SETUP max_rows=%d max_width=%d allocations=1\n", SOFTMAX_ROW_MAX_ROWS, SOFTMAX_ROW_MAX_WIDTH);

    for (uint32_t case_id = 0; case_id < SOFTMAX_ROW_CASES; case_id++) {
        uint32_t rows = softmax_row_cases[case_id].rows;
        uint32_t width = softmax_row_cases[case_id].width;
	        uint32_t logical_words = rows * width;
	        uint32_t mismatches = 0, case_sentinel = 0, case_row_sum = 0;
	        float max_abs_diff = 0.0f;
	        int checksum = 0;
	        printk("SOFTMAX_ROW_PHASE case=%d before_fill\n", (int)case_id);
	        fill_case(case_id, rows, width, softmax_row_cases[case_id].constant_rows, epsilon, gamma, beta);
	        printk("SOFTMAX_ROW_PHASE case=%d after_fill\n", (int)case_id);
	        vc4_dim3 grid = vc4_m2_dim3(rows == 0u ? 0u : rows, 1u, 1u);
	        printk("SOFTMAX_ROW_PHASE case=%d before_htod\n", (int)case_id);

	        if (vc4_m2_copy_htod(program, input_dev, input_values, input_bytes) < 0 ||
	            vc4_m2_copy_htod(program, output_dev, output_values, output_bytes) < 0) {
	            launch_failures++;
	            printk("SOFTMAX_ROW_CASE case=%d rows=%d width=%d copy_htod=FAIL launches=%d allocations=1\n",
	                   (int)case_id, (int)rows, (int)width, (int)runtime_launches);
	            continue;
	        }
	        printk("SOFTMAX_ROW_PHASE case=%d before_launch\n", (int)case_id);
	        if (softmax_row_launch(program, grid, block, input_dev, output_dev, rows, width, width, epsilon, gamma, beta) < 0 ||
	            vc4_m2_copy_dtoh(program, output_values, output_dev, output_bytes) < 0) {
	            launch_failures++;
	            printk("SOFTMAX_ROW_CASE case=%d rows=%d width=%d launch=FAIL launches=%d allocations=1\n",
	                   (int)case_id, (int)rows, (int)width, (int)runtime_launches);
	            continue;
	        }
        runtime_launches++;

        for (uint32_t i = 0; i < logical_words; i++) {
            float diff = output_values[i] - expected_values[i];
            float adiff = abs_f32(diff);
            checksum += (int)(output_values[i] * 1024.0f);
            checked_elements++;
            if (adiff > max_abs_diff)
                max_abs_diff = adiff;
            if (invalid_f32(output_values[i]) || adiff > SOFTMAX_ROW_EPSILON) {
                if (mismatches < 8u)
                    printk("ERROR: case=%d row=%d lane=%d gpu=%f cpu=%f diff=%f\n",
                           (int)case_id, (int)(width == 0u ? 0u : i / width), (int)(width == 0u ? 0u : i % width),
                           output_values[i], expected_values[i], diff);
                mismatches++;
            }
        }
        for (uint32_t i = logical_words; i < SOFTMAX_ROW_MAX_WORDS + SOFTMAX_ROW_GUARD_WORDS; i++)
            if (output_values[i] != SOFTMAX_ROW_SENTINEL)
                case_sentinel++;
        if (width != 0u) {
            for (uint32_t row = 0; row < rows; row++) {
                float row_sum = 0.0f;
                for (uint32_t lane = 0; lane < width; lane++)
                    row_sum += output_values[row * width + lane];
                if (invalid_f32(row_sum) || abs_f32(row_sum - 1.0f) > SOFTMAX_ROW_EPSILON)
                    case_row_sum++;
            }
        }
        if (max_abs_diff > global_max_abs_diff)
            global_max_abs_diff = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        row_sum_mismatches += case_row_sum;
        checksum_accum += checksum;
        printk("SOFTMAX_ROW_CASE case=%d rows=%d width=%d checked=%d mismatches=%d sentinel_mismatches=%d row_sum_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=1\n",
               (int)case_id, (int)rows, (int)width, (int)logical_words, (int)mismatches, (int)case_sentinel,
               (int)case_row_sum, checksum, max_abs_diff, (int)runtime_launches);
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u && row_sum_mismatches == 0u && launch_failures == 0u &&
                          checked_elements > 0u && global_max_abs_diff <= SOFTMAX_ROW_EPSILON) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=softmax_row status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d row_sum_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_rows=%d max_width=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, SOFTMAX_ROW_CASES, (int)checked_elements, (int)total_mismatches, (int)sentinel_mismatches, (int)row_sum_mismatches,
           (int)launch_failures, SOFTMAX_ROW_ACTIVE_QPUS, SOFTMAX_ROW_LANES, SOFTMAX_ROW_MAX_ROWS,
           SOFTMAX_ROW_MAX_WIDTH, checksum_accum, global_max_abs_diff, 1, (int)runtime_launches, elapsed);
    vc4Free(program, input_dev);
    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
