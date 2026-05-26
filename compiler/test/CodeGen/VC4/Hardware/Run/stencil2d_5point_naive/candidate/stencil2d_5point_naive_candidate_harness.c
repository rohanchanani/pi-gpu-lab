#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define V3D_BASE 0x20C00000u
#define V3D_VPMBASE (V3D_BASE + 0x00504u)

#define STENCIL2D_5POINT_NAIVE_MAX_WIDTH 32u
#define STENCIL2D_5POINT_NAIVE_MAX_HEIGHT 32u
#define STENCIL2D_5POINT_NAIVE_ROW_STRIDE 32u
#define STENCIL2D_5POINT_NAIVE_MAX_WORDS \
    (STENCIL2D_5POINT_NAIVE_MAX_WIDTH * STENCIL2D_5POINT_NAIVE_MAX_HEIGHT)
#define STENCIL2D_5POINT_NAIVE_SCRATCH_WORDS \
    (STENCIL2D_5POINT_NAIVE_ROW_STRIDE * STENCIL2D_5POINT_NAIVE_MAX_HEIGHT)
#define STENCIL2D_5POINT_NAIVE_CASES 9u
#define STENCIL2D_5POINT_NAIVE_SENTINEL (-13579.0f)
#define STENCIL2D_5POINT_NAIVE_EPSILON 0.0f
#define STENCIL2D_5POINT_NAIVE_ACTIVE_QPUS 12u
#define STENCIL2D_5POINT_NAIVE_LANES 16u
#define CHECKSUM_SCALE 4096.0f

struct stencil_case {
    uint32_t width;
    uint32_t height;
};

static const struct stencil_case test_cases[STENCIL2D_5POINT_NAIVE_CASES] = {
    {3u, 2u},
    {1u, 1u},
    {2u, 3u},
    {5u, 7u},
    {7u, 5u},
    {16u, 4u},
    {17u, 9u},
    {31u, 19u},
    {32u, 32u},
};

static float input_values[STENCIL2D_5POINT_NAIVE_MAX_WORDS];
static float expected_values[STENCIL2D_5POINT_NAIVE_MAX_WORDS];
static float input_scratch[STENCIL2D_5POINT_NAIVE_SCRATCH_WORDS];
static float output_scratch[STENCIL2D_5POINT_NAIVE_SCRATCH_WORDS];

static float abs_f32(float value) {
    return value < 0.0f ? -value : value;
}

static int invalid_f32(float value) {
    return !(value == value) || value > 3.4e38f || value < -3.4e38f;
}

static uint32_t compact_index(uint32_t width, uint32_t y, uint32_t x) {
    return y * width + x;
}

static uint32_t scratch_index(uint32_t y, uint32_t x) {
    return y * STENCIL2D_5POINT_NAIVE_ROW_STRIDE + x;
}

static float input_value(uint32_t i, uint32_t x, uint32_t y) {
    int raw = (int)((i * 13u + x * 7u + y * 5u + 11u) % 113u) - 56;
    return (float)raw * 0.03125f;
}

static void fill_case(uint32_t width, uint32_t height) {
    for (uint32_t i = 0; i < STENCIL2D_5POINT_NAIVE_MAX_WORDS; i++) {
        input_values[i] = 0.0f;
        expected_values[i] = 0.0f;
    }

    for (uint32_t i = 0; i < STENCIL2D_5POINT_NAIVE_SCRATCH_WORDS; i++) {
        input_scratch[i] = 0.0f;
        output_scratch[i] = STENCIL2D_5POINT_NAIVE_SENTINEL;
    }

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t i = compact_index(width, y, x);
            float value = input_value(i, x, y);
            input_values[i] = value;
            input_scratch[scratch_index(y, x)] = value;
        }
    }
}

static void run_cpu_reference(uint32_t width, uint32_t height,
                              float center_weight, float neighbor_weight) {
    if (width == 0u || height == 0u)
        return;

    for (uint32_t y = 0; y < height; y++) {
        uint32_t up_y = y == 0u ? 0u : y - 1u;
        uint32_t down_y = y + 1u >= height ? height - 1u : y + 1u;
        for (uint32_t x = 0; x < width; x++) {
            uint32_t left_x = x == 0u ? 0u : x - 1u;
            uint32_t right_x = x + 1u >= width ? width - 1u : x + 1u;
            float up = input_values[compact_index(width, up_y, x)];
            float down = input_values[compact_index(width, down_y, x)];
            float left = input_values[compact_index(width, y, left_x)];
            float right = input_values[compact_index(width, y, right_x)];
            float center = input_values[compact_index(width, y, x)];
            float neighbor_sum = ((up + down) + left) + right;
            float neighbor_part = neighbor_weight * neighbor_sum;
            float center_part = center_weight * center;
            expected_values[compact_index(width, y, x)] = neighbor_part + center_part;
        }
    }
}

static int scaled_checksum(uint32_t width, uint32_t height) {
    int checksum = 0;
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            checksum += (int)(output_scratch[scratch_index(y, x)] * CHECKSUM_SCALE);
    return checksum;
}

static int scaled_expected_checksum(uint32_t width, uint32_t height) {
    int checksum = 0;
    uint32_t total = width * height;
    for (uint32_t i = 0; i < total; i++)
        checksum += (int)(expected_values[i] * CHECKSUM_SCALE);
    return checksum;
}

static void verify_case(uint32_t case_id, uint32_t width, uint32_t height,
                        uint32_t *checked, uint32_t *mismatches,
                        uint32_t *sentinel_mismatches, float *max_abs_diff) {
    *checked = 0u;
    *mismatches = 0u;
    *sentinel_mismatches = 0u;
    *max_abs_diff = 0.0f;

    for (uint32_t y = 0; y < STENCIL2D_5POINT_NAIVE_MAX_HEIGHT; y++) {
        for (uint32_t x = 0; x < STENCIL2D_5POINT_NAIVE_ROW_STRIDE; x++) {
            float gpu = output_scratch[scratch_index(y, x)];
            if (y < height && x < width) {
                float cpu = expected_values[compact_index(width, y, x)];
                float diff = gpu - cpu;
                float adiff = abs_f32(diff);
                (*checked)++;
                if (adiff > *max_abs_diff)
                    *max_abs_diff = adiff;
                if (invalid_f32(gpu) || adiff > STENCIL2D_5POINT_NAIVE_EPSILON) {
                    if (*mismatches < 8u)
                        printk("ERROR: case=%d width=%d height=%d y=%d x=%d gpu=%f cpu=%f diff=%f\n",
                               (int)case_id, (int)width, (int)height,
                               (int)y, (int)x, gpu, cpu, diff);
                    (*mismatches)++;
                }
            } else if (gpu != STENCIL2D_5POINT_NAIVE_SENTINEL) {
                if (*sentinel_mismatches < 8u)
                    printk("ERROR: sentinel changed case=%d y=%d x=%d gpu=%f expected=%f\n",
                           (int)case_id, (int)y, (int)x, gpu,
                           STENCIL2D_5POINT_NAIVE_SENTINEL);
                (*sentinel_mismatches)++;
            }
        }
    }
}

void notmain(void) {
    const float center_weight = 0.5f;
    const float neighbor_weight = 0.125f;
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0;
    vc4_deviceptr_t output_dev = 0;
    uint32_t bytes = STENCIL2D_5POINT_NAIVE_SCRATCH_WORDS * sizeof(float);
    uint32_t total_mismatches = 0u;
    uint32_t sentinel_mismatches = 0u;
    uint32_t launch_failures = 0u;
    uint32_t checked_elements = 0u;
    int checksum_accum = 0;
    float global_max_abs_diff = 0.0f;

    if (vc4_program_create(&program, 0u) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &input_dev, bytes) < 0 ||
        vc4_m2_malloc(program, &output_dev, bytes) < 0)
        panic("stencil2d_5point_naive device allocation failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(STENCIL2D_5POINT_NAIVE_ACTIVE_QPUS *
                                     STENCIL2D_5POINT_NAIVE_LANES,
                                 1u, 1u);
    printk("STENCIL2D_5POINT_NAIVE_RUNTIME_SETUP max_width=%d max_height=%d allocations=1 active_qpus=%d lanes=%d\n",
           STENCIL2D_5POINT_NAIVE_MAX_WIDTH, STENCIL2D_5POINT_NAIVE_MAX_HEIGHT,
           STENCIL2D_5POINT_NAIVE_ACTIVE_QPUS, STENCIL2D_5POINT_NAIVE_LANES);
    PUT32(V3D_VPMBASE, 16u);

    int start = timer_get_usec();
    for (uint32_t case_id = 0; case_id < STENCIL2D_5POINT_NAIVE_CASES; case_id++) {
        uint32_t width = test_cases[case_id].width;
        uint32_t height = test_cases[case_id].height;
        uint32_t total = width * height;
        uint32_t mismatches = 0u;
        uint32_t case_sentinel = 0u;
        uint32_t case_checked = 0u;
        float max_abs_diff = 0.0f;

        fill_case(width, height);
        run_cpu_reference(width, height, center_weight, neighbor_weight);

        if (vc4_m2_copy_htod(program, input_dev, input_scratch, bytes) < 0 ||
            vc4_m2_copy_htod(program, output_dev, output_scratch, bytes) < 0 ||
            stencil2d_5point_naive_launch(program, grid, block, input_dev,
                                          output_dev, width, height, total,
                                          center_weight, neighbor_weight) < 0 ||
            vc4_m2_copy_dtoh(program, output_scratch, output_dev, bytes) < 0) {
            launch_failures++;
            printk("STENCIL2D_5POINT_NAIVE_CASE case=%d width=%d height=%d launch=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)width, (int)height, (int)(case_id + 1u));
            continue;
        }

        verify_case(case_id, width, height, &case_checked, &mismatches,
                    &case_sentinel, &max_abs_diff);
        int checksum = scaled_checksum(width, height);
        int expected_checksum = scaled_expected_checksum(width, height);
        if (checksum != expected_checksum) {
            printk("ERROR: checksum mismatch case=%d width=%d height=%d gpu=%d cpu=%d\n",
                   (int)case_id, (int)width, (int)height, checksum,
                   expected_checksum);
            mismatches++;
        }

        if (max_abs_diff > global_max_abs_diff)
            global_max_abs_diff = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checked_elements += case_checked;
        checksum_accum += checksum;

        printk("STENCIL2D_5POINT_NAIVE_CASE case=%d width=%d height=%d n=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=1\n",
               (int)case_id, (int)width, (int)height, (int)total,
               (int)case_checked, (int)mismatches, (int)case_sentinel,
               checksum, max_abs_diff, (int)(case_id + 1u));
    }

    int elapsed = timer_get_usec() - start;
    const char *status =
        (total_mismatches == 0u && sentinel_mismatches == 0u &&
         launch_failures == 0u && checked_elements == 1913u &&
         global_max_abs_diff <= STENCIL2D_5POINT_NAIVE_EPSILON) ? "PASS" : "FAIL";

    printk("VC4_TEST_RESULT name=stencil2d_5point_naive status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_width=%d max_height=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d checksum_accum=%d elapsed_usec=%d\n",
           status, STENCIL2D_5POINT_NAIVE_CASES, (int)checked_elements,
           (int)total_mismatches, (int)sentinel_mismatches,
           (int)launch_failures, STENCIL2D_5POINT_NAIVE_ACTIVE_QPUS,
           STENCIL2D_5POINT_NAIVE_LANES, STENCIL2D_5POINT_NAIVE_MAX_WIDTH,
           STENCIL2D_5POINT_NAIVE_MAX_HEIGHT, global_max_abs_diff, 1,
           STENCIL2D_5POINT_NAIVE_CASES, checksum_accum, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
