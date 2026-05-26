#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MAXPOOL2D_2X2_CASES 8u
#define MAXPOOL2D_2X2_MAX_WIDTH 31u
#define MAXPOOL2D_2X2_MAX_HEIGHT 19u
#define MAXPOOL2D_2X2_MAX_PIXELS (MAXPOOL2D_2X2_MAX_WIDTH * MAXPOOL2D_2X2_MAX_HEIGHT)
#define MAXPOOL2D_2X2_MAX_OUT_W ((MAXPOOL2D_2X2_MAX_WIDTH + 1u) / 2u)
#define MAXPOOL2D_2X2_MAX_OUT_H ((MAXPOOL2D_2X2_MAX_HEIGHT + 1u) / 2u)
#define MAXPOOL2D_2X2_MAX_OUTPUTS (MAXPOOL2D_2X2_MAX_OUT_W * MAXPOOL2D_2X2_MAX_OUT_H)
#define MAXPOOL2D_2X2_GUARD_WORDS 64u
#define MAXPOOL2D_2X2_SENTINEL (-12345.25f)
#define MAXPOOL2D_2X2_ACTIVE_QPUS 12u
#define MAXPOOL2D_2X2_LANES 16u

struct maxpool2d_2x2_case {
    uint32_t width;
    uint32_t height;
    uint32_t pattern;
};

static const struct maxpool2d_2x2_case maxpool2d_2x2_cases[MAXPOOL2D_2X2_CASES] = {
    {0u, 7u, 0u},
    {1u, 1u, 1u},
    {2u, 2u, 2u},
    {3u, 3u, 0u},
    {5u, 7u, 1u},
    {16u, 16u, 2u},
    {17u, 9u, 0u},
    {31u, 19u, 1u},
};

static float input_values[MAXPOOL2D_2X2_MAX_PIXELS];
static float output_values[MAXPOOL2D_2X2_MAX_OUTPUTS + MAXPOOL2D_2X2_GUARD_WORDS];
static float expected_values[MAXPOOL2D_2X2_MAX_OUTPUTS];

static uint32_t ceil_div2_u32(uint32_t value) { return (value + 1u) >> 1; }

static float abs_f32(float value) { return value < 0.0f ? -value : value; }

static float input_pattern(uint32_t pattern, uint32_t x, uint32_t y) {
    int raw;

    switch (pattern) {
    case 0u:
        raw = (int)((x * 7u + y * 13u + x * y * 3u + 5u) % 97u) - 48;
        break;
    case 1u:
        raw = ((x + y) & 1u) ? (int)(21u + ((x * 11u + y * 5u) % 41u))
                              : -(int)(17u + ((x * 3u + y * 19u) % 37u));
        break;
    default:
        raw = (int)((x * 23u + y * 29u + 31u) % 113u) - 56;
        if ((x % 5u) == 0u)
            raw += 9;
        if ((y % 4u) == 0u)
            raw -= 7;
        break;
    }

    return (float)raw * 0.125f;
}

static float ref_one(const float *input, uint32_t width, uint32_t height,
                     uint32_t ox, uint32_t oy) {
    const uint32_t x0 = ox * 2u;
    const uint32_t y0 = oy * 2u;
    float best = input[y0 * width + x0];

    if (x0 + 1u < width) {
        const float v = input[y0 * width + x0 + 1u];
        if (v > best)
            best = v;
    }

    if (y0 + 1u < height) {
        const float v = input[(y0 + 1u) * width + x0];
        if (v > best)
            best = v;
    }

    if (x0 + 1u < width && y0 + 1u < height) {
        const float v = input[(y0 + 1u) * width + x0 + 1u];
        if (v > best)
            best = v;
    }

    return best;
}

static void fill_case(uint32_t width, uint32_t height, uint32_t pattern) {
    for (uint32_t i = 0; i < MAXPOOL2D_2X2_MAX_PIXELS; i++)
        input_values[i] = 0.0f;
    for (uint32_t i = 0; i < MAXPOOL2D_2X2_MAX_OUTPUTS; i++)
        expected_values[i] = MAXPOOL2D_2X2_SENTINEL;
    for (uint32_t i = 0; i < MAXPOOL2D_2X2_MAX_OUTPUTS + MAXPOOL2D_2X2_GUARD_WORDS; i++)
        output_values[i] = MAXPOOL2D_2X2_SENTINEL;

    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            input_values[y * width + x] = input_pattern(pattern, x, y);

    const uint32_t out_w = ceil_div2_u32(width);
    const uint32_t out_h = ceil_div2_u32(height);
    for (uint32_t oy = 0; oy < out_h; oy++)
        for (uint32_t ox = 0; ox < out_w; ox++)
            expected_values[oy * out_w + ox] = ref_one(input_values, width, height, ox, oy);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0, output_dev = 0;
    const uint32_t input_bytes = MAXPOOL2D_2X2_MAX_PIXELS * sizeof(float);
    const uint32_t output_bytes =
        (MAXPOOL2D_2X2_MAX_OUTPUTS + MAXPOOL2D_2X2_GUARD_WORDS) * sizeof(float);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0;
    uint32_t checked_elements = 0;
    float global_max_abs_diff = 0.0f;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &input_dev, input_bytes) < 0 ||
        vc4_m2_malloc(program, &output_dev, output_bytes) < 0)
        panic("maxpool2d_2x2 device allocation failed");

    vc4_dim3 block = vc4_m2_dim3(MAXPOOL2D_2X2_LANES, 1u, 1u);
    printk("MAXPOOL2D_2X2_RUNTIME_SETUP max_width=%d max_height=%d allocations=1\n",
           MAXPOOL2D_2X2_MAX_WIDTH, MAXPOOL2D_2X2_MAX_HEIGHT);

    for (uint32_t case_id = 0; case_id < MAXPOOL2D_2X2_CASES; case_id++) {
        const struct maxpool2d_2x2_case *tc = &maxpool2d_2x2_cases[case_id];
        const uint32_t out_w = ceil_div2_u32(tc->width);
        const uint32_t out_h = ceil_div2_u32(tc->height);
        const uint32_t total_outputs = out_w * out_h;
        uint32_t mismatches = 0, case_sentinel = 0;
        float max_abs_diff = 0.0f;
        int checksum = 0;

        fill_case(tc->width, tc->height, tc->pattern);
        vc4_dim3 grid = vc4_m2_dim3(out_h, 1u, 1u);

        if (vc4_m2_copy_htod(program, input_dev, input_values, input_bytes) < 0 ||
            vc4_m2_copy_htod(program, output_dev, output_values, output_bytes) < 0 ||
            maxpool2d_2x2_launch(program, grid, block, input_dev, output_dev, tc->width,
                                 tc->height) < 0 ||
            vc4_m2_copy_dtoh(program, output_values, output_dev, output_bytes) < 0) {
            launch_failures++;
            printk("MAXPOOL2D_2X2_CASE case=%d width=%d height=%d out_w=%d out_h=%d launch=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)tc->width, (int)tc->height, (int)out_w, (int)out_h,
                   (int)(case_id + 1u));
            continue;
        }

        for (uint32_t i = 0; i < total_outputs; i++) {
            const float diff = output_values[i] - expected_values[i];
            const float adiff = abs_f32(diff);
            checksum += (int)(output_values[i] * 1024.0f);
            checked_elements++;
            if (adiff > max_abs_diff)
                max_abs_diff = adiff;
            if (output_values[i] != expected_values[i]) {
                if (mismatches < 8u) {
                    const uint32_t oy = out_w == 0u ? 0u : i / out_w;
                    const uint32_t ox = out_w == 0u ? 0u : i - oy * out_w;
                    printk("ERROR: case=%d ox=%d oy=%d gpu=%f cpu=%f diff=%f\n",
                           (int)case_id, (int)ox, (int)oy, output_values[i],
                           expected_values[i], diff);
                }
                mismatches++;
            }
        }

        for (uint32_t i = total_outputs;
             i < MAXPOOL2D_2X2_MAX_OUTPUTS + MAXPOOL2D_2X2_GUARD_WORDS; i++) {
            if (output_values[i] != MAXPOOL2D_2X2_SENTINEL)
                case_sentinel++;
        }

        if (max_abs_diff > global_max_abs_diff)
            global_max_abs_diff = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checksum_accum += checksum;

        printk("MAXPOOL2D_2X2_CASE case=%d width=%d height=%d out_w=%d out_h=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=1\n",
               (int)case_id, (int)tc->width, (int)tc->height, (int)out_w, (int)out_h,
               (int)total_outputs, (int)mismatches, (int)case_sentinel, checksum,
               max_abs_diff, (int)(case_id + 1u));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u &&
                          launch_failures == 0u && checked_elements > 0u &&
                          global_max_abs_diff == 0.0f)
                             ? "PASS"
                             : "FAIL";
    printk("VC4_TEST_RESULT name=maxpool2d_2x2 status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_width=%d max_height=%d checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, MAXPOOL2D_2X2_CASES, (int)checked_elements, (int)total_mismatches,
           (int)sentinel_mismatches, (int)launch_failures, MAXPOOL2D_2X2_ACTIVE_QPUS,
           MAXPOOL2D_2X2_LANES, MAXPOOL2D_2X2_MAX_WIDTH, MAXPOOL2D_2X2_MAX_HEIGHT,
           checksum_accum, global_max_abs_diff, 1, MAXPOOL2D_2X2_CASES, elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
