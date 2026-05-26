#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define IMAGE_SOBEL_3X3_CASES 6u
#define IMAGE_SOBEL_3X3_MAX_WIDTH 16u
#define IMAGE_SOBEL_3X3_MAX_HEIGHT 16u
#define IMAGE_SOBEL_3X3_MAX_PIXELS (IMAGE_SOBEL_3X3_MAX_WIDTH * IMAGE_SOBEL_3X3_MAX_HEIGHT)
#define IMAGE_SOBEL_3X3_GUARD_WORDS 32u
#define IMAGE_SOBEL_3X3_SENTINEL 0xdeadbeefu
#define IMAGE_SOBEL_3X3_ACTIVE_QPUS 12u
#define IMAGE_SOBEL_3X3_LANES 16u

struct image_sobel_3x3_case { uint32_t width, height, pattern; };

static const struct image_sobel_3x3_case image_sobel_3x3_cases[IMAGE_SOBEL_3X3_CASES] = {
    {1u, 1u, 0u}, {2u, 3u, 1u}, {7u, 5u, 2u},
    {16u, 16u, 3u}, {15u, 9u, 4u}, {16u, 12u, 5u},
};

static uint32_t input_values[IMAGE_SOBEL_3X3_MAX_PIXELS];
static uint32_t output_values[IMAGE_SOBEL_3X3_MAX_PIXELS + IMAGE_SOBEL_3X3_GUARD_WORDS];
static uint32_t expected_values[IMAGE_SOBEL_3X3_MAX_PIXELS];

static int abs_i32(int value) { return value < 0 ? -value : value; }

static uint32_t pattern_value(uint32_t pattern, uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    switch (pattern) {
    case 0u: return (x * 13u + y * 17u + 7u) & 255u;
    case 1u: return ((x + y) & 1u) ? 255u : 0u;
    case 2u:
        if (x == width / 2u && y == height / 2u) return 255u;
        if ((x == 1u && y + 2u == height) || (y == 1u && x + 2u == width)) return 96u;
        return (x * 5u + y * 3u) & 31u;
    case 3u: return (x * 7u + y * 11u) & 255u;
    case 4u: return (((x / 2u) + (y / 3u)) & 1u) ? 200u : 20u;
    default: return (x * 37u + y * 19u + x * y + 23u) & 255u;
    }
}

static uint32_t clamp_pixel(const uint32_t *input, uint32_t width, uint32_t height, int x, int y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if ((uint32_t)x >= width) x = (int)width - 1;
    if ((uint32_t)y >= height) y = (int)height - 1;
    return input[(uint32_t)y * width + (uint32_t)x] & 255u;
}

static uint32_t sobel_ref_pixel(const uint32_t *input, uint32_t width, uint32_t height, uint32_t x, uint32_t y) {
    int ix = (int)x, iy = (int)y;
    int p00 = (int)clamp_pixel(input, width, height, ix - 1, iy - 1);
    int p01 = (int)clamp_pixel(input, width, height, ix, iy - 1);
    int p02 = (int)clamp_pixel(input, width, height, ix + 1, iy - 1);
    int p10 = (int)clamp_pixel(input, width, height, ix - 1, iy);
    int p12 = (int)clamp_pixel(input, width, height, ix + 1, iy);
    int p20 = (int)clamp_pixel(input, width, height, ix - 1, iy + 1);
    int p21 = (int)clamp_pixel(input, width, height, ix, iy + 1);
    int p22 = (int)clamp_pixel(input, width, height, ix + 1, iy + 1);
    int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
    int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
    int mag = abs_i32(gx) + abs_i32(gy);
    return mag > 255 ? 255u : (uint32_t)mag;
}

static void fill_case(uint32_t width, uint32_t height, uint32_t pattern) {
    for (uint32_t i = 0; i < IMAGE_SOBEL_3X3_MAX_PIXELS; i++) {
        input_values[i] = 0u;
        expected_values[i] = 0u;
    }
    for (uint32_t i = 0; i < IMAGE_SOBEL_3X3_MAX_PIXELS + IMAGE_SOBEL_3X3_GUARD_WORDS; i++)
        output_values[i] = IMAGE_SOBEL_3X3_SENTINEL;
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            input_values[y * width + x] = pattern_value(pattern, x, y, width, height);
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            expected_values[y * width + x] = sobel_ref_pixel(input_values, width, height, x, y);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0, output_dev = 0;
    uint32_t input_bytes = IMAGE_SOBEL_3X3_MAX_PIXELS * sizeof(uint32_t);
    uint32_t output_bytes = (IMAGE_SOBEL_3X3_MAX_PIXELS + IMAGE_SOBEL_3X3_GUARD_WORDS) * sizeof(uint32_t);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0, checked_elements = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &input_dev, input_bytes) < 0 || vc4_m2_malloc(program, &output_dev, output_bytes) < 0)
        panic("image_sobel_3x3 device allocation failed");

    vc4_dim3 block = vc4_m2_dim3(IMAGE_SOBEL_3X3_LANES, 1u, 1u);
    printk("IMAGE_SOBEL_3X3_RUNTIME_SETUP max_width=%d max_height=%d allocations=1\n",
           IMAGE_SOBEL_3X3_MAX_WIDTH, IMAGE_SOBEL_3X3_MAX_HEIGHT);

    for (uint32_t case_id = 0; case_id < IMAGE_SOBEL_3X3_CASES; case_id++) {
        const struct image_sobel_3x3_case *tc = &image_sobel_3x3_cases[case_id];
        uint32_t total = tc->width * tc->height;
        uint32_t mismatches = 0, case_sentinel = 0;
        int checksum = 0;
        fill_case(tc->width, tc->height, tc->pattern);
        vc4_dim3 grid = vc4_m2_dim3(tc->height, 1u, 1u);

        if (vc4_m2_copy_htod(program, input_dev, input_values, input_bytes) < 0 ||
            vc4_m2_copy_htod(program, output_dev, output_values, output_bytes) < 0 ||
            image_sobel_3x3_launch(program, grid, block, input_dev, output_dev, tc->width, tc->height, total) < 0 ||
            vc4_m2_copy_dtoh(program, output_values, output_dev, output_bytes) < 0) {
            launch_failures++;
            printk("IMAGE_SOBEL_3X3_CASE case=%d width=%d height=%d launch=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)tc->width, (int)tc->height, (int)(case_id + 1u));
            continue;
        }

        for (uint32_t i = 0; i < total; i++) {
            checksum += (int)output_values[i];
            checked_elements++;
            if (output_values[i] != expected_values[i]) {
                if (mismatches < 8u) {
                    uint32_t y = i / tc->width;
                    uint32_t x = i - y * tc->width;
                    printk("ERROR: case=%d x=%d y=%d gpu=%x cpu=%x\n",
                           (int)case_id, (int)x, (int)y, output_values[i], expected_values[i]);
                }
                mismatches++;
            }
        }
        for (uint32_t i = 0; i < IMAGE_SOBEL_3X3_GUARD_WORDS; i++)
            if (output_values[total + i] != IMAGE_SOBEL_3X3_SENTINEL)
                case_sentinel++;

        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checksum_accum += checksum;
        printk("IMAGE_SOBEL_3X3_CASE case=%d width=%d height=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d launches=%d allocations=1\n",
               (int)case_id, (int)tc->width, (int)tc->height, (int)total, (int)mismatches,
               (int)case_sentinel, checksum, (int)(case_id + 1u));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u && launch_failures == 0u && checked_elements > 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=image_sobel_3x3 status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_width=%d max_height=%d runtime_allocations=%d runtime_launches=%d checksum_accum=%d elapsed_usec=%d\n",
           status, IMAGE_SOBEL_3X3_CASES, (int)checked_elements, (int)total_mismatches,
           (int)sentinel_mismatches, (int)launch_failures, IMAGE_SOBEL_3X3_ACTIVE_QPUS,
           IMAGE_SOBEL_3X3_LANES, IMAGE_SOBEL_3X3_MAX_WIDTH, IMAGE_SOBEL_3X3_MAX_HEIGHT,
           1, IMAGE_SOBEL_3X3_CASES, checksum_accum, elapsed);
    vc4Free(program, input_dev);
    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
