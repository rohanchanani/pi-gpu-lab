#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define MANDELBROT_MASKED_LOOP_CASES 5u
#define MANDELBROT_MASKED_LOOP_MAX_WIDTH 32u
#define MANDELBROT_MASKED_LOOP_MAX_HEIGHT 32u
#define MANDELBROT_MASKED_LOOP_MAX_PIXELS (MANDELBROT_MASKED_LOOP_MAX_WIDTH * MANDELBROT_MASKED_LOOP_MAX_HEIGHT)
#define MANDELBROT_MASKED_LOOP_GUARD_WORDS 64u
#define MANDELBROT_MASKED_LOOP_SENTINEL 0xdeadbeefu
#define MANDELBROT_MASKED_LOOP_MAX_ITER_SEEN 64u
#define MANDELBROT_MASKED_LOOP_ACTIVE_QPUS 12u
#define MANDELBROT_MASKED_LOOP_LANES 16u

struct mandelbrot_masked_loop_case {
    uint32_t resolution, max_iter;
};

static const struct mandelbrot_masked_loop_case mandelbrot_cases[MANDELBROT_MASKED_LOOP_CASES] = {
    {16u, 8u},
    {16u, 16u},
    {16u, 24u},
    {16u, 32u},
    {16u, 64u},
};

static uint32_t output_values[MANDELBROT_MASKED_LOOP_MAX_PIXELS + MANDELBROT_MASKED_LOOP_GUARD_WORDS];
static uint32_t expected_values[MANDELBROT_MASKED_LOOP_MAX_PIXELS];

static uint32_t mandelbrot_cpu_pixel(float cx, float cy, uint32_t max_iter) {
    float u = 0.0f, v = 0.0f, u2 = 0.0f, v2 = 0.0f;
    uint32_t escaped = 0u;
    for (uint32_t iter = 0; iter < max_iter; iter++) {
        v = 2.0f * v * u + cy;
        u = u2 + cx - v2;
        u2 = u * u;
        v2 = v * v;
        if (u2 + v2 > 4.0f) {
            escaped = 1u;
            break;
        }
    }
    return escaped;
}

static void fill_case(const struct mandelbrot_masked_loop_case *tc) {
    uint32_t width = tc->resolution * 2u;
    uint32_t height = tc->resolution * 2u;
    float inv_resolution = 1.0f / (float)tc->resolution;
    for (uint32_t i = 0; i < MANDELBROT_MASKED_LOOP_MAX_PIXELS + MANDELBROT_MASKED_LOOP_GUARD_WORDS; i++)
        output_values[i] = MANDELBROT_MASKED_LOOP_SENTINEL;
    for (uint32_t y = 0; y < height; y++) {
        float cy = -1.0f + (float)y * inv_resolution;
        for (uint32_t x = 0; x < width; x++) {
            float cx = -1.0f + (float)x * inv_resolution;
            expected_values[y * width + x] = mandelbrot_cpu_pixel(cx, cy, tc->max_iter);
        }
    }
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t output_dev = 0;
    uint32_t output_bytes = (MANDELBROT_MASKED_LOOP_MAX_PIXELS + MANDELBROT_MASKED_LOOP_GUARD_WORDS) * sizeof(uint32_t);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0, checked_elements = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &output_dev, output_bytes) < 0)
        panic("mandelbrot_masked_loop device allocation failed");

    vc4_dim3 block = vc4_m2_dim3(MANDELBROT_MASKED_LOOP_LANES, 1u, 1u);
    printk("MANDELBROT_MASKED_LOOP_RUNTIME_SETUP max_width=%d max_height=%d allocations=1\n",
           MANDELBROT_MASKED_LOOP_MAX_WIDTH, MANDELBROT_MASKED_LOOP_MAX_HEIGHT);

    for (uint32_t case_id = 0; case_id < MANDELBROT_MASKED_LOOP_CASES; case_id++) {
        const struct mandelbrot_masked_loop_case *tc = &mandelbrot_cases[case_id];
        uint32_t width = tc->resolution * 2u;
        uint32_t height = tc->resolution * 2u;
        uint32_t logical_pixels = width * height;
        uint32_t mismatches = 0, case_sentinel = 0;
        int checksum = 0;
        float inv_resolution = 1.0f / (float)tc->resolution;
        printk("MANDELBROT_MASKED_LOOP_CASE_PREP_BEGIN case=%d resolution=%d max_iter=%d\n",
               (int)case_id, (int)tc->resolution, (int)tc->max_iter);
        fill_case(tc);
        printk("MANDELBROT_MASKED_LOOP_CASE_PREP_DONE case=%d\n", (int)case_id);
        vc4_dim3 grid = vc4_m2_dim3(MANDELBROT_MASKED_LOOP_ACTIVE_QPUS, 1u, 1u);

        printk("MANDELBROT_MASKED_LOOP_CASE_BEGIN case=%d resolution=%d max_iter=%d\n",
               (int)case_id, (int)tc->resolution, (int)tc->max_iter);
        if (vc4_m2_copy_htod(program, output_dev, output_values, output_bytes) < 0) {
            launch_failures++;
            printk("MANDELBROT_MASKED_LOOP_CASE case=%d width=%d height=%d resolution=%d max_iter=%d htod=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)width, (int)height, (int)tc->resolution, (int)tc->max_iter, (int)(case_id + 1u));
            continue;
        }
        printk("MANDELBROT_MASKED_LOOP_CASE_HTOD_DONE case=%d\n", (int)case_id);
        if (mandelbrot_masked_loop_launch(program, grid, block, tc->resolution, inv_resolution, tc->max_iter, output_dev) < 0) {
            launch_failures++;
            printk("MANDELBROT_MASKED_LOOP_CASE case=%d width=%d height=%d resolution=%d max_iter=%d launch=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)width, (int)height, (int)tc->resolution, (int)tc->max_iter, (int)(case_id + 1u));
            continue;
        }
        printk("MANDELBROT_MASKED_LOOP_CASE_LAUNCH_DONE case=%d\n", (int)case_id);
        if (vc4_m2_copy_dtoh(program, output_values, output_dev, output_bytes) < 0) {
            launch_failures++;
            printk("MANDELBROT_MASKED_LOOP_CASE case=%d width=%d height=%d resolution=%d max_iter=%d dtoh=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)width, (int)height, (int)tc->resolution, (int)tc->max_iter, (int)(case_id + 1u));
            continue;
        }
        printk("MANDELBROT_MASKED_LOOP_CASE_DTOH_DONE case=%d\n", (int)case_id);

        for (uint32_t i = 0; i < logical_pixels; i++) {
            checksum += (int)(output_values[i] * (i + 1u));
            checked_elements++;
            if (output_values[i] != expected_values[i]) {
                if (mismatches < 32u) {
                    uint32_t y = i / width;
                    uint32_t x = i % width;
                    printk("ERROR: case=%d x=%d y=%d gpu=%d cpu=%d\n",
                           (int)case_id, (int)x, (int)y, (int)output_values[i], (int)expected_values[i]);
                }
                mismatches++;
            }
        }
        for (uint32_t i = logical_pixels; i < MANDELBROT_MASKED_LOOP_MAX_PIXELS + MANDELBROT_MASKED_LOOP_GUARD_WORDS; i++)
            if (output_values[i] != MANDELBROT_MASKED_LOOP_SENTINEL)
                case_sentinel++;

        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checksum_accum += checksum;
        printk("MANDELBROT_MASKED_LOOP_CASE case=%d width=%d height=%d resolution=%d max_iter=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d launches=%d allocations=1\n",
               (int)case_id, (int)width, (int)height, (int)tc->resolution, (int)tc->max_iter, (int)logical_pixels,
               (int)mismatches, (int)case_sentinel, checksum, (int)(case_id + 1u));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u && launch_failures == 0u && checked_elements > 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=mandelbrot_masked_loop status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_width=%d max_height=%d max_iter=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
           status, MANDELBROT_MASKED_LOOP_CASES, (int)checked_elements, (int)total_mismatches,
           (int)sentinel_mismatches, (int)launch_failures, MANDELBROT_MASKED_LOOP_ACTIVE_QPUS,
           MANDELBROT_MASKED_LOOP_LANES, MANDELBROT_MASKED_LOOP_MAX_WIDTH, MANDELBROT_MASKED_LOOP_MAX_HEIGHT,
           MANDELBROT_MASKED_LOOP_MAX_ITER_SEEN, checksum_accum, 1, MANDELBROT_MASKED_LOOP_CASES, elapsed);
    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
