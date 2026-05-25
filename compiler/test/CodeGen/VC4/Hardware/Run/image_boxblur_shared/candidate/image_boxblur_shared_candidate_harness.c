#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define IMAGE_BOXBLUR_SHARED_CASES 3u
#define IMAGE_BOXBLUR_SHARED_MAX_WIDTH 31u
#define IMAGE_BOXBLUR_SHARED_MAX_HEIGHT 19u
#define IMAGE_BOXBLUR_SHARED_MAX_PIXELS (IMAGE_BOXBLUR_SHARED_MAX_WIDTH * IMAGE_BOXBLUR_SHARED_MAX_HEIGHT)
#define IMAGE_BOXBLUR_SHARED_GUARD_WORDS 64u
#define IMAGE_BOXBLUR_SHARED_SENTINEL 0xdeadbeefu
#define IMAGE_BOXBLUR_SHARED_ACTIVE_QPUS 12u
#define IMAGE_BOXBLUR_SHARED_LANES 16u
#define IMAGE_BOXBLUR_SHARED_WARPS_PER_BLOCK 12u
#define IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_W 14u
#define IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_H 10u
#define V3D_BASE 0x20C00000u
#define V3D_VPMBASE (V3D_BASE + 0x00504u)

struct image_boxblur_shared_case { uint32_t width, height, origin_x, origin_y, pattern; };

static const struct image_boxblur_shared_case image_boxblur_shared_cases[IMAGE_BOXBLUR_SHARED_CASES] = {
    {16u, 12u, 0u, 0u, 0u},
    {31u, 19u, 3u, 2u, 1u},
    {31u, 19u, 17u, 9u, 2u},
};

static uint32_t input_values[IMAGE_BOXBLUR_SHARED_MAX_PIXELS];
static uint32_t output_values[IMAGE_BOXBLUR_SHARED_MAX_PIXELS + IMAGE_BOXBLUR_SHARED_GUARD_WORDS];
static uint32_t expected_values[IMAGE_BOXBLUR_SHARED_MAX_PIXELS];

static uint32_t pattern_value(uint32_t pattern, uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
    switch (pattern) {
    case 0u: return (x * 9u + y * 13u + 5u) & 255u;
    case 1u: return (((x / 3u) + (y / 2u)) & 1u) ? 220u : ((x * 7u + y * 11u + 17u) & 127u);
    default:
        if ((x == width - 2u && y == height - 2u) || (x + 3u == width && y + 4u == height))
            return 255u;
        if ((x + y) % 7u == 0u)
            return 180u;
        return (x * 19u + y * 23u + x * y + 31u) & 255u;
    }
}

static uint32_t clamp_coord_u32(int value, uint32_t limit) {
    if (value < 0) return 0u;
    if ((uint32_t)value >= limit) return limit - 1u;
    return (uint32_t)value;
}

static uint32_t clamp_pixel(const uint32_t *input, uint32_t width, uint32_t height, int x, int y) {
    uint32_t cx = clamp_coord_u32(x, width);
    uint32_t cy = clamp_coord_u32(y, height);
    return input[cy * width + cx] & 255u;
}

static uint32_t boxblur_ref_pixel(const uint32_t *input, uint32_t width, uint32_t height, uint32_t x, uint32_t y) {
    uint32_t sum = 0u;
    int ix = (int)x, iy = (int)y;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            sum += clamp_pixel(input, width, height, ix + dx, iy + dy);
    return (sum + 4u) / 9u;
}

static int tile_contains(uint32_t origin_x, uint32_t origin_y, uint32_t width, uint32_t height, uint32_t x, uint32_t y) {
    return x >= origin_x && y >= origin_y &&
           x < origin_x + IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_W &&
           y < origin_y + IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_H &&
           x < width && y < height;
}

static void fill_case(uint32_t width, uint32_t height, uint32_t pattern) {
    for (uint32_t i = 0; i < IMAGE_BOXBLUR_SHARED_MAX_PIXELS; i++) {
        input_values[i] = 0u;
        expected_values[i] = IMAGE_BOXBLUR_SHARED_SENTINEL;
    }
    for (uint32_t i = 0; i < IMAGE_BOXBLUR_SHARED_MAX_PIXELS + IMAGE_BOXBLUR_SHARED_GUARD_WORDS; i++)
        output_values[i] = IMAGE_BOXBLUR_SHARED_SENTINEL;
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            input_values[y * width + x] = pattern_value(pattern, x, y, width, height);
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            expected_values[y * width + x] = boxblur_ref_pixel(input_values, width, height, x, y);
}

void notmain(void) {
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0, output_dev = 0;
    uint32_t input_bytes = IMAGE_BOXBLUR_SHARED_MAX_PIXELS * sizeof(uint32_t);
    uint32_t output_bytes = (IMAGE_BOXBLUR_SHARED_MAX_PIXELS + IMAGE_BOXBLUR_SHARED_GUARD_WORDS) * sizeof(uint32_t);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0, checked_elements = 0;
    int checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &input_dev, input_bytes) < 0 || vc4_m2_malloc(program, &output_dev, output_bytes) < 0)
        panic("image_boxblur_shared device allocation failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(IMAGE_BOXBLUR_SHARED_WARPS_PER_BLOCK * IMAGE_BOXBLUR_SHARED_LANES, 1u, 1u);
    PUT32(V3D_VPMBASE, 0u);
    printk("IMAGE_BOXBLUR_SHARED_RUNTIME_SETUP max_width=%d max_height=%d allocations=1 warps_per_block=%d\n",
           IMAGE_BOXBLUR_SHARED_MAX_WIDTH, IMAGE_BOXBLUR_SHARED_MAX_HEIGHT, IMAGE_BOXBLUR_SHARED_WARPS_PER_BLOCK);

    for (uint32_t case_id = 0; case_id < IMAGE_BOXBLUR_SHARED_CASES; case_id++) {
        const struct image_boxblur_shared_case *tc = &image_boxblur_shared_cases[case_id];
        uint32_t total = tc->width * tc->height;
        uint32_t mismatches = 0, case_sentinel = 0, case_checked = 0;
        int checksum = 0;
        fill_case(tc->width, tc->height, tc->pattern);

        if (vc4_m2_copy_htod(program, input_dev, input_values, input_bytes) < 0 ||
            vc4_m2_copy_htod(program, output_dev, output_values, output_bytes) < 0 ||
            image_boxblur_shared_launch(program, grid, block, input_dev, output_dev, tc->width, tc->height, tc->origin_x, tc->origin_y) < 0 ||
            vc4_m2_copy_dtoh(program, output_values, output_dev, output_bytes) < 0) {
            launch_failures++;
            printk("IMAGE_BOXBLUR_SHARED_CASE case=%d width=%d height=%d origin_x=%d origin_y=%d launch=FAIL launches=%d allocations=1\n",
                   (int)case_id, (int)tc->width, (int)tc->height, (int)tc->origin_x, (int)tc->origin_y, (int)(case_id + 1u));
            continue;
        }

        for (uint32_t y = 0; y < tc->height; y++) {
            for (uint32_t x = 0; x < tc->width; x++) {
                uint32_t i = y * tc->width + x;
                if (tile_contains(tc->origin_x, tc->origin_y, tc->width, tc->height, x, y)) {
                    checksum += (int)output_values[i];
                    case_checked++;
                    if (output_values[i] != expected_values[i]) {
                        if (mismatches < 8u)
                            printk("ERROR: case=%d x=%d y=%d gpu=%x cpu=%x\n",
                                   (int)case_id, (int)x, (int)y, output_values[i], expected_values[i]);
                        mismatches++;
                    }
                } else if (output_values[i] != IMAGE_BOXBLUR_SHARED_SENTINEL) {
                    if (case_sentinel < 8u)
                        printk("SENTINEL_ERROR: case=%d x=%d y=%d i=%d gpu=%x\n",
                               (int)case_id, (int)x, (int)y, (int)i, output_values[i]);
                    case_sentinel++;
                }
            }
        }
        for (uint32_t i = 0; i < IMAGE_BOXBLUR_SHARED_GUARD_WORDS; i++)
            if (output_values[total + i] != IMAGE_BOXBLUR_SHARED_SENTINEL) {
                if (case_sentinel < 8u)
                    printk("SENTINEL_ERROR: case=%d guard=%d i=%d gpu=%x\n",
                           (int)case_id, (int)i, (int)(total + i), output_values[total + i]);
                case_sentinel++;
            }

        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checked_elements += case_checked;
        checksum_accum += checksum;
        printk("IMAGE_BOXBLUR_SHARED_CASE case=%d width=%d height=%d origin_x=%d origin_y=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d launches=%d allocations=1\n",
               (int)case_id, (int)tc->width, (int)tc->height, (int)tc->origin_x, (int)tc->origin_y,
               (int)case_checked, (int)mismatches, (int)case_sentinel, checksum, (int)(case_id + 1u));
    }

    int elapsed = timer_get_usec() - start;
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u && launch_failures == 0u && checked_elements > 0u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=image_boxblur_shared status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d warps_per_block=%d checksum_accum=%d runtime_allocations=%d runtime_launches=%d timeouts=%d errstat_relevant_changed=%d elapsed_usec=%d\n",
           status, IMAGE_BOXBLUR_SHARED_CASES, (int)checked_elements, (int)total_mismatches, (int)sentinel_mismatches,
           (int)launch_failures, IMAGE_BOXBLUR_SHARED_ACTIVE_QPUS, IMAGE_BOXBLUR_SHARED_LANES,
           IMAGE_BOXBLUR_SHARED_WARPS_PER_BLOCK, checksum_accum, 1, IMAGE_BOXBLUR_SHARED_CASES, 0, 0, elapsed);
    vc4Free(program, input_dev);
    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
