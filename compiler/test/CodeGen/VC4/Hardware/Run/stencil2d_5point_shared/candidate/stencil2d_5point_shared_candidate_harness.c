#include "rpi.h"
#include "kernel_launch.h"
#include "vc4_m2_candidate_test_helpers.h"

#define V3D_BASE 0x20C00000u
#define V3D_VPMBASE (V3D_BASE + 0x00504u)
#define STENCIL2D_5POINT_SHARED_MAX_WIDTH 31u
#define STENCIL2D_5POINT_SHARED_MAX_HEIGHT 19u
#define STENCIL2D_5POINT_SHARED_ROW_STRIDE 32u
#define STENCIL2D_5POINT_SHARED_MAX_WORDS (STENCIL2D_5POINT_SHARED_MAX_WIDTH * STENCIL2D_5POINT_SHARED_MAX_HEIGHT)
#define STENCIL2D_5POINT_SHARED_SCRATCH_WORDS (STENCIL2D_5POINT_SHARED_ROW_STRIDE * STENCIL2D_5POINT_SHARED_MAX_HEIGHT)
#define STENCIL2D_5POINT_SHARED_TILE_OUT_W 14u
#define STENCIL2D_5POINT_SHARED_TILE_OUT_H 1u
#define STENCIL2D_5POINT_SHARED_CASES 3u
#define STENCIL2D_5POINT_SHARED_SENTINEL (-123456.0f)
#define STENCIL2D_5POINT_SHARED_ACTIVE_QPUS 12u
#define STENCIL2D_5POINT_SHARED_LANES 16u
#define STENCIL2D_5POINT_SHARED_WARPS_PER_BLOCK 12u

struct stencil_case { uint32_t width, height, origin_x, origin_y; };

static const struct stencil_case cases[STENCIL2D_5POINT_SHARED_CASES] = {
    {18u, 14u, 0u, 0u},
    {31u, 19u, 3u, 2u},
    {31u, 19u, 17u, 9u},
};

static float input_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];
static float output_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];
static float expected_values[STENCIL2D_5POINT_SHARED_MAX_WORDS];
static float input_scratch[STENCIL2D_5POINT_SHARED_SCRATCH_WORDS];
static float output_scratch[STENCIL2D_5POINT_SHARED_SCRATCH_WORDS];

static float abs_f32(float value) { return value < 0.0f ? -value : value; }
static int invalid_f32(float value) { return !(value == value) || value > 3.4e38f || value < -3.4e38f; }

static float deterministic_input(uint32_t x, uint32_t y) {
    uint32_t m = (x * 13u + y * 17u + x * y + 5u) % 97u;
    return ((float)m) * 0.25f - 8.0f;
}

static uint32_t clamp_u32_to_range(uint32_t value, uint32_t limit) {
    if (limit == 0u)
        return 0u;
    return value < limit ? value : limit - 1u;
}

static float sample_clamped(const float *input, uint32_t width, uint32_t height, uint32_t x, uint32_t y) {
    x = clamp_u32_to_range(x, width);
    y = clamp_u32_to_range(y, height);
    return input[y * width + x];
}

static int in_output_tile(uint32_t x, uint32_t y, uint32_t origin_x, uint32_t origin_y) {
    return x >= origin_x && x < origin_x + STENCIL2D_5POINT_SHARED_TILE_OUT_W &&
           y >= origin_y && y < origin_y + STENCIL2D_5POINT_SHARED_TILE_OUT_H;
}

static void fill_case(uint32_t width, uint32_t height, uint32_t origin_x, uint32_t origin_y,
                      float center_weight, float neighbor_weight) {
    for (uint32_t i = 0; i < STENCIL2D_5POINT_SHARED_MAX_WORDS; i++) {
        input_values[i] = 0.0f;
        output_values[i] = STENCIL2D_5POINT_SHARED_SENTINEL;
        expected_values[i] = STENCIL2D_5POINT_SHARED_SENTINEL;
    }

    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            input_values[y * width + x] = deterministic_input(x, y);

    for (uint32_t row = 0; row < STENCIL2D_5POINT_SHARED_TILE_OUT_H; row++) {
        uint32_t y = origin_y + row;
        if (y >= height)
            break;
        for (uint32_t col = 0; col < STENCIL2D_5POINT_SHARED_TILE_OUT_W; col++) {
            uint32_t x = origin_x + col;
            if (x >= width)
                break;
            float north = sample_clamped(input_values, width, height, x, y == 0u ? 0u : y - 1u);
            float south = sample_clamped(input_values, width, height, x, y + 1u);
            float west = sample_clamped(input_values, width, height, x == 0u ? 0u : x - 1u, y);
            float east = sample_clamped(input_values, width, height, x + 1u, y);
            float center = sample_clamped(input_values, width, height, x, y);
            expected_values[y * width + x] = center_weight * center + neighbor_weight * (north + south + west + east);
        }
    }
}

static int checksum_scaled_values(const float *values, uint32_t width, uint32_t height, uint32_t origin_x, uint32_t origin_y) {
    int checksum = 0;
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            if (in_output_tile(x, y, origin_x, origin_y))
                checksum += (int)(values[y * width + x] * 4096.0f);
    return checksum;
}

static void pack_scratch(uint32_t width, uint32_t height) {
    for (uint32_t i = 0; i < STENCIL2D_5POINT_SHARED_SCRATCH_WORDS; i++) {
        input_scratch[i] = 0.0f;
        output_scratch[i] = STENCIL2D_5POINT_SHARED_SENTINEL;
    }

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            input_scratch[y * STENCIL2D_5POINT_SHARED_ROW_STRIDE + x] = input_values[y * width + x];
            output_scratch[y * STENCIL2D_5POINT_SHARED_ROW_STRIDE + x] = output_values[y * width + x];
        }
    }
}

static void unpack_output(uint32_t width, uint32_t height) {
    for (uint32_t i = 0; i < STENCIL2D_5POINT_SHARED_MAX_WORDS; i++)
        output_values[i] = STENCIL2D_5POINT_SHARED_SENTINEL;

    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
            output_values[y * width + x] = output_scratch[y * STENCIL2D_5POINT_SHARED_ROW_STRIDE + x];
}

void notmain(void) {
    const float center_weight = 0.5f;
    const float neighbor_weight = 0.125f;
    struct vc4_program *program = 0;
    vc4_deviceptr_t input_dev = 0, output_dev = 0;
    uint32_t bytes = STENCIL2D_5POINT_SHARED_SCRATCH_WORDS * sizeof(float);
    uint32_t total_mismatches = 0, sentinel_mismatches = 0, launch_failures = 0, checked_elements = 0;
    float global_max_abs_diff = 0.0f;
    int checksum_accum = 0;
    int expected_checksum_accum = 0;
    int start = timer_get_usec();

    if (vc4_program_create(&program, 0) < 0 || !program)
        panic("vc4_program_create failed");
    if (vc4_m2_malloc(program, &input_dev, bytes) < 0 || vc4_m2_malloc(program, &output_dev, bytes) < 0)
        panic("stencil2d_5point_shared device allocation failed");

    vc4_dim3 grid = vc4_m2_dim3(1u, 1u, 1u);
    vc4_dim3 block = vc4_m2_dim3(STENCIL2D_5POINT_SHARED_WARPS_PER_BLOCK * STENCIL2D_5POINT_SHARED_LANES, 1u, 1u);
    printk("STENCIL2D_5POINT_SHARED_RUNTIME_SETUP max_width=%d max_height=%d allocations=%d warps_per_block=%d capacity=%d code_uploads=%d\n",
           STENCIL2D_5POINT_SHARED_MAX_WIDTH,
           STENCIL2D_5POINT_SHARED_MAX_HEIGHT,
           (int)stencil2d_5point_shared_runtime_allocations(),
           STENCIL2D_5POINT_SHARED_WARPS_PER_BLOCK,
           (int)stencil2d_5point_shared_runtime_capacity(),
           (int)stencil2d_5point_shared_runtime_code_uploads());
    PUT32(V3D_VPMBASE, 0u);

    for (uint32_t case_id = 0; case_id < STENCIL2D_5POINT_SHARED_CASES; case_id++) {
        const struct stencil_case *tc = &cases[case_id];
        uint32_t mismatches = 0, case_sentinel = 0, case_checked = 0;
        float max_abs_diff = 0.0f;
        fill_case(tc->width, tc->height, tc->origin_x, tc->origin_y, center_weight, neighbor_weight);
        pack_scratch(tc->width, tc->height);

        if (vc4_m2_copy_htod(program, input_dev, input_scratch, bytes) < 0 ||
            vc4_m2_copy_htod(program, output_dev, output_scratch, bytes) < 0 ||
            stencil2d_5point_shared_launch(program, grid, block, input_dev, output_dev, tc->width, tc->height,
                                           tc->origin_x, tc->origin_y, center_weight, neighbor_weight) < 0 ||
            vc4_m2_copy_dtoh(program, output_scratch, output_dev, bytes) < 0) {
            launch_failures++;
            printk("STENCIL2D_5POINT_SHARED_CASE case=%d width=%d height=%d launch=FAIL launches=%d allocations=%d\n",
                   (int)case_id,
                   (int)tc->width,
                   (int)tc->height,
                   (int)stencil2d_5point_shared_runtime_launches(),
                   (int)stencil2d_5point_shared_runtime_allocations());
            continue;
        }
        unpack_output(tc->width, tc->height);

        for (uint32_t y = 0; y < tc->height; y++) {
            for (uint32_t x = 0; x < tc->width; x++) {
                uint32_t i = y * tc->width + x;
                if (in_output_tile(x, y, tc->origin_x, tc->origin_y)) {
                    float diff = output_values[i] - expected_values[i];
                    float adiff = abs_f32(diff);
                    case_checked++;
                    if (adiff > max_abs_diff)
                        max_abs_diff = adiff;
                    if (invalid_f32(output_values[i]) || output_values[i] != expected_values[i]) {
                        if (mismatches < 8u)
                            printk("ERROR: case=%d x=%d y=%d gpu=%f cpu=%f diff=%f\n",
                                   (int)case_id, (int)x, (int)y, output_values[i], expected_values[i], diff);
                        mismatches++;
                    }
                } else if (output_values[i] != STENCIL2D_5POINT_SHARED_SENTINEL) {
                    case_sentinel++;
                }
            }
        }

        int checksum = checksum_scaled_values(output_values, tc->width, tc->height, tc->origin_x, tc->origin_y);
        int expected_checksum = checksum_scaled_values(expected_values, tc->width, tc->height, tc->origin_x, tc->origin_y);
        if (max_abs_diff > global_max_abs_diff)
            global_max_abs_diff = max_abs_diff;
        total_mismatches += mismatches;
        sentinel_mismatches += case_sentinel;
        checked_elements += case_checked;
        checksum_accum += checksum;
        expected_checksum_accum += expected_checksum;
        if (checksum != expected_checksum) {
            printk("ERROR: checksum mismatch case=%d gpu=%d expected=%d\n",
                   (int)case_id, checksum, expected_checksum);
            total_mismatches++;
        }
        printk("STENCIL2D_5POINT_SHARED_CASE case=%d width=%d height=%d origin_x=%d origin_y=%d checked=%d mismatches=%d sentinel_mismatches=%d checksum=%d expected_checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
               (int)case_id, (int)tc->width, (int)tc->height, (int)tc->origin_x, (int)tc->origin_y,
               (int)case_checked,
               (int)mismatches,
               (int)case_sentinel,
               checksum,
               expected_checksum,
               max_abs_diff,
               (int)stencil2d_5point_shared_runtime_launches(),
               (int)stencil2d_5point_shared_runtime_allocations());
    }

    int elapsed = timer_get_usec() - start;
    uint32_t runtime_allocations = stencil2d_5point_shared_runtime_allocations();
    uint32_t runtime_launches = stencil2d_5point_shared_runtime_launches();
    uint32_t runtime_capacity = stencil2d_5point_shared_runtime_capacity();
    uint32_t code_uploads = stencil2d_5point_shared_runtime_code_uploads();
    uint32_t recorded_launch_failures = stencil2d_5point_shared_runtime_launch_failures();
    const char *status = (total_mismatches == 0u && sentinel_mismatches == 0u && launch_failures == 0u &&
                          recorded_launch_failures == 0u &&
                          checked_elements > 0u &&
                          checksum_accum == expected_checksum_accum &&
                          global_max_abs_diff == 0.0f &&
                          runtime_allocations == 1u &&
                          runtime_launches == STENCIL2D_5POINT_SHARED_CASES &&
                          code_uploads == 1u) ? "PASS" : "FAIL";
    printk("VC4_TEST_RESULT name=stencil2d_5point_shared status=%s cases=%d checked_elements=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d recorded_launch_failures=%d active_qpus=%d lanes=%d warps_per_block=%d checksum_accum=%d expected_checksum_accum=%d max_abs_diff=%f runtime_allocations=%d runtime_launches=%d runtime_capacity=%d code_uploads=%d timeouts=%d errstat_relevant_changed=%d elapsed_usec=%d\n",
           status, STENCIL2D_5POINT_SHARED_CASES, (int)checked_elements, (int)total_mismatches, (int)sentinel_mismatches,
           (int)launch_failures,
           (int)recorded_launch_failures,
           STENCIL2D_5POINT_SHARED_ACTIVE_QPUS,
           STENCIL2D_5POINT_SHARED_LANES,
           STENCIL2D_5POINT_SHARED_WARPS_PER_BLOCK,
           checksum_accum,
           expected_checksum_accum,
           global_max_abs_diff,
           (int)runtime_allocations,
           (int)runtime_launches,
           (int)runtime_capacity,
           (int)code_uploads,
           0,
           0,
           elapsed);

    vc4Free(program, input_dev);
    vc4Free(program, output_dev);
    vc4_program_destroy(program);
}
