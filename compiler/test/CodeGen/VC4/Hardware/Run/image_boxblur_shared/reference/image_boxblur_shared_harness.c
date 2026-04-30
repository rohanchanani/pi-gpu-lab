#include "image_boxblur_shared_launch.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define IMAGE_BOXBLUR_SHARED_CASES 3u
#define IMAGE_BOXBLUR_SHARED_MAX_WIDTH 31u
#define IMAGE_BOXBLUR_SHARED_MAX_HEIGHT 19u
#define IMAGE_BOXBLUR_SHARED_MAX_PIXELS (IMAGE_BOXBLUR_SHARED_MAX_WIDTH * IMAGE_BOXBLUR_SHARED_MAX_HEIGHT)
#define IMAGE_BOXBLUR_SHARED_GUARD_WORDS 64u
#define IMAGE_BOXBLUR_SHARED_SENTINEL 0xdeadbeefu

struct image_boxblur_shared_case {
uint32_t width;
uint32_t height;
uint32_t origin_x;
uint32_t origin_y;
uint32_t pattern;
};

static const struct image_boxblur_shared_case image_boxblur_shared_cases[IMAGE_BOXBLUR_SHARED_CASES] = {
{16u, 12u, 0u, 0u, 0u},
{31u, 19u, 3u, 2u, 1u},
{31u, 19u, 17u, 9u, 2u},
};

static uint32_t input_values[IMAGE_BOXBLUR_SHARED_MAX_PIXELS];
static uint32_t output_values[IMAGE_BOXBLUR_SHARED_MAX_PIXELS + IMAGE_BOXBLUR_SHARED_GUARD_WORDS];
static uint32_t expected_values[IMAGE_BOXBLUR_SHARED_MAX_PIXELS];

static uint32_t pattern_value(uint32_t pattern,
uint32_t x,
uint32_t y,
uint32_t width,
uint32_t height) {
switch (pattern) {
case 0u:
return (x * 9u + y * 13u + 5u) & 255u;
case 1u:
return (((x / 3u) + (y / 2u)) & 1u) ? 220u : (uint32_t)((x * 7u + y * 11u + 17u) & 127u);
default:
if ((x == width - 2u && y == height - 2u) || (x + 3u == width && y + 4u == height))
return 255u;
if ((x + y) % 7u == 0u)
return 180u;
return (x * 19u + y * 23u + x * y + 31u) & 255u;
}
}

static uint32_t clamp_coord_u32(int value, uint32_t limit) {
if (value < 0)
return 0u;
if ((uint32_t)value >= limit)
return limit - 1u;
return (uint32_t)value;
}

static uint32_t clamp_pixel(const uint32_t *input,
uint32_t width,
uint32_t height,
int x,
int y) {
const uint32_t cx = clamp_coord_u32(x, width);
const uint32_t cy = clamp_coord_u32(y, height);
return input[cy * width + cx] & 255u;
}

static uint32_t boxblur_ref_pixel(const uint32_t *input,
uint32_t width,
uint32_t height,
uint32_t x,
uint32_t y) {
uint32_t sum = 0u;
const int ix = (int)x;
const int iy = (int)y;

for (int dy = -1; dy <= 1; dy++) {
for (int dx = -1; dx <= 1; dx++) {
sum += clamp_pixel(input, width, height, ix + dx, iy + dy);
}
}

return (sum + 4u) / 9u;
}

static void fill_case(uint32_t width, uint32_t height, uint32_t pattern) {
for (uint32_t i = 0; i < IMAGE_BOXBLUR_SHARED_MAX_PIXELS; i++) {
input_values[i] = 0u;
expected_values[i] = IMAGE_BOXBLUR_SHARED_SENTINEL;
}

for (uint32_t i = 0; i < IMAGE_BOXBLUR_SHARED_MAX_PIXELS + IMAGE_BOXBLUR_SHARED_GUARD_WORDS; i++)
output_values[i] = IMAGE_BOXBLUR_SHARED_SENTINEL;

for (uint32_t y = 0; y < height; y++) {
for (uint32_t x = 0; x < width; x++) {
input_values[y * width + x] = pattern_value(pattern, x, y, width, height);
}
}

for (uint32_t y = 0; y < height; y++) {
for (uint32_t x = 0; x < width; x++) {
expected_values[y * width + x] =
boxblur_ref_pixel(input_values, width, height, x, y);
}
}
}

static int tile_contains(uint32_t origin_x,
uint32_t origin_y,
uint32_t width,
uint32_t height,
uint32_t x,
uint32_t y) {
if (x < origin_x || y < origin_y)
return 0;
if (x >= origin_x + IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_W)
return 0;
if (y >= origin_y + IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_H)
return 0;
if (x >= width || y >= height)
return 0;
return 1;
}

int main(void) {
struct image_boxblur_shared_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t launch_failures = 0u;
unsigned long long checksum_accum = 0ull;
const clock_t start_clock = clock();

if (image_boxblur_shared_prepare(NULL, &state, IMAGE_BOXBLUR_SHARED_MAX_WIDTH,
IMAGE_BOXBLUR_SHARED_MAX_HEIGHT) != 0) {
printf("VC4_TEST_RESULT name=image_boxblur_shared status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 warps_per_block=12 runtime_allocations=0 runtime_launches=0 timeouts=0 errstat_relevant_changed=0 elapsed_usec=0\n");
return 1;
}

printf("IMAGE_BOXBLUR_SHARED_RUNTIME_SETUP max_width=%u max_height=%u allocations=%u warps_per_block=%u\n",
state.max_width, state.max_height, state.allocation_count,
state.warps_per_block);

for (uint32_t case_id = 0; case_id < IMAGE_BOXBLUR_SHARED_CASES; case_id++) {
const uint32_t width = image_boxblur_shared_cases[case_id].width;
const uint32_t height = image_boxblur_shared_cases[case_id].height;
const uint32_t origin_x = image_boxblur_shared_cases[case_id].origin_x;
const uint32_t origin_y = image_boxblur_shared_cases[case_id].origin_y;
const uint32_t pattern = image_boxblur_shared_cases[case_id].pattern;
const uint32_t total = width * height;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
unsigned long long checksum = 0ull;

fill_case(width, height, pattern);

if (image_boxblur_shared_launch(&state, input_values, output_values, width,
                                height, origin_x, origin_y) != 0) {
  launch_failures++;
  printf("IMAGE_BOXBLUR_SHARED_CASE case=%u width=%u height=%u origin_x=%u origin_y=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, width, height, origin_x, origin_y, state.launch_count,
         state.allocation_count);
  continue;
}

for (uint32_t y = 0; y < height; y++) {
  for (uint32_t x = 0; x < width; x++) {
    const uint32_t i = y * width + x;
    if (tile_contains(origin_x, origin_y, width, height, x, y)) {
      checksum += (unsigned long long)output_values[i];
      if (output_values[i] != expected_values[i]) {
        if (mismatches < 8u) {
          printf("ERROR: case=%u x=%u y=%u gpu=0x%x cpu=0x%x\n",
                 case_id, x, y, output_values[i], expected_values[i]);
        }
        mismatches++;
      }
    } else if (output_values[i] != IMAGE_BOXBLUR_SHARED_SENTINEL) {
      sentinel_mismatches++;
    }
  }
}

for (uint32_t i = 0; i < IMAGE_BOXBLUR_SHARED_GUARD_WORDS; i++) {
  if (output_values[total + i] != IMAGE_BOXBLUR_SHARED_SENTINEL)
    sentinel_mismatches++;
}

if (mismatches != 0u)
  printf("ERROR: mismatch count case=%u mismatches=%u\n", case_id, mismatches);
if (sentinel_mismatches != 0u)
  printf("ERROR: sentinel mismatch count case=%u sentinel_mismatches=%u\n",
         case_id, sentinel_mismatches);

total_mismatches += mismatches;
total_sentinel_mismatches += sentinel_mismatches;
checksum_accum += checksum;

printf("IMAGE_BOXBLUR_SHARED_CASE case=%u width=%u height=%u origin_x=%u origin_y=%u mismatches=%u sentinel_mismatches=%u checksum=%llu launches=%u allocations=%u\n",
       case_id, width, height, origin_x, origin_y, mismatches,
       sentinel_mismatches, checksum, state.launch_count,
       state.allocation_count);

}

image_boxblur_shared_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=image_boxblur_shared status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u warps_per_block=%u runtime_allocations=%u runtime_launches=%u timeouts=%u errstat_relevant_changed=%u checksum_accum=%llu elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && state.timeouts == 0u &&
state.errstat_relevant_changed == 0u)
? "PASS"
: "FAIL",
IMAGE_BOXBLUR_SHARED_CASES, total_mismatches, total_sentinel_mismatches,
launch_failures, state.active_qpus, state.lanes, state.warps_per_block,
state.allocation_count, state.launch_count, state.timeouts,
state.errstat_relevant_changed, checksum_accum, elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && state.timeouts == 0u &&
state.errstat_relevant_changed == 0u)
? 0
: 1;
}
