#include "image_sobel_3x3_launch.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define IMAGE_SOBEL_3X3_CASES 6u
#define IMAGE_SOBEL_3X3_MAX_WIDTH 31u
#define IMAGE_SOBEL_3X3_MAX_HEIGHT 19u
#define IMAGE_SOBEL_3X3_MAX_PIXELS (IMAGE_SOBEL_3X3_MAX_WIDTH * IMAGE_SOBEL_3X3_MAX_HEIGHT)
#define IMAGE_SOBEL_3X3_GUARD_WORDS 32u
#define IMAGE_SOBEL_3X3_SENTINEL 0xdeadbeefu

struct image_sobel_3x3_case {
uint32_t width;
uint32_t height;
uint32_t pattern;
};

static const struct image_sobel_3x3_case image_sobel_3x3_cases[IMAGE_SOBEL_3X3_CASES] = {
{1u, 1u, 0u},
{2u, 3u, 1u},
{7u, 5u, 2u},
{16u, 16u, 3u},
{17u, 9u, 4u},
{31u, 19u, 5u},
};

static uint32_t input_values[IMAGE_SOBEL_3X3_MAX_PIXELS];
static uint32_t output_values[IMAGE_SOBEL_3X3_MAX_PIXELS + IMAGE_SOBEL_3X3_GUARD_WORDS];
static uint32_t expected_values[IMAGE_SOBEL_3X3_MAX_PIXELS];

static uint32_t pattern_value(uint32_t pattern,
uint32_t x,
uint32_t y,
uint32_t width,
uint32_t height) {
switch (pattern) {
case 0u:
return (x * 13u + y * 17u + 7u) & 255u;
case 1u:
return ((x + y) & 1u) ? 255u : 0u;
case 2u:
if (x == width / 2u && y == height / 2u)
return 255u;
if ((x == 1u && y + 2u == height) || (y == 1u && x + 2u == width))
return 96u;
return (x * 5u + y * 3u) & 31u;
case 3u:
return (x * 7u + y * 11u) & 255u;
case 4u:
return (((x / 2u) + (y / 3u)) & 1u) ? 200u : 20u;
default:
return (x * 37u + y * 19u + x * y + 23u) & 255u;
}
}

static uint32_t clamp_pixel(const uint32_t *input,
uint32_t width,
uint32_t height,
int x,
int y) {
if (x < 0)
x = 0;
if (y < 0)
y = 0;
if ((uint32_t)x >= width)
x = (int)width - 1;
if ((uint32_t)y >= height)
y = (int)height - 1;
return input[(uint32_t)y * width + (uint32_t)x] & 255u;
}

static int abs_i32(int value) {
return value < 0 ? -value : value;
}

static uint32_t sobel_ref_pixel(const uint32_t *input,
uint32_t width,
uint32_t height,
uint32_t x,
uint32_t y) {
const int ix = (int)x;
const int iy = (int)y;

const int p00 = (int)clamp_pixel(input, width, height, ix - 1, iy - 1);
const int p01 = (int)clamp_pixel(input, width, height, ix, iy - 1);
const int p02 = (int)clamp_pixel(input, width, height, ix + 1, iy - 1);
const int p10 = (int)clamp_pixel(input, width, height, ix - 1, iy);
const int p12 = (int)clamp_pixel(input, width, height, ix + 1, iy);
const int p20 = (int)clamp_pixel(input, width, height, ix - 1, iy + 1);
const int p21 = (int)clamp_pixel(input, width, height, ix, iy + 1);
const int p22 = (int)clamp_pixel(input, width, height, ix + 1, iy + 1);

const int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
const int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
int mag = abs_i32(gx) + abs_i32(gy);

if (mag > 255)
mag = 255;
return (uint32_t)mag;
}

static void fill_case(uint32_t case_id, uint32_t width, uint32_t height, uint32_t pattern) {
for (uint32_t i = 0; i < IMAGE_SOBEL_3X3_MAX_PIXELS; i++)
input_values[i] = 0u;

for (uint32_t y = 0; y < height; y++) {
for (uint32_t x = 0; x < width; x++) {
input_values[y * width + x] = pattern_value(pattern, x, y, width, height);
}
}

for (uint32_t i = 0; i < IMAGE_SOBEL_3X3_MAX_PIXELS + IMAGE_SOBEL_3X3_GUARD_WORDS; i++)
output_values[i] = IMAGE_SOBEL_3X3_SENTINEL;

for (uint32_t y = 0; y < height; y++) {
for (uint32_t x = 0; x < width; x++) {
expected_values[y * width + x] =
sobel_ref_pixel(input_values, width, height, x, y);
}
}

(void)case_id;
}

int main(void) {
struct image_sobel_3x3_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t launch_failures = 0u;
unsigned long long checksum_accum = 0ull;
const clock_t start_clock = clock();

if (image_sobel_3x3_prepare(NULL, &state, IMAGE_SOBEL_3X3_MAX_WIDTH,
IMAGE_SOBEL_3X3_MAX_HEIGHT) != 0) {
printf("VC4_TEST_RESULT name=image_sobel_3x3 status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 max_width=31 max_height=19 runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
return 1;
}

printf("IMAGE_SOBEL_3X3_RUNTIME_SETUP max_width=%u max_height=%u allocations=%u\n",
state.max_width, state.max_height, state.allocation_count);

for (uint32_t case_id = 0; case_id < IMAGE_SOBEL_3X3_CASES; case_id++) {
const uint32_t width = image_sobel_3x3_cases[case_id].width;
const uint32_t height = image_sobel_3x3_cases[case_id].height;
const uint32_t pattern = image_sobel_3x3_cases[case_id].pattern;
const uint32_t total = width * height;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
unsigned long long checksum = 0ull;

fill_case(case_id, width, height, pattern);

if (image_sobel_3x3_launch(&state, input_values, output_values, width, height) != 0) {
  launch_failures++;
  printf("IMAGE_SOBEL_3X3_CASE case=%u width=%u height=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, width, height, state.launch_count, state.allocation_count);
  continue;
}

for (uint32_t i = 0; i < total; i++) {
  checksum += (unsigned long long)output_values[i];
  if (output_values[i] != expected_values[i]) {
    if (mismatches < 8u) {
      const uint32_t y = i / width;
      const uint32_t x = i - y * width;
      printf("ERROR: case=%u x=%u y=%u gpu=0x%x cpu=0x%x\n",
             case_id, x, y, output_values[i], expected_values[i]);
    }
    mismatches++;
  }
}

for (uint32_t i = 0; i < IMAGE_SOBEL_3X3_GUARD_WORDS; i++) {
  if (output_values[total + i] != IMAGE_SOBEL_3X3_SENTINEL)
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

printf("IMAGE_SOBEL_3X3_CASE case=%u width=%u height=%u mismatches=%u sentinel_mismatches=%u checksum=%llu launches=%u allocations=%u\n",
       case_id, width, height, mismatches, sentinel_mismatches, checksum,
       state.launch_count, state.allocation_count);

}

image_sobel_3x3_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=image_sobel_3x3 status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u max_width=%u max_height=%u runtime_allocations=%u runtime_launches=%u checksum_accum=%llu elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u)
? "PASS"
: "FAIL",
IMAGE_SOBEL_3X3_CASES, total_mismatches, total_sentinel_mismatches,
launch_failures, state.active_qpus, state.lanes, state.max_width,
state.max_height, state.allocation_count, state.launch_count,
checksum_accum, elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u)
? 0
: 1;
}
