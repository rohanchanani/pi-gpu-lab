#include "maxpool2d_2x2_launch.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MAXPOOL2D_2X2_CASES 8u
#define MAXPOOL2D_2X2_MAX_WIDTH 31u
#define MAXPOOL2D_2X2_MAX_HEIGHT 19u
#define MAXPOOL2D_2X2_MAX_PIXELS (MAXPOOL2D_2X2_MAX_WIDTH * MAXPOOL2D_2X2_MAX_HEIGHT)
#define MAXPOOL2D_2X2_MAX_OUT_W ((MAXPOOL2D_2X2_MAX_WIDTH + 1u) / 2u)
#define MAXPOOL2D_2X2_MAX_OUT_H ((MAXPOOL2D_2X2_MAX_HEIGHT + 1u) / 2u)
#define MAXPOOL2D_2X2_MAX_OUTPUTS (MAXPOOL2D_2X2_MAX_OUT_W * MAXPOOL2D_2X2_MAX_OUT_H)
#define MAXPOOL2D_2X2_GUARD_WORDS 64u
#define MAXPOOL2D_2X2_SENTINEL (-12345.25f)

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

static uint32_t ceil_div2_u32(uint32_t value) {
return (value + 1u) >> 1;
}

static float abs_f32(float value) {
return value < 0.0f ? -value : value;
}

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

static float ref_one(const float *input,
uint32_t width,
uint32_t height,
uint32_t ox,
uint32_t oy) {
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

for (uint32_t y = 0; y < height; y++) {
for (uint32_t x = 0; x < width; x++) {
input_values[y * width + x] = input_pattern(pattern, x, y);
}
}

const uint32_t out_w = ceil_div2_u32(width);
const uint32_t out_h = ceil_div2_u32(height);

for (uint32_t oy = 0; oy < out_h; oy++) {
for (uint32_t ox = 0; ox < out_w; ox++) {
expected_values[oy * out_w + ox] = ref_one(input_values, width, height, ox, oy);
}
}
}

int main(void) {
struct maxpool2d_2x2_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t launch_failures = 0u;
float global_max_abs_diff = 0.0f;
double checksum_accum = 0.0;
const clock_t start_clock = clock();

if (maxpool2d_2x2_prepare(NULL, &state, MAXPOOL2D_2X2_MAX_WIDTH,
MAXPOOL2D_2X2_MAX_HEIGHT) != 0) {
printf("VC4_TEST_RESULT name=maxpool2d_2x2 status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 max_width=31 max_height=19 max_abs_diff=0.0 runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
return 1;
}

printf("MAXPOOL2D_2X2_RUNTIME_SETUP max_width=%u max_height=%u allocations=%u\n",
state.max_width, state.max_height, state.allocation_count);

for (uint32_t case_id = 0; case_id < MAXPOOL2D_2X2_CASES; case_id++) {
const uint32_t width = maxpool2d_2x2_cases[case_id].width;
const uint32_t height = maxpool2d_2x2_cases[case_id].height;
const uint32_t pattern = maxpool2d_2x2_cases[case_id].pattern;
const uint32_t out_w = ceil_div2_u32(width);
const uint32_t out_h = ceil_div2_u32(height);
const uint32_t total_outputs = out_w * out_h;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
float max_abs_diff = 0.0f;
double checksum = 0.0;

fill_case(width, height, pattern);

if (maxpool2d_2x2_launch(&state, input_values, output_values, width, height) != 0) {
  launch_failures++;
  printf("MAXPOOL2D_2X2_CASE case=%u width=%u height=%u out_w=%u out_h=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, width, height, out_w, out_h, state.launch_count,
         state.allocation_count);
  continue;
}

for (uint32_t i = 0; i < total_outputs; i++) {
  const float diff = output_values[i] - expected_values[i];
  const float adiff = abs_f32(diff);
  checksum += (double)output_values[i] * 1024.0;

  if (adiff > max_abs_diff)
    max_abs_diff = adiff;

  if (adiff > 0.0001f) {
    if (mismatches < 8u) {
      const uint32_t oy = out_w == 0u ? 0u : i / out_w;
      const uint32_t ox = out_w == 0u ? 0u : i - oy * out_w;
      printf("ERROR: case=%u ox=%u oy=%u gpu=%f cpu=%f diff=%f\n",
             case_id, ox, oy, output_values[i], expected_values[i], diff);
    }
    mismatches++;
  }
}

for (uint32_t i = total_outputs; i < MAXPOOL2D_2X2_MAX_OUTPUTS + MAXPOOL2D_2X2_GUARD_WORDS; i++) {
  if (output_values[i] != MAXPOOL2D_2X2_SENTINEL)
    sentinel_mismatches++;
}

if (max_abs_diff > global_max_abs_diff)
  global_max_abs_diff = max_abs_diff;

total_mismatches += mismatches;
total_sentinel_mismatches += sentinel_mismatches;
checksum_accum += checksum;

printf("MAXPOOL2D_2X2_CASE case=%u width=%u height=%u out_w=%u out_h=%u mismatches=%u sentinel_mismatches=%u checksum=%.0f max_abs_diff=%.6f launches=%u allocations=%u\n",
       case_id, width, height, out_w, out_h, mismatches,
       sentinel_mismatches, checksum, max_abs_diff, state.launch_count,
       state.allocation_count);

}

maxpool2d_2x2_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=maxpool2d_2x2 status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u max_width=%u max_height=%u checksum_accum=%.0f max_abs_diff=%.6f runtime_allocations=%u runtime_launches=%u elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= 0.0001f)
? "PASS"
: "FAIL",
MAXPOOL2D_2X2_CASES, total_mismatches, total_sentinel_mismatches,
launch_failures, state.active_qpus, state.lanes, state.max_width,
state.max_height, checksum_accum, global_max_abs_diff,
state.allocation_count, state.launch_count, elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u &&
launch_failures == 0u && global_max_abs_diff <= 0.0001f)
? 0
: 1;
}
