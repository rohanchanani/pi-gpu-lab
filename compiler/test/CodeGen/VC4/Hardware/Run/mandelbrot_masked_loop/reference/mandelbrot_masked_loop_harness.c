#include "mandelbrot_masked_loop_launch.h"

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MANDELBROT_MASKED_LOOP_CASES 5u
#define MANDELBROT_MASKED_LOOP_MAX_WIDTH 32u
#define MANDELBROT_MASKED_LOOP_MAX_HEIGHT 32u
#define MANDELBROT_MASKED_LOOP_MAX_PIXELS (MANDELBROT_MASKED_LOOP_MAX_WIDTH * MANDELBROT_MASKED_LOOP_MAX_HEIGHT)
#define MANDELBROT_MASKED_LOOP_GUARD_WORDS 64u
#define MANDELBROT_MASKED_LOOP_SENTINEL 0xdeadbeefu
#define MANDELBROT_MASKED_LOOP_MAX_ITER_SEEN 64u

struct mandelbrot_masked_loop_case {
uint32_t width;
uint32_t height;
uint32_t max_iter;
float min_x;
float min_y;
float step_x;
float step_y;
};

static const struct mandelbrot_masked_loop_case mandelbrot_cases[MANDELBROT_MASKED_LOOP_CASES] = {
{1u, 1u, 8u, -2.0f, -1.5f, 3.0f, 3.0f},
{16u, 1u, 16u, -2.0f, -1.25f, 0.1875f, 3.0f},
{17u, 3u, 32u, -2.0f, -1.25f, 0.1764705926f, 0.75f},
{31u, 19u, 64u, -2.0f, -1.5f, 0.0967741907f, 0.1578947306f},
{32u, 32u, 64u, -1.75f, -1.25f, 0.078125f, 0.078125f},
};

static uint32_t output_values[MANDELBROT_MASKED_LOOP_MAX_PIXELS + MANDELBROT_MASKED_LOOP_GUARD_WORDS];
static uint32_t expected_values[MANDELBROT_MASKED_LOOP_MAX_PIXELS];

static uint32_t mandelbrot_cpu_pixel(float cx, float cy, uint32_t max_iter) {
float zx = 0.0f;
float zy = 0.0f;
uint32_t iter = 0u;

while (iter < max_iter) {
const float zx2 = zx * zx;
const float zy2 = zy * zy;

if (zx2 + zy2 > 4.0f)
  break;

{
  const float new_zy = 2.0f * zx * zy + cy;
  const float new_zx = zx2 - zy2 + cx;
  zx = new_zx;
  zy = new_zy;
  iter++;
}

}

return iter;
}

static void fill_expected(const struct mandelbrot_masked_loop_case *tc) {
for (uint32_t y = 0; y < tc->height; y++) {
const float cy = tc->min_y + (float)y * tc->step_y;
for (uint32_t x = 0; x < tc->width; x++) {
const float cx = tc->min_x + (float)x * tc->step_x;
expected_values[y * tc->width + x] = mandelbrot_cpu_pixel(cx, cy, tc->max_iter);
}
}
}

static void reset_output(void) {
for (uint32_t i = 0; i < MANDELBROT_MASKED_LOOP_MAX_PIXELS + MANDELBROT_MASKED_LOOP_GUARD_WORDS; i++)
output_values[i] = MANDELBROT_MASKED_LOOP_SENTINEL;
}

int main(void) {
struct mandelbrot_masked_loop_state state;
uint32_t total_mismatches = 0u;
uint32_t total_sentinel_mismatches = 0u;
uint32_t launch_failures = 0u;
unsigned long long checksum_accum = 0ull;
const clock_t start_clock = clock();

if (mandelbrot_masked_loop_prepare(NULL, &state, MANDELBROT_MASKED_LOOP_MAX_WIDTH,
MANDELBROT_MASKED_LOOP_MAX_HEIGHT) != 0) {
printf("VC4_TEST_RESULT name=mandelbrot_masked_loop status=FAIL cases=0 total_mismatches=0 sentinel_mismatches=0 launch_failures=1 active_qpus=12 lanes=16 max_width=32 max_height=32 max_iter=64 runtime_allocations=0 runtime_launches=0 elapsed_usec=0\n");
return 1;
}

printf("MANDELBROT_MASKED_LOOP_RUNTIME_SETUP max_width=%u max_height=%u allocations=%u\n",
state.max_width, state.max_height, state.allocation_count);

for (uint32_t case_id = 0; case_id < MANDELBROT_MASKED_LOOP_CASES; case_id++) {
const struct mandelbrot_masked_loop_case *tc = &mandelbrot_cases[case_id];
const uint32_t logical_pixels = tc->width * tc->height;
uint32_t mismatches = 0u;
uint32_t sentinel_mismatches = 0u;
unsigned long long checksum = 0ull;

reset_output();
fill_expected(tc);

if (mandelbrot_masked_loop_launch(&state, output_values, tc->width, tc->height,
                                  tc->min_x, tc->min_y, tc->step_x, tc->step_y,
                                  tc->max_iter) != 0) {
  launch_failures++;
  printf("MANDELBROT_MASKED_LOOP_CASE case=%u width=%u height=%u max_iter=%u launch=FAIL launches=%u allocations=%u\n",
         case_id, tc->width, tc->height, tc->max_iter, state.launch_count,
         state.allocation_count);
  continue;
}

for (uint32_t i = 0; i < logical_pixels; i++) {
  checksum += (unsigned long long)output_values[i] * (unsigned long long)(i + 1u);

  if (output_values[i] != expected_values[i]) {
    if (mismatches < 8u) {
      const uint32_t y = tc->width == 0u ? 0u : i / tc->width;
      const uint32_t x = tc->width == 0u ? 0u : i % tc->width;
      printf("ERROR: case=%u x=%u y=%u gpu=%u cpu=%u\n",
             case_id, x, y, output_values[i], expected_values[i]);
    }
    mismatches++;
  }
}

for (uint32_t i = logical_pixels; i < MANDELBROT_MASKED_LOOP_MAX_PIXELS + MANDELBROT_MASKED_LOOP_GUARD_WORDS; i++) {
  if (output_values[i] != MANDELBROT_MASKED_LOOP_SENTINEL)
    sentinel_mismatches++;
}

total_mismatches += mismatches;
total_sentinel_mismatches += sentinel_mismatches;
checksum_accum += checksum;

printf("MANDELBROT_MASKED_LOOP_CASE case=%u width=%u height=%u max_iter=%u mismatches=%u sentinel_mismatches=%u checksum=%llu launches=%u allocations=%u\n",
       case_id, tc->width, tc->height, tc->max_iter, mismatches,
       sentinel_mismatches, checksum, state.launch_count, state.allocation_count);

}

mandelbrot_masked_loop_shutdown(&state);

clock_t end_clock = clock();
unsigned long long elapsed_usec = 1ull;
if (end_clock != (clock_t)-1 && start_clock != (clock_t)-1 && end_clock >= start_clock) {
elapsed_usec = (unsigned long long)(end_clock - start_clock) * 1000000ull /
(unsigned long long)CLOCKS_PER_SEC;
if (elapsed_usec == 0ull)
elapsed_usec = 1ull;
}

printf("VC4_TEST_RESULT name=mandelbrot_masked_loop status=%s cases=%u total_mismatches=%u sentinel_mismatches=%u launch_failures=%u active_qpus=%u lanes=%u max_width=%u max_height=%u max_iter=%u checksum_accum=%llu runtime_allocations=%u runtime_launches=%u elapsed_usec=%llu\n",
(total_mismatches == 0u && total_sentinel_mismatches == 0u && launch_failures == 0u) ? "PASS" : "FAIL",
MANDELBROT_MASKED_LOOP_CASES, total_mismatches, total_sentinel_mismatches,
launch_failures, state.active_qpus, state.lanes, state.max_width,
state.max_height, MANDELBROT_MASKED_LOOP_MAX_ITER_SEEN, checksum_accum,
state.allocation_count, state.launch_count, elapsed_usec);

return (total_mismatches == 0u && total_sentinel_mismatches == 0u && launch_failures == 0u) ? 0 : 1;
}
