#include "rpi.h"
#include "tmu_strided_load_launch.h"

#define TMU_STRIDED_LOAD_EPSILON 0.0001f
#define CHECKSUM_SCALE 4096.0f
#define TMU_STRIDED_LOAD_CASES 11u
#define TMU_STRIDED_LOAD_MAX_N 257u
#define TMU_STRIDED_LOAD_MAX_STRIDE 4u
#define TMU_STRIDED_LOAD_MAX_INPUT_WORDS 1104u
#define TMU_STRIDED_LOAD_BUFFER_N (TMU_STRIDED_LOAD_MAX_N + TMU_STRIDED_LOAD_GUARD_WORDS)

struct tmu_strided_load_case {
uint32_t n;
uint32_t offset;
uint32_t stride;
};

static const struct tmu_strided_load_case cases[TMU_STRIDED_LOAD_CASES] = {
{0u, 0u, 1u},
{1u, 0u, 1u},
{15u, 2u, 1u},
{16u, 3u, 1u},
{17u, 1u, 1u},
{31u, 4u, 2u},
{32u, 5u, 2u},
{33u, 7u, 3u},
{65u, 11u, 4u},
{191u, 13u, 2u},
{257u, 17u, 3u},
};

static float input_values[TMU_STRIDED_LOAD_MAX_INPUT_WORDS];
static float out_values[TMU_STRIDED_LOAD_BUFFER_N];
static float expected_values[TMU_STRIDED_LOAD_BUFFER_N];

static float absf_local(float value)
{
return value < 0.0f ? -value : value;
}

static float make_input_value(uint32_t j)
{
return ((float)((j * 17u + 5u) % 113u) * 0.125f) - 6.0f;
}

static void fill_inputs(void)
{
for (uint32_t i = 0; i < TMU_STRIDED_LOAD_MAX_INPUT_WORDS; i++)
input_values[i] = make_input_value(i);
}

static void fill_output_buffers(void)
{
for (uint32_t i = 0; i < TMU_STRIDED_LOAD_BUFFER_N; i++) {
out_values[i] = TMU_STRIDED_LOAD_SENTINEL;
expected_values[i] = TMU_STRIDED_LOAD_SENTINEL;
}
}

static void run_cpu_reference(
uint32_t n,
uint32_t offset,
uint32_t stride,
float scale,
float bias)
{
for (uint32_t i = 0; i < n; i++)
expected_values[i] = scale * input_values[offset + i * stride] + bias;
}

static void verify_results(
uint32_t case_id,
uint32_t n,
int *mismatches,
float *max_abs_diff)
{
*mismatches = 0;
*max_abs_diff = 0.0f;

for (uint32_t i = 0; i < n; i++) {
    float diff = out_values[i] - expected_values[i];
    float abs_diff = absf_local(diff);
    if (abs_diff > *max_abs_diff)
        *max_abs_diff = abs_diff;

    if (abs_diff > TMU_STRIDED_LOAD_EPSILON) {
        if (*mismatches < 8) {
            printk("ERROR: case=%d i=%d gpu=%f cpu=%f diff=%f\n",
                   (int)case_id,
                   (int)i,
                   out_values[i],
                   expected_values[i],
                   diff);
        }
        (*mismatches)++;
    }
}

}

static int verify_sentinel_tail(uint32_t n)
{
int mismatches = 0;

for (uint32_t i = n; i < n + TMU_STRIDED_LOAD_GUARD_WORDS &&
                     i < TMU_STRIDED_LOAD_BUFFER_N; i++) {
    if (out_values[i] != TMU_STRIDED_LOAD_SENTINEL) {
        if (mismatches < 8) {
            printk("ERROR: sentinel changed i=%d value=%f expected=%f\n",
                   (int)i,
                   out_values[i],
                   TMU_STRIDED_LOAD_SENTINEL);
        }
        mismatches++;
    }
}

return mismatches;

}

static int scaled_checksum(const float *values, uint32_t n)
{
int checksum = 0;
for (uint32_t i = 0; i < n; i++)
checksum += (int)(values[i] * CHECKSUM_SCALE);
return checksum;
}

void notmain(void)
{
struct vc4_runtime rt;
struct tmu_strided_load_state state;
const float scale = -1.75f;
const float bias = 0.5f;

if (vc4_runtime_init(&rt) < 0)
    panic("Failed to initialize VC4 runtime");

uint32_t active_qpus = vc4_runtime_active_qpus(&rt);
uint32_t lane_width = vc4_runtime_lane_width();

if (active_qpus != VC4_RUNTIME_MAX_QPUS)
    panic("Unexpected active QPU count: %d", (int)active_qpus);
if (lane_width != VC4_RUNTIME_LANE_WIDTH)
    panic("Unexpected lane width: %d", (int)lane_width);

if (tmu_strided_load_prepare(&rt,
                             &state,
                             TMU_STRIDED_LOAD_MAX_INPUT_WORDS,
                             TMU_STRIDED_LOAD_MAX_N) < 0)
    panic("tmu_strided_load runtime setup failed");

fill_inputs();

printk("TMU_STRIDED_LOAD_RUNTIME_SETUP max_n=%d max_input_words=%d allocations=%d\n",
       TMU_STRIDED_LOAD_MAX_N,
       TMU_STRIDED_LOAD_MAX_INPUT_WORDS,
       (int)tmu_strided_load_runtime_allocations(&state));

int total_mismatches = 0;
int sentinel_mismatches = 0;
int launch_failures = 0;
int checksum_accum = 0;
float max_abs_diff_overall = 0.0f;

int start = timer_get_usec();

for (uint32_t case_index = 0; case_index < TMU_STRIDED_LOAD_CASES; case_index++) {
    uint32_t n = cases[case_index].n;
    uint32_t offset = cases[case_index].offset;
    uint32_t stride = cases[case_index].stride;

    fill_output_buffers();
    run_cpu_reference(n, offset, stride, scale, bias);

    if (tmu_strided_load_launch(&state,
                                input_values,
                                out_values,
                                n,
                                offset,
                                stride,
                                scale,
                                bias) < 0) {
        printk("ERROR: tmu_strided_load launch failed case=%d n=%d offset=%d stride=%d\n",
               (int)case_index,
               (int)n,
               (int)offset,
               (int)stride);
        launch_failures++;
        continue;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    verify_results(case_index, n, &mismatches, &max_abs_diff);
    int case_sentinel_mismatches = verify_sentinel_tail(n);
    int checksum = scaled_checksum(out_values, n);
    int expected_checksum = scaled_checksum(expected_values, n);
    if (checksum != expected_checksum) {
        printk("ERROR: checksum mismatch case=%d gpu=%d cpu=%d\n",
               (int)case_index,
               checksum,
               expected_checksum);
        mismatches++;
    }

    if (max_abs_diff > max_abs_diff_overall)
        max_abs_diff_overall = max_abs_diff;

    total_mismatches += mismatches;
    sentinel_mismatches += case_sentinel_mismatches;
    checksum_accum += checksum;

    printk("TMU_STRIDED_LOAD_CASE case=%d n=%d offset=%d stride=%d mismatches=%d sentinel_mismatches=%d checksum=%d max_abs_diff=%f launches=%d allocations=%d\n",
           (int)case_index,
           (int)n,
           (int)offset,
           (int)stride,
           mismatches,
           case_sentinel_mismatches,
           checksum,
           max_abs_diff,
           (int)tmu_strided_load_runtime_launches(&state),
           (int)tmu_strided_load_runtime_allocations(&state));
}

int end = timer_get_usec();
int elapsed = end - start;

uint32_t runtime_allocations =
    tmu_strided_load_runtime_allocations(&state);
uint32_t runtime_launches =
    tmu_strided_load_runtime_launches(&state);
uint32_t runtime_capacity_n =
    tmu_strided_load_runtime_capacity_n(&state);

const char *status =
    (total_mismatches == 0 &&
     sentinel_mismatches == 0 &&
     launch_failures == 0 &&
     runtime_allocations == 1u &&
     runtime_launches == TMU_STRIDED_LOAD_CASES &&
     runtime_capacity_n == TMU_STRIDED_LOAD_MAX_N &&
     max_abs_diff_overall <= TMU_STRIDED_LOAD_EPSILON) ? "PASS" : "FAIL";

printk("VC4_TEST_RESULT name=tmu_strided_load status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d max_n=%d max_stride=%d max_abs_diff=%f checksum_accum=%d runtime_allocations=%d runtime_launches=%d elapsed_usec=%d\n",
       status,
       TMU_STRIDED_LOAD_CASES,
       total_mismatches,
       sentinel_mismatches,
       launch_failures,
       (int)active_qpus,
       (int)lane_width,
       TMU_STRIDED_LOAD_MAX_N,
       TMU_STRIDED_LOAD_MAX_STRIDE,
       max_abs_diff_overall,
       checksum_accum,
       (int)runtime_allocations,
       (int)runtime_launches,
       elapsed);

vc4_runtime_shutdown(&rt);

}

