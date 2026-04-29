#include "rpi.h"
#include "shared_transpose_16x16_launch.h"

#define SHARED_TRANSPOSE_16X16_CASES 4u
#define HOST_GUARD_WORDS 16u
#define HOST_SENTINEL 0x6d7e8f90u

static uint32_t input_values[SHARED_TRANSPOSE_16X16_WORDS];
static uint32_t output_values[SHARED_TRANSPOSE_16X16_WORDS + HOST_GUARD_WORDS];
static uint32_t expected_values[SHARED_TRANSPOSE_16X16_WORDS];

static uint32_t matrix_index(uint32_t row, uint32_t col)
{
return row * SHARED_TRANSPOSE_16X16_DIM + col;
}

static uint32_t case_value(uint32_t case_id, uint32_t row, uint32_t col)
{
switch (case_id)
{
case 0:
return row * SHARED_TRANSPOSE_16X16_DIM + col;
case 1:
return 0x70000000u | (row << 8) | col;
case 2:
return row * 37u + col * 11u + 5u;
case 3:
if (row == col)
return 0xf0000000u | row;
if (row + col == SHARED_TRANSPOSE_16X16_DIM - 1u)
return 0xe0000000u | col;
return 0x10000000u | ((row * 19u + col * 23u + 7u) & 0x0000ffffu);
default:
return 0;
}
}

static void fill_case(uint32_t case_id)
{
for (uint32_t row = 0; row < SHARED_TRANSPOSE_16X16_DIM; row++)
{
for (uint32_t col = 0; col < SHARED_TRANSPOSE_16X16_DIM; col++)
{
input_values[matrix_index(row, col)] = case_value(case_id, row, col);
expected_values[matrix_index(row, col)] = case_value(case_id, col, row);
}
}

for (uint32_t i = 0; i < SHARED_TRANSPOSE_16X16_WORDS + HOST_GUARD_WORDS; i++)
    output_values[i] = HOST_SENTINEL;

}

static uint32_t checksum_words(const uint32_t *values, uint32_t count)
{
uint32_t hash = 2166136261u;
for (uint32_t i = 0; i < count; i++)
{
hash ^= values[i];
hash *= 16777619u;
}
return hash;
}

static int verify_case(uint32_t case_id)
{
int mismatches = 0;

for (uint32_t row = 0; row < SHARED_TRANSPOSE_16X16_DIM; row++)
{
    for (uint32_t col = 0; col < SHARED_TRANSPOSE_16X16_DIM; col++)
    {
        uint32_t index = matrix_index(row, col);
        uint32_t got = output_values[index];
        uint32_t expected = expected_values[index];

        if (got != expected)
        {
            if (mismatches < 8)
            {
                printk("ERROR: case=%d row=%d col=%d got=%x expected=%x input_transposed=%x\n",
                       (int)case_id,
                       (int)row,
                       (int)col,
                       got,
                       expected,
                       input_values[matrix_index(col, row)]);
            }
            mismatches++;
        }
    }
}

return mismatches;

}

static int verify_host_guard(void)
{
int mismatches = 0;

for (uint32_t i = 0; i < HOST_GUARD_WORDS; i++)
{
    uint32_t value = output_values[SHARED_TRANSPOSE_16X16_WORDS + i];
    if (value != HOST_SENTINEL)
    {
        if (mismatches < 8)
        {
            printk("ERROR: host output guard changed index=%d value=%x expected=%x\n",
                   (int)i,
                   value,
                   HOST_SENTINEL);
        }
        mismatches++;
    }
}

return mismatches;

}

void notmain(void)
{
struct vc4_runtime rt;
struct shared_transpose_16x16_state state;

if (shared_transpose_16x16_prepare(&rt, &state) < 0)
    panic("shared_transpose_16x16 runtime setup failed");

uint32_t activeQpus = state.active_qpus;
uint32_t laneWidth = state.lane_width;

if (activeQpus != VC4_RUNTIME_MAX_QPUS)
    panic("Unexpected active QPU count: %d", (int)activeQpus);
if (laneWidth != VC4_RUNTIME_LANE_WIDTH)
    panic("Unexpected lane width: %d", (int)laneWidth);

printk("Running VC4 shared_transpose_16x16 reference bundle...\n");
printk("SHARED_TRANSPOSE_16X16_RUNTIME_SETUP allocations=%d active_qpus=%d lanes=%d warps_per_block=%d\n",
       (int)state.runtime_allocations,
       (int)activeQpus,
       (int)laneWidth,
       SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK);

int start = timer_get_usec();
int totalMismatches = 0;
int sentinelMismatches = 0;
int launchFailures = 0;
uint32_t checksumAccum = 0;

for (uint32_t caseId = 0; caseId < SHARED_TRANSPOSE_16X16_CASES; caseId++)
{
    fill_case(caseId);

    if (shared_transpose_16x16_launch(&state, input_values, output_values) < 0)
    {
        printk("ERROR: shared_transpose_16x16 launch failed for case=%d\n",
               (int)caseId);
        launchFailures++;
        continue;
    }

    int mismatches = verify_case(caseId);
    int hostGuardMismatches = verify_host_guard();
    int caseSentinelMismatches =
        hostGuardMismatches + (int)state.last_sentinel_mismatches;
    uint32_t checksum = checksum_words(output_values,
                                       SHARED_TRANSPOSE_16X16_WORDS);
    checksumAccum ^= checksum + (caseId * 0x9e3779b9u);

    totalMismatches += mismatches;
    sentinelMismatches += caseSentinelMismatches;

    printk("SHARED_TRANSPOSE_16X16_CASE case=%d mismatches=%d sentinel_mismatches=%d checksum=%x launches=%d allocations=%d\n",
           (int)caseId,
           mismatches,
           caseSentinelMismatches,
           checksum,
           (int)state.runtime_launches,
           (int)state.runtime_allocations);
}

int end = timer_get_usec();
int elapsed = end - start;

const char *status =
    (totalMismatches == 0 &&
     sentinelMismatches == 0 &&
     launchFailures == 0 &&
     activeQpus == VC4_RUNTIME_MAX_QPUS &&
     laneWidth == VC4_RUNTIME_LANE_WIDTH &&
     state.runtime_allocations == 1u &&
     state.runtime_launches == SHARED_TRANSPOSE_16X16_CASES &&
     state.timeouts == 0u &&
     state.errstat_relevant_changed == 0u) ? "PASS" : "FAIL";

printk("VC4_TEST_RESULT name=shared_transpose_16x16 status=%s cases=%d total_mismatches=%d sentinel_mismatches=%d launch_failures=%d active_qpus=%d lanes=%d warps_per_block=%d runtime_allocations=%d runtime_launches=%d timeouts=%d errstat_relevant_changed=%d checksum_accum=%x elapsed_usec=%d\n",
       status,
       SHARED_TRANSPOSE_16X16_CASES,
       totalMismatches,
       sentinelMismatches,
       launchFailures,
       (int)activeQpus,
       (int)laneWidth,
       SHARED_TRANSPOSE_16X16_WARPS_PER_BLOCK,
       (int)state.runtime_allocations,
       (int)state.runtime_launches,
       (int)state.timeouts,
       (int)state.errstat_relevant_changed,
       checksumAccum,
       elapsed);

shared_transpose_16x16_shutdown(&state);

}

