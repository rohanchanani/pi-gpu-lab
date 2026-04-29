#ifndef CONV1D_3TAP_LAUNCH_H
#define CONV1D_3TAP_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define CONV1D_3TAP_MAX_N 511u
#define CONV1D_3TAP_GUARD_FLOATS 32u

struct conv1d_3tap_state
{
struct vc4_runtime *rt;
uint32_t prepared;
uint32_t active_qpus;
uint32_t lane_width;
uint32_t max_n;
uint32_t padded_capacity_n;
uint32_t runtime_allocations;
uint32_t runtime_launches;
uint32_t last_sentinel_mismatches;
uint32_t total_sentinel_mismatches;
};

/*

Public semantic launcher API for conv1d_3tap.

Computes:

out[i] = c0 * x[max(i - 1, 0)] +

       c1 * x[i] +
       c2 * x[min(i + 1, n - 1)]

for every i in [0, n). The qasm implements clamp-to-edge boundary handling.

The public API exposes only semantic arguments plus the runtime/state handle.

It does not expose qpu_id, num_qpus, raw uniforms, TMU details, scratch

pointers, VPM rows, scheduler registers, or GPU bus addresses.
*/
int conv1d_3tap_prepare(
struct vc4_runtime *rt,
struct conv1d_3tap_state *state,
uint32_t max_n);

int conv1d_3tap_launch(
struct conv1d_3tap_state *state,
const float *x,
float *out,
uint32_t n,
float c0,
float c1,
float c2);

void conv1d_3tap_shutdown(struct conv1d_3tap_state *state);

#endif

