#ifndef STENCIL2D_5POINT_NAIVE_LAUNCH_H
#define STENCIL2D_5POINT_NAIVE_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define STENCIL2D_5POINT_NAIVE_MAX_WIDTH 32u
#define STENCIL2D_5POINT_NAIVE_MAX_HEIGHT 32u
#define STENCIL2D_5POINT_NAIVE_ROW_STRIDE 32u
#define STENCIL2D_5POINT_NAIVE_MAX_WORDS (STENCIL2D_5POINT_NAIVE_MAX_WIDTH * STENCIL2D_5POINT_NAIVE_MAX_HEIGHT)
#define STENCIL2D_5POINT_NAIVE_GUARD_FLOATS 32u

struct stencil2d_5point_naive_state
{
struct vc4_runtime *rt;
uint32_t prepared;
uint32_t active_qpus;
uint32_t lane_width;
uint32_t max_width;
uint32_t max_height;
uint32_t runtime_allocations;
uint32_t runtime_launches;
uint32_t last_sentinel_mismatches;
uint32_t total_sentinel_mismatches;
};

/*

Public semantic launcher API for stencil2d_5point_naive.

Computes a row-major clamp-to-edge 2D five-point stencil. The public API

exposes only semantic shape and coefficient arguments plus the runtime/state

handle. It does not expose qpu_id, num_qpus, raw uniforms, TMU details,

scratch pointers, VPM rows, scheduler registers, or GPU bus addresses.
*/
int stencil2d_5point_naive_prepare(
struct vc4_runtime *rt,
struct stencil2d_5point_naive_state *state,
uint32_t max_width,
uint32_t max_height);

int stencil2d_5point_naive_launch(
struct stencil2d_5point_naive_state *state,
const float *input,
float *output,
uint32_t width,
uint32_t height,
float center_weight,
float neighbor_weight);

void stencil2d_5point_naive_shutdown(
struct stencil2d_5point_naive_state *state);

#endif

