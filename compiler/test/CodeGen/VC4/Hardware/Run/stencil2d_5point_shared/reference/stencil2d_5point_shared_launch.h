#ifndef STENCIL2D_5POINT_SHARED_LAUNCH_H
#define STENCIL2D_5POINT_SHARED_LAUNCH_H

#include <stdint.h>
#include "mailbox.h"

#define STENCIL2D_5POINT_SHARED_MAX_WIDTH 31u
#define STENCIL2D_5POINT_SHARED_MAX_HEIGHT 19u
#define STENCIL2D_5POINT_SHARED_ROW_STRIDE 32u
#define STENCIL2D_5POINT_SHARED_MAX_WORDS (STENCIL2D_5POINT_SHARED_MAX_WIDTH * STENCIL2D_5POINT_SHARED_MAX_HEIGHT)
#define STENCIL2D_5POINT_SHARED_GUARD_FLOATS 32u

struct stencil2d_5point_shared_state
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

int stencil2d_5point_shared_prepare(
    struct vc4_runtime *rt,
    struct stencil2d_5point_shared_state *state,
    uint32_t max_width,
    uint32_t max_height);

int stencil2d_5point_shared_launch(
    struct stencil2d_5point_shared_state *state,
    const float *input,
    float *output,
    uint32_t width,
    uint32_t height,
    uint32_t origin_x,
    uint32_t origin_y,
    float center_weight,
    float neighbor_weight);

void stencil2d_5point_shared_shutdown(
    struct stencil2d_5point_shared_state *state);

#endif
