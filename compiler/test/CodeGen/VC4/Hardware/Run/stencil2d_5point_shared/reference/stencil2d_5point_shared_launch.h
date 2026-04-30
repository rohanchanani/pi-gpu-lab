#ifndef STENCIL2D_5POINT_SHARED_LAUNCH_H
#define STENCIL2D_5POINT_SHARED_LAUNCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct vc4_runtime;

struct stencil2d_5point_shared_state {
struct vc4_runtime *rt;
uint32_t max_width;
uint32_t max_height;
uint32_t allocations;
uint32_t launches;
uint32_t prepared;
};

int stencil2d_5point_shared_prepare(struct vc4_runtime *rt,
struct stencil2d_5point_shared_state *state,
uint32_t max_width,
uint32_t max_height);

int stencil2d_5point_shared_launch(struct stencil2d_5point_shared_state *state,
const float *input,
float *output,
uint32_t width,
uint32_t height,
uint32_t tile_origin_x,
uint32_t tile_origin_y,
float center_weight,
float neighbor_weight);

void stencil2d_5point_shared_shutdown(struct stencil2d_5point_shared_state *state);

#ifdef __cplusplus
}
#endif

#endif
