#ifndef IMAGE_BOXBLUR_SHARED_LAUNCH_H
#define IMAGE_BOXBLUR_SHARED_LAUNCH_H

#include <stdint.h>

#define IMAGE_BOXBLUR_SHARED_ACTIVE_QPUS 12u
#define IMAGE_BOXBLUR_SHARED_LANES 16u
#define IMAGE_BOXBLUR_SHARED_WARPS_PER_BLOCK 12u
#define IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_W 14u
#define IMAGE_BOXBLUR_SHARED_TILE_OUTPUT_H 10u

struct vc4_runtime;

struct image_boxblur_shared_state {
struct vc4_runtime *rt;
uint32_t max_width;
uint32_t max_height;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
uint32_t warps_per_block;
uint32_t timeouts;
uint32_t errstat_relevant_changed;
};

int image_boxblur_shared_prepare(struct vc4_runtime *rt,
struct image_boxblur_shared_state *state,
uint32_t max_width,
uint32_t max_height);

int image_boxblur_shared_launch(struct image_boxblur_shared_state *state,
const uint32_t *input,
uint32_t *output,
uint32_t width,
uint32_t height,
uint32_t tile_origin_x,
uint32_t tile_origin_y);

void image_boxblur_shared_shutdown(struct image_boxblur_shared_state *state);

#endif
