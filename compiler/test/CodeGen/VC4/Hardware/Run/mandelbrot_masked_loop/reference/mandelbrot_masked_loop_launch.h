#ifndef MANDELBROT_MASKED_LOOP_LAUNCH_H
#define MANDELBROT_MASKED_LOOP_LAUNCH_H

#include <stdint.h>

#define MANDELBROT_MASKED_LOOP_ACTIVE_QPUS 12u
#define MANDELBROT_MASKED_LOOP_LANES 16u

struct vc4_runtime;

struct mandelbrot_masked_loop_state {
struct vc4_runtime *rt;
uint32_t max_width;
uint32_t max_height;
uint32_t allocation_count;
uint32_t launch_count;
uint32_t active_qpus;
uint32_t lanes;
};

int mandelbrot_masked_loop_prepare(struct vc4_runtime *rt,
struct mandelbrot_masked_loop_state *state,
uint32_t max_width,
uint32_t max_height);

int mandelbrot_masked_loop_launch(struct mandelbrot_masked_loop_state *state,
uint32_t *output,
uint32_t width,
uint32_t height,
float min_x,
float min_y,
float step_x,
float step_y,
uint32_t max_iter);

void mandelbrot_masked_loop_shutdown(struct mandelbrot_masked_loop_state *state);

#endif
