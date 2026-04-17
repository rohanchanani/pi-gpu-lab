#ifndef VC4_SAXPY_KERNEL_LAUNCH_H
#define VC4_SAXPY_KERNEL_LAUNCH_H

#include <stdint.h>

struct vc4_runtime;

int saxpy_launch(struct vc4_runtime *rt, float * arg0, float * arg1, float arg2, uint32_t arg3);

#endif
