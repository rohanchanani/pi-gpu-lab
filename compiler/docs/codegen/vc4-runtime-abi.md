# VC4 Runtime ABI for M2

The generated public ABI is CUDA-like: users explicitly allocate/copy/free device memory and launch kernels with device pointers.

```c
typedef uint32_t vc4_deviceptr_t;
typedef struct vc4_dim3 { uint32_t x, y, z; } vc4_dim3;
struct vc4_program;

int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes);
void vc4_program_destroy(struct vc4_program *program);
int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes);
int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr);
int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst, const void *src, uint32_t bytes);
int vc4MemcpyDtoH(struct vc4_program *program, void *dst, vc4_deviceptr_t src, uint32_t bytes);
int vc4MemcpyDtoD(struct vc4_program *program, vc4_deviceptr_t dst, vc4_deviceptr_t src, uint32_t bytes);
int vc4MemsetD8(struct vc4_program *program, vc4_deviceptr_t dst, uint8_t value, uint32_t bytes);
```

Generated `<kernel>_launch` functions pack uniforms and enqueue hardware. They must not implicitly copy host buffers.
