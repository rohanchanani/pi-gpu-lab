#ifndef VC4_SAXPY_RUNTIME_ABI_H
#define VC4_SAXPY_RUNTIME_ABI_H

#include <cstdint>
#include <cstring>

// Prototype per-QPU runtime uniform payload for the current compiler SAXPY slice.
//
// Physical uniform word order for one launched QPU:
//   word 0: x GPU base address
//   word 1: y GPU base address
//   word 2: alpha as one scalar f32 value
//   word 3: n as one 32-bit element count
//   word 4: qpu_id
//   word 5: num_qpus
//
// Physical consumption rule:
// - The final kernel-side realization must consume the uniform stream
//   sequentially with repeated `mov ..., unif`.
// - This payload order is therefore part of the runtime ABI.
// - Do not treat these values as random-access physical storage.
//
// In other words, the eventual emitted VC4 program must realize this logical
// order as something structurally equivalent to:
//   mov ..., unif   ; x
//   mov ..., unif   ; y
//   mov ..., unif   ; a
//   mov ..., unif   ; n
//   mov ..., unif   ; qpu_id
//   mov ..., unif   ; num_qpus
//
// Lowered VC4 IR mapping:
//   vc4.get_uniform[0]          -> x
//   vc4.get_uniform[1]          -> y
//   vc4.get_uniform[2]          -> alpha
//   vc4.get_uniform[3]          -> n
//   vc4.get_builtin qpu_id      -> word 4 in the runtime payload
//   vc4.get_builtin num_qpus    -> word 5 in the runtime payload
//
// Important distinction:
// - The runtime payload may physically deliver builtin values through reserved
//   uniform slots.
// - The IR must still model qpu_id and num_qpus as vc4.get_builtin, not as
//   ordinary user uniforms.
//
// Scalar alpha note:
// - The host/runtime packs one scalar alpha value.
// - It does not pack a literal 16-lane vector for alpha.
// - The lowered IR reads alpha as vector<16xf32> because the current VC4 slice
//   assumes lane-broadcast uniform semantics.
//
// Index note:
// - The source kernel ABI uses index for n.
// - The current prototype runtime contract packs n, qpu_id, and num_qpus as
//   single 32-bit uniform words because the VC4 uniform stream is word-based.
// - This prototype therefore assumes these values fit in 32 bits.

namespace vc4::examples {

inline uint32_t packF32(float value) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct SaxpyUniformBlock {
  uint32_t xBase;
  uint32_t yBase;
  uint32_t alphaBits;
  uint32_t n;
  uint32_t qpuId;
  uint32_t numQpus;
};

static_assert(sizeof(SaxpyUniformBlock) == 6 * sizeof(uint32_t));
static_assert(offsetof(SaxpyUniformBlock, xBase) == 0 * sizeof(uint32_t));
static_assert(offsetof(SaxpyUniformBlock, yBase) == 1 * sizeof(uint32_t));
static_assert(offsetof(SaxpyUniformBlock, alphaBits) == 2 * sizeof(uint32_t));
static_assert(offsetof(SaxpyUniformBlock, n) == 3 * sizeof(uint32_t));
static_assert(offsetof(SaxpyUniformBlock, qpuId) == 4 * sizeof(uint32_t));
static_assert(offsetof(SaxpyUniformBlock, numQpus) == 5 * sizeof(uint32_t));

inline SaxpyUniformBlock makeSaxpyUniformBlock(uint32_t xBase, uint32_t yBase,
                                              float alpha, uint32_t n,
                                              uint32_t qpuId,
                                              uint32_t numQpus) {
  return SaxpyUniformBlock{
      xBase,
      yBase,
      packF32(alpha),
      n,
      qpuId,
      numQpus,
  };
}

} // namespace vc4::examples

#endif // VC4_SAXPY_RUNTIME_ABI_H
