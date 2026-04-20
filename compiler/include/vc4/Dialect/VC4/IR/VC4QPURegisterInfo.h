//===- VC4QPURegisterInfo.h - VC4 QPU register-space helpers ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_DIALECT_VC4_IR_VC4QPUREGISTERINFO_H
#define VC4_DIALECT_VC4_IR_VC4QPUREGISTERINFO_H

#include <cstdint>

namespace mlir::vc4 {

inline constexpr int64_t kVC4QPUPhysicalRegfileMin = 0;
inline constexpr int64_t kVC4QPUPhysicalRegfileMax = 31;
inline constexpr int64_t kVC4QPUThreadEndHazardPhysicalRegfileAddr = 14;
inline constexpr int64_t kVC4QPUUniformRead = 32;
inline constexpr int64_t kVC4QPUVaryingRead = 35;
inline constexpr int64_t kVC4QPUTMUNoswap = 36;
inline constexpr int64_t kVC4QPUR5Write = 37;
inline constexpr int64_t kVC4QPUUniformsAddress = 40;
inline constexpr int64_t kVC4QPUVPMVDRVDWMin = 48;
inline constexpr int64_t kVC4QPUVPMVDRVDWMax = 50;
inline constexpr int64_t kVC4QPUMutex = 51;
inline constexpr int64_t kVC4QPUSFUMin = 52;
inline constexpr int64_t kVC4QPUSFUMax = 55;
inline constexpr int64_t kVC4QPUTMUParameterWriteMin = 56;
inline constexpr int64_t kVC4QPUTMUParameterWriteMax = 63;

constexpr bool isVC4QPUPhysicalRegfileAddress(int64_t value) {
  return value >= kVC4QPUPhysicalRegfileMin &&
         value <= kVC4QPUPhysicalRegfileMax;
}

constexpr bool isVC4QPUUniformReadAddress(int64_t value) {
  return value == kVC4QPUUniformRead;
}

constexpr bool isVC4QPUThreadEndHazardPhysicalRegfileAddress(int64_t value) {
  return value == kVC4QPUThreadEndHazardPhysicalRegfileAddr;
}

constexpr bool isVC4QPUVaryingReadAddress(int64_t value) {
  return value == kVC4QPUVaryingRead;
}

constexpr bool isVC4QPUTMUNoswapAddress(int64_t value) {
  return value == kVC4QPUTMUNoswap;
}

constexpr bool isVC4QPUR5WriteAddress(int64_t value) {
  return value == kVC4QPUR5Write;
}

constexpr bool isVC4QPUUniformsAddress(int64_t value) {
  return value == kVC4QPUUniformsAddress;
}

constexpr bool isVC4QPUVPMVDRVDWRegisterSpaceAddress(int64_t value) {
  return value >= kVC4QPUVPMVDRVDWMin && value <= kVC4QPUVPMVDRVDWMax;
}

constexpr bool isVC4QPUMutexAddress(int64_t value) {
  return value == kVC4QPUMutex;
}

constexpr bool isVC4QPUSFUWriteAddress(int64_t value) {
  return value >= kVC4QPUSFUMin && value <= kVC4QPUSFUMax;
}

constexpr bool isVC4QPUTMUParameterWriteAddress(int64_t value) {
  return value >= kVC4QPUTMUParameterWriteMin &&
         value <= kVC4QPUTMUParameterWriteMax;
}

} // namespace mlir::vc4

#endif // VC4_DIALECT_VC4_IR_VC4QPUREGISTERINFO_H
