//===- VC4SideEffects.h - VC4 side effect resources -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_DIALECT_VC4_IR_VC4SIDEEFFECTS_H
#define VC4_DIALECT_VC4_IR_VC4SIDEEFFECTS_H

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::vc4::effects {

// Shared VC4 hardware resources. Most are not consumed by the scheduled-only
// vc4 dialect today, but they remain target resources for future SSAVC4 work.
#define VC4_DECLARE_RESOURCE(ResourceName)                                        \
  struct ResourceName : public SideEffects::Resource::Base<ResourceName> {        \
    StringRef getName() final { return #ResourceName; }                           \
  }

VC4_DECLARE_RESOURCE(UniformStream);
VC4_DECLARE_RESOURCE(MainMemory);
VC4_DECLARE_RESOURCE(TMUReq0);
VC4_DECLARE_RESOURCE(TMUReq1);
VC4_DECLARE_RESOURCE(TMURcv0);
VC4_DECLARE_RESOURCE(TMURcv1);
VC4_DECLARE_RESOURCE(SFU);
VC4_DECLARE_RESOURCE(VPMReadFIFO);
VC4_DECLARE_RESOURCE(VPMWriteFIFO);
VC4_DECLARE_RESOURCE(VDR);
VC4_DECLARE_RESOURCE(VDW);
VC4_DECLARE_RESOURCE(Mutex);
VC4_DECLARE_RESOURCE(Semaphore);
VC4_DECLARE_RESOURCE(HostIRQ);
VC4_DECLARE_RESOURCE(QPUScheduler);
VC4_DECLARE_RESOURCE(V3DSystem);

#undef VC4_DECLARE_RESOURCE

} // namespace mlir::vc4::effects

#endif // VC4_DIALECT_VC4_IR_VC4SIDEEFFECTS_H
