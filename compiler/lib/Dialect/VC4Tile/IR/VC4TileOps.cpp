//===- VC4TileOps.cpp - VC4Tile operation implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4Tile/IR/VC4TileOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace mlir;
using namespace mlir::vc4tile;

namespace {

static LogicalResult emitTypeError(Operation *op, Type type, StringRef role,
                                   StringRef expected) {
  return op->emitOpError() << role << " type must be " << expected << "; got "
                           << type;
}

static LogicalResult verifyScalarId(Operation *op, Type type, StringRef role) {
  if (isVC4TileScalarIdType(type))
    return success();
  return emitTypeError(op, type, role, "i32 or index");
}

static LogicalResult verifyVector16I32(Operation *op, Type type,
                                       StringRef role) {
  if (isVC4TileVector16I32Type(type))
    return success();
  return emitTypeError(op, type, role, "vector<16xi32>");
}

static LogicalResult verifyVector16I1(Operation *op, Type type,
                                      StringRef role) {
  if (isVC4TileVector16I1Type(type))
    return success();
  return emitTypeError(op, type, role, "vector<16xi1>");
}

static LogicalResult verifyVector16Data(Operation *op, Type type,
                                        StringRef role) {
  if (isVC4TileVector16DataType(type))
    return success();
  return emitTypeError(op, type, role,
                       "vector<16xi32> or vector<16xf32>");
}

static LogicalResult verifySharedTile(Operation *op, Type type,
                                      StringRef role) {
  if (isVC4TileSharedTileType(type))
    return success();
  return emitTypeError(op, type, role, "!vc4tile.shared_tile");
}

static LogicalResult verifyElemBytes4(Operation *op, IntegerAttr attr) {
  if (attr && attr.getInt() == 4)
    return success();
  return op->emitOpError("elem_bytes must be 4 for the M4 32-bit contract");
}

static LogicalResult verifyRowsAttr(Operation *op, IntegerAttr attr,
                                    StringRef name) {
  if (!attr)
    return op->emitOpError() << name << " attribute is required";
  int64_t rows = attr.getInt();
  if (rows < 1 || rows > 64)
    return op->emitOpError() << name << " must be in range [1, 64]";
  return success();
}

static std::optional<int64_t> getI32Attr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool getBoolAttr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<BoolAttr>(name);
  return attr && attr.getValue();
}

static std::optional<ScheduleMode> getKernelScheduleMode(KernelOp kernel) {
  if (auto attr = kernel.getScheduleModeAttr())
    return attr.getValue();
  return std::nullopt;
}

static bool isKernelCooperative(KernelOp kernel) {
  std::optional<ScheduleMode> mode = getKernelScheduleMode(kernel);
  return mode && *mode == ScheduleMode::cooperative_block;
}

static bool isKernelIndependent(KernelOp kernel) {
  std::optional<ScheduleMode> mode = getKernelScheduleMode(kernel);
  return !mode || *mode == ScheduleMode::independent_vector;
}

static LogicalResult verifyInsideKernel(Operation *op) {
  if (op->getParentOfType<KernelOp>())
    return success();
  return op->emitOpError("must appear inside vc4tile.kernel");
}

static LogicalResult verifySharedKernelContract(Operation *op) {
  auto kernel = op->getParentOfType<KernelOp>();
  if (!kernel)
    return op->emitOpError("must appear inside vc4tile.kernel");
  if (!getBoolAttr(kernel.getOperation(), "uses_shared_vpm")) {
    return op->emitOpError(
        "requires parent vc4tile.kernel to set uses_shared_vpm = true");
  }
  if (!isKernelCooperative(kernel)) {
    return op->emitOpError(
        "requires parent vc4tile.kernel schedule_mode = "
        "#vc4tile.schedule_mode<cooperative_block>");
  }
  return success();
}

static LogicalResult verifyMemoryAccessNotGeneric(Operation *op,
                                                  MemoryAccessAttr accessAttr) {
  if (accessAttr && accessAttr.getValue() == MemoryAccess::generic) {
    return op->emitOpError(
        "generic per-lane store access is outside the M4 hardware-lowered "
        "contract; use coalesced or affine_contiguous");
  }
  return success();
}

static LogicalResult verifyMemorySpace(Operation *op, MemorySpaceAttr spaceAttr,
                                       MemorySpace expected,
                                       StringRef expectedName) {
  if (!spaceAttr || spaceAttr.getValue() == expected)
    return success();
  return op->emitOpError() << "memory_space must be #vc4tile.memory_space<"
                           << expectedName << ">";
}

} // namespace

LogicalResult KernelOp::verify() {
  Operation *op = getOperation();
  std::optional<ScheduleMode> mode = getKernelScheduleMode(*this);
  int64_t warpsPerBlock = getI32Attr(op, "warps_per_block_max").value_or(
      mode && *mode == ScheduleMode::cooperative_block ? 12 : 1);
  int64_t vpmRows = getI32Attr(op, "vpm_rows_per_block").value_or(0);
  int64_t vpmBytes = getI32Attr(op, "vpm_bytes_per_block").value_or(0);
  int64_t semaphores = getI32Attr(op, "semaphores_per_block").value_or(0);
  bool usesShared = getBoolAttr(op, "uses_shared_vpm");
  bool usesBarrier = getBoolAttr(op, "uses_barrier");
  bool requireFullResidency = getBoolAttr(op, "require_full_block_residency");

  if (warpsPerBlock < 1 || warpsPerBlock > 12)
    return emitOpError("warps_per_block_max must be in range [1, 12]");
  if (vpmRows < 0 || vpmRows > 64)
    return emitOpError("vpm_rows_per_block must be in range [0, 64]");
  if (vpmBytes < 0 || vpmBytes > 4096)
    return emitOpError("vpm_bytes_per_block must be in range [0, 4096]");
  if (semaphores < 0 || semaphores > 16)
    return emitOpError("semaphores_per_block must be in range [0, 16]");
  if (vpmBytes > vpmRows * 64)
    return emitOpError("vpm_bytes_per_block must fit vpm_rows_per_block");

  if (mode && *mode == ScheduleMode::independent_vector) {
    if (warpsPerBlock != 1)
      return emitOpError(
          "independent_vector kernels require warps_per_block_max = 1");
    if (usesShared)
      return emitOpError("independent_vector kernels must not use shared VPM");
    if (usesBarrier)
      return emitOpError("independent_vector kernels must not use barriers");
  }

  if (usesShared && (vpmRows == 0 || vpmBytes == 0)) {
    return emitOpError(
        "uses_shared_vpm requires positive vpm_rows_per_block and "
        "vpm_bytes_per_block");
  }

  if (usesBarrier) {
    if (!isKernelCooperative(*this)) {
      return emitOpError(
          "uses_barrier requires schedule_mode = "
          "#vc4tile.schedule_mode<cooperative_block>");
    }
    if (semaphores != 4)
      return emitOpError("uses_barrier requires semaphores_per_block = 4");
    if (!requireFullResidency) {
      return emitOpError(
          "uses_barrier requires require_full_block_residency = true");
    }
  }

  bool sawBarrier = false;
  bool sawSharedOp = false;
  getBody().walk([&](Operation *nested) {
    if (isa<BarrierOp>(nested))
      sawBarrier = true;
    if (isa<SharedAllocOp, SharedLoadOp, SharedStoreOp>(nested))
      sawSharedOp = true;
  });
  if (sawBarrier && !usesBarrier)
    return emitOpError("contains vc4tile.barrier but uses_barrier is not true");
  if (sawSharedOp && !usesShared)
    return emitOpError("contains shared VPM ops but uses_shared_vpm is not true");

  return success();
}

LogicalResult ReturnOp::verify() { return verifyInsideKernel(getOperation()); }

LogicalResult ProgramIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyScalarId(op, getResult().getType(), "result");
}

LogicalResult BlockIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  auto kernel = op->getParentOfType<KernelOp>();
  if (isKernelIndependent(kernel))
    return emitOpError("is only legal in cooperative_block kernels");
  return verifyScalarId(op, getResult().getType(), "result");
}

LogicalResult WarpIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  auto kernel = op->getParentOfType<KernelOp>();
  if (isKernelIndependent(kernel))
    return emitOpError("is only legal in cooperative_block kernels");
  return verifyScalarId(op, getResult().getType(), "result");
}

LogicalResult LaneIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyScalarId(op, getResult().getType(), "result");
}

LogicalResult LaneRangeOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyVector16I32(op, getResult().getType(), "result");
}

LogicalResult ThreadIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyVector16I32(op, getResult().getType(), "result");
}

LogicalResult MaskAllOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyVector16I1(op, getResult().getType(), "result");
}

LogicalResult TailMaskOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyScalarId(op, getBase().getType(), "base")) ||
      failed(verifyScalarId(op, getLimit().getType(), "limit")))
    return failure();
  return verifyVector16I1(op, getResult().getType(), "result");
}

LogicalResult MaskedLoadGlobalOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyScalarId(op, getBase().getType(), "base")) ||
      failed(verifyVector16I32(op, getOffsets().getType(), "offsets")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyVector16Data(op, getResult().getType(), "result")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(), MemorySpace::global,
                               "global")))
    return failure();
  return success();
}

LogicalResult MaskedStoreGlobalOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyScalarId(op, getBase().getType(), "base")) ||
      failed(verifyVector16I32(op, getOffsets().getType(), "offsets")) ||
      failed(verifyVector16Data(op, getValue().getType(), "value")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(), MemorySpace::global,
                               "global")) ||
      failed(verifyMemoryAccessNotGeneric(op, getAccessAttr())))
    return failure();
  return success();
}

LogicalResult RotateOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyVector16Data(op, inputType, "input")) ||
      failed(verifyVector16Data(op, resultType, "result")))
    return failure();
  if (inputType != resultType)
    return emitOpError() << "input and result types must match; got "
                         << inputType << " and " << resultType;
  int64_t amount = getAmountAttr().getInt();
  if (amount < 0 || amount > 15)
    return emitOpError("amount must be in range [0, 15]");
  return success();
}

LogicalResult ReduceOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyVector16Data(op, inputType, "input")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyVector16Data(op, resultType, "result")))
    return failure();
  if (inputType != resultType)
    return emitOpError() << "input and result types must match; got "
                         << inputType << " and " << resultType;
  bool isInteger = isVC4TileVector16I32Type(inputType);
  switch (getKind()) {
  case ReduceKind::add:
  case ReduceKind::min:
  case ReduceKind::max:
    return success();
  case ReduceKind::bit_and:
  case ReduceKind::bit_or:
  case ReduceKind::bit_xor:
    if (isInteger)
      return success();
    return emitOpError("bitwise reductions require vector<16xi32>");
  }
  llvm_unreachable("unhandled VC4Tile reduction kind");
}

LogicalResult SharedAllocOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySharedKernelContract(op)) ||
      failed(verifyRowsAttr(op, getRowsAttr(), "rows")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(),
                               MemorySpace::shared_vpm, "shared_vpm")) ||
      failed(verifySharedTile(op, getResult().getType(), "result")))
    return failure();
  return success();
}

LogicalResult SharedLoadOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySharedKernelContract(op)) ||
      failed(verifySharedTile(op, getHandle().getType(), "handle")) ||
      failed(verifyScalarId(op, getRow().getType(), "row")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyVector16Data(op, getResult().getType(), "result")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(),
                               MemorySpace::shared_vpm, "shared_vpm")))
    return failure();
  return success();
}

LogicalResult SharedStoreOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySharedKernelContract(op)) ||
      failed(verifySharedTile(op, getHandle().getType(), "handle")) ||
      failed(verifyScalarId(op, getRow().getType(), "row")) ||
      failed(verifyVector16Data(op, getValue().getType(), "value")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(),
                               MemorySpace::shared_vpm, "shared_vpm")))
    return failure();
  return success();
}

LogicalResult BarrierOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  auto kernel = op->getParentOfType<KernelOp>();
  if (!isKernelCooperative(kernel)) {
    return emitOpError(
        "requires parent vc4tile.kernel schedule_mode = "
        "#vc4tile.schedule_mode<cooperative_block>");
  }
  if (!getBoolAttr(kernel.getOperation(), "uses_barrier")) {
    return emitOpError(
        "requires parent vc4tile.kernel to set uses_barrier = true");
  }
  if (!getBoolAttr(kernel.getOperation(), "require_full_block_residency")) {
    return emitOpError(
        "requires parent vc4tile.kernel to set "
        "require_full_block_residency = true");
  }
  int64_t semaphores =
      getI32Attr(kernel.getOperation(), "semaphores_per_block").value_or(0);
  if (semaphores != 4)
    return emitOpError("requires parent semaphores_per_block = 4");
  return success();
}

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4Tile/IR/VC4TileOps.cpp.inc"
