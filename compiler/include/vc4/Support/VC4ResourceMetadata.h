//===- VC4ResourceMetadata.h - Semantic vc4.resource helpers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VC4_SUPPORT_VC4RESOURCEMETADATA_H
#define VC4_SUPPORT_VC4RESOURCEMETADATA_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <optional>

namespace mlir::vc4 {

struct SemanticResourceInfo {
  llvm::StringRef scheduleMode = "independent_vector";
  int64_t warpsPerBlock = 1;
  int64_t userVPMRowsPerBlock = 0;
  int64_t compilerVPMStagingRowsPerWarp = 0;
  int64_t compilerVPMStagingRowsPerBlock = 0;
  int64_t spillVPMRowsPerBlock = 0;
  bool hasSpillVPMRowsPerBlock = false;
  int64_t totalVPMRowsPerBlock = 0;
  bool usesTMU = false;
  bool usesVPM = false;
  bool usesVPMQPURead = false;
  bool usesVPMQPUWrite = false;
  bool usesVDR = false;
  bool usesVDW = false;
  bool usesBarrier = false;
  int64_t semaphoreCountPerBlock = 0;
  bool requiresVPMBaseRowBuiltin = false;
  bool requiresSemaphoreBaseBuiltin = false;
};

inline std::optional<int64_t>
getSemanticResourceI32(DictionaryAttr resource, llvm::StringRef name) {
  if (!resource)
    return std::nullopt;
  auto attr = llvm::dyn_cast_or_null<IntegerAttr>(resource.get(name));
  if (!attr || !attr.getType().isSignlessInteger(32))
    return std::nullopt;
  return attr.getInt();
}

inline std::optional<bool>
getSemanticResourceBool(DictionaryAttr resource, llvm::StringRef name) {
  if (!resource)
    return std::nullopt;
  auto attr = llvm::dyn_cast_or_null<BoolAttr>(resource.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

inline std::optional<llvm::StringRef>
getSemanticResourceString(DictionaryAttr resource, llvm::StringRef name) {
  if (!resource)
    return std::nullopt;
  auto attr = llvm::dyn_cast_or_null<StringAttr>(resource.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

inline LogicalResult emitSemanticResourceError(Operation *op,
                                               const llvm::Twine &message) {
  return op->emitOpError() << "\"vc4.resource\" " << message;
}

inline LogicalResult parseSemanticResourceMetadata(
    Operation *op, DictionaryAttr resource, SemanticResourceInfo &info,
    bool allowAbsent = true) {
  if (!resource) {
    if (allowAbsent) {
      info = SemanticResourceInfo();
      return success();
    }
    return emitSemanticResourceError(op, "requires a dictionary attribute");
  }

  SemanticResourceInfo parsed;
  std::optional<llvm::StringRef> scheduleMode =
      getSemanticResourceString(resource, "schedule_mode");
  if (scheduleMode)
    parsed.scheduleMode = *scheduleMode;
  if (parsed.scheduleMode != "independent_vector" &&
      parsed.scheduleMode != "cooperative_block") {
    return emitSemanticResourceError(
        op,
        "requires schedule_mode = \"independent_vector\" or \"cooperative_block\"");
  }

  std::optional<int64_t> warpsPerBlock =
      getSemanticResourceI32(resource, "warps_per_block");
  if (!warpsPerBlock)
    return emitSemanticResourceError(op, "requires signless i32 'warps_per_block'");
  parsed.warpsPerBlock = *warpsPerBlock;
  if (parsed.warpsPerBlock < 1 || parsed.warpsPerBlock > 12) {
    return emitSemanticResourceError(
        op, llvm::Twine("warps_per_block must be in range [1, 12]; got ") +
                llvm::Twine(parsed.warpsPerBlock));
  }
  if (parsed.scheduleMode == "independent_vector" &&
      parsed.warpsPerBlock != 1) {
    return emitSemanticResourceError(
        op, "independent_vector requires warps_per_block = 1");
  }

  parsed.userVPMRowsPerBlock =
      getSemanticResourceI32(resource, "user_vpm_rows_per_block").value_or(0);
  parsed.compilerVPMStagingRowsPerWarp =
      getSemanticResourceI32(resource, "compiler_vpm_staging_rows_per_warp")
          .value_or(0);
  parsed.compilerVPMStagingRowsPerBlock =
      getSemanticResourceI32(resource, "compiler_vpm_staging_rows_per_block")
          .value_or(0);
  if (std::optional<int64_t> spillRows =
          getSemanticResourceI32(resource, "spill_vpm_rows_per_block")) {
    parsed.spillVPMRowsPerBlock = *spillRows;
    parsed.hasSpillVPMRowsPerBlock = true;
  }
  parsed.totalVPMRowsPerBlock =
      getSemanticResourceI32(resource, "total_vpm_rows_per_block").value_or(0);

  parsed.usesTMU = getSemanticResourceBool(resource, "uses_tmu").value_or(false);
  parsed.usesVPM = getSemanticResourceBool(resource, "uses_vpm").value_or(false);
  parsed.usesVPMQPURead =
      getSemanticResourceBool(resource, "uses_vpm_qpu_read").value_or(false);
  parsed.usesVPMQPUWrite =
      getSemanticResourceBool(resource, "uses_vpm_qpu_write").value_or(false);
  parsed.usesVDR = getSemanticResourceBool(resource, "uses_vdr").value_or(false);
  parsed.usesVDW = getSemanticResourceBool(resource, "uses_vdw").value_or(false);
  parsed.usesBarrier =
      getSemanticResourceBool(resource, "uses_barrier").value_or(false);
  parsed.semaphoreCountPerBlock =
      getSemanticResourceI32(resource, "semaphore_count_per_block").value_or(0);
  parsed.requiresVPMBaseRowBuiltin =
      getSemanticResourceBool(resource, "requires_vpm_base_row_builtin")
          .value_or(false);
  parsed.requiresSemaphoreBaseBuiltin =
      getSemanticResourceBool(resource, "requires_semaphore_base_builtin")
          .value_or(false);

  if (parsed.userVPMRowsPerBlock < 0 ||
      parsed.compilerVPMStagingRowsPerWarp < 0 ||
      parsed.compilerVPMStagingRowsPerBlock < 0 ||
      parsed.spillVPMRowsPerBlock < 0 || parsed.totalVPMRowsPerBlock < 0) {
    return emitSemanticResourceError(op, "VPM row counts must be non-negative");
  }
  int64_t expectedRows =
      parsed.userVPMRowsPerBlock + parsed.compilerVPMStagingRowsPerBlock +
      parsed.warpsPerBlock * parsed.compilerVPMStagingRowsPerWarp +
      (parsed.hasSpillVPMRowsPerBlock ? parsed.spillVPMRowsPerBlock : 0);
  if (parsed.totalVPMRowsPerBlock != expectedRows) {
    return emitSemanticResourceError(
        op, llvm::Twine("total_vpm_rows_per_block must equal user + "
                        "compiler_block + warps_per_block * compiler_per_warp"
                        " + spill rows; expected ") +
                llvm::Twine(expectedRows) + ", got " +
                llvm::Twine(parsed.totalVPMRowsPerBlock));
  }
  if (parsed.totalVPMRowsPerBlock > 64) {
    return emitSemanticResourceError(
        op, llvm::Twine("total_vpm_rows_per_block must fit the 64 row VPM; got ") +
                llvm::Twine(parsed.totalVPMRowsPerBlock));
  }

  bool computedUsesVPM = parsed.usesVPMQPURead || parsed.usesVPMQPUWrite ||
                         parsed.usesVDR || parsed.usesVDW ||
                         parsed.totalVPMRowsPerBlock > 0;
  if (parsed.usesVPM != computedUsesVPM) {
    return emitSemanticResourceError(
        op, "uses_vpm must equal VPM feature usage or nonzero total VPM rows");
  }
  if (parsed.requiresVPMBaseRowBuiltin !=
      (parsed.totalVPMRowsPerBlock > 0)) {
    return emitSemanticResourceError(
        op, "requires_vpm_base_row_builtin must equal total_vpm_rows_per_block > 0");
  }
  if (!parsed.usesBarrier && parsed.semaphoreCountPerBlock != 0) {
    return emitSemanticResourceError(
        op, "semaphore_count_per_block must be zero unless uses_barrier = true");
  }
  if (parsed.usesBarrier && parsed.scheduleMode != "cooperative_block") {
    return emitSemanticResourceError(
        op, "uses_barrier requires schedule_mode = \"cooperative_block\"");
  }
  if (parsed.usesBarrier && parsed.semaphoreCountPerBlock == 0) {
    return emitSemanticResourceError(
        op, "uses_barrier requires semaphore_count_per_block > 0");
  }
  if (parsed.semaphoreCountPerBlock < 0) {
    return emitSemanticResourceError(
        op, "semaphore_count_per_block must be non-negative");
  }
  if (parsed.semaphoreCountPerBlock > 16) {
    return emitSemanticResourceError(
        op, llvm::Twine("semaphore_count_per_block must fit the target "
                        "hardware semaphore limit 16; got ") +
                llvm::Twine(parsed.semaphoreCountPerBlock));
  }
  if (parsed.requiresSemaphoreBaseBuiltin !=
      (parsed.semaphoreCountPerBlock > 0)) {
    return emitSemanticResourceError(
        op, "requires_semaphore_base_builtin must equal semaphore_count_per_block > 0");
  }

  info = parsed;
  return success();
}

inline int64_t getMaxResidentBlocksForSemanticResource(
    const SemanticResourceInfo &info, int64_t targetActiveQPUs = 12,
    int64_t targetSemaphores = 16, int64_t targetVPMRows = 64) {
  int64_t byQPU = targetActiveQPUs / std::max<int64_t>(info.warpsPerBlock, 1);
  int64_t bySem = info.semaphoreCountPerBlock > 0
                      ? targetSemaphores / info.semaphoreCountPerBlock
                      : targetActiveQPUs;
  int64_t byVPM = info.totalVPMRowsPerBlock > 0
                      ? targetVPMRows / info.totalVPMRowsPerBlock
                      : targetActiveQPUs;
  return std::min(byQPU, std::min(bySem, byVPM));
}

} // namespace mlir::vc4

#endif // VC4_SUPPORT_VC4RESOURCEMETADATA_H
