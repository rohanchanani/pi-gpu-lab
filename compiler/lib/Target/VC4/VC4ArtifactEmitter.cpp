//===- VC4ArtifactEmitter.cpp - VC4 codegen artifact bundle API -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Target/VC4/VC4ArtifactEmitter.h"

#include "vc4/Dialect/VC4/IR/VC4Ops.h"
#include "vc4/Support/VC4ResourceMetadata.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

using namespace mlir;

namespace {

enum class LaunchABIArgumentKind { Scalar, Buffer };

struct LaunchABIArgumentModel {
  std::string name;
  LaunchABIArgumentKind kind = LaunchABIArgumentKind::Scalar;
  std::string direction;
  std::string scalarType;
  std::string elementType;
  std::string cType;
  int64_t uniformIndex = -1;
};

struct LaunchABIBuiltinModel {
  std::string name;
  std::string kind;
  std::string materialization;
  std::optional<int64_t> uniformIndex;
};

struct LaunchABIModel {
  std::string publicName;
  std::string codeSymbol;
  std::string tailPolicy;
  int64_t uniformWordsPerQPU = 0;
  llvm::SmallVector<LaunchABIArgumentModel, 8> arguments;
  llvm::SmallVector<LaunchABIBuiltinModel, 4> builtins;
};

struct KernelResourceModel {
  std::string scheduleMode = "independent_vector";
  bool usesBarrier = false;
  bool usesTMU = false;
  bool usesVPM = false;
  bool usesVPMQPURead = false;
  bool usesVPMQPUWrite = false;
  bool usesVDR = false;
  bool usesVDW = false;
  bool requiresVPMBaseRowBuiltin = false;
  bool requiresSemaphoreBaseBuiltin = false;
  int64_t vpmBytesPerBlock = 0;
  int64_t userVPMRowsPerBlock = 0;
  int64_t compilerVPMStagingRowsPerWarp = 0;
  int64_t compilerVPMStagingRowsPerBlock = 0;
  int64_t spillVPMRowsPerBlock = 0;
  int64_t totalVPMRowsPerBlock = 0;
  int64_t semaphoreCountPerBlock = 0;
  int64_t warpsPerBlock = 1;
  int64_t maxResidentBlocks = 12;
};

struct KernelSpillFrameModel {
  uint64_t frameBytes = 0;
  uint64_t strideBytes = 0;
  uint64_t frameCount = 0;
  uint64_t arenaBytes = 0;
};

struct KernelRecord {
  explicit KernelRecord(mlir::vc4::FuncOp func) : func(func) {}

  mlir::vc4::FuncOp func;
  mlir::vc4::VC4ArtifactKernelInfo info;
  LaunchABIModel launchABI;
  KernelResourceModel resources;
  KernelSpillFrameModel spill;
  unsigned kernelId = 0;
  std::string qasmPath;
  llvm::SmallVector<mlir::Operation *, 16> scheduledStream;
};

struct ProgramLayoutRegion {
  std::string name;
  std::string kind;
  uint64_t offset = 0;
  uint64_t size = 0;
  uint64_t alignment = 8;
  std::optional<unsigned> kernelId;
};

struct KernelLayoutRecord {
  unsigned kernelId = 0;
  std::string publicName;
  std::string codeSymbol;
  uint64_t descriptorOffset = 0;
  uint64_t descriptorSize = 0;
  uint64_t codeOffset = 0;
  uint64_t codeSize = 0;
  uint64_t codeWords = 0;
  uint64_t uniformsOffset = 0;
  uint64_t uniformsSize = 0;
  uint64_t unifPtrOffset = 0;
  uint64_t unifPtrSize = 0;
  uint64_t uniformWordsPerRequest = 0;
  uint64_t maxRequestsPerWave = 12;
  uint64_t spillFrameBytes = 0;
  uint64_t spillFrameStrideBytes = 0;
  uint64_t spillFrameCount = 0;
  uint64_t spillArenaBytes = 0;
};

struct ProgramLayoutModel {
  uint64_t alignment = 8;
  uint64_t programBytes = 0;
  uint64_t staticBytes = 0;
  uint64_t spillArenaOffset = 0;
  uint64_t spillArenaBytes = 0;
  uint64_t heapOffset = 0;
  uint64_t heapBytes = 4096;
  llvm::SmallVector<ProgramLayoutRegion, 16> regions;
  llvm::SmallVector<KernelLayoutRecord, 8> kernels;
};

struct QASMEmissionState {
  std::optional<unsigned> preparedVectorRotateDest;
  std::optional<int64_t> preparedVectorRotateSelector;
};

static bool isScheduledQPUKernel(mlir::vc4::FuncOp func) {
  if (func.isExternal())
    return false;

  std::optional<mlir::vc4::ExecutionDomain> domain = func.getDomain();
  std::optional<mlir::vc4::FunctionForm> form = func.getForm();
  if (!domain || *domain != mlir::vc4::ExecutionDomain::qpu || !form ||
      *form != mlir::vc4::FunctionForm::scheduled)
    return false;

  return func->hasAttr("kernel");
}

static bool isCIdentifierHead(char value) {
  return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') ||
         value == '_';
}

static bool isCIdentifierTail(char value) {
  return isCIdentifierHead(value) || (value >= '0' && value <= '9');
}

static bool isCIdentifier(llvm::StringRef value) {
  if (value.empty() || !isCIdentifierHead(value.front()))
    return false;
  for (char c : value.drop_front()) {
    if (!isCIdentifierTail(c))
      return false;
  }
  return true;
}

static LogicalResult emitLaunchABIModelError(mlir::vc4::FuncOp func,
                                             const llvm::Twine &message) {
  return func.emitOpError() << "vc4.launch_abi " << message;
}

static LogicalResult emitResourceModelError(mlir::vc4::FuncOp func,
                                            const llvm::Twine &message) {
  return func.emitOpError() << "vc4.resource " << message;
}

static mlir::StringAttr getDictionaryStringAttr(mlir::DictionaryAttr dict,
                                                llvm::StringRef name) {
  return llvm::dyn_cast_or_null<mlir::StringAttr>(dict.get(name));
}

static std::optional<int64_t>
getDictionaryIntegerAttrValue(mlir::DictionaryAttr dict, llvm::StringRef name) {
  auto integerAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(dict.get(name));
  if (!integerAttr)
    return std::nullopt;
  return integerAttr.getInt();
}

static std::optional<std::string> getScalarCType(llvm::StringRef type) {
  if (type == "i32")
    return std::string("int32_t");
  if (type == "u32" || type == "index")
    return std::string("uint32_t");
  if (type == "f32")
    return std::string("float");
  return std::nullopt;
}

static std::optional<std::string> getElementCType(llvm::StringRef elemType) {
  if (elemType == "i8")
    return std::string("int8_t");
  if (elemType == "u8")
    return std::string("uint8_t");
  if (elemType == "i16")
    return std::string("int16_t");
  if (elemType == "u16")
    return std::string("uint16_t");
  if (elemType == "i32")
    return std::string("int32_t");
  if (elemType == "u32")
    return std::string("uint32_t");
  if (elemType == "f32")
    return std::string("float");
  return std::nullopt;
}

static std::optional<std::string> getBufferCType(llvm::StringRef direction,
                                                 llvm::StringRef elemType) {
  if (!getElementCType(elemType))
    return std::nullopt;
  if (direction == "in" || direction == "out" || direction == "inout")
    return std::string("vc4_deviceptr_t");
  return std::nullopt;
}

static bool hasLaunchABIArgumentName(const LaunchABIModel &launchABI,
                                     llvm::StringRef name) {
  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    if (arg.name == name)
      return true;
  }
  return false;
}

static bool isRuntimeLaunchABIName(llvm::StringRef name) {
  return name == "logical_request" || name == "total_requests" ||
         name == "logical_block_id" || name == "logical_warp_id" ||
         name == "program_id_x" || name == "program_id_y" ||
         name == "program_id_z" || name == "num_programs_x" ||
         name == "num_programs_y" || name == "num_programs_z" ||
         name == "warps_per_block" || name == "vpm_base_row" ||
         name == "vpm_rows" || name == "semaphore_base" ||
         name == "barrier_arrive_sem" || name == "barrier_go_sem" ||
         name == "barrier_depart_sem" || name == "barrier_reset_sem" ||
         name == "resident_request_id" || name == "spill_frame_base" ||
         name == "spill_frame_bytes" || name == "spill_frame_stride_bytes" ||
         name == "spill_vpm_row";
}

static bool isLaneIdentityLaunchABIName(llvm::StringRef name) {
  return name == "ELEMENT_NUMBER" || name == "element_number" ||
         name == "elem_num" || name == "lane_id" || name == "lane_range";
}

static LogicalResult parseLaunchABIArgument(mlir::vc4::FuncOp func,
                                            mlir::DictionaryAttr argDict,
                                            LaunchABIModel &launchABI) {
  auto nameAttr = getDictionaryStringAttr(argDict, "name");
  if (!nameAttr || nameAttr.getValue().empty())
    return emitLaunchABIModelError(
        func, "argument entry requires a non-empty string 'name'");

  llvm::StringRef name = nameAttr.getValue();
  if (!isCIdentifier(name)) {
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' requires a C identifier name for generated launcher API");
  }
  if (name == "rt" || name == "qpu_id" || name == "num_qpus" ||
      isRuntimeLaunchABIName(name) || isLaneIdentityLaunchABIName(name) ||
      name.starts_with("__vc4_")) {
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' uses a reserved launcher/runtime or builtin name");
  }
  if (hasLaunchABIArgumentName(launchABI, name)) {
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' duplicates a generated public API parameter name");
  }

  auto kindAttr = getDictionaryStringAttr(argDict, "kind");
  auto directionAttr = getDictionaryStringAttr(argDict, "direction");
  if (!kindAttr || !directionAttr) {
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' requires string 'kind' and 'direction' metadata");
  }

  std::optional<int64_t> uniformIndex =
      getDictionaryIntegerAttrValue(argDict, "uniform_index");
  if (!uniformIndex) {
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' requires signless i32 'uniform_index'");
  }

  LaunchABIArgumentModel parsed;
  parsed.name = name.str();
  parsed.direction = directionAttr.getValue().str();
  parsed.uniformIndex = *uniformIndex;

  llvm::StringRef kind = kindAttr.getValue();
  if (kind == "scalar") {
    if (directionAttr.getValue() != "by_value") {
      return emitLaunchABIModelError(func,
                                     llvm::Twine("scalar argument '") + name +
                                         "' requires direction = \"by_value\"");
    }
    auto typeAttr = getDictionaryStringAttr(argDict, "type");
    if (!typeAttr) {
      return emitLaunchABIModelError(func, llvm::Twine("scalar argument '") +
                                               name +
                                               "' requires string 'type'");
    }
    std::optional<std::string> cType = getScalarCType(typeAttr.getValue());
    if (!cType) {
      return emitLaunchABIModelError(
          func, llvm::Twine("scalar argument '") + name +
                    "' has unsupported type for generated launcher API");
    }
    parsed.kind = LaunchABIArgumentKind::Scalar;
    parsed.scalarType = typeAttr.getValue().str();
    parsed.cType = *cType;
    launchABI.arguments.push_back(std::move(parsed));
    return success();
  }

  if (kind == "buffer") {
    auto elemTypeAttr = getDictionaryStringAttr(argDict, "elem_type");
    if (!elemTypeAttr) {
      return emitLaunchABIModelError(func, llvm::Twine("buffer argument '") +
                                               name +
                                               "' requires string 'elem_type'");
    }
    std::optional<std::string> cType =
        getBufferCType(directionAttr.getValue(), elemTypeAttr.getValue());
    if (!cType) {
      return emitLaunchABIModelError(
          func, llvm::Twine("buffer argument '") + name +
                    "' has unsupported direction or elem_type for generated "
                    "launcher API");
    }
    parsed.kind = LaunchABIArgumentKind::Buffer;
    parsed.elementType = elemTypeAttr.getValue().str();
    parsed.cType = *cType;
    launchABI.arguments.push_back(std::move(parsed));
    return success();
  }

  return emitLaunchABIModelError(
      func, llvm::Twine("argument '") + name +
                "' requires kind = \"scalar\" or \"buffer\"");
}

static const char *getBuiltinKindName(mlir::vc4::BuiltinKind kind) {
  switch (kind) {
  case mlir::vc4::BuiltinKind::logical_request:
    return "logical_request";
  case mlir::vc4::BuiltinKind::total_requests:
    return "total_requests";
  case mlir::vc4::BuiltinKind::logical_block_id:
    return "logical_block_id";
  case mlir::vc4::BuiltinKind::logical_warp_id:
    return "logical_warp_id";
  case mlir::vc4::BuiltinKind::warps_per_block:
    return "warps_per_block";
  case mlir::vc4::BuiltinKind::vpm_base_row:
    return "vpm_base_row";
  case mlir::vc4::BuiltinKind::vpm_rows:
    return "vpm_rows";
  case mlir::vc4::BuiltinKind::semaphore_base:
    return "semaphore_base";
  case mlir::vc4::BuiltinKind::barrier_arrive_sem:
    return "barrier_arrive_sem";
  case mlir::vc4::BuiltinKind::barrier_go_sem:
    return "barrier_go_sem";
  case mlir::vc4::BuiltinKind::barrier_depart_sem:
    return "barrier_depart_sem";
  case mlir::vc4::BuiltinKind::barrier_reset_sem:
    return "barrier_reset_sem";
  case mlir::vc4::BuiltinKind::resident_request_id:
    return "resident_request_id";
  case mlir::vc4::BuiltinKind::spill_frame_base:
    return "spill_frame_base";
  case mlir::vc4::BuiltinKind::spill_frame_bytes:
    return "spill_frame_bytes";
  case mlir::vc4::BuiltinKind::spill_frame_stride_bytes:
    return "spill_frame_stride_bytes";
  case mlir::vc4::BuiltinKind::spill_vpm_row:
    return "spill_vpm_row";
  case mlir::vc4::BuiltinKind::program_id_x:
    return "program_id_x";
  case mlir::vc4::BuiltinKind::program_id_y:
    return "program_id_y";
  case mlir::vc4::BuiltinKind::program_id_z:
    return "program_id_z";
  case mlir::vc4::BuiltinKind::num_programs_x:
    return "num_programs_x";
  case mlir::vc4::BuiltinKind::num_programs_y:
    return "num_programs_y";
  case mlir::vc4::BuiltinKind::num_programs_z:
    return "num_programs_z";
  }
  return "unknown";
}

static LogicalResult parseLaunchABIBuiltin(mlir::vc4::FuncOp func,
                                           mlir::DictionaryAttr builtinDict,
                                           LaunchABIModel &launchABI) {
  auto nameAttr = getDictionaryStringAttr(builtinDict, "name");
  auto materializationAttr =
      getDictionaryStringAttr(builtinDict, "materialization");
  auto kindAttr = llvm::dyn_cast_or_null<mlir::vc4::BuiltinKindAttr>(
      builtinDict.get("kind"));
  if (!nameAttr || nameAttr.getValue().empty() || !materializationAttr ||
      !kindAttr) {
    return emitLaunchABIModelError(
        func,
        "builtin entry requires name, kind, and materialization metadata");
  }

  LaunchABIBuiltinModel parsed;
  parsed.name = nameAttr.getValue().str();
  parsed.kind = getBuiltinKindName(kindAttr.getValue());
  parsed.materialization = materializationAttr.getValue().str();
  if (parsed.materialization != "uniform_suffix") {
    return emitLaunchABIModelError(
        func, llvm::Twine("builtin '") + parsed.name +
                  "' requires materialization = \"uniform_suffix\"");
  }
  parsed.uniformIndex =
      getDictionaryIntegerAttrValue(builtinDict, "uniform_index");
  launchABI.builtins.push_back(std::move(parsed));
  return success();
}

static std::string canonicalizeKernelPublicName(llvm::StringRef rawName) {
  // vc4.launch_abi.public_name names the public kernel. Historically some
  // inputs used a *_launch spelling because the only public API was the
  // generated launch stub. Canonical M2 program artifacts name the kernel,
  // QASM artifact, manifest entry, layout regions, and default code symbol
  // without that launcher suffix; the generated C API adds *_launch at the
  // function boundary.
  constexpr const char *launchSuffix = "_launch";
  constexpr size_t launchSuffixLen = 7;
  std::string name = rawName.str();
  if (name.size() > launchSuffixLen &&
      name.compare(name.size() - launchSuffixLen, launchSuffixLen,
                   launchSuffix) == 0)
    name.resize(name.size() - launchSuffixLen);
  return name;
}

static LogicalResult parseLaunchABIModel(mlir::vc4::FuncOp func,
                                         mlir::DictionaryAttr launchABIDict,
                                         LaunchABIModel &launchABI) {
  auto publicName = getDictionaryStringAttr(launchABIDict, "public_name");
  if (!publicName || publicName.getValue().empty()) {
    return emitLaunchABIModelError(func,
                                   "requires non-empty string 'public_name'");
  }
  if (!isCIdentifier(publicName.getValue())) {
    return emitLaunchABIModelError(
        func, "public_name must be a C identifier for generated launcher API");
  }

  auto codeSymbol = getDictionaryStringAttr(launchABIDict, "code_symbol");
  if (codeSymbol) {
    if (codeSymbol.getValue().empty()) {
      return emitLaunchABIModelError(
          func, "code_symbol must be a non-empty C identifier when present");
    }
    if (!isCIdentifier(codeSymbol.getValue())) {
      return emitLaunchABIModelError(
          func, "code_symbol must be a C identifier for generated code arrays");
    }
  }

  auto tailPolicy = getDictionaryStringAttr(launchABIDict, "tail_policy");
  if (!tailPolicy || tailPolicy.getValue().empty()) {
    return emitLaunchABIModelError(func,
                                   "requires string 'tail_policy' metadata");
  }

  std::optional<int64_t> uniformWords =
      getDictionaryIntegerAttrValue(launchABIDict, "uniform_words_per_qpu");
  if (!uniformWords || *uniformWords < 0) {
    return emitLaunchABIModelError(
        func, "requires non-negative signless i32 'uniform_words_per_qpu'");
  }

  auto args =
      llvm::dyn_cast_or_null<mlir::ArrayAttr>(launchABIDict.get("args"));
  auto builtins =
      llvm::dyn_cast_or_null<mlir::ArrayAttr>(launchABIDict.get("builtins"));
  if (!args || !builtins) {
    return emitLaunchABIModelError(func,
                                   "requires array 'args' and 'builtins'");
  }

  LaunchABIModel parsed;
  parsed.publicName = canonicalizeKernelPublicName(publicName.getValue());
  parsed.codeSymbol =
      codeSymbol ? codeSymbol.getValue().str() : parsed.publicName + "_shader";
  parsed.tailPolicy = tailPolicy.getValue().str();
  parsed.uniformWordsPerQPU = *uniformWords;

  for (mlir::Attribute argAttr : args) {
    auto argDict = llvm::dyn_cast<mlir::DictionaryAttr>(argAttr);
    if (!argDict) {
      return emitLaunchABIModelError(
          func, "argument entries must be dictionary attributes");
    }
    if (failed(parseLaunchABIArgument(func, argDict, parsed)))
      return failure();
  }

  for (mlir::Attribute builtinAttr : builtins) {
    auto builtinDict = llvm::dyn_cast<mlir::DictionaryAttr>(builtinAttr);
    if (!builtinDict) {
      return emitLaunchABIModelError(
          func, "builtin entries must be dictionary attributes");
    }
    if (failed(parseLaunchABIBuiltin(func, builtinDict, parsed)))
      return failure();
  }

  launchABI = std::move(parsed);
  return success();
}

static LogicalResult
markLaunchABIUniformIndex(mlir::vc4::FuncOp func, const llvm::Twine &ownerName,
                          int64_t uniformIndex,
                          llvm::SmallVectorImpl<char> &seenUniformIndices) {
  if (uniformIndex < 0 ||
      uniformIndex >= static_cast<int64_t>(seenUniformIndices.size())) {
    return emitLaunchABIModelError(
        func,
        ownerName + " uniform_index is outside [0, uniform_words_per_qpu)");
  }

  if (seenUniformIndices[uniformIndex]) {
    return emitLaunchABIModelError(
        func, ownerName + " duplicates another physical uniform_index");
  }

  seenUniformIndices[uniformIndex] = 1;
  return success();
}

static LogicalResult
validateLaunchABIUniformPacking(mlir::vc4::FuncOp func,
                                const LaunchABIModel &launchABI) {
  llvm::SmallVector<char, 16> seenUniformIndices;
  seenUniformIndices.resize(static_cast<size_t>(launchABI.uniformWordsPerQPU));
  for (char &seen : seenUniformIndices)
    seen = 0;

  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    if (failed(markLaunchABIUniformIndex(
            func, llvm::Twine("argument '") + arg.name + "'", arg.uniformIndex,
            seenUniformIndices)))
      return failure();
  }

  for (const LaunchABIBuiltinModel &builtin : launchABI.builtins) {
    if (builtin.materialization != "uniform_suffix") {
      if (builtin.uniformIndex) {
        return emitLaunchABIModelError(
            func, llvm::Twine("non-uniform builtin '") + builtin.name +
                      "' must not carry a uniform_index for launcher packing");
      }
      continue;
    }

    if (!builtin.uniformIndex) {
      return emitLaunchABIModelError(
          func, llvm::Twine("uniform_suffix builtin '") + builtin.name +
                    "' requires a uniform_index for launcher packing");
    }

    if (failed(markLaunchABIUniformIndex(
            func, llvm::Twine("builtin '") + builtin.name + "'",
            *builtin.uniformIndex, seenUniformIndices)))
      return failure();
  }

  for (size_t index = 0; index != seenUniformIndices.size(); ++index) {
    if (seenUniformIndices[index])
      continue;
    return emitLaunchABIModelError(
        func, llvm::Twine("launcher packing requires dense physical "
                          "uniform_index coverage; missing index ") +
                  llvm::Twine(static_cast<unsigned>(index)));
  }

  return success();
}

static constexpr int64_t kVC4TargetActiveQPUs = 12;
static constexpr int64_t kVC4TargetVPMBytes = 4096;
static constexpr int64_t kVC4TargetSemaphores = 16;
static constexpr int64_t kVC4VPMRowBytes = 64;

static int64_t resourceLimitFor(int64_t total, int64_t perBlock) {
  if (perBlock <= 0)
    return total;
  return total / perBlock;
}

static int64_t min3(int64_t a, int64_t b, int64_t c) {
  return std::min(a, std::min(b, c));
}

static LogicalResult parseResourceModel(mlir::vc4::FuncOp func,
                                        mlir::DictionaryAttr resourceDict,
                                        KernelResourceModel &resources) {
  KernelResourceModel parsed;
  if (!resourceDict) {
    resources = std::move(parsed);
    return success();
  }

  mlir::vc4::SemanticResourceInfo info;
  if (failed(mlir::vc4::parseSemanticResourceMetadata(
          func.getOperation(), resourceDict, info, /*allowAbsent=*/false)))
    return failure();

  parsed.scheduleMode = info.scheduleMode.str();
  parsed.usesBarrier = info.usesBarrier;
  parsed.usesTMU = info.usesTMU;
  parsed.usesVPM = info.usesVPM;
  parsed.usesVPMQPURead = info.usesVPMQPURead;
  parsed.usesVPMQPUWrite = info.usesVPMQPUWrite;
  parsed.usesVDR = info.usesVDR;
  parsed.usesVDW = info.usesVDW;
  parsed.requiresVPMBaseRowBuiltin = info.requiresVPMBaseRowBuiltin;
  parsed.requiresSemaphoreBaseBuiltin = info.requiresSemaphoreBaseBuiltin;
  parsed.warpsPerBlock = info.warpsPerBlock;
  parsed.semaphoreCountPerBlock = info.semaphoreCountPerBlock;
  parsed.userVPMRowsPerBlock = info.userVPMRowsPerBlock;
  parsed.compilerVPMStagingRowsPerWarp = info.compilerVPMStagingRowsPerWarp;
  parsed.compilerVPMStagingRowsPerBlock = info.compilerVPMStagingRowsPerBlock;
  parsed.spillVPMRowsPerBlock = info.spillVPMRowsPerBlock;
  parsed.totalVPMRowsPerBlock = info.totalVPMRowsPerBlock;
  parsed.vpmBytesPerBlock = parsed.totalVPMRowsPerBlock * kVC4VPMRowBytes;

  int64_t spillFrameBytes = getDictionaryIntegerAttrValue(
                                func->getAttrDictionary(), "spill_frame_bytes")
                                .value_or(0);
  if (parsed.scheduleMode == "cooperative_block" && spillFrameBytes > 0 &&
      parsed.spillVPMRowsPerBlock == 0) {
    parsed.spillVPMRowsPerBlock = parsed.warpsPerBlock;
    parsed.totalVPMRowsPerBlock += parsed.spillVPMRowsPerBlock;
    parsed.vpmBytesPerBlock = parsed.totalVPMRowsPerBlock * kVC4VPMRowBytes;
    parsed.usesVPM = true;
    parsed.requiresVPMBaseRowBuiltin = parsed.totalVPMRowsPerBlock > 0;
  }

  int64_t byQPU = resourceLimitFor(kVC4TargetActiveQPUs, parsed.warpsPerBlock);
  int64_t bySem =
      resourceLimitFor(kVC4TargetSemaphores, parsed.semaphoreCountPerBlock);
  int64_t byVPM = resourceLimitFor(kVC4TargetVPMBytes / kVC4VPMRowBytes,
                                   parsed.totalVPMRowsPerBlock);
  parsed.maxResidentBlocks = min3(byQPU, bySem, byVPM);
  if (parsed.maxResidentBlocks <= 0)
    return emitResourceModelError(
        func,
        "resource request leaves zero resident_blocks; check warps_per_block, "
        "total_vpm_rows_per_block, and semaphore_count_per_block");

  resources = std::move(parsed);
  return success();
}

static LogicalResult populateResourceInfo(KernelRecord &kernel) {
  auto resource =
      kernel.func->getAttrOfType<mlir::DictionaryAttr>("vc4.resource");
  return parseResourceModel(kernel.func, resource, kernel.resources);
}

static const char *getScheduleModeMacro(const KernelResourceModel &resources) {
  if (resources.scheduleMode == "cooperative_block")
    return "VC4_SCHEDULE_COOPERATIVE_BLOCK";
  return "VC4_SCHEDULE_INDEPENDENT_VECTOR";
}

static void appendJSONEscapedString(llvm::raw_ostream &os,
                                    llvm::StringRef value) {
  static constexpr char hex[] = "0123456789abcdef";
  os << '"';
  for (unsigned char c : value) {
    switch (c) {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\b':
      os << "\\b";
      break;
    case '\f':
      os << "\\f";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      if (c < 0x20) {
        os << "\\u00" << hex[c >> 4] << hex[c & 0xf];
      } else {
        os << c;
      }
      break;
    }
  }
  os << '"';
}

static LogicalResult
appendScheduledSinkOp(mlir::Operation *op,
                      llvm::SmallVectorImpl<mlir::Operation *> &stream);

static bool isThreadEndScheduledOp(mlir::Operation *op) {
  auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op);
  return bundle && bundle.getSig() == mlir::vc4::QPUSignal::thrend;
}

static bool
isNonBranchScheduledDelaySlotOpWithoutThreadEnd(mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op))
    return bundle.getSig() != mlir::vc4::QPUSignal::thrend;
  return llvm::isa<mlir::vc4::QPULDIOp, mlir::vc4::QPUSemaOp,
                   mlir::vc4::QPUVPMVCDSetupOp, mlir::vc4::QPUVPMVCDSetupLDIOp,
                   mlir::vc4::QPUVPMVCDAddrOp, mlir::vc4::QPUVPMVCDWaitOp>(op);
}

static mlir::InFlightDiagnostic
emitInvalidQASMEpilogueDiag(mlir::vc4::FuncOp func) {
  return func.emitOpError(
      "is not directly emittable: qasm input requires an explicit thrend plus "
      "two delay-slot instructions at the end of the flattened scheduled "
      "instruction stream");
}

static LogicalResult
appendBranchDelaySlotSinkOps(mlir::vc4::QPUBranchOp branch,
                             llvm::SmallVectorImpl<mlir::Operation *> &stream) {
  if (!branch.getDelaySlots().hasOneBlock()) {
    return branch.emitOpError()
           << "requires exactly one delay-slot block for artifact emission";
  }

  for (mlir::Operation &delaySlotOp : branch.getDelaySlots().front()) {
    if (mlir::failed(appendScheduledSinkOp(&delaySlotOp, stream)))
      return failure();
  }
  return success();
}

static LogicalResult
appendScheduledSinkOp(mlir::Operation *op,
                      llvm::SmallVectorImpl<mlir::Operation *> &stream) {
  if (llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPULDIOp,
                mlir::vc4::QPUSemaOp, mlir::vc4::QPUVPMVCDSetupOp,
                mlir::vc4::QPUVPMVCDSetupLDIOp, mlir::vc4::QPUVPMVCDAddrOp,
                mlir::vc4::QPUVPMVCDWaitOp>(op)) {
    stream.push_back(op);
    return success();
  }

  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    stream.push_back(op);
    return appendBranchDelaySlotSinkOps(branch, stream);
  }

  return op->emitOpError()
         << "is not a supported final scheduled VC4 QPU sink op for "
            "artifact emission";
}

static LogicalResult
verifyThreadEndEpilogue(mlir::vc4::FuncOp func,
                        llvm::ArrayRef<mlir::Operation *> stream) {
  if (stream.size() < 3) {
    auto diag = emitInvalidQASMEpilogueDiag(func);
    diag << "; found only " << static_cast<unsigned>(stream.size())
         << " scheduled instruction slot(s)";
    return failure();
  }

  const size_t threadEndIndex = stream.size() - 3;
  auto threadEndBundle =
      llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[threadEndIndex]);
  if (!threadEndBundle ||
      threadEndBundle.getSig() != mlir::vc4::QPUSignal::thrend) {
    auto diag = emitInvalidQASMEpilogueDiag(func);
    diag << "; slot N-3 must be a vc4.qpu.bundle with sig = "
            "#vc4.qpu_signal<thrend>";
    return failure();
  }

  for (size_t i = 0; i != threadEndIndex; ++i) {
    if (!isThreadEndScheduledOp(stream[i]))
      continue;
    auto diag = emitInvalidQASMEpilogueDiag(func);
    diag << "; found an earlier vc4.qpu.bundle with sig = "
            "#vc4.qpu_signal<thrend> before slot N-3";
    return failure();
  }

  if (!isNonBranchScheduledDelaySlotOpWithoutThreadEnd(
          stream[threadEndIndex + 1]) ||
      !isNonBranchScheduledDelaySlotOpWithoutThreadEnd(
          stream[threadEndIndex + 2])) {
    auto diag = emitInvalidQASMEpilogueDiag(func);
    diag << "; only slot N-3 may carry sig = #vc4.qpu_signal<thrend>; "
            "slots N-2 and N-1 must be non-branch scheduled ops without "
            "another thread-end signal";
    return failure();
  }

  return success();
}

static LogicalResult populateScheduledSinkInfo(KernelRecord &kernel) {
  if (!kernel.func.getBody().hasOneBlock()) {
    return kernel.func.emitOpError()
           << "requires a single top-level block for artifact emission";
  }

  llvm::SmallVector<mlir::Operation *, 16> stream;
  for (mlir::Operation &op : kernel.func.getBody().front()) {
    if (mlir::failed(appendScheduledSinkOp(&op, stream)))
      return failure();
  }

  if (mlir::failed(verifyThreadEndEpilogue(kernel.func, stream)))
    return failure();

  kernel.info.scheduledOpCount = static_cast<unsigned>(stream.size());
  kernel.scheduledStream.clear();
  kernel.scheduledStream.append(stream.begin(), stream.end());
  return success();
}

static LogicalResult populateLaunchABIInfo(KernelRecord &kernel) {
  auto launchABI =
      kernel.func->getAttrOfType<mlir::DictionaryAttr>("vc4.launch_abi");
  if (!launchABI) {
    return kernel.func.emitOpError()
           << "requires a dictionary \"vc4.launch_abi\" attribute";
  }

  if (failed(parseLaunchABIModel(kernel.func, launchABI, kernel.launchABI)))
    return failure();
  if (failed(validateLaunchABIUniformPacking(kernel.func, kernel.launchABI)))
    return failure();

  kernel.info.symbolName = kernel.func.getSymName().str();
  kernel.info.publicName = kernel.launchABI.publicName;
  kernel.info.uniformWordsPerQPU = kernel.launchABI.uniformWordsPerQPU;
  return success();
}

static std::string makeKernelQASMPath(llvm::StringRef publicName) {
  return (llvm::Twine("kernels/") + publicName + ".qasm").str();
}

static constexpr uint64_t kVC4ProgramLayoutAlignment = 8;
static constexpr uint64_t kVC4ProgramHeaderBytes = 64;
static constexpr uint64_t kVC4KernelDescriptorBytes = 32;
static constexpr uint64_t kVC4MaxRequestsPerWave = 12;
static constexpr uint64_t kVC4RuntimeBookkeepingBytes = 48;
static constexpr uint64_t kVC4HeapAlignment = 16;
static constexpr uint64_t kVC4SpillFrameAlignment = 64;
static constexpr uint64_t kVC4ReservedHeapBytes = 65536;

static uint64_t alignUpTo(uint64_t value, uint64_t alignment) {
  if (alignment <= 1)
    return value;
  return ((value + alignment - 1) / alignment) * alignment;
}

static LogicalResult emitSpillFrameModelError(mlir::vc4::FuncOp func,
                                              const llvm::Twine &message) {
  return func.emitOpError() << "spill frame metadata " << message;
}

static LogicalResult populateSpillFrameInfo(KernelRecord &kernel) {
  std::optional<int64_t> frameBytes = getDictionaryIntegerAttrValue(
      kernel.func->getAttrDictionary(), "spill_frame_bytes");
  std::optional<int64_t> explicitStride = getDictionaryIntegerAttrValue(
      kernel.func->getAttrDictionary(), "spill_frame_stride_bytes");

  KernelSpillFrameModel parsed;
  if (frameBytes) {
    if (*frameBytes < 0)
      return emitSpillFrameModelError(
          kernel.func, "requires non-negative spill_frame_bytes");
    parsed.frameBytes = static_cast<uint64_t>(*frameBytes);
  }

  uint64_t computedStride =
      parsed.frameBytes == 0
          ? 0
          : alignUpTo(parsed.frameBytes, kVC4SpillFrameAlignment);
  if (explicitStride) {
    if (*explicitStride < 0)
      return emitSpillFrameModelError(
          kernel.func, "requires non-negative spill_frame_stride_bytes");
    if (static_cast<uint64_t>(*explicitStride) != computedStride) {
      return emitSpillFrameModelError(
          kernel.func,
          llvm::Twine("spill_frame_stride_bytes must equal align_up("
                      "spill_frame_bytes, 64); expected ") +
              llvm::Twine(computedStride));
    }
  }

  parsed.strideBytes = computedStride;
  parsed.frameCount = parsed.frameBytes == 0 ? 0 : kVC4MaxRequestsPerWave;
  parsed.arenaBytes = parsed.strideBytes * parsed.frameCount;
  kernel.spill = parsed;
  kernel.info.spillFrameBytes = parsed.frameBytes;
  kernel.info.spillFrameStrideBytes = parsed.strideBytes;
  kernel.info.spillFrameCount = parsed.frameCount;
  kernel.info.spillArenaBytes = parsed.arenaBytes;
  return success();
}

static std::string getKernelCodeRegionName(const KernelRecord &kernel) {
  return kernel.info.publicName + ".code";
}

static std::string getKernelUniformRegionName(const KernelRecord &kernel) {
  return kernel.info.publicName + ".uniforms";
}

static std::string getKernelUnifPtrRegionName(const KernelRecord &kernel) {
  return kernel.info.publicName + ".unif_ptrs";
}

static void
appendProgramLayoutRegion(ProgramLayoutModel &layout, llvm::StringRef name,
                          llvm::StringRef kind, uint64_t offset, uint64_t size,
                          std::optional<unsigned> kernelId = std::nullopt) {
  ProgramLayoutRegion region;
  region.name = name.str();
  region.kind = kind.str();
  region.offset = offset;
  region.size = size;
  region.alignment = kVC4ProgramLayoutAlignment;
  region.kernelId = kernelId;
  layout.regions.push_back(std::move(region));
}

static ProgramLayoutModel
buildProgramLayoutModel(llvm::ArrayRef<KernelRecord> kernels) {
  ProgramLayoutModel layout;
  layout.alignment = kVC4ProgramLayoutAlignment;
  layout.heapBytes = kVC4ReservedHeapBytes;
  for (const KernelRecord &kernel : kernels)
    layout.spillArenaBytes =
        std::max(layout.spillArenaBytes, kernel.spill.arenaBytes);

  uint64_t offset = 0;
  offset = alignUpTo(offset, layout.alignment);
  appendProgramLayoutRegion(layout, "program_header", "program_header", offset,
                            kVC4ProgramHeaderBytes);
  offset += kVC4ProgramHeaderBytes;

  offset = alignUpTo(offset, layout.alignment);
  uint64_t descriptorTableOffset = offset;
  uint64_t descriptorTableSize = kernels.size() * kVC4KernelDescriptorBytes;
  appendProgramLayoutRegion(layout, "kernel_descriptor_table",
                            "kernel_descriptor_table", offset,
                            descriptorTableSize);
  offset += descriptorTableSize;

  for (const KernelRecord &kernel : kernels) {
    KernelLayoutRecord kernelLayout;
    kernelLayout.kernelId = kernel.kernelId;
    kernelLayout.publicName = kernel.info.publicName;
    kernelLayout.codeSymbol = kernel.launchABI.codeSymbol;
    kernelLayout.descriptorOffset =
        descriptorTableOffset + kernel.kernelId * kVC4KernelDescriptorBytes;
    kernelLayout.descriptorSize = kVC4KernelDescriptorBytes;
    kernelLayout.uniformWordsPerRequest = kernel.info.uniformWordsPerQPU;
    kernelLayout.maxRequestsPerWave = kVC4MaxRequestsPerWave;
    kernelLayout.spillFrameBytes = kernel.spill.frameBytes;
    kernelLayout.spillFrameStrideBytes = kernel.spill.strideBytes;
    kernelLayout.spillFrameCount = kernel.spill.frameCount;
    kernelLayout.spillArenaBytes = kernel.spill.arenaBytes;

    offset = alignUpTo(offset, layout.alignment);
    kernelLayout.codeOffset = offset;
    kernelLayout.codeWords =
        static_cast<uint64_t>(kernel.info.scheduledOpCount) * 2u;
    kernelLayout.codeSize = kernelLayout.codeWords * sizeof(uint32_t);
    appendProgramLayoutRegion(layout, getKernelCodeRegionName(kernel), "code",
                              kernelLayout.codeOffset, kernelLayout.codeSize,
                              kernel.kernelId);
    offset += kernelLayout.codeSize;

    offset = alignUpTo(offset, layout.alignment);
    kernelLayout.uniformsOffset = offset;
    kernelLayout.uniformsSize = kVC4MaxRequestsPerWave *
                                kernelLayout.uniformWordsPerRequest *
                                sizeof(uint32_t);
    appendProgramLayoutRegion(layout, getKernelUniformRegionName(kernel),
                              "uniforms", kernelLayout.uniformsOffset,
                              kernelLayout.uniformsSize, kernel.kernelId);
    offset += kernelLayout.uniformsSize;

    offset = alignUpTo(offset, layout.alignment);
    kernelLayout.unifPtrOffset = offset;
    kernelLayout.unifPtrSize = kVC4MaxRequestsPerWave * sizeof(uint32_t);
    appendProgramLayoutRegion(layout, getKernelUnifPtrRegionName(kernel),
                              "unif_ptrs", kernelLayout.unifPtrOffset,
                              kernelLayout.unifPtrSize, kernel.kernelId);
    offset += kernelLayout.unifPtrSize;

    layout.kernels.push_back(std::move(kernelLayout));
  }

  offset = alignUpTo(offset, layout.alignment);
  appendProgramLayoutRegion(layout, "runtime_bookkeeping",
                            "runtime_bookkeeping", offset,
                            kVC4RuntimeBookkeepingBytes);
  offset += kVC4RuntimeBookkeepingBytes;

  if (layout.spillArenaBytes != 0) {
    offset = alignUpTo(offset, kVC4SpillFrameAlignment);
    layout.spillArenaOffset = offset;
    appendProgramLayoutRegion(layout, "hidden_spill_arena",
                              "hidden_spill_arena", offset,
                              layout.spillArenaBytes);
    offset += layout.spillArenaBytes;
  }

  offset = alignUpTo(offset, kVC4HeapAlignment);
  layout.heapOffset = offset;
  appendProgramLayoutRegion(layout, "heap", "heap", layout.heapOffset,
                            layout.heapBytes);
  offset += layout.heapBytes;

  layout.staticBytes = offset;
  layout.programBytes = offset;
  return layout;
}

static LogicalResult
rejectDuplicateKernelPublicNames(mlir::vc4::ModuleOp vc4Module,
                                 llvm::ArrayRef<KernelRecord> kernels) {
  for (size_t i = 0; i != kernels.size(); ++i) {
    for (size_t j = 0; j != i; ++j) {
      if (kernels[i].info.publicName != kernels[j].info.publicName)
        continue;
      return vc4Module.emitOpError()
             << "duplicate public_name '" << kernels[i].info.publicName
             << "' in VC4 QPU program artifact";
    }
  }
  return success();
}

static LogicalResult
rejectDuplicateKernelCodeSymbols(mlir::vc4::ModuleOp vc4Module,
                                 llvm::ArrayRef<KernelRecord> kernels) {
  for (size_t i = 0; i != kernels.size(); ++i) {
    for (size_t j = 0; j != i; ++j) {
      if (kernels[i].launchABI.codeSymbol != kernels[j].launchABI.codeSymbol)
        continue;
      return vc4Module.emitOpError()
             << "duplicate code_symbol '" << kernels[i].launchABI.codeSymbol
             << "' in VC4 QPU program artifact";
    }
  }
  return success();
}

static LogicalResult
collectProgramKernels(mlir::vc4::ModuleOp vc4Module,
                      llvm::SmallVectorImpl<KernelRecord> &kernels) {
  vc4Module.walk([&](mlir::vc4::FuncOp func) {
    if (isScheduledQPUKernel(func))
      kernels.emplace_back(func);
  });

  if (kernels.empty()) {
    return vc4Module.emitOpError()
           << "expected at least one eligible VC4 QPU kernel: one or more "
              "kernel qpu scheduled vc4.func ops for program artifact emission";
  }

  for (KernelRecord &kernel : kernels) {
    if (failed(populateLaunchABIInfo(kernel)))
      return failure();
    if (failed(populateResourceInfo(kernel)))
      return failure();
    if (failed(populateSpillFrameInfo(kernel)))
      return failure();
  }

  if (failed(rejectDuplicateKernelPublicNames(vc4Module, kernels)))
    return failure();
  if (failed(rejectDuplicateKernelCodeSymbols(vc4Module, kernels)))
    return failure();

  for (size_t i = 0; i != kernels.size(); ++i) {
    KernelRecord &kernel = kernels[i];
    kernel.kernelId = static_cast<unsigned>(i);
    kernel.qasmPath = makeKernelQASMPath(kernel.info.publicName);
    if (failed(populateScheduledSinkInfo(kernel)))
      return failure();
  }

  return success();
}

static LogicalResult
writeTextFile(mlir::Operation *diagOp, llvm::StringRef path,
              llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::CD_CreateAlways,
                          llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Text);
  if (ec) {
    return diagOp->emitError()
           << "failed to open artifact file '" << path << "': " << ec.message();
  }

  emit(os);
  os.flush();
  if (os.has_error()) {
    std::error_code writeError = os.error();
    os.clear_error();
    return diagOp->emitError() << "failed to write artifact file '" << path
                               << "': " << writeError.message();
  }

  return success();
}

static LogicalResult
writeBundleFile(mlir::Operation *diagOp, llvm::StringRef bundleDir,
                llvm::StringRef fileName,
                llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  llvm::SmallString<256> path(bundleDir);
  llvm::sys::path::append(path, fileName);

  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  std::error_code ec = llvm::sys::fs::create_directories(parent);
  if (ec) {
    return diagOp->emitError() << "failed to create artifact directory '"
                               << parent << "': " << ec.message();
  }

  return writeTextFile(diagOp, path, emit);
}

static LogicalResult ensureAdjacentVC4ASMTemplates(mlir::Operation *diagOp,
                                                   llvm::StringRef bundleDir) {
  llvm::SmallString<256> templateDir(bundleDir);
  llvm::sys::path::remove_filename(templateDir);
  llvm::sys::path::append(templateDir, "share", "vc4tmpl");

  std::error_code ec = llvm::sys::fs::create_directories(templateDir);
  if (ec) {
    return diagOp->emitError() << "failed to create vc4asm template directory '"
                               << templateDir << "': " << ec.message();
  }

  llvm::SmallString<256> templateHeader(templateDir);
  llvm::sys::path::append(templateHeader, "template.h");
  if (failed(writeTextFile(diagOp, templateHeader, [](llvm::raw_ostream &os) {
        os << "#ifndef ___SYMBOLNAME____H\n";
        os << "#define ___SYMBOLNAME____H\n\n";
        os << "#include <stdint.h>\n\n";
        os << "#ifdef __cplusplus\n";
        os << "extern \"C\" {\n";
        os << "#endif\n\n";
        os << "extern uint32_t ___SYMBOLNAME___[___INSTCOUNT2___];\n";
        os << "___SYMBOLDEFS___\n";
        os << "#ifdef __cplusplus\n";
        os << "}\n";
        os << "#endif\n\n";
        os << "#endif\n";
      })))
    return failure();

  llvm::SmallString<256> templateSource(templateDir);
  llvm::sys::path::append(templateSource, "template.c");
  if (failed(writeTextFile(diagOp, templateSource, [](llvm::raw_ostream &os) {
        os << "#include \"___HEADERNAME___\"\n\n";
        os << "#ifdef __cplusplus\n";
        os << "extern \"C\" {\n";
        os << "#endif\n\n";
        os << "#ifdef _MSC_VER\n";
        os << "__declspec(align(8))\n";
        os << "#elif defined(__GNUC__)\n";
        os << "__attribute__((aligned(8)))\n";
        os << "#endif\n";
        os << "uint32_t ___SYMBOLNAME___[___INSTCOUNT2___] = {\n";
        os << "___HEXDATA___";
        os << "};\n\n";
        os << "#ifdef __cplusplus\n";
        os << "}\n";
        os << "#endif\n";
      })))
    return failure();

  llvm::SmallString<256> templateHeaderNoInline(templateDir);
  llvm::sys::path::append(templateHeaderNoInline, "template2.h");
  if (failed(writeTextFile(diagOp, templateHeaderNoInline,
                           [](llvm::raw_ostream &os) {
                             os << "#ifndef ___SYMBOLNAME____H\n";
                             os << "#define ___SYMBOLNAME____H\n\n";
                             os << "#include <stdint.h>\n\n";
                             os << "struct unspecified__;\n\n";
                             os << "#ifdef __cplusplus\n";
                             os << "extern \"C\" {\n";
                             os << "#endif\n\n";
                             os << "___SYMBOLIMPORTS___\n";
                             os << "#ifdef __cplusplus\n";
                             os << "}\n";
                             os << "#endif\n\n";
                             os << "___SYMBOLPROXIES___\n";
                             os << "#endif\n";
                           })))
    return failure();

  return success();
}

static int64_t getIntegerAttrValue(mlir::Operation *op,
                                   llvm::StringRef attrName) {
  return llvm::cast<mlir::IntegerAttr>(op->getAttr(attrName)).getInt();
}

static std::optional<int64_t>
getOptionalIntegerAttrValue(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static const char *getAddOpcodeMnemonic(mlir::vc4::AddOpcode opcode) {
  switch (opcode) {
  case mlir::vc4::AddOpcode::nop:
    return "nop";
  case mlir::vc4::AddOpcode::fadd:
    return "fadd";
  case mlir::vc4::AddOpcode::fsub:
    return "fsub";
  case mlir::vc4::AddOpcode::fmin:
    return "fmin";
  case mlir::vc4::AddOpcode::fmax:
    return "fmax";
  case mlir::vc4::AddOpcode::fminabs:
    return "fminabs";
  case mlir::vc4::AddOpcode::fmaxabs:
    return "fmaxabs";
  case mlir::vc4::AddOpcode::ftoi:
    return "ftoi";
  case mlir::vc4::AddOpcode::itof:
    return "itof";
  case mlir::vc4::AddOpcode::add:
    return "add";
  case mlir::vc4::AddOpcode::sub:
    return "sub";
  case mlir::vc4::AddOpcode::shr:
    return "shr";
  case mlir::vc4::AddOpcode::asr:
    return "asr";
  case mlir::vc4::AddOpcode::ror:
    return "ror";
  case mlir::vc4::AddOpcode::shl:
    return "shl";
  case mlir::vc4::AddOpcode::min:
    return "min";
  case mlir::vc4::AddOpcode::max:
    return "max";
  case mlir::vc4::AddOpcode::bit_and:
    return "and";
  case mlir::vc4::AddOpcode::bit_or:
    return "or";
  case mlir::vc4::AddOpcode::bit_xor:
    return "xor";
  case mlir::vc4::AddOpcode::bit_not:
    return "not";
  case mlir::vc4::AddOpcode::clz:
    return "clz";
  case mlir::vc4::AddOpcode::v8adds:
    return "v8adds";
  case mlir::vc4::AddOpcode::v8subs:
    return "v8subs";
  }
  return "unknown";
}

static const char *getMulOpcodeMnemonic(mlir::vc4::MulOpcode opcode) {
  switch (opcode) {
  case mlir::vc4::MulOpcode::nop:
    return "nop";
  case mlir::vc4::MulOpcode::fmul:
    return "fmul";
  case mlir::vc4::MulOpcode::mul24:
    return "mul24";
  case mlir::vc4::MulOpcode::v8muld:
    return "v8muld";
  case mlir::vc4::MulOpcode::v8min:
    return "v8min";
  case mlir::vc4::MulOpcode::v8max:
    return "v8max";
  case mlir::vc4::MulOpcode::v8adds:
    return "v8adds";
  case mlir::vc4::MulOpcode::v8subs:
    return "v8subs";
  }
  return "unknown";
}

static bool isUnaryAddOpcode(mlir::vc4::AddOpcode opcode) {
  return opcode == mlir::vc4::AddOpcode::ftoi ||
         opcode == mlir::vc4::AddOpcode::itof ||
         opcode == mlir::vc4::AddOpcode::bit_not ||
         opcode == mlir::vc4::AddOpcode::clz;
}

static const char *getConditionSuffix(mlir::vc4::Cond cond) {
  switch (cond) {
  case mlir::vc4::Cond::never:
  case mlir::vc4::Cond::always:
    return "";
  case mlir::vc4::Cond::zs:
    return ".ifz";
  case mlir::vc4::Cond::zc:
    return ".ifnz";
  case mlir::vc4::Cond::ns:
    return ".ifn";
  case mlir::vc4::Cond::nc:
    return ".ifnn";
  case mlir::vc4::Cond::cs:
    return ".ifc";
  case mlir::vc4::Cond::cc:
    return ".ifcc";
  }
  return "";
}

static std::string formatSmallImmSelector(int64_t selector) {
  if (selector >= 0 && selector <= 15)
    return std::to_string(selector);
  if (selector >= 16 && selector <= 31)
    return std::to_string(selector - 32);
  if (selector >= 32 && selector <= 39) {
    int64_t value = int64_t{1} << (selector - 32);
    return std::to_string(value) + ".0";
  }
  if (selector >= 40 && selector <= 47) {
    int64_t denominator = int64_t{1} << (48 - selector);
    return "1./" + std::to_string(denominator);
  }
  return "<unsupported-vector-rotate-small-imm>";
}

static bool isVectorRotateSmallImm(int64_t selector) {
  return selector >= 48 && selector <= 63;
}

static std::string formatRegFileAddress(char regFile, int64_t address) {
  std::string result;
  result.push_back('r');
  result.push_back(regFile);
  result += std::to_string(address);
  return result;
}

static std::string getVPMSetupName(mlir::vc4::VPMVCDSide side) {
  return side == mlir::vc4::VPMVCDSide::read ? std::string("vr_setup")
                                             : std::string("vw_setup");
}

static std::string getVPMAddressName(mlir::vc4::VPMVCDSide side) {
  return side == mlir::vc4::VPMVCDSide::read ? std::string("vr_addr")
                                             : std::string("vw_addr");
}

static std::string getVPMWaitName(mlir::vc4::VPMVCDSide side) {
  return side == mlir::vc4::VPMVCDSide::read ? std::string("vr_wait")
                                             : std::string("vw_wait");
}

static std::string formatReadAddress(char regFile, int64_t address) {
  // vc4asm names several read-side peripheral addresses symbolically.  Do not
  // print them as ordinary regfile locations: e.g. ra32 is not a uniform read
  // in the source language the hardware reference uses.
  switch (address) {
  case 32:
    return "unif";
  case 38:
    return regFile == 'a' ? "elem_num" : "qpu_num";
  case 48:
    return "vpm";
  case 51:
    return "mutex_acq";
  default:
    return formatRegFileAddress(regFile, address);
  }
}

static std::string formatWriteAddress(int64_t address, bool forAddALU,
                                      bool writeSwap) {
  if (address >= 32 && address <= 36)
    return "r" + std::to_string(address - 32);
  if (address == 37)
    return "r5quad";
  if (address == 39)
    return "-";

  // vc4asm also requires symbolic names for side-effecting peripheral writes.
  // Address-49/50 VPM/VCD/VDW setup/address writes must use explicit
  // vc4.qpu.vpmvcd_* pseudo-ops, where the read/write side is carried by the
  // op itself.  Raw address-49/50 scheduled ops are rejected before emission.
  switch (address) {
  case 48:
    return "vpm";
  case 51:
    return "mutex_rel";
  case 52:
    return "recip";
  case 53:
    return "recipsqrt";
  case 54:
    return "exp";
  case 55:
    return "log";
  case 56:
    return "t0s";
  default:
    break;
  }

  char regFile = forAddALU ? (writeSwap ? 'b' : 'a') : (writeSwap ? 'a' : 'b');
  return formatRegFileAddress(regFile, address);
}

static std::string printAttributeToString(mlir::Attribute attr) {
  std::string result;
  llvm::raw_string_ostream os(result);
  attr.print(os);
  return result;
}

static std::optional<std::string> getPackSuffix(mlir::Attribute attr) {
  if (!attr)
    return std::string();
  std::string text = printAttributeToString(attr);
  if (text.find("<none>") != std::string::npos)
    return std::string();
  if (text.find("<to_16a>") != std::string::npos)
    return std::string(".16a");
  if (text.find("<to_16b>") != std::string::npos)
    return std::string(".16b");
  if (text.find("<to_8888>") != std::string::npos)
    return std::string(".8888");
  if (text.find("<to_8a>") != std::string::npos)
    return std::string(".8a");
  if (text.find("<to_8b>") != std::string::npos)
    return std::string(".8b");
  if (text.find("<to_8c>") != std::string::npos)
    return std::string(".8c");
  if (text.find("<to_8d>") != std::string::npos)
    return std::string(".8d");
  if (text.find("<sat32>") != std::string::npos)
    return std::string(".32s");
  if (text.find("<sat16a>") != std::string::npos)
    return std::string(".16as");
  if (text.find("<sat16b>") != std::string::npos)
    return std::string(".16bs");
  if (text.find("<sat8888>") != std::string::npos)
    return std::string(".8888s");
  if (text.find("<sat8a>") != std::string::npos)
    return std::string(".8as");
  if (text.find("<sat8b>") != std::string::npos)
    return std::string(".8bs");
  if (text.find("<sat8c>") != std::string::npos)
    return std::string(".8cs");
  if (text.find("<sat8d>") != std::string::npos)
    return std::string(".8ds");
  return std::nullopt;
}

static std::optional<std::string> getUnpackSuffix(mlir::Attribute attr) {
  if (!attr)
    return std::string();
  std::string text = printAttributeToString(attr);
  if (text.find("<none>") != std::string::npos)
    return std::string();
  if (text.find("<f16a_or_i16a>") != std::string::npos ||
      text.find("<f16a>") != std::string::npos)
    return std::string(".16a");
  if (text.find("<f16b_or_i16b>") != std::string::npos ||
      text.find("<f16b>") != std::string::npos)
    return std::string(".16b");
  if (text.find("<replicate_8d>") != std::string::npos)
    return std::string(".8dr");
  if (text.find("<color8a>") != std::string::npos)
    return std::string(".8a");
  if (text.find("<color8b>") != std::string::npos)
    return std::string(".8b");
  if (text.find("<color8c>") != std::string::npos)
    return std::string(".8c");
  if (text.find("<color8d>") != std::string::npos)
    return std::string(".8d");
  return std::nullopt;
}

static std::optional<std::string>
formatVectorRotateSmallImmSource(mlir::vc4::QPUBundleOp bundle,
                                 int64_t selector);

static LogicalResult formatMuxSource(mlir::vc4::QPUBundleOp bundle,
                                     mlir::vc4::QPUMux mux,
                                     std::string unpackSuffix,
                                     std::string &out) {
  int64_t raddrA = getIntegerAttrValue(bundle.getOperation(), "raddr_a");
  std::optional<int64_t> raddrB =
      getOptionalIntegerAttrValue(bundle.getOperation(), "raddr_b");
  std::optional<int64_t> smallImm =
      getOptionalIntegerAttrValue(bundle.getOperation(), "small_imm");

  switch (mux) {
  case mlir::vc4::QPUMux::r0:
    out = "r0";
    return success();
  case mlir::vc4::QPUMux::r1:
    out = "r1";
    return success();
  case mlir::vc4::QPUMux::r2:
    out = "r2";
    return success();
  case mlir::vc4::QPUMux::r3:
    out = "r3";
    return success();
  case mlir::vc4::QPUMux::r4:
    out = "r4" + unpackSuffix;
    return success();
  case mlir::vc4::QPUMux::r5:
    out = "r5";
    return success();
  case mlir::vc4::QPUMux::a:
    out = formatReadAddress('a', raddrA) + unpackSuffix;
    return success();
  case mlir::vc4::QPUMux::b:
    if (smallImm) {
      if (isVectorRotateSmallImm(*smallImm)) {
        std::optional<std::string> rotated =
            formatVectorRotateSmallImmSource(bundle, *smallImm);
        if (!rotated) {
          return bundle.emitOpError()
                 << "cannot emit vector-rotate small_imm selector " << *smallImm
                 << "; expected matching MUL accumulator operands r0-r3";
        }
        out = *rotated;
        return success();
      }
      out = formatSmallImmSelector(*smallImm);
      return success();
    }
    if (!raddrB) {
      return bundle.emitOpError()
             << "cannot emit source mux #vc4.qpu_mux<b> without raddr_b or "
                "small_imm";
    }
    out = formatReadAddress('b', *raddrB);
    return success();
  }
  return bundle.emitOpError() << "cannot emit unknown qpu source mux";
}

static std::optional<unsigned>
getAccumulatorIndexForMux(mlir::vc4::QPUMux mux) {
  switch (mux) {
  case mlir::vc4::QPUMux::r0:
    return 0;
  case mlir::vc4::QPUMux::r1:
    return 1;
  case mlir::vc4::QPUMux::r2:
    return 2;
  case mlir::vc4::QPUMux::r3:
    return 3;
  case mlir::vc4::QPUMux::r4:
    return 4;
  case mlir::vc4::QPUMux::r5:
    return 5;
  default:
    return std::nullopt;
  }
}

static std::optional<std::string>
formatVectorRotateSmallImmSource(mlir::vc4::QPUBundleOp bundle,
                                 int64_t selector) {
  // VC4 encodes horizontal vector rotates in the small-immediate field.
  // vc4asm spells fixed rotates as `register << amount` and the r5-selected
  // rotate as `register >> r5`.  Full-width rotates require the MUL source to
  // be an accumulator r0-r3; the scheduled stream records that source through
  // the MUL mux operands even when the arithmetic result is consumed by the
  // ADD pipe as mux-b.
  if (!isVectorRotateSmallImm(selector))
    return std::nullopt;

  if (bundle.getMulA() != bundle.getMulB())
    return std::nullopt;

  std::optional<unsigned> accumulator =
      getAccumulatorIndexForMux(bundle.getMulA());
  if (!accumulator || *accumulator > 3)
    return std::nullopt;

  std::string source = "r" + std::to_string(*accumulator);
  if (selector == 48)
    return source + " >> r5";
  return source + " << " + std::to_string(selector - 48);
}

static std::optional<std::string>
formatAccumulatorVectorRotateSource(mlir::vc4::QPUMux sourceMux,
                                    int64_t selector) {
  if (!isVectorRotateSmallImm(selector))
    return std::nullopt;

  std::optional<unsigned> accumulator = getAccumulatorIndexForMux(sourceMux);
  if (!accumulator || *accumulator > 3)
    return std::nullopt;

  std::string source = "r" + std::to_string(*accumulator);
  if (selector == 48)
    return source + " >> r5";
  return source + " << " + std::to_string(selector - 48);
}

static LogicalResult formatPseudoVPMVCDMuxSource(mlir::Operation *op,
                                                 mlir::vc4::QPUMux mux,
                                                 mlir::vc4::QPUMux mulA,
                                                 mlir::vc4::QPUMux mulB,
                                                 std::string &out) {
  switch (mux) {
  case mlir::vc4::QPUMux::r0:
    out = "r0";
    return success();
  case mlir::vc4::QPUMux::r1:
    out = "r1";
    return success();
  case mlir::vc4::QPUMux::r2:
    out = "r2";
    return success();
  case mlir::vc4::QPUMux::r3:
    out = "r3";
    return success();
  case mlir::vc4::QPUMux::r4:
    out = "r4";
    return success();
  case mlir::vc4::QPUMux::r5:
    out = "r5";
    return success();
  case mlir::vc4::QPUMux::a:
    out = formatReadAddress('a', getIntegerAttrValue(op, "raddr_a"));
    return success();
  case mlir::vc4::QPUMux::b:
    if (std::optional<int64_t> smallImm =
            getOptionalIntegerAttrValue(op, "small_imm")) {
      if (isVectorRotateSmallImm(*smallImm)) {
        if (mulA != mulB)
          return op->emitOpError()
                 << "cannot emit vector-rotate small_imm selector " << *smallImm
                 << "; expected matching MUL accumulator operands r0-r3";
        if (std::optional<std::string> rotated =
                formatAccumulatorVectorRotateSource(mulA, *smallImm)) {
          out = *rotated;
          return success();
        }
        return op->emitOpError()
               << "cannot emit vector-rotate small_imm selector " << *smallImm
               << "; expected matching MUL accumulator operands r0-r3";
      }
      out = formatSmallImmSelector(*smallImm);
      return success();
    }
    if (std::optional<int64_t> raddrB =
            getOptionalIntegerAttrValue(op, "raddr_b")) {
      out = formatReadAddress('b', *raddrB);
      return success();
    }
    return op->emitOpError()
           << "cannot emit source mux #vc4.qpu_mux<b> without raddr_b or "
              "small_imm";
  }
  return op->emitOpError() << "cannot emit unknown qpu source mux";
}

static bool qpuBundleWritesAddress(mlir::vc4::QPUBundleOp bundle,
                                   int64_t address) {
  mlir::Operation *op = bundle.getOperation();
  return (bundle.getCondAdd() != mlir::vc4::Cond::never &&
          getIntegerAttrValue(op, "waddr_add") == address) ||
         (bundle.getCondMul() != mlir::vc4::Cond::never &&
          getIntegerAttrValue(op, "waddr_mul") == address);
}

static bool qpuBundleCarriesReadAddress(mlir::vc4::QPUBundleOp bundle,
                                        int64_t address) {
  mlir::Operation *op = bundle.getOperation();
  if (getIntegerAttrValue(op, "raddr_a") == address)
    return true;
  std::optional<int64_t> raddrB = getOptionalIntegerAttrValue(op, "raddr_b");
  return raddrB && *raddrB == address;
}

static std::optional<int64_t>
getVectorRotateSmallImmSelectorForMux(mlir::vc4::QPUBundleOp bundle,
                                      mlir::vc4::QPUMux mux) {
  if (mux != mlir::vc4::QPUMux::b)
    return std::nullopt;
  std::optional<int64_t> smallImm =
      getOptionalIntegerAttrValue(bundle.getOperation(), "small_imm");
  if (!smallImm || !isVectorRotateSmallImm(*smallImm))
    return std::nullopt;
  return smallImm;
}

static std::optional<int64_t>
getAddVectorRotateSmallImmSelector(mlir::vc4::QPUBundleOp bundle) {
  if (bundle.getOpAdd() == mlir::vc4::AddOpcode::nop)
    return std::nullopt;
  if (std::optional<int64_t> selector =
          getVectorRotateSmallImmSelectorForMux(bundle, bundle.getAddA()))
    return selector;
  if (!isUnaryAddOpcode(bundle.getOpAdd()))
    return getVectorRotateSmallImmSelectorForMux(bundle, bundle.getAddB());
  return std::nullopt;
}

static std::optional<std::string>
formatStandaloneAddVectorRotateMove(mlir::vc4::QPUBundleOp bundle) {
  // Some scheduled reduction streams materialize a rotate as the canonical
  // ADD-side move shape `or rD, rS, rotate(rS)`.  vc4asm spells this as a
  // unary rotate move.  Treat only that pure ADD shape as a move so reductions
  // can be assembled without needing a stream-specific rewrite.
  if (bundle.getOpAdd() != mlir::vc4::AddOpcode::bit_or ||
      bundle.getOpMul() != mlir::vc4::MulOpcode::nop ||
      bundle.getCondAdd() == mlir::vc4::Cond::never ||
      bundle.getOperation()->hasAttr("set_flags"))
    return std::nullopt;

  std::optional<int64_t> selector =
      getVectorRotateSmallImmSelectorForMux(bundle, bundle.getAddB());
  if (!selector)
    return std::nullopt;

  return formatAccumulatorVectorRotateSource(bundle.getAddA(), *selector);
}

static bool
canEmitStandaloneAddVectorRotateMove(mlir::vc4::QPUBundleOp bundle) {
  return formatStandaloneAddVectorRotateMove(bundle).has_value();
}

static bool accumulatorIsForbidden(unsigned accumulator,
                                   llvm::ArrayRef<unsigned> forbidden) {
  for (unsigned value : forbidden) {
    if (value == accumulator)
      return true;
  }
  return false;
}

static void
forbidNonRotateAddAccumulator(mlir::vc4::QPUBundleOp bundle,
                              mlir::vc4::QPUMux mux,
                              llvm::SmallVectorImpl<unsigned> &forbidden) {
  if (mux == mlir::vc4::QPUMux::b &&
      getVectorRotateSmallImmSelectorForMux(bundle, mux))
    return;
  if (std::optional<unsigned> accumulator = getAccumulatorIndexForMux(mux))
    forbidden.push_back(*accumulator);
}

static std::optional<unsigned>
getPreparedVectorRotateDestination(mlir::vc4::QPUBundleOp bundle) {
  if (!getAddVectorRotateSmallImmSelector(bundle))
    return std::nullopt;
  if (bundle.getMulA() != bundle.getMulB())
    return std::nullopt;

  std::optional<unsigned> source = getAccumulatorIndexForMux(bundle.getMulA());
  if (!source || *source > 3)
    return std::nullopt;

  llvm::SmallVector<unsigned, 4> forbidden;
  forbidden.push_back(*source);
  forbidNonRotateAddAccumulator(bundle, bundle.getAddA(), forbidden);
  if (!isUnaryAddOpcode(bundle.getOpAdd()))
    forbidNonRotateAddAccumulator(bundle, bundle.getAddB(), forbidden);

  const unsigned candidates[] = {2u, 3u, 1u, 0u};
  for (unsigned candidate : candidates) {
    if (!accumulatorIsForbidden(candidate, forbidden))
      return candidate;
  }
  return std::nullopt;
}

static bool isPureNopBundle(mlir::vc4::QPUBundleOp bundle) {
  return bundle.getOpAdd() == mlir::vc4::AddOpcode::nop &&
         bundle.getOpMul() == mlir::vc4::MulOpcode::nop &&
         bundle.getSig() == mlir::vc4::QPUSignal::none;
}

static bool canPrepareVectorRotateFromSpacer(mlir::vc4::QPUBundleOp spacer,
                                             mlir::vc4::QPUBundleOp next) {
  return isPureNopBundle(spacer) && getAddVectorRotateSmallImmSelector(next) &&
         !canEmitStandaloneAddVectorRotateMove(next) &&
         getPreparedVectorRotateDestination(next);
}

static LogicalResult emitVectorRotatePrepBundle(mlir::vc4::QPUBundleOp spacer,
                                                mlir::vc4::QPUBundleOp next,
                                                QASMEmissionState &state,
                                                llvm::raw_ostream &os) {
  std::optional<int64_t> selector = getAddVectorRotateSmallImmSelector(next);
  std::optional<unsigned> source = getAccumulatorIndexForMux(next.getMulA());
  std::optional<unsigned> destination =
      getPreparedVectorRotateDestination(next);
  if (!selector || !source || !destination) {
    return spacer.emitOpError()
           << "cannot prepare vector-rotate small_imm operand for following "
              "ADD instruction";
  }

  // vc4asm accepts horizontal vector rotates only on the MUL side.  When the
  // scheduled stream has an ADD reduction that consumes a rotated accumulator,
  // use the immediately preceding pure spacer slot to materialize the rotate
  // through a scratch accumulator.  The following ADD then reads that scratch
  // accumulator as an ordinary operand, preserving the scheduled slot count and
  // symbolic branch labels.
  os << "mov r" << *destination << ", r" << *source;
  if (*selector == 48)
    os << " >> r5";
  else
    os << " << " << (*selector - 48);
  os << "\n";

  state.preparedVectorRotateDest = *destination;
  state.preparedVectorRotateSelector = *selector;
  return success();
}

static LogicalResult appendAddInstruction(mlir::vc4::QPUBundleOp bundle,
                                          QASMEmissionState &state,
                                          std::string &line,
                                          bool &needSeparator) {
  mlir::vc4::AddOpcode opcode = bundle.getOpAdd();
  if (opcode == mlir::vc4::AddOpcode::nop)
    return success();

  std::optional<std::string> packSuffix =
      getPackSuffix(bundle.getOperation()->getAttr("pack"));
  std::optional<std::string> unpackSuffix =
      getUnpackSuffix(bundle.getOperation()->getAttr("unpack"));
  if (!packSuffix) {
    return bundle.emitOpError()
           << "cannot emit unsupported qpu.bundle pack attribute '"
           << bundle.getOperation()->getAttr("pack") << "'";
  }
  if (!unpackSuffix) {
    return bundle.emitOpError()
           << "cannot emit unsupported qpu.bundle unpack attribute '"
           << bundle.getOperation()->getAttr("unpack") << "'";
  }

  if (std::optional<std::string> rotateMove =
          formatStandaloneAddVectorRotateMove(bundle)) {
    if (needSeparator)
      line += "; ";
    needSeparator = true;
    line += "mov";
    if (bundle.getCondAdd() != mlir::vc4::Cond::never)
      line += getConditionSuffix(bundle.getCondAdd());
    line += " ";
    line +=
        formatWriteAddress(
            getIntegerAttrValue(bundle.getOperation(), "waddr_add"),
            /*forAddALU=*/true, bundle.getOperation()->hasAttr("write_swap")) +
        *packSuffix;
    line += ", ";
    line += *rotateMove;
    state.preparedVectorRotateDest = std::nullopt;
    state.preparedVectorRotateSelector = std::nullopt;
    return success();
  }

  std::string src0;
  std::string src1;
  bool printUnpackOnSecondSource =
      !unpackSuffix->empty() && !isUnaryAddOpcode(opcode);
  if (failed(formatMuxSource(
          bundle, bundle.getAddA(),
          printUnpackOnSecondSource ? std::string() : *unpackSuffix, src0)))
    return failure();
  if (!isUnaryAddOpcode(opcode) &&
      failed(formatMuxSource(
          bundle, bundle.getAddB(),
          printUnpackOnSecondSource ? *unpackSuffix : std::string(), src1)))
    return failure();

  if (std::optional<int64_t> rotateSelector =
          getAddVectorRotateSmallImmSelector(bundle)) {
    if (!canEmitStandaloneAddVectorRotateMove(bundle)) {
      std::optional<unsigned> destination =
          getPreparedVectorRotateDestination(bundle);
      if (!destination || !state.preparedVectorRotateDest ||
          !state.preparedVectorRotateSelector ||
          *state.preparedVectorRotateDest != *destination ||
          *state.preparedVectorRotateSelector != *rotateSelector) {
        return bundle.emitOpError()
               << "ADD-side vector-rotate small_imm selector "
               << *rotateSelector
               << " requires a prepared MUL-side rotate in the preceding "
                  "spacer";
      }
      std::string preparedSource = "r" + std::to_string(*destination);
      if (bundle.getAddA() == mlir::vc4::QPUMux::b)
        src0 = preparedSource;
      if (!isUnaryAddOpcode(opcode) && bundle.getAddB() == mlir::vc4::QPUMux::b)
        src1 = preparedSource;
      state.preparedVectorRotateDest = std::nullopt;
      state.preparedVectorRotateSelector = std::nullopt;
    }
  }

  if (needSeparator)
    line += "; ";
  needSeparator = true;

  mlir::vc4::Cond cond = bundle.getCondAdd();
  mlir::Operation *op = bundle.getOperation();

  // The scheduled stream sometimes represents a VPM DMA wait as an ADD-side
  // instruction with a dummy second source, e.g.
  //   add ra31, vw_wait, rb0
  //   add ra31, vr_wait, rb31
  // vc4asm maps some wait aliases through the same regfile as the dummy source
  // and rejects that spelling with A20.  The dummy source read is not part of
  // the wait effect, so emit the canonical unary wait move while preserving the
  // scheduled destination. This is generic over the bundle shape.
  if (bundle.getOpMul() == mlir::vc4::MulOpcode::nop &&
      cond == mlir::vc4::Cond::always && opcode == mlir::vc4::AddOpcode::add &&
      !op->hasAttr("set_flags") &&
      (src0 == "vw_wait" || src1 == "vw_wait" || src0 == "vr_wait" ||
       src1 == "vr_wait")) {
    std::string waitSource =
        (src0 == "vw_wait" || src1 == "vw_wait") ? "vw_wait" : "vr_wait";
    std::string dest =
        formatWriteAddress(getIntegerAttrValue(op, "waddr_add"),
                           /*forAddALU=*/true, op->hasAttr("write_swap")) +
        *packSuffix;
    line += "mov ";
    line += dest;
    line += ", ";
    line += waitSource;
    return success();
  }

  bool useMovAlias = false;
  std::string movAliasSource = src0;
  if (cond == mlir::vc4::Cond::always && !op->hasAttr("set_flags")) {
    if (opcode == mlir::vc4::AddOpcode::bit_or && !unpackSuffix->empty() &&
        bundle.getAddA() == bundle.getAddB()) {
      useMovAlias = true;
      movAliasSource = src1;
    } else if (opcode == mlir::vc4::AddOpcode::bit_or && src0 == src1) {
      useMovAlias = true;
      movAliasSource = src0;
    } else if (opcode == mlir::vc4::AddOpcode::add && src1 == "0") {
      // Canonicalize scheduled identity adds into vc4asm's unary move form.
      // This is especially important for peripheral reads/writes such as
      // uniform, VPM, VPM DMA address, and VPM wait registers: the IR often
      // carries them as add-with-zero because it is already in final scheduled
      // slot form, but the hardware reference spellings use mov and avoid
      // unintended second-source register-file traffic.
      useMovAlias = true;
      movAliasSource = src0;
    } else if (opcode == mlir::vc4::AddOpcode::add && src0 == "0") {
      useMovAlias = true;
      movAliasSource = src1;
    }
  }

  line += useMovAlias ? "mov" : getAddOpcodeMnemonic(opcode);
  if (bundle.getOperation()->hasAttr("set_flags"))
    line += ".setf";
  if (cond != mlir::vc4::Cond::never)
    line += getConditionSuffix(cond);

  std::string dest =
      cond == mlir::vc4::Cond::never
          ? std::string("-")
          : formatWriteAddress(
                getIntegerAttrValue(bundle.getOperation(), "waddr_add"),
                /*forAddALU=*/true,
                bundle.getOperation()->hasAttr("write_swap")) +
                *packSuffix;
  line += " ";
  line += dest;
  line += ", ";
  line += useMovAlias ? movAliasSource : src0;
  if (!useMovAlias && !isUnaryAddOpcode(opcode)) {
    line += ", ";
    line += src1;
  }
  return success();
}

static LogicalResult appendMulInstruction(mlir::vc4::QPUBundleOp bundle,
                                          std::string &line,
                                          bool &needSeparator,
                                          bool setFlagsAlreadyUsed) {
  mlir::vc4::MulOpcode opcode = bundle.getOpMul();
  if (opcode == mlir::vc4::MulOpcode::nop)
    return success();

  std::optional<std::string> packSuffix =
      getPackSuffix(bundle.getOperation()->getAttr("pack"));
  std::optional<std::string> unpackSuffix =
      getUnpackSuffix(bundle.getOperation()->getAttr("unpack"));
  if (!packSuffix) {
    return bundle.emitOpError()
           << "cannot emit unsupported qpu.bundle pack attribute '"
           << bundle.getOperation()->getAttr("pack") << "'";
  }
  if (!unpackSuffix) {
    return bundle.emitOpError()
           << "cannot emit unsupported qpu.bundle unpack attribute '"
           << bundle.getOperation()->getAttr("unpack") << "'";
  }

  std::string src0;
  std::string src1;
  // The r4 unpack path is selected by pm=true.  Regfile-A unpack suffixes do
  // not apply to MUL accumulator operands, so only pass the textual unpack
  // suffix through when the pm-selected r4 path could use it.
  std::string maybeR4Unpack = bundle.getPm() ? *unpackSuffix : std::string();
  if (failed(formatMuxSource(bundle, bundle.getMulA(), maybeR4Unpack, src0)))
    return failure();
  if (failed(formatMuxSource(bundle, bundle.getMulB(), maybeR4Unpack, src1)))
    return failure();

  if (needSeparator)
    line += "; ";
  needSeparator = true;

  mlir::vc4::Cond cond = bundle.getCondMul();
  line += getMulOpcodeMnemonic(opcode);
  if (bundle.getOperation()->hasAttr("set_flags") && !setFlagsAlreadyUsed)
    line += ".setf";
  if (cond != mlir::vc4::Cond::never)
    line += getConditionSuffix(cond);

  std::string dest =
      cond == mlir::vc4::Cond::never
          ? std::string("-")
          : formatWriteAddress(
                getIntegerAttrValue(bundle.getOperation(), "waddr_mul"),
                /*forAddALU=*/false,
                bundle.getOperation()->hasAttr("write_swap")) +
                *packSuffix;
  line += " ";
  line += dest;
  line += ", ";
  line += src0;
  line += ", ";
  line += src1;
  return success();
}

static LogicalResult appendSignalInstruction(mlir::vc4::QPUBundleOp bundle,
                                             std::string &line,
                                             bool &needSeparator) {
  const char *signal = nullptr;
  switch (bundle.getSig()) {
  case mlir::vc4::QPUSignal::none:
  case mlir::vc4::QPUSignal::small_imm:
    return success();
  case mlir::vc4::QPUSignal::bkpt:
    signal = "bkpt";
    break;
  case mlir::vc4::QPUSignal::thrsw:
    signal = "thrsw";
    break;
  case mlir::vc4::QPUSignal::thrend:
    signal = "thrend";
    break;
  case mlir::vc4::QPUSignal::last_thread_switch:
    signal = "lthrsw";
    break;
  case mlir::vc4::QPUSignal::ldtmu0:
    signal = "ldtmu0";
    break;
  case mlir::vc4::QPUSignal::ldtmu1:
    signal = "ldtmu1";
    break;
  case mlir::vc4::QPUSignal::load_imm:
    return bundle.emitOpError()
           << "sig = #vc4.qpu_signal<load_imm> must be represented as "
              "a scheduled vc4.qpu.ldi operation for artifact emission";
  case mlir::vc4::QPUSignal::branch:
    return bundle.emitOpError()
           << "sig = #vc4.qpu_signal<branch> must be represented as "
              "a scheduled vc4.qpu.branch operation for artifact emission";
  }

  if (needSeparator)
    line += "; ";
  needSeparator = true;
  line += signal;
  return success();
}

static std::optional<std::string>
formatReadOnlyRegisterAccess(mlir::vc4::QPUBundleOp bundle) {
  // Some scheduled sink slots intentionally perform only a read-side effect.
  // VPM/VCD/VDW wait slots must use vc4.qpu.vpmvcd_wait; this compatibility
  // path is only for unambiguous non-VPM peripheral reads.
  mlir::Operation *op = bundle.getOperation();
  if (bundle.getSig() != mlir::vc4::QPUSignal::none ||
      bundle.getOpAdd() != mlir::vc4::AddOpcode::nop ||
      bundle.getOpMul() != mlir::vc4::MulOpcode::nop ||
      bundle.getCondAdd() != mlir::vc4::Cond::never ||
      bundle.getCondMul() != mlir::vc4::Cond::never ||
      op->hasAttr("set_flags")) {
    return std::nullopt;
  }

  auto formatPseudoRead = [](int64_t raddr) -> std::optional<std::string> {
    switch (raddr) {
    case 51:
      return std::string("read mutex_acq");
    default:
      return std::nullopt;
    }
  };

  if (std::optional<int64_t> raddrB =
          getOptionalIntegerAttrValue(op, "raddr_b")) {
    if (std::optional<std::string> read = formatPseudoRead(*raddrB))
      return read;
  }

  return formatPseudoRead(getIntegerAttrValue(op, "raddr_a"));
}

static std::string formatU32Immediate(int64_t signedValue);

static LogicalResult emitQPUBundleQASM(mlir::vc4::QPUBundleOp bundle,
                                       QASMEmissionState &state,
                                       llvm::raw_ostream &os) {
  if (qpuBundleWritesAddress(bundle, 49)) {
    return bundle.emitOpError()
           << "raw writes to VPM/VCD/VDW setup address 49 require "
              "vc4.qpu.vpmvcd_setup";
  }
  if (qpuBundleWritesAddress(bundle, 50)) {
    return bundle.emitOpError()
           << "raw writes to VPM/VCD/VDW address register 50 require "
              "vc4.qpu.vpmvcd_addr";
  }
  if (qpuBundleCarriesReadAddress(bundle, 50)) {
    return bundle.emitOpError()
           << "raw reads from VPM/VCD/VDW wait address 50 require "
              "vc4.qpu.vpmvcd_wait";
  }

  std::string line;
  bool needSeparator = false;

  const bool addActive = bundle.getOpAdd() != mlir::vc4::AddOpcode::nop;
  if (failed(appendAddInstruction(bundle, state, line, needSeparator)))
    return failure();
  if (failed(appendMulInstruction(bundle, line, needSeparator, addActive)))
    return failure();
  if (failed(appendSignalInstruction(bundle, line, needSeparator)))
    return failure();

  if (!needSeparator) {
    if (std::optional<std::string> readOnlyAccess =
            formatReadOnlyRegisterAccess(bundle))
      line = *readOnlyAccess;
    else
      line = "nop";
  }

  os << line << "\n";
  return success();
}

static LogicalResult emitQPUVPMVCDWritePseudoQASM(
    mlir::Operation *op, mlir::vc4::Cond cond, mlir::vc4::AddOpcode opcode,
    mlir::vc4::QPUMux addA, mlir::vc4::QPUMux addB, mlir::vc4::QPUMux mulA,
    mlir::vc4::QPUMux mulB, llvm::StringRef destination,
    llvm::raw_ostream &os) {
  if (cond == mlir::vc4::Cond::never || opcode == mlir::vc4::AddOpcode::nop) {
    return op->emitOpError()
           << "cannot emit inactive VPM/VCD/VDW control pseudo-op";
  }

  std::string src0;
  std::string src1;
  if (failed(formatPseudoVPMVCDMuxSource(op, addA, mulA, mulB, src0)))
    return failure();
  if (!isUnaryAddOpcode(opcode) &&
      failed(formatPseudoVPMVCDMuxSource(op, addB, mulA, mulB, src1)))
    return failure();

  bool useMovAlias = false;
  std::string movAliasSource = src0;
  if (cond == mlir::vc4::Cond::always && !op->hasAttr("set_flags")) {
    if (opcode == mlir::vc4::AddOpcode::bit_or && src0 == src1) {
      useMovAlias = true;
      movAliasSource = src0;
    } else if (opcode == mlir::vc4::AddOpcode::add && src1 == "0") {
      useMovAlias = true;
      movAliasSource = src0;
    } else if (opcode == mlir::vc4::AddOpcode::add && src0 == "0") {
      useMovAlias = true;
      movAliasSource = src1;
    }
  }

  os << (useMovAlias ? "mov" : getAddOpcodeMnemonic(opcode));
  if (op->hasAttr("set_flags"))
    os << ".setf";
  if (cond != mlir::vc4::Cond::never)
    os << getConditionSuffix(cond);
  os << " " << destination << ", " << (useMovAlias ? movAliasSource : src0);
  if (!useMovAlias && !isUnaryAddOpcode(opcode))
    os << ", " << src1;
  os << "\n";
  return success();
}

static LogicalResult emitQPUVPMVCDSetupQASM(mlir::vc4::QPUVPMVCDSetupOp setup,
                                            llvm::raw_ostream &os) {
  return emitQPUVPMVCDWritePseudoQASM(
      setup.getOperation(), setup.getCondAdd(), setup.getOpAdd(),
      setup.getAddA(), setup.getAddB(), setup.getMulA(), setup.getMulB(),
      getVPMSetupName(setup.getSide()), os);
}

static LogicalResult
emitQPUVPMVCDSetupLDIQASM(mlir::vc4::QPUVPMVCDSetupLDIOp setup,
                          llvm::raw_ostream &os) {
  if (setup.getCondAdd() == mlir::vc4::Cond::never) {
    return setup.emitOpError()
           << "cannot emit inactive VPM/VCD/VDW setup LDI pseudo-op";
  }
  if (setup.getCondMul() != mlir::vc4::Cond::never) {
    return setup.emitOpError()
           << "cannot emit VPM/VCD/VDW setup LDI pseudo-op with an active "
              "MUL-side write";
  }

  std::string opcode = "ldi";
  if (setup.getCondAdd() != mlir::vc4::Cond::never)
    opcode += getConditionSuffix(setup.getCondAdd());

  os << opcode << " " << getVPMSetupName(setup.getSide()) << ", "
     << formatU32Immediate(static_cast<uint32_t>(setup.getValueAttr().getInt()))
     << "\n";
  return success();
}

static LogicalResult emitQPUVPMVCDAddrQASM(mlir::vc4::QPUVPMVCDAddrOp addr,
                                           llvm::raw_ostream &os) {
  return emitQPUVPMVCDWritePseudoQASM(
      addr.getOperation(), addr.getCondAdd(), addr.getOpAdd(), addr.getAddA(),
      addr.getAddB(), addr.getMulA(), addr.getMulB(),
      getVPMAddressName(addr.getSide()), os);
}

static LogicalResult emitQPUVPMVCDWaitQASM(mlir::vc4::QPUVPMVCDWaitOp wait,
                                           llvm::raw_ostream &os) {
  os << "read " << getVPMWaitName(wait.getSide()) << "\n";
  return success();
}

static std::string formatU32Immediate(int64_t signedValue) {
  static constexpr char hexDigits[] = "0123456789abcdef";
  uint32_t value = static_cast<uint32_t>(signedValue);
  std::string result = "0x00000000";
  for (unsigned i = 0; i != 8; ++i)
    result[2 + i] = hexDigits[(value >> ((7 - i) * 4)) & 0xf];
  return result;
}

static std::string
formatPerElementLDIImmediate(llvm::ArrayRef<int32_t> values) {
  std::string result = "[";
  for (unsigned i = 0; i != values.size(); ++i) {
    if (i != 0)
      result += ",";
    result += std::to_string(values[i]);
  }
  result += "]";
  return result;
}

static LogicalResult
buildLoadLikeDestinationList(mlir::Operation *op, mlir::vc4::Cond condAdd,
                             mlir::vc4::Cond condMul, bool pm,
                             llvm::StringRef packSuffix, std::string &destList,
                             mlir::vc4::Cond &emittedCond) {
  const bool addWrites = condAdd != mlir::vc4::Cond::never;
  const bool mulWrites = condMul != mlir::vc4::Cond::never;
  const bool writeSwap = op->hasAttr("write_swap");

  if (!addWrites && !mulWrites) {
    destList = "-";
    emittedCond = mlir::vc4::Cond::always;
    return success();
  }

  if (addWrites && mulWrites && condAdd != condMul) {
    return op->emitOpError()
           << "cannot emit load-immediate/semaphore qasm when ADD and MUL "
              "conditions differ; split this into separately scheduled "
              "instructions before the artifact boundary";
  }

  std::string addDest = formatWriteAddress(getIntegerAttrValue(op, "waddr_add"),
                                           /*forAddALU=*/true, writeSwap);
  if (!pm)
    addDest += packSuffix.str();

  std::string mulDest = formatWriteAddress(getIntegerAttrValue(op, "waddr_mul"),
                                           /*forAddALU=*/false, writeSwap);
  if (pm)
    mulDest += packSuffix.str();

  if (addWrites && mulWrites) {
    destList = addDest + ", " + mulDest;
    emittedCond = condAdd;
    return success();
  }

  if (addWrites) {
    destList = addDest;
    emittedCond = condAdd;
    return success();
  }

  // vc4asm's two-destination load-immediate/mov forms let a single scheduled
  // operation target only the MUL write path by discarding the ADD-side result.
  destList = std::string("-, ") + mulDest;
  emittedCond = condMul;
  return success();
}

static void appendLoadLikeOpcodeSuffix(mlir::Operation *op,
                                       mlir::vc4::Cond cond,
                                       std::string &opcode) {
  if (op->hasAttr("set_flags"))
    opcode += ".setf";
  if (cond != mlir::vc4::Cond::never)
    opcode += getConditionSuffix(cond);
}

static LogicalResult emitQPULDIQASM(mlir::vc4::QPULDIOp ldi,
                                    llvm::raw_ostream &os) {
  std::optional<std::string> packSuffix =
      getPackSuffix(ldi.getOperation()->getAttr("pack"));
  if (!packSuffix) {
    return ldi.emitOpError()
           << "cannot emit unsupported vc4.qpu.ldi pack attribute '"
           << ldi.getOperation()->getAttr("pack") << "'";
  }

  std::string destinations;
  mlir::vc4::Cond emittedCond = mlir::vc4::Cond::always;
  if (failed(buildLoadLikeDestinationList(
          ldi.getOperation(), ldi.getCondAdd(), ldi.getCondMul(), ldi.getPm(),
          *packSuffix, destinations, emittedCond)))
    return failure();

  std::string opcode = "ldi";
  appendLoadLikeOpcodeSuffix(ldi.getOperation(), emittedCond, opcode);

  switch (ldi.getMode()) {
  case mlir::vc4::LoadImmMode::splat32: {
    auto valueAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
        ldi.getOperation()->getAttr("value"));
    if (!valueAttr) {
      return ldi.emitOpError()
             << "requires an integer 'value' attribute for splat32 qasm "
                "emission";
    }

    os << opcode << " " << destinations << ", "
       << formatU32Immediate(static_cast<uint32_t>(valueAttr.getInt())) << "\n";
    return success();
  }
  case mlir::vc4::LoadImmMode::per_elem_i2:
  case mlir::vc4::LoadImmMode::per_elem_u2: {
    auto valueAttr = llvm::dyn_cast_or_null<mlir::DenseI32ArrayAttr>(
        ldi.getOperation()->getAttr("value"));
    if (!valueAttr) {
      return ldi.emitOpError()
             << "requires a dense i32 array 'value' attribute for "
                "per-element qasm emission";
    }
    if (valueAttr.asArrayRef().size() != 16) {
      return ldi.emitOpError()
             << "requires exactly 16 lane values for per-element qasm "
                "emission";
    }

    os << opcode << " " << destinations << ", "
       << formatPerElementLDIImmediate(valueAttr.asArrayRef()) << "\n";
    return success();
  }
  }

  llvm_unreachable("unhandled vc4.qpu.ldi mode");
}

static const char *getSemaphoreSourceMnemonic(mlir::vc4::SemaphoreMode mode) {
  switch (mode) {
  case mlir::vc4::SemaphoreMode::acquire:
    return "sacq";
  case mlir::vc4::SemaphoreMode::release:
    return "srel";
  }
  return "sema";
}

static const char *getSemaphoreCommentName(mlir::vc4::SemaphoreMode mode) {
  switch (mode) {
  case mlir::vc4::SemaphoreMode::acquire:
    return "acquire";
  case mlir::vc4::SemaphoreMode::release:
    return "release";
  }
  return "unknown";
}

static LogicalResult emitQPUSemaQASM(mlir::vc4::QPUSemaOp sema,
                                     llvm::raw_ostream &os) {
  int64_t id = getIntegerAttrValue(sema.getOperation(), "id");
  if (id < 0 || id > 15) {
    return sema.emitOpError()
           << "cannot emit semaphore id " << id
           << "; vc4asm semaphore immediates must be in range [0, 15]";
  }

  std::optional<std::string> packSuffix =
      getPackSuffix(sema.getOperation()->getAttr("pack"));
  if (!packSuffix) {
    return sema.emitOpError()
           << "cannot emit unsupported vc4.qpu.sema pack attribute '"
           << sema.getOperation()->getAttr("pack") << "'";
  }

  std::string destinations;
  mlir::vc4::Cond emittedCond = mlir::vc4::Cond::always;
  if (failed(buildLoadLikeDestinationList(
          sema.getOperation(), sema.getCondAdd(), sema.getCondMul(),
          sema.getPm(), *packSuffix, destinations, emittedCond)))
    return failure();

  // For side-effect-only semaphore ops, prefer the direct vc4asm spelling used
  // by the hardware references.  Destination-writing semaphore moves are still
  // available for tests that intentionally model that form.
  if (destinations == "-") {
    os << getSemaphoreSourceMnemonic(sema.getMode()) << " -, " << id << "\n";
    return success();
  }

  // vc4asm documents both the direct `sacq`/`srel` syntax and the Broadcom
  // compatible `mov dest, sacqN` form.  Use the latter when destinations are
  // present so conditions, flags, pack suffixes, write-swap, and dual ADD/MUL
  // destinations share the same deterministic printer path as ordinary
  // load-immediate output.
  std::string opcode = "mov";
  appendLoadLikeOpcodeSuffix(sema.getOperation(), emittedCond, opcode);

  os << opcode << " " << destinations << ", "
     << getSemaphoreSourceMnemonic(sema.getMode()) << id << " # sema "
     << getSemaphoreCommentName(sema.getMode()) << " " << id << "\n";
  return success();
}

static const char *getBranchConditionSuffix(mlir::vc4::BranchCond cond) {
  switch (cond) {
  case mlir::vc4::BranchCond::all_z_set:
    return ".allz";
  case mlir::vc4::BranchCond::all_z_clear:
    return ".allnz";
  case mlir::vc4::BranchCond::any_z_set:
    return ".anyz";
  case mlir::vc4::BranchCond::any_z_clear:
    return ".anynz";
  case mlir::vc4::BranchCond::all_n_set:
    return ".alln";
  case mlir::vc4::BranchCond::all_n_clear:
    return ".allnn";
  case mlir::vc4::BranchCond::any_n_set:
    return ".anyn";
  case mlir::vc4::BranchCond::any_n_clear:
    return ".anynn";
  case mlir::vc4::BranchCond::all_c_set:
    return ".allc";
  case mlir::vc4::BranchCond::all_c_clear:
    return ".allnc";
  case mlir::vc4::BranchCond::any_c_set:
    return ".anyc";
  case mlir::vc4::BranchCond::any_c_clear:
    return ".anync";
  case mlir::vc4::BranchCond::always:
    return "";
  }
  return "";
}

static const char *getBranchModeMnemonic(bool relative) {
  return relative ? "brr" : "bra";
}

static bool getBoolAttrValue(mlir::Operation *op, llvm::StringRef attrName) {
  if (auto attr = op->getAttrOfType<mlir::BoolAttr>(attrName))
    return attr.getValue();
  return false;
}

static std::string getQPUInstructionLabel(unsigned slotIndex) {
  return "vc4_qpu_slot_" + std::to_string(slotIndex);
}

static bool hasBranchInstruction(llvm::ArrayRef<mlir::Operation *> stream) {
  for (mlir::Operation *op : stream) {
    if (llvm::isa<mlir::vc4::QPUBranchOp>(op))
      return true;
  }
  return false;
}

static LogicalResult buildBranchDestinationList(mlir::vc4::QPUBranchOp branch,
                                                std::string &destinations) {
  mlir::Operation *op = branch.getOperation();
  const bool writeSwap = op->hasAttr("write_swap");
  std::string addDest = formatWriteAddress(getIntegerAttrValue(op, "waddr_add"),
                                           /*forAddALU=*/true, writeSwap);
  std::string mulDest = formatWriteAddress(getIntegerAttrValue(op, "waddr_mul"),
                                           /*forAddALU=*/false, writeSwap);

  // The VC4 branch encoding has independent ADD/MUL-side link destinations.
  // Emit both deterministically for non-no-link branches so the scheduled
  // write-address contract is not silently narrowed at the artifact boundary.
  destinations = addDest + ", " + mulDest;
  return success();
}

static bool isBranchNoLinkDestination(mlir::Operation *op) {
  // The scheduled IR uses write-address 39 on both sides as the ordinary
  // no-link sentinel.  Use vc4asm's compact no-link branch syntax for this
  // case so generated kernels match the hardware reference style:
  //   brr.anync -, :target
  // instead of materializing sentinel write addresses as real ra39/rb39 link
  // destinations.
  return getIntegerAttrValue(op, "waddr_add") == 39 &&
         getIntegerAttrValue(op, "waddr_mul") == 39;
}

static std::optional<unsigned>
resolveRelativeBranchImmediateToSlot(unsigned slotIndex, int64_t immediate,
                                     unsigned streamSize) {
  // The scheduled MLIR carries relative branch immediates in bytes.  The
  // flattened stream is one 64-bit QPU instruction per slot, so exact multiples
  // of 8 can be resolved to deterministic instruction labels.  Let vc4asm do
  // the final PC-relative encoding from the label; do not bake our own relative
  // PC convention into the qasm text.
  if (immediate % 8 != 0)
    return std::nullopt;

  const int64_t targetSlot = static_cast<int64_t>(slotIndex) + immediate / 8;
  if (targetSlot < 0 || targetSlot >= static_cast<int64_t>(streamSize))
    return std::nullopt;

  return static_cast<unsigned>(targetSlot);
}

static std::string
formatBranchTargetExpression(bool relative, bool useReg, int64_t immediate,
                             unsigned slotIndex, unsigned streamSize,
                             std::optional<unsigned> &targetSlot) {
  targetSlot = std::nullopt;
  if (relative && !useReg) {
    targetSlot =
        resolveRelativeBranchImmediateToSlot(slotIndex, immediate, streamSize);
    if (targetSlot)
      return ":" + getQPUInstructionLabel(*targetSlot);
  }

  return std::to_string(immediate);
}

static LogicalResult emitQPUBranchQASM(mlir::vc4::QPUBranchOp branch,
                                       unsigned slotIndex, unsigned streamSize,
                                       llvm::raw_ostream &os) {
  mlir::Operation *op = branch.getOperation();
  const bool relative = getBoolAttrValue(op, "relative");
  const bool useReg = getBoolAttrValue(op, "use_reg");
  const int64_t immediate = getIntegerAttrValue(op, "immediate");

  std::string opcode = getBranchModeMnemonic(relative);
  opcode += getBranchConditionSuffix(branch.getCond());

  std::optional<unsigned> targetSlot;
  std::string targetExpr = formatBranchTargetExpression(
      relative, useReg, immediate, slotIndex, streamSize, targetSlot);

  if (isBranchNoLinkDestination(op)) {
    os << opcode << " -";
    if (useReg) {
      int64_t raddrA = getIntegerAttrValue(op, "raddr_a");
      os << ", " << formatReadAddress('a', raddrA);
    }
    os << ", " << targetExpr;
  } else {
    std::string destinations;
    if (failed(buildBranchDestinationList(branch, destinations)))
      return failure();

    os << opcode << " " << destinations << ", ";
    if (useReg) {
      int64_t raddrA = getIntegerAttrValue(op, "raddr_a");
      os << formatReadAddress('a', raddrA) << ", ";
    } else {
      os << "-, ";
    }
    os << targetExpr;
  }

  os << " # qpu.branch label=" << getQPUInstructionLabel(slotIndex)
     << " target=" << (relative ? "relative" : "absolute") << ":" << immediate;
  if (targetSlot)
    os << " target_label=" << getQPUInstructionLabel(*targetSlot);
  os << " delay_slots=3\n";
  return success();
}

static LogicalResult writeQASM(KernelRecord &kernel,
                               llvm::StringRef bundleDir) {
  std::string qasm;
  llvm::raw_string_ostream qasmOS(qasm);
  QASMEmissionState qasmState;
  const bool emitInstructionLabels =
      hasBranchInstruction(kernel.scheduledStream);

  for (unsigned slotIndex = 0; slotIndex != kernel.scheduledStream.size();
       ++slotIndex) {
    mlir::Operation *op = kernel.scheduledStream[slotIndex];

    if (emitInstructionLabels)
      qasmOS << ":" << getQPUInstructionLabel(slotIndex) << "\n";

    if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
      if (failed(emitQPUBranchQASM(
              branch, slotIndex,
              static_cast<unsigned>(kernel.scheduledStream.size()), qasmOS)))
        return failure();
      continue;
    }

    if (auto setup = llvm::dyn_cast<mlir::vc4::QPUVPMVCDSetupOp>(op)) {
      if (failed(emitQPUVPMVCDSetupQASM(setup, qasmOS)))
        return failure();
      continue;
    }

    if (auto setup = llvm::dyn_cast<mlir::vc4::QPUVPMVCDSetupLDIOp>(op)) {
      if (failed(emitQPUVPMVCDSetupLDIQASM(setup, qasmOS)))
        return failure();
      continue;
    }

    if (auto addr = llvm::dyn_cast<mlir::vc4::QPUVPMVCDAddrOp>(op)) {
      if (failed(emitQPUVPMVCDAddrQASM(addr, qasmOS)))
        return failure();
      continue;
    }

    if (auto wait = llvm::dyn_cast<mlir::vc4::QPUVPMVCDWaitOp>(op)) {
      if (failed(emitQPUVPMVCDWaitQASM(wait, qasmOS)))
        return failure();
      continue;
    }

    if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
      if (slotIndex + 1 < kernel.scheduledStream.size()) {
        if (auto nextBundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(
                kernel.scheduledStream[slotIndex + 1])) {
          if (canPrepareVectorRotateFromSpacer(bundle, nextBundle)) {
            if (failed(emitVectorRotatePrepBundle(bundle, nextBundle, qasmState,
                                                  qasmOS)))
              return failure();
            continue;
          }
        }
      }
      if (failed(emitQPUBundleQASM(bundle, qasmState, qasmOS)))
        return failure();
      continue;
    }

    if (auto ldi = llvm::dyn_cast<mlir::vc4::QPULDIOp>(op)) {
      if (failed(emitQPULDIQASM(ldi, qasmOS)))
        return failure();
      continue;
    }

    if (auto sema = llvm::dyn_cast<mlir::vc4::QPUSemaOp>(op)) {
      if (failed(emitQPUSemaQASM(sema, qasmOS)))
        return failure();
      continue;
    }

    return op->emitOpError()
           << "cannot be emitted as qasm by the VC4 artifact emitter";
  }
  qasmOS.flush();

  return writeBundleFile(kernel.func.getOperation(), bundleDir, kernel.qasmPath,
                         [&](llvm::raw_ostream &os) { os << qasm; });
}

static void appendLauncherParameter(llvm::raw_ostream &os,
                                    const LaunchABIArgumentModel &arg) {
  os << arg.cType;
  if (!arg.cType.empty() && arg.cType.back() == '*')
    os << arg.name;
  else
    os << " " << arg.name;
}

static std::string getLaunchAPIBaseName(const LaunchABIModel &launchABI) {
  std::string base = launchABI.publicName;
  constexpr const char *suffix = "_launch";
  constexpr size_t suffixLen = 7;
  if (base.size() > suffixLen &&
      base.compare(base.size() - suffixLen, suffixLen, suffix) == 0) {
    base.resize(base.size() - suffixLen);
  }
  return base;
}

static std::string getLaunchFunctionName(const LaunchABIModel &launchABI) {
  constexpr const char *suffix = "_launch";
  constexpr size_t suffixLen = 7;
  if (launchABI.publicName.size() > suffixLen &&
      launchABI.publicName.compare(launchABI.publicName.size() - suffixLen,
                                   suffixLen, suffix) == 0)
    return launchABI.publicName;
  return launchABI.publicName + "_launch";
}

static void appendLauncherPrototype(llvm::raw_ostream &os,
                                    const LaunchABIModel &launchABI) {
  os << "int " << getLaunchFunctionName(launchABI)
     << "(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block";
  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    os << ", ";
    appendLauncherParameter(os, arg);
  }
  os << ")";
}

static std::string
getRuntimeAllocationsFunctionName(const LaunchABIModel &launchABI) {
  return getLaunchAPIBaseName(launchABI) + "_runtime_allocations";
}

static std::string
getRuntimeLaunchesFunctionName(const LaunchABIModel &launchABI) {
  return getLaunchAPIBaseName(launchABI) + "_runtime_launches";
}

static std::string
getRuntimeCapacityFunctionName(const LaunchABIModel &launchABI) {
  return getLaunchAPIBaseName(launchABI) + "_runtime_capacity";
}

static std::string
getRuntimeCodeUploadsFunctionName(const LaunchABIModel &launchABI) {
  return getLaunchAPIBaseName(launchABI) + "_runtime_code_uploads";
}

static std::string
getRuntimeLaunchFailuresFunctionName(const LaunchABIModel &launchABI) {
  return getLaunchAPIBaseName(launchABI) + "_runtime_launch_failures";
}

static bool isLaunchABIBufferArgument(const LaunchABIArgumentModel &arg) {
  return arg.kind == LaunchABIArgumentKind::Buffer;
}

static llvm::SmallVector<const LaunchABIArgumentModel *, 4>
collectLaunchABIBufferArguments(const LaunchABIModel &launchABI) {
  llvm::SmallVector<const LaunchABIArgumentModel *, 4> buffers;
  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    if (isLaunchABIBufferArgument(arg))
      buffers.push_back(&arg);
  }
  return buffers;
}

static std::string
getBufferElementCTypeForCodegen(const LaunchABIArgumentModel &arg) {
  std::optional<std::string> elemType = getElementCType(arg.elementType);
  if (elemType)
    return *elemType;
  return "uint32_t";
}

static std::optional<llvm::StringRef>
getRuntimeBuiltinRequestInfoField(llvm::StringRef kind) {
  if (kind == "logical_request")
    return llvm::StringRef("logical_request");
  if (kind == "total_requests")
    return llvm::StringRef("total_requests");
  if (kind == "logical_block_id")
    return llvm::StringRef("logical_block_id");
  if (kind == "program_id_x")
    return llvm::StringRef("program_id_x");
  if (kind == "program_id_y")
    return llvm::StringRef("program_id_y");
  if (kind == "program_id_z")
    return llvm::StringRef("program_id_z");
  if (kind == "num_programs_x")
    return llvm::StringRef("num_programs_x");
  if (kind == "num_programs_y")
    return llvm::StringRef("num_programs_y");
  if (kind == "num_programs_z")
    return llvm::StringRef("num_programs_z");
  if (kind == "logical_warp_id")
    return llvm::StringRef("logical_warp_id");
  if (kind == "warps_per_block")
    return llvm::StringRef("warps_per_block");
  if (kind == "vpm_base_row")
    return llvm::StringRef("vpm_base_row");
  if (kind == "vpm_rows")
    return llvm::StringRef("vpm_rows");
  if (kind == "semaphore_base")
    return llvm::StringRef("semaphore_base");
  if (kind == "barrier_arrive_sem")
    return llvm::StringRef("barrier_arrive_sem");
  if (kind == "barrier_go_sem")
    return llvm::StringRef("barrier_go_sem");
  if (kind == "barrier_depart_sem")
    return llvm::StringRef("barrier_depart_sem");
  if (kind == "barrier_reset_sem")
    return llvm::StringRef("barrier_reset_sem");
  if (kind == "resident_request_id")
    return llvm::StringRef("resident_request_id");
  if (kind == "spill_frame_base")
    return llvm::StringRef("spill_frame_base");
  if (kind == "spill_frame_bytes")
    return llvm::StringRef("spill_frame_bytes");
  if (kind == "spill_frame_stride_bytes")
    return llvm::StringRef("spill_frame_stride_bytes");
  return std::nullopt;
}

static void appendRuntimeAPIDeclarations(llvm::raw_ostream &os) {
  os << "int vc4_program_create(struct vc4_program **out, uint32_t "
        "requested_bytes);\n";
  os << "\n";
}

static LogicalResult writeLauncherHeader(llvm::ArrayRef<KernelRecord> kernels,
                                         llvm::StringRef bundleDir) {
  if (kernels.empty())
    return failure();
  mlir::vc4::FuncOp firstFunc = kernels.front().func;
  return writeBundleFile(
      firstFunc.getOperation(), bundleDir, "kernel_launch.h",
      [&](llvm::raw_ostream &os) {
        os << "#ifndef VC4_CODEGEN_KERNEL_LAUNCH_H\n";
        os << "#define VC4_CODEGEN_KERNEL_LAUNCH_H\n\n";
        os << "#include <stdint.h>\n";
        os << "#include \"vc4_runtime.h\"\n\n";
        os << "#ifdef __cplusplus\n";
        os << "extern \"C\" {\n";
        os << "#endif\n\n";
        appendRuntimeAPIDeclarations(os);
        for (const KernelRecord &kernel : kernels) {
          appendLauncherPrototype(os, kernel.launchABI);
          os << ";\n";
        }
        os << "\n";
        os << "uint32_t "
           << getRuntimeAllocationsFunctionName(kernels.front().launchABI)
           << "(void);\n";
        os << "uint32_t "
           << getRuntimeLaunchesFunctionName(kernels.front().launchABI)
           << "(void);\n";
        os << "uint32_t "
           << getRuntimeCapacityFunctionName(kernels.front().launchABI)
           << "(void);\n";
        os << "uint32_t "
           << getRuntimeCodeUploadsFunctionName(kernels.front().launchABI)
           << "(void);\n";
        os << "uint32_t "
           << getRuntimeLaunchFailuresFunctionName(kernels.front().launchABI)
           << "(void);\n\n";
        os << "#ifdef __cplusplus\n";
        os << "}\n";
        os << "#endif\n\n";
        os << "#endif // VC4_CODEGEN_KERNEL_LAUNCH_H\n";
      });
}

static bool launchABIRequiresF32Packing(const LaunchABIModel &launchABI) {
  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    if (arg.kind == LaunchABIArgumentKind::Scalar && arg.scalarType == "f32")
      return true;
  }
  return false;
}

static const LaunchABIArgumentModel *
findLaunchABIArgumentForUniformIndex(const LaunchABIModel &launchABI,
                                     int64_t uniformIndex) {
  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    if (arg.uniformIndex == uniformIndex)
      return &arg;
  }
  return nullptr;
}

static const LaunchABIBuiltinModel *
findLaunchABIBuiltinForUniformIndex(const LaunchABIModel &launchABI,
                                    int64_t uniformIndex) {
  for (const LaunchABIBuiltinModel &builtin : launchABI.builtins) {
    if (builtin.materialization == "uniform_suffix" && builtin.uniformIndex &&
        *builtin.uniformIndex == uniformIndex)
      return &builtin;
  }
  return nullptr;
}

static void
appendLauncherUniformLayoutComment(llvm::raw_ostream &os,
                                   const LaunchABIModel &launchABI) {
  os << "  /*\n";
  os << "   * Dense physical uniform layout per logical request, ordered by ";
  os << "vc4.launch_abi uniform_index:\n";
  for (int64_t index = 0; index != launchABI.uniformWordsPerQPU; ++index) {
    os << "   *   [" << index << "] ";
    if (const LaunchABIArgumentModel *arg =
            findLaunchABIArgumentForUniformIndex(launchABI, index)) {
      os << "arg " << arg->name;
    } else if (const LaunchABIBuiltinModel *builtin =
                   findLaunchABIBuiltinForUniformIndex(launchABI, index)) {
      os << "builtin " << builtin->name << " (" << builtin->kind << ")";
    } else {
      os << "<missing>";
    }
    os << "\n";
  }
  os << "   */\n";
}

static LogicalResult writeLauncherSource(llvm::ArrayRef<KernelRecord> kernels,
                                         llvm::StringRef bundleDir) {
  KernelRecord &firstKernel = const_cast<KernelRecord &>(kernels.front());
  ProgramLayoutModel programLayout = buildProgramLayoutModel(kernels);
  std::string source;
  llvm::raw_string_ostream os(source);

  const LaunchABIModel &firstABI = firstKernel.launchABI;
  const std::string apiBase = getLaunchAPIBaseName(firstABI);
  const std::string stateName = apiBase + "_program_state";
  const std::string allocationsName =
      getRuntimeAllocationsFunctionName(firstABI);
  const std::string launchesName = getRuntimeLaunchesFunctionName(firstABI);
  const std::string capacityName = getRuntimeCapacityFunctionName(firstABI);
  const std::string codeUploadsName =
      getRuntimeCodeUploadsFunctionName(firstABI);
  const std::string launchFailuresName =
      getRuntimeLaunchFailuresFunctionName(firstABI);

  bool requiresF32Packing = false;
  bool requiresLaunchElements = false;
  bool requiresCooperativeScheduling = false;
  for (const KernelRecord &kernel : kernels) {
    requiresF32Packing |= launchABIRequiresF32Packing(kernel.launchABI);
    requiresCooperativeScheduling |=
        kernel.resources.scheduleMode == "cooperative_block";
    // Scalar arguments are semantic uniforms, even when named n/m/rows/cols.
    // CUDA-like launch geometry is the single host-side authority for the
    // logical element count; in-kernel predicates consume scalar extents.
    requiresLaunchElements |=
        kernel.resources.scheduleMode != "cooperative_block";
  }
  bool requiresSaturatingMul =
      requiresLaunchElements || requiresCooperativeScheduling;

  os << "#define VC4_CODEGEN_KERNEL_LAUNCH_IMPLEMENTATION 1\n";
  os << "#include \"kernel_launch.h\"\n";
  os << "#undef VC4_CODEGEN_KERNEL_LAUNCH_IMPLEMENTATION\n\n";
  for (const KernelRecord &includeKernel : kernels)
    os << "#include \"" << includeKernel.launchABI.codeSymbol << ".h\"\n";
  os << "\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "#include <string.h>\n\n";
  os << "#define VC4_CODEGEN_PROGRAM_HEAP_BYTES " << programLayout.heapBytes
     << "u\n";
  os << "#define VC4_CODEGEN_SPILL_FRAME_ALIGNMENT " << kVC4SpillFrameAlignment
     << "u\n";
  os << "#define VC4_CODEGEN_SPILL_ARENA_BYTES "
     << programLayout.spillArenaBytes << "u\n";
  os << "#define VC4_CODEGEN_TARGET_VPM_BYTES 4096u\n";
  os << "#define VC4_CODEGEN_TARGET_SEMAPHORES 16u\n";
  for (const KernelRecord &macroKernel : kernels) {
    os << "#define KERNEL_" << macroKernel.kernelId << "_NUM_UNIFS "
       << macroKernel.info.uniformWordsPerQPU << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId << "_SPILL_FRAME_BYTES "
       << macroKernel.spill.frameBytes << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId
       << "_SPILL_FRAME_STRIDE_BYTES " << macroKernel.spill.strideBytes
       << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId << "_SPILL_FRAME_COUNT "
       << macroKernel.spill.frameCount << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId << "_SPILL_ARENA_BYTES "
       << macroKernel.spill.arenaBytes << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId << "_WARPS_PER_BLOCK "
       << macroKernel.resources.warpsPerBlock << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId
       << "_SEMAPHORE_COUNT_PER_BLOCK "
       << macroKernel.resources.semaphoreCountPerBlock << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId
       << "_USER_VPM_ROWS_PER_BLOCK "
       << macroKernel.resources.userVPMRowsPerBlock << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId
       << "_COMPILER_VPM_STAGING_ROWS_PER_WARP "
       << macroKernel.resources.compilerVPMStagingRowsPerWarp << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId
       << "_COMPILER_VPM_STAGING_ROWS_PER_BLOCK "
       << macroKernel.resources.compilerVPMStagingRowsPerBlock << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId
       << "_SPILL_VPM_ROWS_PER_BLOCK "
       << macroKernel.resources.spillVPMRowsPerBlock << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId
       << "_TOTAL_VPM_ROWS_PER_BLOCK "
       << macroKernel.resources.totalVPMRowsPerBlock << "u\n";
    os << "#define KERNEL_" << macroKernel.kernelId << "_MAX_RESIDENT_BLOCKS "
       << macroKernel.resources.maxResidentBlocks << "u\n";
  }
  os << "\n";
  os << "/* VC4_RUNTIME_LAYOUT program_bytes=" << programLayout.programBytes
     << " static_bytes=" << programLayout.staticBytes
     << " heap_offset=" << programLayout.heapOffset
     << " heap_bytes=" << programLayout.heapBytes
     << " hidden_spill_arena_bytes=" << programLayout.spillArenaBytes
     << " kernels=" << kernels.size() << " */\n";
  os << "/* VC4_SPILL_FRAME_ABI alignment=" << kVC4SpillFrameAlignment
     << " max_spill_arena_bytes=" << programLayout.spillArenaBytes
     << " hidden_arena_before_public_heap="
     << (programLayout.spillArenaBytes ? 1 : 0)
     << " request_info_fields=resident_request_id,spill_frame_base,"
        "spill_frame_bytes,spill_frame_stride_bytes,spill_vpm_row,"
        "program_id_x,program_id_y,program_id_z,num_programs_x,"
        "num_programs_y,num_programs_z */\n";
  os << "/* Generated launches pack kernel-specific uniforms; libpi owns "
        "allocation, code residency, queueing, and heap/copy APIs. */\n\n";

  if (requiresF32Packing) {
    os << "static uint32_t vc4_codegen_pack_f32(float value) {\n";
    os << "  uint32_t bits;\n";
    os << "  memcpy(&bits, &value, sizeof(bits));\n";
    os << "  return bits;\n";
    os << "}\n\n";
  }

  os << "static uint32_t vc4_codegen_ceil_div_u32(uint32_t value, uint32_t "
        "divisor) {\n";
  os << "  if (value == 0u)\n";
  os << "    return 0u;\n";
  os << "  if (divisor == 0u)\n";
  os << "    return 0xffffffffu;\n";
  os << "  return 1u + (value - 1u) / divisor;\n";
  os << "}\n\n";

  if (requiresSaturatingMul) {
    os << "static uint32_t vc4_codegen_saturating_mul_u32(uint32_t lhs, "
          "uint32_t rhs) {\n";
    os << "  if (lhs != 0u && rhs > 0xffffffffu / lhs)\n";
    os << "    return 0xffffffffu;\n";
    os << "  return lhs * rhs;\n";
    os << "}\n\n";
  }

  if (requiresLaunchElements) {
    os << "static uint32_t vc4_codegen_launch_elements(vc4_dim3 grid, vc4_dim3 "
          "block) {\n";
    os << "  uint32_t total = 1u;\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, grid.x);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, grid.y);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, grid.z);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, block.x);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, block.y);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, block.z);\n";
    os << "  return total;\n";
    os << "}\n\n";
  }

  if (requiresCooperativeScheduling) {
    os << "static uint32_t vc4_codegen_min_u32(uint32_t lhs, uint32_t rhs) {\n";
    os << "  return lhs < rhs ? lhs : rhs;\n";
    os << "}\n\n";
    os << "static uint32_t vc4_codegen_grid_blocks(vc4_dim3 grid) {\n";
    os << "  uint32_t total = 1u;\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, grid.x);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, grid.y);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, grid.z);\n";
    os << "  return total;\n";
    os << "}\n\n";
    os << "static uint32_t vc4_codegen_block_warps(vc4_dim3 block) {\n";
    os << "  uint32_t total = 1u;\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, block.x);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, block.y);\n";
    os << "  total = vc4_codegen_saturating_mul_u32(total, block.z);\n";
    os << "  return vc4_codegen_ceil_div_u32(total, VC4_RUNTIME_LANE_WIDTH);\n";
    os << "}\n\n";
  }

  os << "static const struct vc4_kernel_image vc4_codegen_kernels[] = {\n";
  for (const KernelRecord &kernel : kernels) {
    os << "  {\n";
    os << "    .name = \"" << kernel.launchABI.publicName << "\",\n";
    os << "    .code = " << kernel.launchABI.codeSymbol << ",\n";
    os << "    .code_words = (uint32_t)(sizeof(" << kernel.launchABI.codeSymbol
       << ") / sizeof(uint32_t)),\n";
    os << "    .uniform_words_per_request = KERNEL_" << kernel.kernelId
       << "_NUM_UNIFS,\n";
    os << "    .max_requests_per_wave = VC4_RUNTIME_MAX_QPUS,\n";
    os << "    .resource = {\n";
    os << "      .schedule_mode = " << getScheduleModeMacro(kernel.resources)
       << ",\n";
    os << "      .warps_per_block = KERNEL_" << kernel.kernelId
       << "_WARPS_PER_BLOCK,\n";
    os << "      .uses_tmu = " << (kernel.resources.usesTMU ? "1u" : "0u")
       << ",\n";
    os << "      .uses_vpm = " << (kernel.resources.usesVPM ? "1u" : "0u")
       << ",\n";
    os << "      .uses_vpm_qpu_read = "
       << (kernel.resources.usesVPMQPURead ? "1u" : "0u") << ",\n";
    os << "      .uses_vpm_qpu_write = "
       << (kernel.resources.usesVPMQPUWrite ? "1u" : "0u") << ",\n";
    os << "      .uses_vdr = " << (kernel.resources.usesVDR ? "1u" : "0u")
       << ",\n";
    os << "      .uses_vdw = " << (kernel.resources.usesVDW ? "1u" : "0u")
       << ",\n";
    os << "      .uses_barrier = "
       << (kernel.resources.usesBarrier ? "1u" : "0u") << ",\n";
    os << "      .user_vpm_rows_per_block = KERNEL_" << kernel.kernelId
       << "_USER_VPM_ROWS_PER_BLOCK,\n";
    os << "      .compiler_vpm_staging_rows_per_warp = KERNEL_"
       << kernel.kernelId << "_COMPILER_VPM_STAGING_ROWS_PER_WARP,\n";
    os << "      .compiler_vpm_staging_rows_per_block = KERNEL_"
       << kernel.kernelId << "_COMPILER_VPM_STAGING_ROWS_PER_BLOCK,\n";
    os << "      .total_vpm_rows_per_block = KERNEL_" << kernel.kernelId
       << "_TOTAL_VPM_ROWS_PER_BLOCK,\n";
    os << "      .semaphore_count_per_block = KERNEL_" << kernel.kernelId
       << "_SEMAPHORE_COUNT_PER_BLOCK,\n";
    os << "      .requires_vpm_base_row_builtin = "
       << (kernel.resources.requiresVPMBaseRowBuiltin ? "1u" : "0u") << ",\n";
    os << "      .requires_semaphore_base_builtin = "
       << (kernel.resources.requiresSemaphoreBaseBuiltin ? "1u" : "0u")
       << ",\n";
    os << "    },\n";
    os << "    .spill_frame_bytes = KERNEL_" << kernel.kernelId
       << "_SPILL_FRAME_BYTES,\n";
    os << "    .spill_frame_stride_bytes = KERNEL_" << kernel.kernelId
       << "_SPILL_FRAME_STRIDE_BYTES,\n";
    os << "    .spill_frame_count = KERNEL_" << kernel.kernelId
       << "_SPILL_FRAME_COUNT,\n";
    os << "    .spill_arena_bytes = KERNEL_" << kernel.kernelId
       << "_SPILL_ARENA_BYTES,\n";
    os << "  },\n";
  }
  os << "};\n\n";
  for (const KernelRecord &resourceKernel : kernels) {
    os << "/* VC4_KERNEL_RESOURCE kernel_id=" << resourceKernel.kernelId
       << " public_name=" << resourceKernel.launchABI.publicName
       << " schedule_mode=" << resourceKernel.resources.scheduleMode
       << " resident_blocks=" << resourceKernel.resources.maxResidentBlocks
       << " warps_per_block=" << resourceKernel.resources.warpsPerBlock
       << " semaphore_base=resident_slot*"
       << resourceKernel.resources.semaphoreCountPerBlock
       << " vpm_base_row=resident_slot*"
       << resourceKernel.resources.totalVPMRowsPerBlock
       << " user_vpm_rows=" << resourceKernel.resources.userVPMRowsPerBlock
       << " compiler_vpm_staging_rows="
       << resourceKernel.resources.compilerVPMStagingRowsPerBlock
       << " spill_vpm_rows=" << resourceKernel.resources.spillVPMRowsPerBlock
       << " resident_wave_wait_before_resource_reuse="
       << (resourceKernel.resources.scheduleMode == "cooperative_block" ? 1 : 0)
       << " */\n";
  }
  for (const KernelRecord &spillKernel : kernels) {
    os << "/* VC4_KERNEL_SPILL kernel_id=" << spillKernel.kernelId
       << " public_name=" << spillKernel.launchABI.publicName
       << " spill_frame_bytes=" << spillKernel.spill.frameBytes
       << " spill_frame_stride_bytes=" << spillKernel.spill.strideBytes
       << " spill_frame_count=" << spillKernel.spill.frameCount
       << " spill_arena_bytes=" << spillKernel.spill.arenaBytes;
    if (spillKernel.resources.scheduleMode == "cooperative_block") {
      os << " resident_request_id=resident_block_slot*warps_per_block+"
            "logical_warp_id";
      if (spillKernel.resources.spillVPMRowsPerBlock != 0) {
        os << " spill_vpm_row=vpm_base_row+"
           << (spillKernel.resources.userVPMRowsPerBlock +
               spillKernel.resources.compilerVPMStagingRowsPerBlock +
               spillKernel.resources.warpsPerBlock *
                   spillKernel.resources.compilerVPMStagingRowsPerWarp)
           << "+logical_warp_id";
      }
    } else {
      os << " resident_request_id=wave_local_request_index";
      if (spillKernel.spill.frameBytes != 0)
        os << " spill_vpm_row=vpm_base_row+"
           << (spillKernel.resources.userVPMRowsPerBlock +
               spillKernel.resources.compilerVPMStagingRowsPerBlock +
               spillKernel.resources.warpsPerBlock *
                   spillKernel.resources.compilerVPMStagingRowsPerWarp);
    }
    os << " spill_frame_base=spill_arena_base+resident_request_id*"
          "spill_frame_stride_bytes */\n";
  }
  os << "\n";
  os << "static const struct vc4_module_image vc4_codegen_module = {\n";
  os << "  VC4_RUNTIME_MODULE_VERSION,\n";
  os << "  (uint32_t)(sizeof(vc4_codegen_kernels) / "
        "sizeof(vc4_codegen_kernels[0])),\n";
  os << "  VC4_CODEGEN_PROGRAM_HEAP_BYTES,\n";
  os << "  vc4_codegen_kernels,\n";
  os << "};\n\n";
  os << "int vc4_program_create(struct vc4_program **out, uint32_t "
        "requested_bytes) {\n";
  os << "  return vc4ProgramCreateFromImage(out, &vc4_codegen_module, "
        "requested_bytes);\n";
  os << "}\n\n";

  for (const KernelRecord &launchKernel : kernels) {
    const LaunchABIModel &kernelABI = launchKernel.launchABI;
    const std::string kernelBase = getLaunchAPIBaseName(kernelABI);
    os << "struct " << kernelBase << "_pack_ctx {\n";
    os << "  uint32_t __vc4_unused;\n";
    for (const LaunchABIArgumentModel &arg : kernelABI.arguments) {
      if (arg.kind == LaunchABIArgumentKind::Buffer) {
        os << "  vc4_deviceptr_t " << arg.name << ";\n";
      } else {
        std::optional<std::string> cType = getScalarCType(arg.scalarType);
        if (!cType)
          return emitLaunchABIModelError(
              launchKernel.func,
              llvm::Twine("unsupported scalar type for launcher context: ") +
                  arg.scalarType);
        os << "  " << *cType << " " << arg.name << ";\n";
      }
    }
    os << "};\n\n";

    os << "static int " << kernelBase
       << "_pack_uniforms(void *opaque, const struct vc4_launch_request_info "
          "*requestInfo, "
          "uint32_t *uniformWords, uint32_t uniformWordsPerRequest) {\n";
    os << "  struct " << kernelBase << "_pack_ctx *ctx = (struct " << kernelBase
       << "_pack_ctx *)opaque;\n";
    os << "  if (!ctx || !requestInfo || !uniformWords || "
          "uniformWordsPerRequest < KERNEL_"
       << launchKernel.kernelId << "_NUM_UNIFS)\n";
    os << "    return -1;\n";
    appendLauncherUniformLayoutComment(os, kernelABI);
    for (int64_t index = 0; index != kernelABI.uniformWordsPerQPU; ++index) {
      if (const LaunchABIArgumentModel *arg =
              findLaunchABIArgumentForUniformIndex(kernelABI, index)) {
        if (arg->kind == LaunchABIArgumentKind::Buffer) {
          os << "  uniformWords[" << index << "] = (uint32_t)ctx->" << arg->name
             << "; /* arg " << arg->name << " */\n";
        } else if (arg->scalarType == "f32") {
          os << "  uniformWords[" << index << "] = vc4_codegen_pack_f32(ctx->"
             << arg->name << "); /* arg " << arg->name << " */\n";
        } else {
          os << "  uniformWords[" << index << "] = (uint32_t)ctx->" << arg->name
             << "; /* arg " << arg->name << " */\n";
        }
        continue;
      }

      if (const LaunchABIBuiltinModel *builtin =
              findLaunchABIBuiltinForUniformIndex(kernelABI, index)) {
        if (builtin->kind == "spill_vpm_row") {
          if (launchKernel.resources.scheduleMode == "cooperative_block") {
            os << "  uniformWords[" << index
               << "] = requestInfo->vpm_base_row + KERNEL_"
               << launchKernel.kernelId << "_USER_VPM_ROWS_PER_BLOCK + KERNEL_"
               << launchKernel.kernelId
               << "_COMPILER_VPM_STAGING_ROWS_PER_BLOCK + KERNEL_"
               << launchKernel.kernelId << "_WARPS_PER_BLOCK * KERNEL_"
               << launchKernel.kernelId
               << "_COMPILER_VPM_STAGING_ROWS_PER_WARP + "
                  "requestInfo->logical_warp_id; /* builtin "
               << builtin->name << " */\n";
          } else {
            os << "  uniformWords[" << index
               << "] = requestInfo->vpm_base_row + KERNEL_"
               << launchKernel.kernelId << "_USER_VPM_ROWS_PER_BLOCK + KERNEL_"
               << launchKernel.kernelId
               << "_COMPILER_VPM_STAGING_ROWS_PER_BLOCK + KERNEL_"
               << launchKernel.kernelId << "_WARPS_PER_BLOCK * KERNEL_"
               << launchKernel.kernelId
               << "_COMPILER_VPM_STAGING_ROWS_PER_WARP; /* builtin "
               << builtin->name << " */\n";
          }
          continue;
        }
        std::optional<llvm::StringRef> field =
            getRuntimeBuiltinRequestInfoField(builtin->kind);
        if (!field) {
          return emitLaunchABIModelError(
              launchKernel.func,
              llvm::Twine("builtin '") + builtin->name +
                  "' has unsupported kind for launcher uniform packing");
        }
        os << "  uniformWords[" << index << "] = requestInfo->" << *field
           << "; /* builtin " << builtin->name << " */\n";
        continue;
      }

      return emitLaunchABIModelError(
          launchKernel.func,
          llvm::Twine("missing launcher uniform assignment for index ") +
              std::to_string(index));
    }
    os << "  return 0;\n";
    os << "}\n\n";
  }

  for (const KernelRecord &launchKernel : kernels) {
    const LaunchABIModel &kernelABI = launchKernel.launchABI;
    const std::string kernelBase = getLaunchAPIBaseName(kernelABI);
    llvm::SmallVector<const LaunchABIArgumentModel *, 4> bufferArgs =
        collectLaunchABIBufferArguments(kernelABI);

    appendLauncherPrototype(os, kernelABI);
    os << " {\n";
    if (launchKernel.resources.scheduleMode == "cooperative_block") {
      os << "  uint32_t gridBlocks = vc4_codegen_grid_blocks(grid);\n";
      os << "  uint32_t warpsPerBlock = vc4_codegen_block_warps(block);\n";
      os << "  uint32_t logicalN = gridBlocks;\n";
      os << "  if (warpsPerBlock == 0u || warpsPerBlock > KERNEL_"
         << launchKernel.kernelId
         << "_WARPS_PER_BLOCK || warpsPerBlock > VC4_RUNTIME_MAX_QPUS) {\n";
      os << "    vc4ProgramRecordLaunchFailure(program);\n";
      os << "    return -1;\n";
      os << "  }\n";
      os << "  /* schedule_mode=cooperative_block; runtime computes "
            "resident_blocks and waits before VPM/semaphore resource reuse. "
            "*/\n";
      os << "  uint32_t totalRequests = "
            "vc4_codegen_saturating_mul_u32(gridBlocks, warpsPerBlock);\n";
    } else {
      os << "  uint32_t logicalN = vc4_codegen_launch_elements(grid, block);\n";
      os << "  uint32_t totalRequests = vc4_codegen_ceil_div_u32(logicalN, "
            "VC4_RUNTIME_LANE_WIDTH);\n";
      os << "  uint32_t warpsPerBlock = 1u;\n";
      os << "  (void)grid;\n";
      os << "  (void)block;\n";
    }
    os << "  (void)logicalN;\n";
    for (const LaunchABIArgumentModel *arg : bufferArgs) {
      std::string elemType = getBufferElementCTypeForCodegen(*arg);
      os << "  if (totalRequests != 0u) {\n";
      os << "    size_t arg_bytes = sizeof(" << elemType << ");\n";
      os << "    if (arg_bytes > 0xffffffffu || "
            "!vc4DeviceRangeIsAllocated(program, "
         << arg->name << ", (uint32_t)arg_bytes)) {\n";
      os << "      vc4ProgramRecordLaunchFailure(program);\n";
      os << "      return -1;\n";
      os << "    }\n";
      os << "  }\n";
    }
    os << "  struct " << kernelBase << "_pack_ctx ctx;\n";
    os << "  memset(&ctx, 0, sizeof(ctx));\n";
    for (const LaunchABIArgumentModel &arg : kernelABI.arguments)
      os << "  ctx." << arg.name << " = " << arg.name << ";\n";
    os << "  return vc4LaunchKernel(program, " << launchKernel.kernelId
       << "u, grid, block, totalRequests, warpsPerBlock, " << kernelBase
       << "_pack_uniforms, &ctx);\n";
    os << "}\n\n";
  }

  os << "uint32_t " << allocationsName << "(void) {\n";
  os << "  return vc4RuntimeProgramAllocations();\n";
  os << "}\n\n";

  os << "uint32_t " << launchesName << "(void) {\n";
  os << "  return vc4ProgramLaunches(vc4RuntimeCurrentProgram());\n";
  os << "}\n\n";

  os << "uint32_t " << capacityName << "(void) {\n";
  os << "  return vc4ProgramCapacity(vc4RuntimeCurrentProgram());\n";
  os << "}\n\n";

  os << "uint32_t " << codeUploadsName << "(void) {\n";
  os << "  return vc4ProgramCodeUploads(vc4RuntimeCurrentProgram());\n";
  os << "}\n\n";

  os << "uint32_t " << launchFailuresName << "(void) {\n";
  os << "  return vc4ProgramLaunchFailures(vc4RuntimeCurrentProgram());\n";
  os << "}\n";
  os.flush();

  return writeBundleFile(firstKernel.func.getOperation(), bundleDir,
                         "kernel_launch.c",
                         [&](llvm::raw_ostream &fileOS) { fileOS << source; });
}

static void appendManifestLaunchABIArgument(llvm::raw_ostream &os,
                                            const LaunchABIArgumentModel &arg,
                                            bool trailingComma) {
  os << "      {\"name\": ";
  appendJSONEscapedString(os, arg.name);
  os << ", \"kind\": ";
  appendJSONEscapedString(
      os, arg.kind == LaunchABIArgumentKind::Scalar ? "scalar" : "buffer");
  os << ", \"direction\": ";
  appendJSONEscapedString(os, arg.direction);
  if (arg.kind == LaunchABIArgumentKind::Scalar) {
    os << ", \"type\": ";
    appendJSONEscapedString(os, arg.scalarType);
  } else {
    os << ", \"elem_type\": ";
    appendJSONEscapedString(os, arg.elementType);
  }
  os << ", \"c_type\": ";
  appendJSONEscapedString(os, arg.cType);
  os << ", \"uniform_index\": " << arg.uniformIndex << "}";
  if (trailingComma)
    os << ",";
  os << "\n";
}

static void appendManifestLaunchABIBuiltin(llvm::raw_ostream &os,
                                           const LaunchABIBuiltinModel &builtin,
                                           bool trailingComma) {
  os << "      {\"name\": ";
  appendJSONEscapedString(os, builtin.name);
  os << ", \"kind\": ";
  appendJSONEscapedString(os, builtin.kind);
  os << ", \"materialization\": ";
  appendJSONEscapedString(os, builtin.materialization);
  if (builtin.uniformIndex)
    os << ", \"uniform_index\": " << *builtin.uniformIndex;
  os << "}";
  if (trailingComma)
    os << ",";
  os << "\n";
}

static void appendManifestKernelArguments(llvm::raw_ostream &os,
                                          const LaunchABIModel &launchABI) {
  for (size_t i = 0; i != launchABI.arguments.size(); ++i) {
    appendManifestLaunchABIArgument(os, launchABI.arguments[i],
                                    i + 1 != launchABI.arguments.size());
  }
}

static void appendManifestKernelBuiltins(llvm::raw_ostream &os,
                                         const LaunchABIModel &launchABI) {
  for (size_t i = 0; i != launchABI.builtins.size(); ++i) {
    appendManifestLaunchABIBuiltin(os, launchABI.builtins[i],
                                   i + 1 != launchABI.builtins.size());
  }
}

static void appendManifestKernelEntry(llvm::raw_ostream &os,
                                      const KernelRecord &kernel,
                                      bool trailingComma) {
  os << "    {\n";
  os << "      \"kernel_id\": " << kernel.kernelId << ",\n";
  os << "      \"symbol_name\": ";
  appendJSONEscapedString(os, kernel.info.symbolName);
  os << ",\n";
  os << "      \"public_name\": ";
  appendJSONEscapedString(os, kernel.info.publicName);
  os << ",\n";
  os << "      \"qasm_path\": ";
  appendJSONEscapedString(os, kernel.qasmPath);
  os << ",\n";
  os << "      \"code_symbol\": ";
  appendJSONEscapedString(os, kernel.launchABI.codeSymbol);
  os << ",\n";
  os << "      \"scheduled_sink_ops\": " << kernel.info.scheduledOpCount
     << ",\n";
  os << "      \"uniform_words_per_request\": "
     << kernel.info.uniformWordsPerQPU << ",\n";
  os << "      \"uniform_words_per_qpu\": " << kernel.info.uniformWordsPerQPU
     << ",\n";
  os << "      \"max_requests_per_wave\": 12,\n";
  os << "      \"spill_frame_bytes\": " << kernel.info.spillFrameBytes << ",\n";
  os << "      \"spill_frame_stride_bytes\": "
     << kernel.info.spillFrameStrideBytes << ",\n";
  os << "      \"spill_frame_count\": " << kernel.info.spillFrameCount << ",\n";
  os << "      \"spill_arena_bytes\": " << kernel.info.spillArenaBytes << ",\n";
  os << "      \"tail_policy\": ";
  appendJSONEscapedString(os, kernel.launchABI.tailPolicy);
  os << ",\n";
  os << "      \"schedule_mode\": ";
  appendJSONEscapedString(os, kernel.resources.scheduleMode);
  os << ",\n";
  os << "      \"args\": [\n";
  appendManifestKernelArguments(os, kernel.launchABI);
  os << "      ],\n";
  os << "      \"builtins\": [\n";
  appendManifestKernelBuiltins(os, kernel.launchABI);
  os << "      ],\n";
  os << "      \"resources\": {\n";
  os << "        \"schedule_mode\": ";
  appendJSONEscapedString(os, kernel.resources.scheduleMode);
  os << ",\n";
  os << "        \"warps_per_block\": " << kernel.resources.warpsPerBlock
     << ",\n";
  os << "        \"user_vpm_rows_per_block\": "
     << kernel.resources.userVPMRowsPerBlock << ",\n";
  os << "        \"compiler_vpm_staging_rows_per_warp\": "
     << kernel.resources.compilerVPMStagingRowsPerWarp << ",\n";
  os << "        \"compiler_vpm_staging_rows_per_block\": "
     << kernel.resources.compilerVPMStagingRowsPerBlock << ",\n";
  os << "        \"spill_vpm_rows_per_block\": "
     << kernel.resources.spillVPMRowsPerBlock << ",\n";
  os << "        \"total_vpm_rows_per_block\": "
     << kernel.resources.totalVPMRowsPerBlock << ",\n";
  os << "        \"uses_tmu\": "
     << (kernel.resources.usesTMU ? "true" : "false") << ",\n";
  os << "        \"uses_vpm\": "
     << (kernel.resources.usesVPM ? "true" : "false") << ",\n";
  os << "        \"uses_vpm_qpu_read\": "
     << (kernel.resources.usesVPMQPURead ? "true" : "false") << ",\n";
  os << "        \"uses_vpm_qpu_write\": "
     << (kernel.resources.usesVPMQPUWrite ? "true" : "false") << ",\n";
  os << "        \"uses_vdr\": "
     << (kernel.resources.usesVDR ? "true" : "false") << ",\n";
  os << "        \"uses_vdw\": "
     << (kernel.resources.usesVDW ? "true" : "false") << ",\n";
  os << "        \"uses_barrier\": "
     << (kernel.resources.usesBarrier ? "true" : "false") << ",\n";
  os << "        \"semaphore_count_per_block\": "
     << kernel.resources.semaphoreCountPerBlock << ",\n";
  os << "        \"requires_vpm_base_row_builtin\": "
     << (kernel.resources.requiresVPMBaseRowBuiltin ? "true" : "false")
     << ",\n";
  os << "        \"requires_semaphore_base_builtin\": "
     << (kernel.resources.requiresSemaphoreBaseBuiltin ? "true" : "false")
     << ",\n";
  os << "        \"max_resident_blocks\": "
     << kernel.resources.maxResidentBlocks << "\n";
  os << "      }\n";
  os << "    }";
  if (trailingComma)
    os << ",";
  os << "\n";
}

static void appendLayoutRegionJSON(llvm::raw_ostream &os,
                                   const ProgramLayoutRegion &region,
                                   bool trailingComma) {
  os << "    {\"name\": ";
  appendJSONEscapedString(os, region.name);
  os << ", \"kind\": ";
  appendJSONEscapedString(os, region.kind);
  os << ", \"offset\": " << region.offset;
  os << ", \"size\": " << region.size;
  os << ", \"alignment\": " << region.alignment;
  if (region.kernelId)
    os << ", \"kernel_id\": " << *region.kernelId;
  os << "}";
  if (trailingComma)
    os << ",";
  os << "\n";
}

static void appendLayoutByteRange(llvm::raw_ostream &os, uint64_t offset,
                                  uint64_t size) {
  os << "{\"offset\": " << offset << ", \"size\": " << size
     << ", \"alignment\": " << kVC4ProgramLayoutAlignment << "}";
}

static void appendKernelLayoutJSON(llvm::raw_ostream &os,
                                   const KernelLayoutRecord &kernel,
                                   bool trailingComma) {
  os << "    {\n";
  os << "      \"kernel_id\": " << kernel.kernelId << ",\n";
  os << "      \"public_name\": ";
  appendJSONEscapedString(os, kernel.publicName);
  os << ",\n";
  os << "      \"code_symbol\": ";
  appendJSONEscapedString(os, kernel.codeSymbol);
  os << ",\n";
  os << "      \"descriptor\": ";
  appendLayoutByteRange(os, kernel.descriptorOffset, kernel.descriptorSize);
  os << ",\n";
  os << "      \"code\": ";
  appendLayoutByteRange(os, kernel.codeOffset, kernel.codeSize);
  os << ",\n";
  os << "      \"uniforms\": ";
  appendLayoutByteRange(os, kernel.uniformsOffset, kernel.uniformsSize);
  os << ",\n";
  os << "      \"uniform_stream\": ";
  appendLayoutByteRange(os, kernel.uniformsOffset, kernel.uniformsSize);
  os << ",\n";
  os << "      \"unif_ptrs\": ";
  appendLayoutByteRange(os, kernel.unifPtrOffset, kernel.unifPtrSize);
  os << ",\n";
  os << "      \"uniform_pointer_array\": ";
  appendLayoutByteRange(os, kernel.unifPtrOffset, kernel.unifPtrSize);
  os << ",\n";
  os << "      \"code_words\": " << kernel.codeWords << ",\n";
  os << "      \"uniform_words_per_request\": " << kernel.uniformWordsPerRequest
     << ",\n";
  os << "      \"max_requests_per_wave\": " << kernel.maxRequestsPerWave
     << ",\n";
  os << "      \"spill_frame_bytes\": " << kernel.spillFrameBytes << ",\n";
  os << "      \"spill_frame_stride_bytes\": " << kernel.spillFrameStrideBytes
     << ",\n";
  os << "      \"spill_frame_count\": " << kernel.spillFrameCount << ",\n";
  os << "      \"spill_arena_bytes\": " << kernel.spillArenaBytes << "\n";
  os << "    }";
  if (trailingComma)
    os << ",";
  os << "\n";
}

static LogicalResult writeProgramLayout(mlir::vc4::ModuleOp vc4Module,
                                        llvm::ArrayRef<KernelRecord> kernels,
                                        llvm::StringRef bundleDir) {
  ProgramLayoutModel layout = buildProgramLayoutModel(kernels);
  return writeBundleFile(
      vc4Module.getOperation(), bundleDir, "layout.json",
      [&](llvm::raw_ostream &os) {
        os << "{\n";
        os << "  \"schema_version\": 1,\n";
        os << "  \"kind\": \"vc4-program-layout\",\n";
        os << "  \"program_name\": ";
        appendJSONEscapedString(os, vc4Module.getSymName());
        os << ",\n";
        os << "  \"alignment\": " << layout.alignment << ",\n";
        os << "  \"kernel_count\": " << kernels.size() << ",\n";
        os << "  \"max_active_qpus\": " << kVC4MaxRequestsPerWave << ",\n";
        os << "  \"program_bytes\": " << layout.programBytes << ",\n";
        os << "  \"static_bytes\": " << layout.staticBytes << ",\n";
        os << "  \"spill_frame_alignment_bytes\": " << kVC4SpillFrameAlignment
           << ",\n";
        os << "  \"max_spill_arena_bytes\": " << layout.spillArenaBytes
           << ",\n";
        os << "  \"hidden_spill_arena_bytes\": " << layout.spillArenaBytes
           << ",\n";
        os << "  \"hidden_spill_arena\": ";
        appendLayoutByteRange(os, layout.spillArenaOffset,
                              layout.spillArenaBytes);
        os << ",\n";
        os << "  \"heap_offset\": " << layout.heapOffset << ",\n";
        os << "  \"heap_offset_bytes\": " << layout.heapOffset << ",\n";
        os << "  \"heap_bytes\": " << layout.heapBytes << ",\n";
        os << "  \"heap_size_bytes\": " << layout.heapBytes << ",\n";
        os << "  \"heap\": ";
        appendLayoutByteRange(os, layout.heapOffset, layout.heapBytes);
        os << ",\n";
        os << "  \"regions\": [\n";
        for (size_t i = 0; i != layout.regions.size(); ++i) {
          appendLayoutRegionJSON(os, layout.regions[i],
                                 i + 1 != layout.regions.size());
        }
        os << "  ],\n";
        os << "  \"kernels\": [\n";
        for (size_t i = 0; i != layout.kernels.size(); ++i) {
          appendKernelLayoutJSON(os, layout.kernels[i],
                                 i + 1 != layout.kernels.size());
        }
        os << "  ]\n";
        os << "}\n";
      });
}

static LogicalResult writeManifest(mlir::vc4::ModuleOp vc4Module,
                                   llvm::ArrayRef<KernelRecord> kernels,
                                   llvm::StringRef bundleDir) {
  ProgramLayoutModel layout = buildProgramLayoutModel(kernels);
  return writeBundleFile(
      vc4Module.getOperation(), bundleDir, "manifest.json",
      [&](llvm::raw_ostream &os) {
        os << "{\n";
        os << "  \"schema_version\": 2,\n";
        os << "  \"kind\": \"vc4-codegen-artifact-bundle\",\n";
        os << "  \"program_name\": ";
        appendJSONEscapedString(os, vc4Module.getSymName());
        os << ",\n";
        os << "  \"target\": {\n";
        os << "    \"name\": \"vc4-bcm2835-user-qpu\",\n";
        os << "    \"warp_size\": 16,\n";
        os << "    \"max_active_qpus\": 12,\n";
        os << "    \"total_vpm_bytes\": 4096,\n";
        os << "    \"semaphores\": 16\n";
        os << "  },\n";
        os << "  \"spill_frame_alignment_bytes\": " << kVC4SpillFrameAlignment
           << ",\n";
        os << "  \"max_spill_arena_bytes\": " << layout.spillArenaBytes
           << ",\n";
        os << "  \"kernels\": [\n";
        for (size_t i = 0; i != kernels.size(); ++i) {
          appendManifestKernelEntry(os, kernels[i], i + 1 != kernels.size());
        }
        os << "  ],\n";
        os << "  \"artifacts\": [\n";
        os << "    \"kernel_launch.c\",\n";
        os << "    \"kernel_launch.h\",\n";
        os << "    \"layout.json\"";
        for (const KernelRecord &kernel : kernels) {
          os << ",\n    ";
          appendJSONEscapedString(os, kernel.qasmPath);
        }
        os << "\n  ]\n";
        os << "}\n";
      });
}

} // namespace

LogicalResult mlir::vc4::emitVC4ArtifactBundle(mlir::ModuleOp module,
                                               llvm::StringRef bundleDir) {
  llvm::SmallVector<mlir::vc4::ModuleOp, 1> vc4Modules;
  module.walk(
      [&](mlir::vc4::ModuleOp vc4Module) { vc4Modules.push_back(vc4Module); });
  if (vc4Modules.size() != 1) {
    return module.emitError()
           << "expected exactly one vc4.module for artifact emission";
  }

  mlir::vc4::ModuleOp vc4Module = vc4Modules.front();
  llvm::SmallVector<KernelRecord, 1> kernels;
  if (failed(collectProgramKernels(vc4Module, kernels)))
    return failure();

  std::error_code ec = llvm::sys::fs::create_directories(bundleDir);
  if (ec) {
    return module.emitError() << "failed to create artifact bundle directory '"
                              << bundleDir << "': " << ec.message();
  }

  if (failed(ensureAdjacentVC4ASMTemplates(module.getOperation(), bundleDir)))
    return failure();

  for (KernelRecord &kernel : kernels) {
    if (failed(writeQASM(kernel, bundleDir)))
      return failure();
  }
  if (failed(writeLauncherSource(kernels, bundleDir)))
    return failure();
  if (failed(writeLauncherHeader(kernels, bundleDir)))
    return failure();
  if (failed(writeProgramLayout(vc4Module, kernels, bundleDir)))
    return failure();
  if (failed(writeManifest(vc4Module, kernels, bundleDir)))
    return failure();

  return success();
}
