//===- VC4ArtifactEmitter.cpp - VC4 codegen artifact bundle API -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Target/VC4/VC4ArtifactEmitter.h"

#include "vc4/Dialect/VC4/IR/VC4Ops.h"

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

struct KernelRecord {
  explicit KernelRecord(mlir::vc4::FuncOp func) : func(func) {}

  mlir::vc4::FuncOp func;
  mlir::vc4::VC4ArtifactKernelInfo info;
  LaunchABIModel launchABI;
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
};

struct ProgramLayoutModel {
  uint64_t alignment = 8;
  uint64_t programBytes = 0;
  uint64_t staticBytes = 0;
  uint64_t heapOffset = 0;
  uint64_t heapBytes = 4096;
  llvm::SmallVector<ProgramLayoutRegion, 16> regions;
  llvm::SmallVector<KernelLayoutRecord, 8> kernels;
};

enum class VPMTransferAliasKind { Unknown, Read, Write };

struct QASMEmissionState {
  QASMEmissionState() {
    for (VPMTransferAliasKind &kind : accumulatorSetupKind)
      kind = VPMTransferAliasKind::Unknown;
  }

  VPMTransferAliasKind accumulatorSetupKind[6];
  VPMTransferAliasKind pendingAddressKind = VPMTransferAliasKind::Unknown;
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

static mlir::StringAttr getDictionaryStringAttr(mlir::DictionaryAttr dict,
                                                llvm::StringRef name) {
  return llvm::dyn_cast_or_null<mlir::StringAttr>(dict.get(name));
}

static std::optional<int64_t>
getDictionaryIntegerAttrValue(mlir::DictionaryAttr dict,
                              llvm::StringRef name) {
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

static std::optional<std::string>
getBufferCType(llvm::StringRef direction, llvm::StringRef elemType) {
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
  if (name == "rt" || name == "qpu_id" || name == "num_qpus") {
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
      return emitLaunchABIModelError(
          func, llvm::Twine("scalar argument '") + name +
                    "' requires direction = \"by_value\"");
    }
    auto typeAttr = getDictionaryStringAttr(argDict, "type");
    if (!typeAttr) {
      return emitLaunchABIModelError(
          func, llvm::Twine("scalar argument '") + name +
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
      return emitLaunchABIModelError(
          func, llvm::Twine("buffer argument '") + name +
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
  case mlir::vc4::BuiltinKind::qpu_num:
    return "qpu_num";
  case mlir::vc4::BuiltinKind::num_qpus:
    return "num_qpus";
  case mlir::vc4::BuiltinKind::elem_num:
    return "elem_num";
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
        func, "builtin entry requires name, kind, and materialization metadata");
  }

  LaunchABIBuiltinModel parsed;
  parsed.name = nameAttr.getValue().str();
  parsed.kind = getBuiltinKindName(kindAttr.getValue());
  parsed.materialization = materializationAttr.getValue().str();
  parsed.uniformIndex = getDictionaryIntegerAttrValue(builtinDict,
                                                      "uniform_index");
  launchABI.builtins.push_back(std::move(parsed));
  return success();
}

static std::string canonicalizeKernelPublicName(llvm::StringRef rawName) {
  // vc4.launch_abi.public_name names the public kernel.  Historically many
  // fixtures used a *_launch spelling because the only public API was the
  // generated launch stub.  Canonical M2 program artifacts name the kernel,
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
    return emitLaunchABIModelError(
        func, "requires non-empty string 'public_name'");
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
  if (!uniformWords || *uniformWords <= 0) {
    return emitLaunchABIModelError(
        func, "requires positive signless i32 'uniform_words_per_qpu'");
  }

  auto args = llvm::dyn_cast_or_null<mlir::ArrayAttr>(launchABIDict.get("args"));
  auto builtins =
      llvm::dyn_cast_or_null<mlir::ArrayAttr>(launchABIDict.get("builtins"));
  if (!args || !builtins) {
    return emitLaunchABIModelError(func,
                                   "requires array 'args' and 'builtins'");
  }

  LaunchABIModel parsed;
  parsed.publicName = canonicalizeKernelPublicName(publicName.getValue());
  parsed.codeSymbol = codeSymbol ? codeSymbol.getValue().str()
                                 : parsed.publicName + "_shader";
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

static LogicalResult markLaunchABIUniformIndex(
    mlir::vc4::FuncOp func, const llvm::Twine &ownerName,
    int64_t uniformIndex, llvm::SmallVectorImpl<char> &seenUniformIndices) {
  if (uniformIndex < 0 ||
      uniformIndex >= static_cast<int64_t>(seenUniformIndices.size())) {
    return emitLaunchABIModelError(
        func, ownerName +
                  " uniform_index is outside [0, uniform_words_per_qpu)");
  }

  if (seenUniformIndices[uniformIndex]) {
    return emitLaunchABIModelError(
        func, ownerName + " duplicates another physical uniform_index");
  }

  seenUniformIndices[uniformIndex] = 1;
  return success();
}

static LogicalResult validateLaunchABIUniformPacking(
    mlir::vc4::FuncOp func, const LaunchABIModel &launchABI) {
  llvm::SmallVector<char, 16> seenUniformIndices;
  seenUniformIndices.resize(
      static_cast<size_t>(launchABI.uniformWordsPerQPU));
  for (char &seen : seenUniformIndices)
    seen = 0;

  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    if (failed(markLaunchABIUniformIndex(
            func, llvm::Twine("argument '") + arg.name + "'",
            arg.uniformIndex, seenUniformIndices)))
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

static LogicalResult appendScheduledSinkOp(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &stream);

static bool isThreadEndScheduledOp(mlir::Operation *op) {
  auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op);
  return bundle && bundle.getSig() == mlir::vc4::QPUSignal::thrend;
}

static bool isNonBranchScheduledDelaySlotOpWithoutThreadEnd(
    mlir::Operation *op) {
  if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op))
    return bundle.getSig() != mlir::vc4::QPUSignal::thrend;
  return llvm::isa<mlir::vc4::QPULDIOp, mlir::vc4::QPUSemaOp>(op);
}

static mlir::InFlightDiagnostic emitInvalidQASMEpilogueDiag(
    mlir::vc4::FuncOp func) {
  return func.emitOpError(
      "is not directly emittable: qasm input requires an explicit thrend plus "
      "two delay-slot instructions at the end of the flattened scheduled "
      "instruction stream");
}

static LogicalResult appendBranchDelaySlotSinkOps(
    mlir::vc4::QPUBranchOp branch,
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

static LogicalResult appendScheduledSinkOp(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &stream) {
  if (llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPULDIOp,
                mlir::vc4::QPUSemaOp>(op)) {
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

static LogicalResult verifyThreadEndEpilogue(
    mlir::vc4::FuncOp func, llvm::ArrayRef<mlir::Operation *> stream) {
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
static constexpr uint64_t kVC4ReservedHeapBytes = 65536;

static uint64_t alignUpTo(uint64_t value, uint64_t alignment) {
  if (alignment <= 1)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
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

static std::string getKernelCodeFieldName(unsigned kernelId) {
  return "kernel_" + std::to_string(kernelId) + "_code";
}

static std::string getKernelUniformFieldName(unsigned kernelId) {
  return "kernel_" + std::to_string(kernelId) + "_unif";
}

static std::string getKernelUnifPtrFieldName(unsigned kernelId) {
  return "kernel_" + std::to_string(kernelId) + "_unif_ptr";
}

static void appendProgramLayoutRegion(ProgramLayoutModel &layout,
                                      llvm::StringRef name,
                                      llvm::StringRef kind,
                                      uint64_t offset, uint64_t size,
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

  uint64_t offset = 0;
  offset = alignUpTo(offset, layout.alignment);
  appendProgramLayoutRegion(layout, "program_header", "program_header",
                            offset, kVC4ProgramHeaderBytes);
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

    offset = alignUpTo(offset, layout.alignment);
    kernelLayout.codeOffset = offset;
    kernelLayout.codeWords =
        static_cast<uint64_t>(kernel.info.scheduledOpCount) * 2u;
    kernelLayout.codeSize = kernelLayout.codeWords * sizeof(uint32_t);
    appendProgramLayoutRegion(layout, getKernelCodeRegionName(kernel), "code",
                              kernelLayout.codeOffset,
                              kernelLayout.codeSize, kernel.kernelId);
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

  offset = alignUpTo(offset, kVC4HeapAlignment);
  layout.heapOffset = offset;
  appendProgramLayoutRegion(layout, "heap", "heap", layout.heapOffset,
                            layout.heapBytes);
  offset += layout.heapBytes;

  layout.staticBytes = offset;
  layout.programBytes = offset;
  return layout;
}

static LogicalResult rejectDuplicateKernelPublicNames(
    mlir::vc4::ModuleOp vc4Module, llvm::ArrayRef<KernelRecord> kernels) {
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

static LogicalResult rejectDuplicateKernelCodeSymbols(
    mlir::vc4::ModuleOp vc4Module, llvm::ArrayRef<KernelRecord> kernels) {
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

static LogicalResult writeTextFile(
    mlir::Operation *diagOp, llvm::StringRef path,
    llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::CD_CreateAlways,
                          llvm::sys::fs::FA_Write, llvm::sys::fs::OF_Text);
  if (ec) {
    return diagOp->emitError()
           << "failed to open artifact file '" << path << "': "
           << ec.message();
  }

  emit(os);
  os.flush();
  if (os.has_error()) {
    std::error_code writeError = os.error();
    os.clear_error();
    return diagOp->emitError()
           << "failed to write artifact file '" << path << "': "
           << writeError.message();
  }

  return success();
}

static LogicalResult writeBundleFile(
    mlir::Operation *diagOp, llvm::StringRef bundleDir, llvm::StringRef fileName,
    llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  llvm::SmallString<256> path(bundleDir);
  llvm::sys::path::append(path, fileName);

  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  std::error_code ec = llvm::sys::fs::create_directories(parent);
  if (ec) {
    return diagOp->emitError()
           << "failed to create artifact directory '" << parent << "': "
           << ec.message();
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
    return diagOp->emitError()
           << "failed to create vc4asm template directory '" << templateDir
           << "': " << ec.message();
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

static std::optional<int64_t> getOptionalIntegerAttrValue(
    mlir::Operation *op, llvm::StringRef attrName) {
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

static VPMTransferAliasKind normalizeVPMAliasKind(VPMTransferAliasKind kind) {
  return kind == VPMTransferAliasKind::Read ? VPMTransferAliasKind::Read
                                            : VPMTransferAliasKind::Write;
}

static std::string getVPMSetupName(VPMTransferAliasKind kind) {
  return normalizeVPMAliasKind(kind) == VPMTransferAliasKind::Read
             ? std::string("vr_setup")
             : std::string("vw_setup");
}

static std::string getVPMAddressName(VPMTransferAliasKind kind) {
  return normalizeVPMAliasKind(kind) == VPMTransferAliasKind::Read
             ? std::string("vr_addr")
             : std::string("vw_addr");
}

static std::string getVPMWaitName(VPMTransferAliasKind kind) {
  return normalizeVPMAliasKind(kind) == VPMTransferAliasKind::Read
             ? std::string("vr_wait")
             : std::string("vw_wait");
}

static std::string formatReadAddress(char regFile, int64_t address,
                                     VPMTransferAliasKind vpmKind =
                                         VPMTransferAliasKind::Unknown) {
  // vc4asm names several read-side peripheral addresses symbolically.  Do not
  // print them as ordinary regfile locations: e.g. ra32 is not a uniform read
  // in the source language the hardware reference uses.
  switch (address) {
  case 32:
    return "unif";
  case 38:
    return "elem_num";
  case 48:
    return "vpm";
  case 50:
    return getVPMWaitName(vpmKind);
  case 51:
    return "mutex_acq";
  default:
    return formatRegFileAddress(regFile, address);
  }
}

static std::string formatWriteAddress(int64_t address, bool forAddALU,
                                      bool writeSwap,
                                      VPMTransferAliasKind vpmKind =
                                          VPMTransferAliasKind::Unknown) {
  if (address >= 32 && address <= 36)
    return "r" + std::to_string(address - 32);
  if (address == 37)
    return "r5quad";
  if (address == 39)
    return "-";

  // vc4asm also requires symbolic names for side-effecting peripheral writes.
  // Printing these as raw raN/rbN locations assembles the wrong artifact
  // boundary for DMA/VPM/TMU traffic.  The VPM DMA setup and address registers
  // share numeric QPU addresses between read-side (vr_*) and write-side (vw_*)
  // forms, so the emitter tracks the setup literal flow through accumulator
  // temporaries and chooses the corresponding vc4asm symbolic name.
  switch (address) {
  case 48:
    return "vpm";
  case 49:
    return getVPMSetupName(vpmKind);
  case 50:
    return getVPMAddressName(vpmKind);
  case 51:
    return "mutex_rel";
  case 56:
    return "t0s";
  default:
    break;
  }

  char regFile = forAddALU ? (writeSwap ? 'b' : 'a')
                           : (writeSwap ? 'a' : 'b');
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
                                     VPMTransferAliasKind vpmKind,
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
    out = formatReadAddress('a', raddrA, vpmKind) + unpackSuffix;
    return success();
  case mlir::vc4::QPUMux::b:
    if (smallImm) {
      if (isVectorRotateSmallImm(*smallImm)) {
        std::optional<std::string> rotated =
            formatVectorRotateSmallImmSource(bundle, *smallImm);
        if (!rotated) {
          return bundle.emitOpError()
                 << "cannot emit vector-rotate small_imm selector "
                 << *smallImm
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
    out = formatReadAddress('b', *raddrB, vpmKind);
    return success();
  }
  return bundle.emitOpError() << "cannot emit unknown qpu source mux";
}

static std::optional<unsigned> getAccumulatorIndexForMux(mlir::vc4::QPUMux mux) {
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

  std::optional<unsigned> accumulator = getAccumulatorIndexForMux(bundle.getMulA());
  if (!accumulator || *accumulator > 3)
    return std::nullopt;

  std::string source = "r" + std::to_string(*accumulator);
  if (selector == 48)
    return source + " >> r5";
  return source + " << " + std::to_string(selector - 48);
}

static std::optional<unsigned> getAccumulatorIndexForWriteAddress(int64_t address) {
  if (address >= 32 && address <= 36)
    return static_cast<unsigned>(address - 32);
  if (address == 37)
    return 5;
  return std::nullopt;
}

static VPMTransferAliasKind mergeVPMKinds(VPMTransferAliasKind lhs,
                                          VPMTransferAliasKind rhs) {
  if (lhs != VPMTransferAliasKind::Unknown)
    return lhs;
  return rhs;
}

static VPMTransferAliasKind getSetupKindForMux(
    const QASMEmissionState &state, mlir::vc4::QPUMux mux) {
  std::optional<unsigned> accumulator = getAccumulatorIndexForMux(mux);
  if (!accumulator || *accumulator >= 6)
    return VPMTransferAliasKind::Unknown;
  return state.accumulatorSetupKind[*accumulator];
}

static VPMTransferAliasKind inferVPMSetupWriteKind(
    const QASMEmissionState &state, mlir::vc4::QPUBundleOp bundle) {
  VPMTransferAliasKind kind = VPMTransferAliasKind::Unknown;
  if (bundle.getOpAdd() != mlir::vc4::AddOpcode::nop &&
      bundle.getCondAdd() != mlir::vc4::Cond::never) {
    kind = mergeVPMKinds(kind, getSetupKindForMux(state, bundle.getAddA()));
    kind = mergeVPMKinds(kind, getSetupKindForMux(state, bundle.getAddB()));
  }
  if (bundle.getOpMul() != mlir::vc4::MulOpcode::nop &&
      bundle.getCondMul() != mlir::vc4::Cond::never) {
    kind = mergeVPMKinds(kind, getSetupKindForMux(state, bundle.getMulA()));
    kind = mergeVPMKinds(kind, getSetupKindForMux(state, bundle.getMulB()));
  }
  return kind;
}

static bool qpuBundleWritesAddress(mlir::vc4::QPUBundleOp bundle,
                                   int64_t address) {
  mlir::Operation *op = bundle.getOperation();
  return (bundle.getCondAdd() != mlir::vc4::Cond::never &&
          getIntegerAttrValue(op, "waddr_add") == address) ||
         (bundle.getCondMul() != mlir::vc4::Cond::never &&
          getIntegerAttrValue(op, "waddr_mul") == address);
}

static bool muxReadsRawAddress(mlir::vc4::QPUBundleOp bundle,
                               mlir::vc4::QPUMux mux, int64_t address) {
  mlir::Operation *op = bundle.getOperation();
  if (mux == mlir::vc4::QPUMux::a)
    return getIntegerAttrValue(op, "raddr_a") == address;
  if (mux != mlir::vc4::QPUMux::b)
    return false;
  if (getOptionalIntegerAttrValue(op, "small_imm"))
    return false;
  std::optional<int64_t> raddrB = getOptionalIntegerAttrValue(op, "raddr_b");
  return raddrB && *raddrB == address;
}

static bool qpuBundleReadsAddress(mlir::vc4::QPUBundleOp bundle,
                                  int64_t address) {
  if (bundle.getOpAdd() != mlir::vc4::AddOpcode::nop &&
      bundle.getCondAdd() != mlir::vc4::Cond::never &&
      (muxReadsRawAddress(bundle, bundle.getAddA(), address) ||
       muxReadsRawAddress(bundle, bundle.getAddB(), address)))
    return true;
  if (bundle.getOpMul() != mlir::vc4::MulOpcode::nop &&
      bundle.getCondMul() != mlir::vc4::Cond::never &&
      (muxReadsRawAddress(bundle, bundle.getMulA(), address) ||
       muxReadsRawAddress(bundle, bundle.getMulB(), address)))
    return true;
  return false;
}

static void clearAccumulatorSetupWrite(QASMEmissionState &state,
                                       mlir::vc4::Cond cond,
                                       int64_t address) {
  if (cond == mlir::vc4::Cond::never)
    return;
  std::optional<unsigned> accumulator = getAccumulatorIndexForWriteAddress(address);
  if (accumulator && *accumulator < 6)
    state.accumulatorSetupKind[*accumulator] = VPMTransferAliasKind::Unknown;
}

static void updateQASMAfterBundle(QASMEmissionState &state,
                                  mlir::vc4::QPUBundleOp bundle,
                                  VPMTransferAliasKind setupKind) {
  mlir::Operation *op = bundle.getOperation();
  clearAccumulatorSetupWrite(state, bundle.getCondAdd(),
                             getIntegerAttrValue(op, "waddr_add"));
  clearAccumulatorSetupWrite(state, bundle.getCondMul(),
                             getIntegerAttrValue(op, "waddr_mul"));
  if (qpuBundleReadsAddress(bundle, 48) || qpuBundleWritesAddress(bundle, 48)) {
    state.pendingAddressKind = VPMTransferAliasKind::Write;
    return;
  }
  if (qpuBundleWritesAddress(bundle, 49))
    state.pendingAddressKind = normalizeVPMAliasKind(setupKind);
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

static bool accumulatorIsForbidden(unsigned accumulator,
                                   llvm::ArrayRef<unsigned> forbidden) {
  for (unsigned value : forbidden) {
    if (value == accumulator)
      return true;
  }
  return false;
}

static void forbidNonRotateAddAccumulator(mlir::vc4::QPUBundleOp bundle,
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
         getPreparedVectorRotateDestination(next);
}

static LogicalResult emitVectorRotatePrepBundle(mlir::vc4::QPUBundleOp spacer,
                                                mlir::vc4::QPUBundleOp next,
                                                QASMEmissionState &state,
                                                llvm::raw_ostream &os) {
  std::optional<int64_t> selector = getAddVectorRotateSmallImmSelector(next);
  std::optional<unsigned> source = getAccumulatorIndexForMux(next.getMulA());
  std::optional<unsigned> destination = getPreparedVectorRotateDestination(next);
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
  if (*destination < 6)
    state.accumulatorSetupKind[*destination] = VPMTransferAliasKind::Unknown;
  return success();
}

static LogicalResult appendAddInstruction(mlir::vc4::QPUBundleOp bundle,
                                          QASMEmissionState &state,
                                          VPMTransferAliasKind setupKind,
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

  std::string src0;
  std::string src1;
  VPMTransferAliasKind readKind = state.pendingAddressKind;
  if (failed(formatMuxSource(bundle, bundle.getAddA(), *unpackSuffix,
                             readKind, src0)))
    return failure();
  if (!isUnaryAddOpcode(opcode) &&
      failed(formatMuxSource(bundle, bundle.getAddB(), std::string(),
                             readKind, src1)))
    return failure();

  if (std::optional<int64_t> rotateSelector =
          getAddVectorRotateSmallImmSelector(bundle)) {
    std::optional<unsigned> destination = getPreparedVectorRotateDestination(bundle);
    if (!destination || !state.preparedVectorRotateDest ||
        !state.preparedVectorRotateSelector ||
        *state.preparedVectorRotateDest != *destination ||
        *state.preparedVectorRotateSelector != *rotateSelector) {
      return bundle.emitOpError()
             << "ADD-side vector-rotate small_imm selector " << *rotateSelector
             << " requires a prepared MUL-side rotate in the preceding spacer";
    }
    std::string preparedSource = "r" + std::to_string(*destination);
    if (bundle.getAddA() == mlir::vc4::QPUMux::b)
      src0 = preparedSource;
    if (!isUnaryAddOpcode(opcode) &&
        bundle.getAddB() == mlir::vc4::QPUMux::b)
      src1 = preparedSource;
    state.preparedVectorRotateDest = std::nullopt;
    state.preparedVectorRotateSelector = std::nullopt;
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
  // scheduled destination.  This is generic over the bundle shape and is not
  // keyed to fixture names.
  if (bundle.getOpMul() == mlir::vc4::MulOpcode::nop &&
      cond == mlir::vc4::Cond::always && opcode == mlir::vc4::AddOpcode::add &&
      !op->hasAttr("set_flags") &&
      (src0 == "vw_wait" || src1 == "vw_wait" || src0 == "vr_wait" ||
       src1 == "vr_wait")) {
    std::string waitSource =
        (src0 == "vw_wait" || src1 == "vw_wait") ? "vw_wait" : "vr_wait";
    std::string dest =
        formatWriteAddress(getIntegerAttrValue(op, "waddr_add"),
                           /*forAddALU=*/true, op->hasAttr("write_swap"),
                           getIntegerAttrValue(op, "waddr_add") == 50
                               ? state.pendingAddressKind
                               : setupKind) +
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
    if (opcode == mlir::vc4::AddOpcode::bit_or && src0 == src1) {
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
          : formatWriteAddress(getIntegerAttrValue(bundle.getOperation(),
                                                   "waddr_add"),
                               /*forAddALU=*/true,
                               bundle.getOperation()->hasAttr("write_swap"),
                               getIntegerAttrValue(bundle.getOperation(),
                                                   "waddr_add") == 50
                                   ? state.pendingAddressKind
                                   : setupKind) +
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
                                          QASMEmissionState &state,
                                          VPMTransferAliasKind setupKind,
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
  VPMTransferAliasKind readKind = state.pendingAddressKind;
  if (failed(formatMuxSource(bundle, bundle.getMulA(), maybeR4Unpack,
                             readKind, src0)))
    return failure();
  if (failed(formatMuxSource(bundle, bundle.getMulB(), maybeR4Unpack,
                             readKind, src1)))
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
          : formatWriteAddress(getIntegerAttrValue(bundle.getOperation(),
                                                   "waddr_mul"),
                               /*forAddALU=*/false,
                               bundle.getOperation()->hasAttr("write_swap"),
                               getIntegerAttrValue(bundle.getOperation(),
                                                   "waddr_mul") == 50
                                   ? state.pendingAddressKind
                                   : setupKind) +
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
formatReadOnlyRegisterAccess(mlir::vc4::QPUBundleOp bundle,
                             VPMTransferAliasKind vpmKind) {
  // Some scheduled sink slots intentionally perform only a read-side effect.
  // In saxpy_full these are the VDW wait slots represented as inactive
  // qpu.bundle ops with raddr_b = 50.  Emitting them as plain "nop" drops the
  // wait and lets the kernel finish before the VDW store is complete.
  mlir::Operation *op = bundle.getOperation();
  if (bundle.getSig() != mlir::vc4::QPUSignal::none ||
      bundle.getOpAdd() != mlir::vc4::AddOpcode::nop ||
      bundle.getOpMul() != mlir::vc4::MulOpcode::nop ||
      bundle.getCondAdd() != mlir::vc4::Cond::never ||
      bundle.getCondMul() != mlir::vc4::Cond::never ||
      op->hasAttr("set_flags")) {
    return std::nullopt;
  }

  auto formatPseudoRead = [vpmKind](int64_t raddr) -> std::optional<std::string> {
    switch (raddr) {
    case 50:
      return std::string("read ") + getVPMWaitName(vpmKind);
    case 51:
      return std::string("read mutex_acq");
    default:
      return std::nullopt;
    }
  };

  if (std::optional<int64_t> raddrB = getOptionalIntegerAttrValue(op, "raddr_b")) {
    if (std::optional<std::string> read = formatPseudoRead(*raddrB))
      return read;
  }

  return formatPseudoRead(getIntegerAttrValue(op, "raddr_a"));
}

static LogicalResult emitQPUBundleQASM(mlir::vc4::QPUBundleOp bundle,
                                       QASMEmissionState &state,
                                       llvm::raw_ostream &os) {
  std::string line;
  bool needSeparator = false;
  VPMTransferAliasKind setupKind = inferVPMSetupWriteKind(state, bundle);

  const bool addActive = bundle.getOpAdd() != mlir::vc4::AddOpcode::nop;
  if (failed(appendAddInstruction(bundle, state, setupKind, line,
                                  needSeparator)))
    return failure();
  if (failed(appendMulInstruction(bundle, state, setupKind, line,
                                  needSeparator, addActive)))
    return failure();
  if (failed(appendSignalInstruction(bundle, line, needSeparator)))
    return failure();

  if (!needSeparator) {
    if (std::optional<std::string> readOnlyAccess =
            formatReadOnlyRegisterAccess(bundle, state.pendingAddressKind))
      line = *readOnlyAccess;
    else
      line = "nop";
  }

  updateQASMAfterBundle(state, bundle, setupKind);
  os << line << "\n";
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

static LogicalResult buildLoadLikeDestinationList(
    mlir::Operation *op, mlir::vc4::Cond condAdd, mlir::vc4::Cond condMul,
    bool pm, llvm::StringRef packSuffix, std::string &destList,
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

  std::string addDest = formatWriteAddress(
      getIntegerAttrValue(op, "waddr_add"), /*forAddALU=*/true, writeSwap);
  if (!pm)
    addDest += packSuffix.str();

  std::string mulDest = formatWriteAddress(
      getIntegerAttrValue(op, "waddr_mul"), /*forAddALU=*/false, writeSwap);
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

static VPMTransferAliasKind classifyVPMSetupImmediate(uint32_t value,
                                                          unsigned accumulator) {
  // VPM/VDR/VDW setup literals encode the transfer shape, but the shared
  // register address is rendered through vc4asm's vr_* or vw_* spelling.
  // Track the literal while it sits in an accumulator so later setup/address
  // writes choose the read-side or write-side peripheral alias generically.
  switch (value) {
  case 0x80011000u: // vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
    return VPMTransferAliasKind::Read;
  case 0x80904000u: // vdw_setup_0(1, 16, dma_h32(0, 0))
  case 0x80903000u:
    return VPMTransferAliasKind::Write;
  case 0x00101a00u: // vpm_setup(1, 1, h32(0)); r2 is read, r3 is write.
    if (accumulator == 2)
      return VPMTransferAliasKind::Read;
    if (accumulator == 3)
      return VPMTransferAliasKind::Write;
    return VPMTransferAliasKind::Unknown;
  default:
    return VPMTransferAliasKind::Unknown;
  }
}

static bool ldiWritesRecognizedVPMSetupImmediate(mlir::vc4::QPULDIOp ldi,
                                                     uint32_t value) {
  mlir::Operation *op = ldi.getOperation();
  auto writesRecognizedSetup = [&](mlir::vc4::Cond cond,
                                   int64_t address) -> bool {
    if (cond == mlir::vc4::Cond::never)
      return false;
    std::optional<unsigned> accumulator =
        getAccumulatorIndexForWriteAddress(address);
    if (!accumulator || *accumulator >= 6)
      return false;
    return classifyVPMSetupImmediate(value, *accumulator) !=
           VPMTransferAliasKind::Unknown;
  };
  return writesRecognizedSetup(ldi.getCondAdd(),
                               getIntegerAttrValue(op, "waddr_add")) ||
         writesRecognizedSetup(ldi.getCondMul(),
                               getIntegerAttrValue(op, "waddr_mul"));
}

static uint32_t canonicalizeLDIImmediateForQASM(mlir::vc4::QPULDIOp ldi,
                                                uint32_t value) {
  // Older scheduled test inputs encode the depth-16 horizontal VDW setup shape
  // with the legacy literal 0x80903000.  vc4asm's hardware reference spelling
  // for vdw_setup_0(1, 16, dma_h32(0, 0)) assembles to 0x80904000, and using
  // the stale literal corrupts the VDW store path for otherwise generic
  // scheduled streams.  Canonicalize only while emitting a recognized VPM/VDW
  // setup immediate, rather than rewriting arbitrary scalar constants.
  if (value == 0x80903000u &&
      ldiWritesRecognizedVPMSetupImmediate(ldi, value))
    return 0x80904000u;
  return value;
}

static void recordLDIAccumulatorSetupKind(QASMEmissionState &state,
                                          mlir::vc4::Cond cond,
                                          int64_t address, uint32_t value) {
  if (cond == mlir::vc4::Cond::never)
    return;
  std::optional<unsigned> accumulator = getAccumulatorIndexForWriteAddress(address);
  if (!accumulator || *accumulator >= 6)
    return;
  state.accumulatorSetupKind[*accumulator] =
      classifyVPMSetupImmediate(value, *accumulator);
}

static void updateQASMAfterLDI(QASMEmissionState &state,
                               mlir::vc4::QPULDIOp ldi, uint32_t value) {
  mlir::Operation *op = ldi.getOperation();
  recordLDIAccumulatorSetupKind(state, ldi.getCondAdd(),
                                getIntegerAttrValue(op, "waddr_add"), value);
  recordLDIAccumulatorSetupKind(state, ldi.getCondMul(),
                                getIntegerAttrValue(op, "waddr_mul"), value);
}

static LogicalResult emitQPULDIQASM(mlir::vc4::QPULDIOp ldi,
                                    QASMEmissionState &state,
                                    llvm::raw_ostream &os) {
  if (ldi.getMode() != mlir::vc4::LoadImmMode::splat32) {
    return ldi.emitOpError()
           << "only splat32 vc4.qpu.ldi qasm emission is supported in this "
              "slice";
  }

  auto valueAttr = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
      ldi.getOperation()->getAttr("value"));
  if (!valueAttr) {
    return ldi.emitOpError()
           << "requires an integer 'value' attribute for splat32 qasm "
              "emission";
  }

  std::optional<std::string> packSuffix =
      getPackSuffix(ldi.getOperation()->getAttr("pack"));
  if (!packSuffix) {
    return ldi.emitOpError()
           << "cannot emit unsupported vc4.qpu.ldi pack attribute '"
           << ldi.getOperation()->getAttr("pack") << "'";
  }

  std::string destinations;
  mlir::vc4::Cond emittedCond = mlir::vc4::Cond::always;
  if (failed(buildLoadLikeDestinationList(ldi.getOperation(), ldi.getCondAdd(),
                                          ldi.getCondMul(), ldi.getPm(),
                                          *packSuffix, destinations,
                                          emittedCond)))
    return failure();

  std::string opcode = "ldi";
  appendLoadLikeOpcodeSuffix(ldi.getOperation(), emittedCond, opcode);

  uint32_t immediateValue = static_cast<uint32_t>(valueAttr.getInt());
  uint32_t emittedImmediateValue =
      canonicalizeLDIImmediateForQASM(ldi, immediateValue);
  os << opcode << " " << destinations << ", "
     << formatU32Immediate(emittedImmediateValue) << "\n";
  updateQASMAfterLDI(state, ldi, emittedImmediateValue);
  return success();
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

  // vc4asm documents both the direct `sacq`/`srel` syntax and the Broadcom
  // compatible `mov dest, sacqN` form.  Use the latter uniformly so conditions,
  // flags, pack suffixes, write-swap, and dual ADD/MUL destinations share the
  // same deterministic printer path as ordinary load-immediate output.
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

static std::string formatBranchTargetExpression(bool relative, bool useReg,
                                                int64_t immediate,
                                                unsigned slotIndex,
                                                unsigned streamSize,
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
                                       unsigned slotIndex,
                                       unsigned streamSize,
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
     << " target=" << (relative ? "relative" : "absolute") << ":"
     << immediate;
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
  const bool emitInstructionLabels = hasBranchInstruction(kernel.scheduledStream);

  for (unsigned slotIndex = 0; slotIndex != kernel.scheduledStream.size();
       ++slotIndex) {
    mlir::Operation *op = kernel.scheduledStream[slotIndex];

    if (emitInstructionLabels)
      qasmOS << ":" << getQPUInstructionLabel(slotIndex) << "\n";

    if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
      if (failed(emitQPUBranchQASM(branch, slotIndex, static_cast<unsigned>(kernel.scheduledStream.size()), qasmOS)))
        return failure();
      continue;
    }

    if (auto bundle = llvm::dyn_cast<mlir::vc4::QPUBundleOp>(op)) {
      if (slotIndex + 1 < kernel.scheduledStream.size()) {
        if (auto nextBundle =
                llvm::dyn_cast<mlir::vc4::QPUBundleOp>(kernel.scheduledStream[slotIndex + 1])) {
          if (canPrepareVectorRotateFromSpacer(bundle, nextBundle)) {
            if (failed(emitVectorRotatePrepBundle(bundle, nextBundle, qasmState, qasmOS)))
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
      if (failed(emitQPULDIQASM(ldi, qasmState, qasmOS)))
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

  return writeBundleFile(kernel.func.getOperation(), bundleDir,
                         kernel.qasmPath,
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

static const LaunchABIArgumentModel *
findLaunchABILogicalCountArgument(const LaunchABIModel &launchABI) {
  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    if (arg.kind == LaunchABIArgumentKind::Scalar && arg.name == "n")
      return &arg;
  }
  return nullptr;
}

static const LaunchABIArgumentModel *
findLaunchABIScalarArgumentNamed(const LaunchABIModel &launchABI,
                                 llvm::StringRef name) {
  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    if (arg.kind == LaunchABIArgumentKind::Scalar && arg.name == name)
      return &arg;
  }
  return nullptr;
}

static std::string getBufferElementCTypeForCodegen(
    const LaunchABIArgumentModel &arg) {
  std::optional<std::string> elemType = getElementCType(arg.elementType);
  if (elemType)
    return *elemType;
  return "uint32_t";
}

static void appendRuntimeAPIDeclarations(llvm::raw_ostream &os) {
  os << "typedef uint32_t vc4_deviceptr_t;\n\n";
  os << "typedef struct vc4_dim3 {\n";
  os << "  uint32_t x;\n";
  os << "  uint32_t y;\n";
  os << "  uint32_t z;\n";
  os << "} vc4_dim3;\n\n";
  os << "struct vc4_program;\n\n";
  os << "int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes);\n";
  os << "void vc4_program_destroy(struct vc4_program *program);\n";
  os << "int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes);\n";
  os << "int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr);\n";
  os << "int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst, const void *src, uint32_t bytes);\n";
  os << "int vc4MemcpyDtoH(struct vc4_program *program, void *dst, vc4_deviceptr_t src, uint32_t bytes);\n";
  os << "int vc4MemcpyDtoD(struct vc4_program *program, vc4_deviceptr_t dst, vc4_deviceptr_t src, uint32_t bytes);\n";
  os << "int vc4MemsetD8(struct vc4_program *program, vc4_deviceptr_t dst, uint8_t value, uint32_t bytes);\n\n";
}

static LogicalResult writeLauncherHeader(llvm::ArrayRef<KernelRecord> kernels,
                                         llvm::StringRef bundleDir) {
  if (kernels.empty())
    return failure();
  mlir::vc4::FuncOp firstFunc = kernels.front().func;
  return writeBundleFile(firstFunc.getOperation(), bundleDir,
                         "kernel_launch.h", [&](llvm::raw_ostream &os) {
                           os << "#ifndef VC4_CODEGEN_KERNEL_LAUNCH_H\n";
                           os << "#define VC4_CODEGEN_KERNEL_LAUNCH_H\n\n";
                           os << "#include <stdint.h>\n";
                           os << "#include \"mailbox.h\"\n\n";
                           os << "#ifndef VC4_RUNTIME_MAX_QPUS\n";
                           os << "#define VC4_RUNTIME_MAX_QPUS 12u\n";
                           os << "#endif\n";
                           os << "#ifndef VC4_RUNTIME_LANE_WIDTH\n";
                           os << "#define VC4_RUNTIME_LANE_WIDTH 16u\n";
                           os << "#endif\n\n";
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

static std::string
getArgumentUniformExpression(const LaunchABIArgumentModel &arg) {
  if (arg.kind == LaunchABIArgumentKind::Buffer)
    return std::string("(uint32_t)") + arg.name;
  if (arg.scalarType == "f32")
    return std::string("vc4_codegen_pack_f32(") + arg.name + ")";
  return std::string("(uint32_t)") + arg.name;
}

static std::optional<std::string>
getBuiltinUniformExpression(const LaunchABIBuiltinModel &builtin) {
  if (builtin.kind == "qpu_num")
    return std::string("logicalRequest");
  if (builtin.kind == "num_qpus")
    return std::string("totalRequests");
  return std::nullopt;
}

static void appendLauncherUniformLayoutComment(
    llvm::raw_ostream &os, const LaunchABIModel &launchABI) {
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
  const std::string allocationsName = getRuntimeAllocationsFunctionName(firstABI);
  const std::string launchesName = getRuntimeLaunchesFunctionName(firstABI);
  const std::string capacityName = getRuntimeCapacityFunctionName(firstABI);
  const std::string codeUploadsName = getRuntimeCodeUploadsFunctionName(firstABI);
  const std::string launchFailuresName =
      getRuntimeLaunchFailuresFunctionName(firstABI);

  bool requiresF32Packing = false;
  bool requiresLaunchElements = false;
  for (const KernelRecord &kernel : kernels) {
    requiresF32Packing |= launchABIRequiresF32Packing(kernel.launchABI);
    const LaunchABIArgumentModel *rowCountArgForLaunchShape =
        findLaunchABIScalarArgumentNamed(kernel.launchABI, "m");
    // Kernels with both m and n, such as GEMV, use m/n as semantic dimensions;
    // neither scalar is necessarily the one-dimensional launch element count.
    // Those kernels must use CUDA-like grid/block launch geometry to choose
    // logical QPU requests.  Elementwise kernels with only an n scalar keep the
    // convenient n-as-logical-count behavior.
    requiresLaunchElements |=
        rowCountArgForLaunchShape ||
        !findLaunchABILogicalCountArgument(kernel.launchABI);
  }

  os << "#define VC4_CODEGEN_KERNEL_LAUNCH_IMPLEMENTATION 1\n";
  os << "#include \"kernel_launch.h\"\n";
  os << "#undef VC4_CODEGEN_KERNEL_LAUNCH_IMPLEMENTATION\n\n";
  os << "#include \"rpi.h\"\n";
  os << "#include \"mailbox.h\"\n";
  for (const KernelRecord &includeKernel : kernels)
    os << "#include \"" << includeKernel.launchABI.codeSymbol << ".h\"\n";
  os << "\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "#include <string.h>\n\n";

  // kernel_launch.h and mailbox.h provide libpi mailbox/QPU prototypes.  Keep
  // matching declarations here so hardware builds fail at link time if the
  // real libpi VC4 runtime is not linked.
  os << "extern uint32_t qpu_enable(uint32_t enable);\n";
  os << "extern uint32_t mem_alloc(uint32_t size, uint32_t align, uint32_t flags);\n";
  os << "extern uint32_t mem_lock(uint32_t handle);\n";
  os << "extern uint32_t mem_unlock(uint32_t handle);\n";
  os << "extern uint32_t mem_free(uint32_t handle);\n";
  os << "extern unsigned gpu_fft_base_exec_direct(uint32_t code, uint32_t unifs[], int num_qpus);\n";
  os << "\n";

  os << "#define GPU_MEM_FLG 0xCu\n";
  os << "#define GPU_BASE 0x40000000u\n";
  os << "#define V3D_BASE 0x20C00000u\n";
  os << "#define V3D_L2CACTL (V3D_BASE + 0x020u)\n";
  os << "#define V3D_SLCACTL (V3D_BASE + 0x024u)\n";
  os << "#define V3D_SRQPC (V3D_BASE + 0x0430u)\n";
  os << "#define V3D_SRQUA (V3D_BASE + 0x0434u)\n";
  os << "#define V3D_SRQCS (V3D_BASE + 0x043cu)\n";
  os << "#define VC4_CODEGEN_QPU_WAIT_MAX_POLLS 10000000u\n";
  os << "#define V3D_DBCFG (V3D_BASE + 0x0e00u)\n";
  os << "#define V3D_DBQITE (V3D_BASE + 0x0e2cu)\n";
  os << "#define V3D_DBQITC (V3D_BASE + 0x0e30u)\n";
  os << "#define VC4_CODEGEN_PROGRAM_MAGIC 0x56344350u /* VC4P */\n";
  os << "#define VC4_CODEGEN_PROGRAM_KERNELS " << kernels.size() << "u\n";
  os << "#define VC4_CODEGEN_PROGRAM_LAYOUT_ALIGNMENT "
     << programLayout.alignment << "u\n";
  os << "#define VC4_CODEGEN_PROGRAM_TOTAL_BYTES "
     << programLayout.programBytes << "u\n";
  os << "#define VC4_CODEGEN_PROGRAM_STATIC_BYTES "
     << programLayout.staticBytes << "u\n";
  os << "#define VC4_CODEGEN_PROGRAM_HEAP_OFFSET "
     << programLayout.heapOffset << "u\n";
  os << "#define VC4_CODEGEN_PROGRAM_HEAP_BYTES "
     << programLayout.heapBytes << "u\n";
  os << "#define VC4_HEAP_BLOCK_MAGIC 0x48454150u /* HEAP */\n";
  os << "#define VC4_HEAP_NO_NEXT 0xffffffffu\n";
  os << "#define VC4_HEAP_ALIGNMENT " << kVC4HeapAlignment << "u\n";
  os << "#define NUM_UNIFS " << firstABI.uniformWordsPerQPU << "u\n";
  for (const KernelRecord &macroKernel : kernels) {
    os << "#define KERNEL_" << macroKernel.kernelId << "_NUM_UNIFS "
       << macroKernel.info.uniformWordsPerQPU << "u\n";
  }
  os << "\n";
  os << "/* VC4_RUNTIME_LAYOUT program_bytes=" << programLayout.programBytes
     << " static_bytes=" << programLayout.staticBytes
     << " heap_offset=" << programLayout.heapOffset
     << " heap_bytes=" << programLayout.heapBytes
     << " kernels=" << kernels.size() << " */\n";
  os << "/* VC4_HEAP_STATS allocs/frees/failures/high_water are tracked in "
        "the persistent program image. */\n";
  os << "/* Generated launches accept vc4_deviceptr_t buffers and never copy "
        "host buffers implicitly. */\n\n";

  if (requiresF32Packing) {
    os << "static uint32_t vc4_codegen_pack_f32(float value) {\n";
    os << "  uint32_t bits;\n";
    os << "  memcpy(&bits, &value, sizeof(bits));\n";
    os << "  return bits;\n";
    os << "}\n\n";
  }

  os << "struct vc4_codegen_kernel_desc {\n";
  os << "  uint32_t code_word_offset;\n";
  os << "  uint32_t code_word_count;\n";
  os << "  uint32_t unif_word_offset;\n";
  os << "  uint32_t unif_words_per_request;\n";
  os << "  uint32_t max_requests_per_wave;\n";
  os << "  uint32_t unif_ptr_word_offset;\n";
  os << "  uint32_t code_gpu_addr;\n";
  os << "  uint32_t flags;\n";
  os << "};\n\n";

  os << "struct " << stateName << " {\n";
  os << "  uint32_t magic;\n";
  os << "  uint32_t total_size_bytes;\n";
  os << "  uint32_t num_kernels;\n";
  os << "  uint32_t active_qpus;\n";
  os << "  uint32_t warp_size;\n";
  os << "  uint32_t descriptor_table_offset;\n";
  os << "  uint32_t header_reserved[10];\n";
  os << "  struct vc4_codegen_kernel_desc kernel_descs[VC4_CODEGEN_PROGRAM_KERNELS];\n";
  for (const KernelRecord &stateKernel : kernels) {
    os << "  uint32_t " << getKernelCodeFieldName(stateKernel.kernelId)
       << "[sizeof(" << stateKernel.launchABI.codeSymbol
       << ") / sizeof(uint32_t)];\n";
    os << "  uint32_t " << getKernelUniformFieldName(stateKernel.kernelId)
       << "[VC4_RUNTIME_MAX_QPUS][KERNEL_" << stateKernel.kernelId
       << "_NUM_UNIFS];\n";
    os << "  uint32_t " << getKernelUnifPtrFieldName(stateKernel.kernelId)
       << "[VC4_RUNTIME_MAX_QPUS];\n";
  }
  os << "  uint32_t handle;\n";
  os << "  uint32_t launch_count;\n";
  os << "  uint32_t launch_failures;\n";
  os << "  uint32_t code_uploads;\n";
  os << "  uint32_t heap_allocs;\n";
  os << "  uint32_t heap_frees;\n";
  os << "  uint32_t heap_failures;\n";
  os << "  uint32_t heap_live_bytes;\n";
  os << "  uint32_t heap_high_water;\n";
  os << "  uint32_t runtime_reserved[3];\n";
  os << "  uint8_t heap[VC4_CODEGEN_PROGRAM_HEAP_BYTES] __attribute__((aligned(16)));\n";
  os << "};\n\n";

  os << "struct vc4_codegen_heap_block {\n";
  os << "  uint32_t size;\n";
  os << "  uint32_t next;\n";
  os << "  uint32_t free;\n";
  os << "  uint32_t magic;\n";
  os << "};\n\n";

  os << "struct vc4_program {\n";
  os << "  volatile struct " << stateName << " *state;\n";
  os << "  uint32_t handle;\n";
  os << "  uint32_t active_qpus;\n";
  os << "  uint32_t heap_bytes;\n";
  os << "  uint32_t heap_head;\n";
  os << "};\n\n";

  os << "static struct vc4_program g_program_storage;\n";
  os << "static uint32_t g_program_live;\n";
  os << "static uint32_t g_program_allocations;\n";
  os << "\n";


  os << "static void vc4_codegen_prepare_v3d_queue(void) {\n";
  os << "  PUT32(V3D_DBCFG, 0u);\n";
  os << "  PUT32(V3D_DBQITE, 0u);\n";
  os << "  PUT32(V3D_DBQITC, 0xffffffffu);\n";
  os << "  PUT32(V3D_L2CACTL, 1u << 2);\n";
  os << "  PUT32(V3D_SLCACTL, 0xffffffffu);\n";
  os << "  PUT32(V3D_SRQCS, (1u << 7) | (1u << 8) | (1u << 16));\n";
  os << "}\n\n";

  os << "static void vc4_codegen_launch_failure(struct vc4_program *program) {\n";
  os << "  if (program && program->state)\n";
  os << "    program->state->launch_failures++;\n";
  os << "}\n\n";

  os << "static int vc4_codegen_wait_for_qpus(struct vc4_program *program, uint32_t activeQpus) {\n";
  os << "  uint32_t max_polls = VC4_CODEGEN_QPU_WAIT_MAX_POLLS;\n";
  os << "  while (max_polls-- != 0u) {\n";
  os << "    if (((GET32(V3D_SRQCS) >> 16) & 0xffu) == activeQpus)\n";
  os << "      return 0;\n";
  os << "  }\n";
  os << "  vc4_codegen_launch_failure(program);\n";
  os << "  PUT32(V3D_SRQCS, (1u << 7) | (1u << 8) | (1u << 16));\n";
  os << "  return -1;\n";
  os << "}\n\n";

  os << "static uint32_t vc4_codegen_launch_code_gpu_addr(uint32_t code_cpu_addr) {\n";
  os << "  return GPU_BASE + code_cpu_addr;\n";
  os << "}\n\n";

  os << "static uint32_t vc4_codegen_align_u32(uint32_t value, uint32_t alignment) {\n";
  os << "  if (alignment <= 1u)\n";
  os << "    return value;\n";
  os << "  return (value + alignment - 1u) & ~(alignment - 1u);\n";
  os << "}\n\n";

  os << "static uint32_t vc4_codegen_ceil_div_u32(uint32_t value, uint32_t divisor) {\n";
  os << "  if (value == 0u)\n";
  os << "    return 0u;\n";
  os << "  if (divisor == 0u)\n";
  os << "    return 0xffffffffu;\n";
  os << "  return 1u + (value - 1u) / divisor;\n";
  os << "}\n\n";

  if (requiresLaunchElements) {
    os << "static uint32_t vc4_codegen_saturating_mul_u32(uint32_t lhs, uint32_t rhs) {\n";
    os << "  if (lhs != 0u && rhs > 0xffffffffu / lhs)\n";
    os << "    return 0xffffffffu;\n";
    os << "  return lhs * rhs;\n";
    os << "}\n\n";

    os << "static uint32_t vc4_codegen_launch_elements(vc4_dim3 grid, vc4_dim3 block) {\n";
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

  os << "static int vc4_program_is_live(const struct vc4_program *program) {\n";
  os << "  return program && g_program_live && program == &g_program_storage && program->state;\n";
  os << "}\n\n";

  os << "static unsigned char *vc4_program_heap_base(struct vc4_program *program) {\n";
  os << "  return (unsigned char *)&program->state->heap[0];\n";
  os << "}\n\n";

  os << "static uint32_t vc4_program_heap_gpu_base(struct vc4_program *program) {\n";
  os << "  return GPU_BASE + (uint32_t)(uintptr_t)vc4_program_heap_base(program);\n";
  os << "}\n\n";

  os << "static void vc4_heap_failure(struct vc4_program *program) {\n";
  os << "  if (vc4_program_is_live(program))\n";
  os << "    program->state->heap_failures++;\n";
  os << "}\n\n";

  os << "static struct vc4_codegen_heap_block *vc4_heap_block_at(struct vc4_program *program, uint32_t offset) {\n";
  os << "  if (!vc4_program_is_live(program))\n";
  os << "    return 0;\n";
  os << "  if (offset > program->heap_bytes ||\n";
  os << "      program->heap_bytes - offset < sizeof(struct vc4_codegen_heap_block))\n";
  os << "    return 0;\n";
  os << "  struct vc4_codegen_heap_block *block =\n";
  os << "      (struct vc4_codegen_heap_block *)(vc4_program_heap_base(program) + offset);\n";
  os << "  if (block->magic != VC4_HEAP_BLOCK_MAGIC)\n";
  os << "    return 0;\n";
  os << "  return block;\n";
  os << "}\n\n";

  os << "static void vc4_heap_init(struct vc4_program *program) {\n";
  os << "  program->heap_head = 0u;\n";
  os << "  struct vc4_codegen_heap_block *head =\n";
  os << "      (struct vc4_codegen_heap_block *)vc4_program_heap_base(program);\n";
  os << "  head->size = program->heap_bytes - (uint32_t)sizeof(struct vc4_codegen_heap_block);\n";
  os << "  head->next = VC4_HEAP_NO_NEXT;\n";
  os << "  head->free = 1u;\n";
  os << "  head->magic = VC4_HEAP_BLOCK_MAGIC;\n";
  os << "  program->state->heap_allocs = 0u;\n";
  os << "  program->state->heap_frees = 0u;\n";
  os << "  program->state->heap_failures = 0u;\n";
  os << "  program->state->heap_live_bytes = 0u;\n";
  os << "  program->state->heap_high_water = 0u;\n";
  os << "}\n\n";

  os << "static void vc4_heap_coalesce_next(struct vc4_program *program, struct vc4_codegen_heap_block *block) {\n";
  os << "  while (block->next != VC4_HEAP_NO_NEXT) {\n";
  os << "    struct vc4_codegen_heap_block *next = vc4_heap_block_at(program, block->next);\n";
  os << "    if (!next || !next->free)\n";
  os << "      return;\n";
  os << "    block->size += (uint32_t)sizeof(struct vc4_codegen_heap_block) + next->size;\n";
  os << "    block->next = next->next;\n";
  os << "  }\n";
  os << "}\n\n";

  os << "static int vc4_device_range_offset(struct vc4_program *program, vc4_deviceptr_t ptr, uint32_t bytes, uint32_t *offset_out) {\n";
  os << "  if (!vc4_program_is_live(program))\n";
  os << "    return 0;\n";
  os << "  uint32_t gpu_base = vc4_program_heap_gpu_base(program);\n";
  os << "  if (ptr < gpu_base)\n";
  os << "    return 0;\n";
  os << "  uint32_t offset = ptr - gpu_base;\n";
  os << "  if (offset > program->heap_bytes)\n";
  os << "    return 0;\n";
  os << "  if (bytes > program->heap_bytes - offset)\n";
  os << "    return 0;\n";
  os << "  if (offset_out)\n";
  os << "    *offset_out = offset;\n";
  os << "  return 1;\n";
  os << "}\n\n";

  os << "static int vc4_device_range_is_allocated(struct vc4_program *program, vc4_deviceptr_t ptr, uint32_t bytes) {\n";
  os << "  uint32_t range_offset = 0u;\n";
  os << "  if (!vc4_device_range_offset(program, ptr, bytes, &range_offset))\n";
  os << "    return 0;\n";
  os << "  uint32_t current = program->heap_head;\n";
  os << "  while (current != VC4_HEAP_NO_NEXT) {\n";
  os << "    struct vc4_codegen_heap_block *block = vc4_heap_block_at(program, current);\n";
  os << "    if (!block)\n";
  os << "      return 0;\n";
  os << "    uint32_t payload_begin = current + (uint32_t)sizeof(struct vc4_codegen_heap_block);\n";
  os << "    if (!block->free && range_offset >= payload_begin &&\n";
  os << "        range_offset - payload_begin <= block->size &&\n";
  os << "        bytes <= block->size - (range_offset - payload_begin))\n";
  os << "      return 1;\n";
  os << "    current = block->next;\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "static void *vc4_deviceptr_to_host(struct vc4_program *program, vc4_deviceptr_t ptr, uint32_t bytes) {\n";
  os << "  if (!vc4_device_range_is_allocated(program, ptr, bytes))\n";
  os << "    return 0;\n";
  os << "  uint32_t offset = ptr - vc4_program_heap_gpu_base(program);\n";
  os << "  return vc4_program_heap_base(program) + offset;\n";
  os << "}\n\n";

  os << "int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes) {\n";
  os << "  if (!out)\n";
  os << "    return -1;\n";
  os << "  *out = 0;\n";
  os << "  if (g_program_live)\n";
  os << "    return -1;\n";
  os << "  if (requested_bytes > VC4_CODEGEN_PROGRAM_HEAP_BYTES - sizeof(struct vc4_codegen_heap_block))\n";
  os << "    return -1;\n\n";
  os << "  uint32_t activeQpus = VC4_RUNTIME_MAX_QPUS;\n";
  os << "  if (activeQpus == 0u || activeQpus > VC4_RUNTIME_MAX_QPUS)\n";
  os << "    return -1;\n\n";
  os << "  size_t allocSize = sizeof(struct " << stateName << ");\n";
  os << "  if (allocSize > 0xffffffffu)\n";
  os << "    return -1;\n\n";
  os << "#ifdef __RPI__\n";
  os << "  if (qpu_enable(1))\n";
  os << "    return -1;\n";
  os << "#endif\n\n";
  os << "  uint32_t handle = mem_alloc((uint32_t)allocSize, 4096u, GPU_MEM_FLG);\n";
  os << "  if (!handle) {\n";
  os << "#ifdef __RPI__\n";
  os << "    qpu_enable(0);\n";
  os << "#endif\n";
  os << "    return -1;\n";
  os << "  }\n\n";
  os << "  uint32_t vc = mem_lock(handle);\n";
  os << "  if (!vc) {\n";
  os << "    mem_free(handle);\n";
  os << "#ifdef __RPI__\n";
  os << "    qpu_enable(0);\n";
  os << "#endif\n";
  os << "    return -1;\n";
  os << "  }\n\n";
  os << "  volatile struct " << stateName << " *state =\n";
  os << "      (volatile struct " << stateName << " *)(vc - GPU_BASE);\n";
  os << "  memset((void *)state, 0, allocSize);\n";
  os << "  state->magic = VC4_CODEGEN_PROGRAM_MAGIC;\n";
  os << "  state->total_size_bytes = (uint32_t)allocSize;\n";
  os << "  state->num_kernels = VC4_CODEGEN_PROGRAM_KERNELS;\n";
  os << "  state->active_qpus = activeQpus;\n";
  os << "  state->warp_size = VC4_RUNTIME_LANE_WIDTH;\n";
  os << "  state->descriptor_table_offset = (uint32_t)offsetof(struct "
     << stateName << ", kernel_descs);\n";
  os << "  state->handle = handle;\n";
  os << "  state->launch_count = 0u;\n";
  os << "  state->launch_failures = 0u;\n";
  os << "  state->code_uploads = 0u;\n";
  for (const KernelRecord &copyKernel : kernels) {
    const unsigned id = copyKernel.kernelId;
    os << "  memcpy((void *)state->" << getKernelCodeFieldName(id) << ", "
       << copyKernel.launchABI.codeSymbol << ", sizeof state->"
       << getKernelCodeFieldName(id) << ");\n";
    os << "  state->code_uploads++;\n";
    os << "  state->kernel_descs[" << id << "].code_word_offset = "
       << "(uint32_t)(offsetof(struct " << stateName << ", "
       << getKernelCodeFieldName(id) << ") / sizeof(uint32_t));\n";
    os << "  state->kernel_descs[" << id << "].code_word_count = "
       << "(uint32_t)(sizeof state->" << getKernelCodeFieldName(id)
       << " / sizeof(uint32_t));\n";
    os << "  state->kernel_descs[" << id << "].unif_word_offset = "
       << "(uint32_t)(offsetof(struct " << stateName << ", "
       << getKernelUniformFieldName(id) << ") / sizeof(uint32_t));\n";
    os << "  state->kernel_descs[" << id
       << "].unif_words_per_request = KERNEL_" << id
       << "_NUM_UNIFS;\n";
    os << "  state->kernel_descs[" << id
       << "].max_requests_per_wave = VC4_RUNTIME_MAX_QPUS;\n";
    os << "  state->kernel_descs[" << id << "].unif_ptr_word_offset = "
       << "(uint32_t)(offsetof(struct " << stateName << ", "
       << getKernelUnifPtrFieldName(id) << ") / sizeof(uint32_t));\n";
    os << "  state->kernel_descs[" << id
       << "].code_gpu_addr = (uint32_t)(uintptr_t)&state->"
       << getKernelCodeFieldName(id) << "[0];\n";
    os << "  state->kernel_descs[" << id << "].flags = 0u;\n";
    os << "  for (uint32_t qpu = 0; qpu < VC4_RUNTIME_MAX_QPUS; ++qpu)\n";
    os << "    state->" << getKernelUnifPtrFieldName(id)
       << "[qpu] = GPU_BASE + (uint32_t)(uintptr_t)&state->"
       << getKernelUniformFieldName(id) << "[qpu][0];\n";
  }
  os << "\n";
  os << "  memset((void *)&g_program_storage, 0, sizeof(g_program_storage));\n";
  os << "  g_program_storage.state = state;\n";
  os << "  g_program_storage.handle = handle;\n";
  os << "  g_program_storage.active_qpus = activeQpus;\n";
  os << "  g_program_storage.heap_bytes = VC4_CODEGEN_PROGRAM_HEAP_BYTES;\n";
  os << "  g_program_live = 1u;\n";
  os << "  vc4_heap_init(&g_program_storage);\n";
  os << "  g_program_allocations++;\n";
  os << "  printk(\"VC4_RUNTIME_LAYOUT program_allocations=%u code_uploads=%u heap_bytes=%u kernels=%u\\n\",\n";
  os << "         g_program_allocations, state->code_uploads,\n";
  os << "         g_program_storage.heap_bytes, VC4_CODEGEN_PROGRAM_KERNELS);\n";
  os << "  *out = &g_program_storage;\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "void vc4_program_destroy(struct vc4_program *program) {\n";
  os << "  if (!vc4_program_is_live(program))\n";
  os << "    return;\n";
  os << "  printk(\"VC4_HEAP_STATS allocs=%u frees=%u failures=%u high_water=%u runtime_launches=%u launch_failures=%u\\n\",\n";
  os << "         program->state->heap_allocs, program->state->heap_frees,\n";
  os << "         program->state->heap_failures, program->state->heap_high_water,\n";
  os << "         program->state->launch_count, program->state->launch_failures);\n";
  os << "  mem_unlock(program->handle);\n";
  os << "  mem_free(program->handle);\n";
  os << "#ifdef __RPI__\n";
  os << "  qpu_enable(0);\n";
  os << "#endif\n";
  os << "  memset((void *)&g_program_storage, 0, sizeof(g_program_storage));\n";
  os << "  g_program_live = 0u;\n";
  os << "}\n\n";

  os << "int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes) {\n";
  os << "  if (!vc4_program_is_live(program) || !out) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  *out = 0u;\n";
  os << "  if (bytes == 0u)\n";
  os << "    return 0;\n";
  os << "  uint32_t aligned = vc4_codegen_align_u32(bytes, VC4_HEAP_ALIGNMENT);\n";
  os << "  if (aligned < bytes) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  uint32_t current = program->heap_head;\n";
  os << "  while (current != VC4_HEAP_NO_NEXT) {\n";
  os << "    struct vc4_codegen_heap_block *block = vc4_heap_block_at(program, current);\n";
  os << "    if (!block)\n";
  os << "      break;\n";
  os << "    if (block->free && block->size >= aligned) {\n";
  os << "      uint32_t remaining = block->size - aligned;\n";
  os << "      if (remaining > sizeof(struct vc4_codegen_heap_block) + VC4_HEAP_ALIGNMENT) {\n";
  os << "        uint32_t new_offset = current + (uint32_t)sizeof(struct vc4_codegen_heap_block) + aligned;\n";
  os << "        struct vc4_codegen_heap_block *split = vc4_heap_block_at(program, new_offset);\n";
  os << "        if (!split)\n";
  os << "          split = (struct vc4_codegen_heap_block *)(vc4_program_heap_base(program) + new_offset);\n";
  os << "        split->size = remaining - (uint32_t)sizeof(struct vc4_codegen_heap_block);\n";
  os << "        split->next = block->next;\n";
  os << "        split->free = 1u;\n";
  os << "        split->magic = VC4_HEAP_BLOCK_MAGIC;\n";
  os << "        block->size = aligned;\n";
  os << "        block->next = new_offset;\n";
  os << "      }\n";
  os << "      block->free = 0u;\n";
  os << "      program->state->heap_allocs++;\n";
  os << "      program->state->heap_live_bytes += block->size;\n";
  os << "      if (program->state->heap_live_bytes > program->state->heap_high_water)\n";
  os << "        program->state->heap_high_water = program->state->heap_live_bytes;\n";
  os << "      *out = vc4_program_heap_gpu_base(program) + current +\n";
  os << "             (uint32_t)sizeof(struct vc4_codegen_heap_block);\n";
  os << "      return 0;\n";
  os << "    }\n";
  os << "    current = block->next;\n";
  os << "  }\n";
  os << "  vc4_heap_failure(program);\n";
  os << "  return -1;\n";
  os << "}\n\n";

  os << "int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr) {\n";
  os << "  if (!vc4_program_is_live(program)) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  if (ptr == 0u)\n";
  os << "    return 0;\n";
  os << "  uint32_t range_offset = 0u;\n";
  os << "  if (!vc4_device_range_offset(program, ptr, 0u, &range_offset) ||\n";
  os << "      range_offset < sizeof(struct vc4_codegen_heap_block)) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  uint32_t block_offset = range_offset - (uint32_t)sizeof(struct vc4_codegen_heap_block);\n";
  os << "  uint32_t current = program->heap_head;\n";
  os << "  uint32_t prev = VC4_HEAP_NO_NEXT;\n";
  os << "  while (current != VC4_HEAP_NO_NEXT) {\n";
  os << "    struct vc4_codegen_heap_block *block = vc4_heap_block_at(program, current);\n";
  os << "    if (!block)\n";
  os << "      break;\n";
  os << "    if (current == block_offset) {\n";
  os << "      if (block->free) {\n";
  os << "        vc4_heap_failure(program);\n";
  os << "        return -1;\n";
  os << "      }\n";
  os << "      block->free = 1u;\n";
  os << "      program->state->heap_frees++;\n";
  os << "      if (program->state->heap_live_bytes >= block->size)\n";
  os << "        program->state->heap_live_bytes -= block->size;\n";
  os << "      else\n";
  os << "        program->state->heap_live_bytes = 0u;\n";
  os << "      vc4_heap_coalesce_next(program, block);\n";
  os << "      if (prev != VC4_HEAP_NO_NEXT) {\n";
  os << "        struct vc4_codegen_heap_block *prev_block = vc4_heap_block_at(program, prev);\n";
  os << "        if (prev_block && prev_block->free)\n";
  os << "          vc4_heap_coalesce_next(program, prev_block);\n";
  os << "      }\n";
  os << "      return 0;\n";
  os << "    }\n";
  os << "    prev = current;\n";
  os << "    current = block->next;\n";
  os << "  }\n";
  os << "  vc4_heap_failure(program);\n";
  os << "  return -1;\n";
  os << "}\n\n";

  os << "int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst, const void *src, uint32_t bytes) {\n";
  os << "  if (bytes == 0u)\n";
  os << "    return 0;\n";
  os << "  if (!src) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  void *dst_host = vc4_deviceptr_to_host(program, dst, bytes);\n";
  os << "  if (!dst_host) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  memcpy(dst_host, src, bytes);\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int vc4MemcpyDtoH(struct vc4_program *program, void *dst, vc4_deviceptr_t src, uint32_t bytes) {\n";
  os << "  if (bytes == 0u)\n";
  os << "    return 0;\n";
  os << "  if (!dst) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  void *src_host = vc4_deviceptr_to_host(program, src, bytes);\n";
  os << "  if (!src_host) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  memcpy(dst, src_host, bytes);\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int vc4MemcpyDtoD(struct vc4_program *program, vc4_deviceptr_t dst, vc4_deviceptr_t src, uint32_t bytes) {\n";
  os << "  if (bytes == 0u)\n";
  os << "    return 0;\n";
  os << "  void *dst_host = vc4_deviceptr_to_host(program, dst, bytes);\n";
  os << "  void *src_host = vc4_deviceptr_to_host(program, src, bytes);\n";
  os << "  if (!dst_host || !src_host) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  memmove(dst_host, src_host, bytes);\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int vc4MemsetD8(struct vc4_program *program, vc4_deviceptr_t dst, uint8_t value, uint32_t bytes) {\n";
  os << "  if (bytes == 0u)\n";
  os << "    return 0;\n";
  os << "  void *dst_host = vc4_deviceptr_to_host(program, dst, bytes);\n";
  os << "  if (!dst_host) {\n";
  os << "    vc4_heap_failure(program);\n";
  os << "    return -1;\n";
  os << "  }\n";
  os << "  memset(dst_host, value, bytes);\n";
  os << "  return 0;\n";
  os << "}\n\n";

  for (const KernelRecord &launchKernel : kernels) {
    const LaunchABIModel &kernelABI = launchKernel.launchABI;
    llvm::SmallVector<const LaunchABIArgumentModel *, 4> bufferArgs =
        collectLaunchABIBufferArguments(kernelABI);
    const LaunchABIArgumentModel *rowCountArg =
        findLaunchABIScalarArgumentNamed(kernelABI, "m");
    const LaunchABIArgumentModel *logicalCountArg =
        rowCountArg ? nullptr : findLaunchABILogicalCountArgument(kernelABI);
    const LaunchABIArgumentModel *validationCountArg = logicalCountArg;

    appendLauncherPrototype(os, kernelABI);
    os << " {\n";
    os << "  if (!vc4_program_is_live(program))\n";
    os << "    return -1;\n";
    os << "  uint32_t activeQpus = program->active_qpus;\n";
    os << "  if (activeQpus == 0u || activeQpus > VC4_RUNTIME_MAX_QPUS) {\n";
    os << "    vc4_codegen_launch_failure(program);\n";
    os << "    return -1;\n";
    os << "  }\n";
    if (validationCountArg) {
      os << "  uint32_t logicalN = (uint32_t)" << validationCountArg->name
         << ";\n";
    } else {
      os << "  uint32_t logicalN = vc4_codegen_launch_elements(grid, block);\n";
    }
    if (kernelABI.tailPolicy == "exact_multiple") {
      os << "  uint32_t totalRequests = logicalN == 0u ? 0u : activeQpus;\n";
    } else {
      os << "  uint32_t totalRequests = vc4_codegen_ceil_div_u32(logicalN, VC4_RUNTIME_LANE_WIDTH);\n";
    }
    os << "  uint32_t totalWaves = vc4_codegen_ceil_div_u32(totalRequests, activeQpus);\n";
    os << "  (void)grid;\n";
    os << "  (void)block;\n";
    os << "  (void)logicalN;\n";
    for (const LaunchABIArgumentModel *arg : bufferArgs) {
      std::string elemType = getBufferElementCTypeForCodegen(*arg);
      os << "  if (totalRequests != 0u) {\n";
      os << "    size_t arg_bytes = sizeof(" << elemType << ");\n";
      os << "    if (arg_bytes > 0xffffffffu || !vc4_device_range_is_allocated(program, "
         << arg->name << ", (uint32_t)arg_bytes)) {\n";
      os << "      vc4_codegen_launch_failure(program);\n";
      os << "      return -1;\n";
      os << "    }\n";
      os << "  }\n";
    }

    os << "  printk(\"VC4_KERNEL_LAUNCH name=" << kernelABI.publicName
       << " kernel_id=" << launchKernel.kernelId
       << " schedule_mode=independent_vector requests=%u waves=%u runtime_launches=%u launch_failures=%u\\n\",\n";
    os << "         totalRequests, totalWaves, program->state->launch_count + 1u,\n";
    os << "         program->state->launch_failures);\n";
    os << "  uint32_t max_wait_polls = VC4_CODEGEN_QPU_WAIT_MAX_POLLS;\n";
    os << "  (void)max_wait_polls;\n";
    os << "  if (totalRequests == 0u) {\n";
    os << "    program->state->launch_count++;\n";
    os << "    return 0;\n";
    os << "  }\n\n";

    appendLauncherUniformLayoutComment(os, kernelABI);
    os << "  for (uint32_t waveBase = 0u; waveBase < totalRequests; waveBase += activeQpus) {\n";
    os << "    uint32_t waveRequests = totalRequests - waveBase;\n";
    os << "    if (waveRequests > activeQpus)\n";
    os << "      waveRequests = activeQpus;\n";
    os << "    for (uint32_t qpu = 0; qpu < waveRequests; ++qpu) {\n";
    os << "      uint32_t logicalRequest = waveBase + qpu;\n";
    for (int64_t index = 0; index != kernelABI.uniformWordsPerQPU; ++index) {
      if (const LaunchABIArgumentModel *arg =
              findLaunchABIArgumentForUniformIndex(kernelABI, index)) {
        if (arg->kind == LaunchABIArgumentKind::Buffer) {
          os << "      program->state->" << getKernelUniformFieldName(launchKernel.kernelId)
             << "[qpu][" << index << "] = (uint32_t)" << arg->name
             << "; /* arg " << arg->name << " */\n";
        } else {
          os << "      program->state->" << getKernelUniformFieldName(launchKernel.kernelId)
             << "[qpu][" << index << "] = "
             << getArgumentUniformExpression(*arg) << "; /* arg "
             << arg->name << " */\n";
        }
        continue;
      }

      if (const LaunchABIBuiltinModel *builtin =
              findLaunchABIBuiltinForUniformIndex(kernelABI, index)) {
        std::optional<std::string> expression = getBuiltinUniformExpression(*builtin);
        if (!expression) {
          return emitLaunchABIModelError(
              launchKernel.func, llvm::Twine("builtin '") + builtin->name +
                            "' has unsupported kind for launcher uniform packing");
        }
        os << "      program->state->" << getKernelUniformFieldName(launchKernel.kernelId)
           << "[qpu][" << index << "] = " << *expression
           << "; /* builtin " << builtin->name << " */\n";
        continue;
      }

      return emitLaunchABIModelError(
          launchKernel.func,
          llvm::Twine("missing launcher uniform assignment for index ") +
              std::to_string(index));
    }
    os << "      program->state->" << getKernelUnifPtrFieldName(launchKernel.kernelId)
       << "[qpu] = GPU_BASE + (uint32_t)(uintptr_t)&program->state->"
       << getKernelUniformFieldName(launchKernel.kernelId) << "[qpu][0];\n";
    os << "    }\n";
    os << "#ifdef VC4_CODEGEN_USE_RAW_SRQ_QUEUE\n";
    os << "    vc4_codegen_prepare_v3d_queue();\n";
    os << "    for (uint32_t qpu = 0; qpu < waveRequests; ++qpu) {\n";
    os << "      PUT32(V3D_SRQUA, program->state->"
       << getKernelUnifPtrFieldName(launchKernel.kernelId) << "[qpu]);\n";
    os << "      PUT32(V3D_SRQPC, vc4_codegen_launch_code_gpu_addr(program->state->kernel_descs["
       << launchKernel.kernelId << "].code_gpu_addr));\n";
    os << "    }\n";
    os << "    if (vc4_codegen_wait_for_qpus(program, waveRequests) < 0)\n";
    os << "      return -1;\n";
    os << "#else\n";
    os << "    gpu_fft_base_exec_direct(program->state->kernel_descs["
       << launchKernel.kernelId << "].code_gpu_addr, (uint32_t *)program->state->"
       << getKernelUnifPtrFieldName(launchKernel.kernelId)
       << ", waveRequests);\n";
    os << "#endif\n";
    os << "  }\n";
    os << "  program->state->launch_count++;\n";
    os << "  return 0;\n";
    os << "}\n\n";
  }

  os << "uint32_t " << allocationsName << "(void) {\n";
  os << "  return g_program_allocations;\n";
  os << "}\n\n";

  os << "uint32_t " << launchesName << "(void) {\n";
  os << "  if (!g_program_live || !g_program_storage.state)\n";
  os << "    return 0u;\n";
  os << "  return g_program_storage.state->launch_count;\n";
  os << "}\n\n";

  os << "uint32_t " << capacityName << "(void) {\n";
  os << "  if (!g_program_live)\n";
  os << "    return 0u;\n";
  os << "  return g_program_storage.heap_bytes;\n";
  os << "}\n\n";

  os << "uint32_t " << codeUploadsName << "(void) {\n";
  os << "  if (!g_program_live || !g_program_storage.state)\n";
  os << "    return 0u;\n";
  os << "  return g_program_storage.state->code_uploads;\n";
  os << "}\n\n";

  os << "uint32_t " << launchFailuresName << "(void) {\n";
  os << "  if (!g_program_live || !g_program_storage.state)\n";
  os << "    return 0u;\n";
  os << "  return g_program_storage.state->launch_failures;\n";
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
  appendJSONEscapedString(os, arg.kind == LaunchABIArgumentKind::Scalar
                                  ? "scalar"
                                  : "buffer");
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
  os << "      \"scheduled_sink_ops\": "
     << kernel.info.scheduledOpCount << ",\n";
  os << "      \"uniform_words_per_request\": "
     << kernel.info.uniformWordsPerQPU << ",\n";
  os << "      \"uniform_words_per_qpu\": "
     << kernel.info.uniformWordsPerQPU << ",\n";
  os << "      \"max_requests_per_wave\": 12,\n";
  os << "      \"tail_policy\": ";
  appendJSONEscapedString(os, kernel.launchABI.tailPolicy);
  os << ",\n";
  os << "      \"schedule_mode\": \"independent_vector\",\n";
  os << "      \"args\": [\n";
  appendManifestKernelArguments(os, kernel.launchABI);
  os << "      ],\n";
  os << "      \"builtins\": [\n";
  appendManifestKernelBuiltins(os, kernel.launchABI);
  os << "      ],\n";
  os << "      \"resources\": {\n";
  os << "        \"uses_barrier\": false,\n";
  os << "        \"uses_shared_vpm\": false,\n";
  os << "        \"vpm_bytes_per_block\": 0,\n";
  os << "        \"semaphores_per_block\": 0,\n";
  os << "        \"warps_per_block_max\": 1\n";
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
  os << "      \"uniform_words_per_request\": "
     << kernel.uniformWordsPerRequest << ",\n";
  os << "      \"max_requests_per_wave\": "
     << kernel.maxRequestsPerWave << "\n";
  os << "    }";
  if (trailingComma)
    os << ",";
  os << "\n";
}

static LogicalResult writeProgramLayout(mlir::vc4::ModuleOp vc4Module,
                                        llvm::ArrayRef<KernelRecord> kernels,
                                        llvm::StringRef bundleDir) {
  ProgramLayoutModel layout = buildProgramLayoutModel(kernels);
  return writeBundleFile(vc4Module.getOperation(), bundleDir, "layout.json",
                         [&](llvm::raw_ostream &os) {
                           os << "{\n";
                           os << "  \"schema_version\": 1,\n";
                           os << "  \"kind\": \"vc4-program-layout\",\n";
                           os << "  \"program_name\": ";
                           appendJSONEscapedString(os, vc4Module.getSymName());
                           os << ",\n";
                           os << "  \"alignment\": " << layout.alignment << ",\n";
                           os << "  \"kernel_count\": " << kernels.size() << ",\n";
                           os << "  \"max_active_qpus\": "
                              << kVC4MaxRequestsPerWave << ",\n";
                           os << "  \"program_bytes\": "
                              << layout.programBytes << ",\n";
                           os << "  \"static_bytes\": "
                              << layout.staticBytes << ",\n";
                           os << "  \"heap_offset\": "
                              << layout.heapOffset << ",\n";
                           os << "  \"heap_offset_bytes\": "
                              << layout.heapOffset << ",\n";
                           os << "  \"heap_bytes\": "
                              << layout.heapBytes << ",\n";
                           os << "  \"heap_size_bytes\": "
                              << layout.heapBytes << ",\n";
                           os << "  \"heap\": ";
                           appendLayoutByteRange(os, layout.heapOffset,
                                                 layout.heapBytes);
                           os << ",\n";
                           os << "  \"regions\": [\n";
                           for (size_t i = 0; i != layout.regions.size(); ++i) {
                             appendLayoutRegionJSON(
                                 os, layout.regions[i],
                                 i + 1 != layout.regions.size());
                           }
                           os << "  ],\n";
                           os << "  \"kernels\": [\n";
                           for (size_t i = 0; i != layout.kernels.size(); ++i) {
                             appendKernelLayoutJSON(
                                 os, layout.kernels[i],
                                 i + 1 != layout.kernels.size());
                           }
                           os << "  ]\n";
                           os << "}\n";
                         });
}

static LogicalResult writeManifest(mlir::vc4::ModuleOp vc4Module,
                                   llvm::ArrayRef<KernelRecord> kernels,
                                   llvm::StringRef bundleDir) {
  return writeBundleFile(vc4Module.getOperation(), bundleDir, "manifest.json",
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
                           os << "    \"shared_vpm_bytes\": 4096,\n";
                           os << "    \"semaphores\": 16\n";
                           os << "  },\n";
                           os << "  \"kernels\": [\n";
                           for (size_t i = 0; i != kernels.size(); ++i) {
                             appendManifestKernelEntry(
                                 os, kernels[i], i + 1 != kernels.size());
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
  module.walk([&](mlir::vc4::ModuleOp vc4Module) {
    vc4Modules.push_back(vc4Module);
  });
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
