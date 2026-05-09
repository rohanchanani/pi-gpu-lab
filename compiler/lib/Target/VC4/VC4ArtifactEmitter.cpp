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
  std::optional<std::string> base = getElementCType(elemType);
  if (!base)
    return std::nullopt;
  if (direction == "in")
    return std::string("const ") + *base + " *";
  if (direction == "out" || direction == "inout")
    return *base + " *";
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
  parsed.publicName = publicName.getValue().str();
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

static bool isUnsupportedVectorRotateSmallImm(int64_t selector) {
  return selector >= 48 && selector <= 63;
}

static std::string formatRegFileAddress(char regFile, int64_t address) {
  std::string result;
  result.push_back('r');
  result.push_back(regFile);
  result += std::to_string(address);
  return result;
}

static std::string formatReadAddress(char regFile, int64_t address) {
  // vc4asm names several read-side peripheral addresses symbolically.  Do not
  // print them as ordinary regfile locations: e.g. ra32 is not a uniform read
  // in the source language the hardware reference uses.
  switch (address) {
  case 32:
    return "unif";
  case 38:
    return "elem_num";
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
  // Printing these as raN/rbN assembles to the wrong artifact boundary for the
  // saxpy_full kernel: uniform/TMU/VPM/VDW traffic then silently disappears or
  // hits the wrong endpoint.
  switch (address) {
  case 48:
    return "vpm";
  case 49:
    return "vw_setup";
  case 50:
    return "vw_addr";
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
      if (isUnsupportedVectorRotateSmallImm(*smallImm)) {
        return bundle.emitOpError()
               << "cannot emit vector-rotate small_imm selector " << *smallImm
               << " in this qpu.bundle qasm slice";
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

static LogicalResult appendAddInstruction(mlir::vc4::QPUBundleOp bundle,
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
  if (failed(formatMuxSource(bundle, bundle.getAddA(), *unpackSuffix, src0)))
    return failure();
  if (!isUnaryAddOpcode(opcode) &&
      failed(formatMuxSource(bundle, bundle.getAddB(), std::string(), src1)))
    return failure();

  if (needSeparator)
    line += "; ";
  needSeparator = true;

  mlir::vc4::Cond cond = bundle.getCondAdd();
  const bool useMovAlias =
      opcode == mlir::vc4::AddOpcode::bit_or && src0 == src1 &&
      cond == mlir::vc4::Cond::always &&
      !bundle.getOperation()->hasAttr("set_flags");

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
                               bundle.getOperation()->hasAttr("write_swap")) +
                *packSuffix;
  line += " ";
  line += dest;
  line += ", ";
  line += src0;
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
          : formatWriteAddress(getIntegerAttrValue(bundle.getOperation(),
                                                   "waddr_mul"),
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

  auto formatPseudoRead = [](int64_t raddr) -> std::optional<std::string> {
    switch (raddr) {
    case 50:
      return std::string("read vw_wait");
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
                                       llvm::raw_ostream &os) {
  std::string line;
  bool needSeparator = false;

  const bool addActive = bundle.getOpAdd() != mlir::vc4::AddOpcode::nop;
  if (failed(appendAddInstruction(bundle, line, needSeparator)))
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

static LogicalResult emitQPULDIQASM(mlir::vc4::QPULDIOp ldi,
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

  os << opcode << " " << destinations << ", "
     << formatU32Immediate(valueAttr.getInt()) << "\n";
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
      if (failed(emitQPUBundleQASM(bundle, qasmOS)))
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

static void appendLauncherPrototype(llvm::raw_ostream &os,
                                    const LaunchABIModel &launchABI) {
  os << "int " << launchABI.publicName << "(struct vc4_runtime *rt";
  for (const LaunchABIArgumentModel &arg : launchABI.arguments) {
    os << ", ";
    appendLauncherParameter(os, arg);
  }
  os << ")";
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

static std::string getPrepareFunctionName(const LaunchABIModel &launchABI) {
  return getLaunchAPIBaseName(launchABI) + "_prepare";
}

static std::string getReleaseFunctionName(const LaunchABIModel &launchABI) {
  return getLaunchAPIBaseName(launchABI) + "_release";
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

static bool isLaunchABIBufferArgument(const LaunchABIArgumentModel &arg) {
  return arg.kind == LaunchABIArgumentKind::Buffer;
}

static bool isInputLikeBufferDirection(llvm::StringRef direction) {
  return direction == "in" || direction == "inout";
}

static bool isOutputLikeBufferDirection(llvm::StringRef direction) {
  return direction == "out" || direction == "inout";
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

static std::string getBufferElementCTypeForCodegen(
    const LaunchABIArgumentModel &arg) {
  std::optional<std::string> elemType = getElementCType(arg.elementType);
  if (elemType)
    return *elemType;
  return "uint32_t";
}

static void appendPreparePrototype(llvm::raw_ostream &os,
                                   const LaunchABIModel &launchABI) {
  os << "int " << getPrepareFunctionName(launchABI)
     << "(struct vc4_runtime *rt, uint32_t max_n)";
}

static void appendReleasePrototype(llvm::raw_ostream &os,
                                   const LaunchABIModel &launchABI) {
  os << "void " << getReleaseFunctionName(launchABI)
     << "(struct vc4_runtime *rt)";
}

static LogicalResult writeLauncherHeader(KernelRecord &kernel,
                                         llvm::StringRef bundleDir) {
  return writeBundleFile(kernel.func.getOperation(), bundleDir,
                         "kernel_launch.h", [&](llvm::raw_ostream &os) {
                           os << "#ifndef VC4_CODEGEN_KERNEL_LAUNCH_H\n";
                           os << "#define VC4_CODEGEN_KERNEL_LAUNCH_H\n\n";
                           os << "#include <stdint.h>\n";
                           os << "#include \"mailbox.h\"\n\n";
                           os << "#ifdef __cplusplus\n";
                           os << "extern \"C\" {\n";
                           os << "#endif\n\n";
                           appendPreparePrototype(os, kernel.launchABI);
                           os << ";\n\n";
                           appendLauncherPrototype(os, kernel.launchABI);
                           os << ";\n\n";
                           appendReleasePrototype(os, kernel.launchABI);
                           os << ";\n\n";
                           os << "uint32_t "
                              << getRuntimeAllocationsFunctionName(kernel.launchABI)
                              << "(void);\n";
                           os << "uint32_t "
                              << getRuntimeLaunchesFunctionName(kernel.launchABI)
                              << "(void);\n";
                           os << "uint32_t "
                              << getRuntimeCapacityFunctionName(kernel.launchABI)
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
    return std::string("(uint32_t)(uintptr_t)") + arg.name;
  if (arg.scalarType == "f32")
    return std::string("vc4_codegen_pack_f32(") + arg.name + ")";
  return std::string("(uint32_t)") + arg.name;
}

static std::optional<std::string>
getBuiltinUniformExpression(const LaunchABIBuiltinModel &builtin) {
  if (builtin.kind == "qpu_num")
    return std::string("qpu");
  if (builtin.kind == "num_qpus")
    return std::string("activeQpus");
  return std::nullopt;
}

static void appendLauncherUniformLayoutComment(
    llvm::raw_ostream &os, const LaunchABIModel &launchABI) {
  os << "  /*\n";
  os << "   * Dense physical uniform layout per QPU, ordered by ";
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

static LogicalResult writeLauncherSource(KernelRecord &kernel,
                                         llvm::StringRef bundleDir) {
  std::string source;
  llvm::raw_string_ostream os(source);

  const LaunchABIModel &launchABI = kernel.launchABI;
  const std::string apiBase = getLaunchAPIBaseName(launchABI);
  const std::string stateName = apiBase + "_launch_state";
  const std::string prepareName = getPrepareFunctionName(launchABI);
  const std::string allocationsName = getRuntimeAllocationsFunctionName(launchABI);
  const std::string launchesName = getRuntimeLaunchesFunctionName(launchABI);
  const std::string capacityName = getRuntimeCapacityFunctionName(launchABI);
  llvm::SmallVector<const LaunchABIArgumentModel *, 4> bufferArgs =
      collectLaunchABIBufferArguments(launchABI);
  const LaunchABIArgumentModel *logicalCountArg =
      findLaunchABILogicalCountArgument(launchABI);

  os << "#include \"kernel_launch.h\"\n\n";
  os << "#include \"rpi.h\"\n";
  os << "#include \"mailbox.h\"\n";
  os << "#include \"" << launchABI.codeSymbol << ".h\"\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "#include <string.h>\n\n";

  os << "#define GPU_MEM_FLG 0xCu\n";
  os << "#define GPU_BASE 0x40000000u\n";
  os << "#define NUM_UNIFS " << launchABI.uniformWordsPerQPU << "u\n\n";

  if (launchABIRequiresF32Packing(launchABI)) {
    os << "static uint32_t vc4_codegen_pack_f32(float value) {\n";
    os << "  uint32_t bits = 0;\n";
    os << "  memcpy(&bits, &value, sizeof(bits));\n";
    os << "  return bits;\n";
    os << "}\n\n";
  }

  os << "struct " << stateName << " {\n";
  os << "  uint32_t code[sizeof(" << launchABI.codeSymbol
     << ") / sizeof(uint32_t)];\n";
  os << "  uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];\n";
  os << "  uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];\n";
  os << "  uint32_t handle;\n";
  os << "  uint32_t max_n;\n";
  os << "  uint32_t padded_capacity_n;\n";
  os << "  uint32_t launch_count;\n";
  os << "  uint8_t payload[];\n";
  os << "};\n\n";

  os << "static volatile struct " << stateName << " *g_state;\n";
  os << "static uint32_t g_handle;\n";
  os << "static uint32_t g_allocations;\n\n";

  os << "static size_t " << apiBase
     << "_align_up_size(size_t value, size_t alignment) {\n";
  os << "  if (alignment <= 1u)\n";
  os << "    return value;\n";
  os << "  return (value + alignment - 1u) & ~(alignment - 1u);\n";
  os << "}\n\n";

  os << "static uint32_t " << apiBase << "_round_up_to_lane_width(uint32_t n) {\n";
  os << "  const uint32_t laneWidth = VC4_RUNTIME_LANE_WIDTH;\n";
  os << "  if (n == 0u)\n";
  os << "    return 0u;\n";
  os << "  return (n + laneWidth - 1u) & ~(laneWidth - 1u);\n";
  os << "}\n\n";

  os << "static size_t " << apiBase
     << "_payload_bytes(uint32_t padded_capacity_n) {\n";
  os << "  size_t offset = 0u;\n";
  for (const LaunchABIArgumentModel *arg : bufferArgs) {
    std::string elemType = getBufferElementCTypeForCodegen(*arg);
    os << "  offset = " << apiBase << "_align_up_size(offset, sizeof("
       << elemType << "));\n";
    os << "  offset += (size_t)padded_capacity_n * sizeof(" << elemType
       << ");\n";
  }
  os << "  return offset;\n";
  os << "}\n\n";

  for (size_t i = 0; i != bufferArgs.size(); ++i) {
    const LaunchABIArgumentModel &arg = *bufferArgs[i];
    std::string elemType = getBufferElementCTypeForCodegen(arg);
    os << "static size_t " << apiBase << "_" << arg.name
       << "_offset(uint32_t padded_capacity_n) {\n";
    os << "  size_t offset = 0u;\n";
    for (size_t j = 0; j != i; ++j) {
      const LaunchABIArgumentModel &prev = *bufferArgs[j];
      std::string prevElemType = getBufferElementCTypeForCodegen(prev);
      os << "  offset = " << apiBase << "_align_up_size(offset, sizeof("
         << prevElemType << "));\n";
      os << "  offset += (size_t)padded_capacity_n * sizeof(" << prevElemType
         << ");\n";
    }
    os << "  return " << apiBase << "_align_up_size(offset, sizeof("
       << elemType << "));\n";
    os << "}\n\n";

    os << "static " << elemType << " *" << apiBase << "_" << arg.name
       << "_ptr(volatile struct " << stateName << " *state) {\n";
    os << "  return (" << elemType << " *)(state->payload + " << apiBase
       << "_" << arg.name << "_offset(state->padded_capacity_n));\n";
    os << "}\n\n";
  }

  os << "static size_t " << apiBase << "_state_size(uint32_t padded_capacity_n) {\n";
  os << "  return offsetof(struct " << stateName << ", payload) + "
     << apiBase << "_payload_bytes(padded_capacity_n);\n";
  os << "}\n\n";

  appendPreparePrototype(os, launchABI);
  os << " {\n";
  os << "  if (!rt || !rt->isInitialized)\n";
  os << "    return -1;\n\n";
  os << "  uint32_t activeQpus = vc4_runtime_active_qpus(rt);\n";
  os << "  if (activeQpus == 0u || activeQpus > VC4_RUNTIME_MAX_QPUS)\n";
  os << "    return -1;\n\n";
  os << "  uint32_t paddedCapacity = " << apiBase
     << "_round_up_to_lane_width(max_n);\n";
  os << "  if (paddedCapacity < max_n)\n";
  os << "    return -1;\n\n";
  os << "  if (g_state) {\n";
  os << "    if (max_n <= g_state->max_n)\n";
  os << "      return 0;\n";
  os << "    return -1;\n";
  os << "  }\n\n";
  os << "  size_t allocSize = " << apiBase
     << "_state_size(paddedCapacity);\n";
  os << "  if (allocSize > 0xffffffffu)\n";
  os << "    return -1;\n\n";
  os << "  uint32_t handle = mem_alloc((uint32_t)allocSize, 4096u, GPU_MEM_FLG);\n";
  os << "  if (!handle)\n";
  os << "    return -1;\n\n";
  os << "  uint32_t vc = mem_lock(handle);\n";
  os << "  if (!vc) {\n";
  os << "    mem_free(handle);\n";
  os << "    return -1;\n";
  os << "  }\n\n";
  os << "  volatile struct " << stateName << " *state =\n";
  os << "      (volatile struct " << stateName << " *)(vc - GPU_BASE);\n";
  os << "  if (!state) {\n";
  os << "    mem_unlock(handle);\n";
  os << "    mem_free(handle);\n";
  os << "    return -1;\n";
  os << "  }\n\n";
  os << "  memset((void *)state, 0, allocSize);\n";
  os << "  state->handle = handle;\n";
  os << "  state->max_n = max_n;\n";
  os << "  state->padded_capacity_n = paddedCapacity;\n";
  os << "  memcpy((void *)state->code, " << launchABI.codeSymbol
     << ", sizeof state->code);\n";
  os << "  for (uint32_t qpu = 0; qpu < VC4_RUNTIME_MAX_QPUS; ++qpu)\n";
  os << "    state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&state->unif[qpu][0];\n\n";
  os << "  g_handle = handle;\n";
  os << "  g_state = state;\n";
  os << "  g_allocations++;\n";
  os << "  return 0;\n";
  os << "}\n\n";

  appendLauncherPrototype(os, launchABI);
  os << " {\n";
  os << "  if (!rt || !rt->isInitialized)\n";
  os << "    return -1;\n\n";
  os << "  uint32_t activeQpus = vc4_runtime_active_qpus(rt);\n";
  os << "  if (activeQpus == 0u || activeQpus > VC4_RUNTIME_MAX_QPUS)\n";
  os << "    return -1;\n\n";
  if (logicalCountArg) {
    os << "  uint32_t logicalN = (uint32_t)" << logicalCountArg->name << ";\n";
  } else {
    os << "  uint32_t logicalN = g_state ? g_state->max_n : 0u;\n";
  }
  for (const LaunchABIArgumentModel *arg : bufferArgs) {
    os << "  if (logicalN != 0u && !" << arg->name << ")\n";
    os << "    return -1;\n";
  }
  os << "  if (!g_state) {\n";
  os << "    if (" << prepareName << "(rt, logicalN) < 0)\n";
  os << "      return -1;\n";
  os << "  }\n";
  os << "  if (logicalN > g_state->max_n)\n";
  os << "    return -1;\n\n";
  os << "  uint32_t paddedN = " << apiBase << "_round_up_to_lane_width(logicalN);\n";
  os << "  if (paddedN < logicalN || paddedN > g_state->padded_capacity_n)\n";
  os << "    return -1;\n\n";

  for (const LaunchABIArgumentModel *arg : bufferArgs) {
    std::string elemType = getBufferElementCTypeForCodegen(*arg);
    os << "  " << elemType << " *gpu_" << arg->name << " = " << apiBase
       << "_" << arg->name << "_ptr(g_state);\n";
  }
  if (!bufferArgs.empty())
    os << "\n";

  for (const LaunchABIArgumentModel *arg : bufferArgs) {
    std::string elemType = getBufferElementCTypeForCodegen(*arg);
    if (isInputLikeBufferDirection(arg->direction)) {
      os << "  if (logicalN != 0u)\n";
      os << "    memcpy(gpu_" << arg->name << ", " << arg->name
         << ", (size_t)logicalN * sizeof(" << elemType << "));\n";
    }
    os << "  for (uint32_t i = logicalN; i < paddedN; ++i)\n";
    os << "    gpu_" << arg->name << "[i] = (" << elemType << ")0;\n";
  }
  if (!bufferArgs.empty())
    os << "\n";

  appendLauncherUniformLayoutComment(os, launchABI);
  os << "  for (uint32_t qpu = 0; qpu < activeQpus; ++qpu) {\n";
  for (int64_t index = 0; index != launchABI.uniformWordsPerQPU; ++index) {
    if (const LaunchABIArgumentModel *arg =
            findLaunchABIArgumentForUniformIndex(launchABI, index)) {
      if (arg->kind == LaunchABIArgumentKind::Buffer) {
        os << "    g_state->unif[qpu][" << index << "] = GPU_BASE + "
           << "(uint32_t)gpu_" << arg->name << "; /* arg " << arg->name
           << " */\n";
      } else {
        os << "    g_state->unif[qpu][" << index << "] = "
           << getArgumentUniformExpression(*arg) << "; /* arg " << arg->name
           << " */\n";
      }
      continue;
    }

    if (const LaunchABIBuiltinModel *builtin =
            findLaunchABIBuiltinForUniformIndex(launchABI, index)) {
      std::optional<std::string> expression = getBuiltinUniformExpression(*builtin);
      if (!expression) {
        return emitLaunchABIModelError(
            kernel.func, llvm::Twine("builtin '") + builtin->name +
                         "' has unsupported kind for launcher uniform packing");
      }
      os << "    g_state->unif[qpu][" << index << "] = " << *expression
         << "; /* builtin " << builtin->name << " */\n";
      continue;
    }

    return emitLaunchABIModelError(
        kernel.func, llvm::Twine("missing launcher uniform assignment for index ") +
                         std::to_string(index));
  }
  os << "    g_state->unif_ptr[qpu] = GPU_BASE + "
        "(uint32_t)&g_state->unif[qpu][0];\n";
  os << "  }\n\n";
  os << "  gpu_fft_base_exec_direct((uint32_t)g_state->code,\n";
  os << "                           (uint32_t *)g_state->unif_ptr, activeQpus);\n\n";

  for (const LaunchABIArgumentModel *arg : bufferArgs) {
    std::string elemType = getBufferElementCTypeForCodegen(*arg);
    if (!isOutputLikeBufferDirection(arg->direction))
      continue;
    os << "  if (logicalN != 0u)\n";
    os << "    memcpy(" << arg->name << ", gpu_" << arg->name
       << ", (size_t)logicalN * sizeof(" << elemType << "));\n";
  }
  if (!bufferArgs.empty())
    os << "\n";

  os << "  g_state->launch_count++;\n";
  os << "  return 0;\n";
  os << "}\n\n";

  appendReleasePrototype(os, launchABI);
  os << " {\n";
  os << "  (void)rt;\n";
  os << "  if (!g_state)\n";
  os << "    return;\n";
  os << "  mem_unlock(g_handle);\n";
  os << "  mem_free(g_handle);\n";
  os << "  g_state = 0;\n";
  os << "  g_handle = 0;\n";
  os << "}\n\n";

  os << "uint32_t " << allocationsName << "(void) {\n";
  os << "  return g_allocations;\n";
  os << "}\n\n";

  os << "uint32_t " << launchesName << "(void) {\n";
  os << "  if (!g_state)\n";
  os << "    return 0u;\n";
  os << "  return g_state->launch_count;\n";
  os << "}\n\n";

  os << "uint32_t " << capacityName << "(void) {\n";
  os << "  if (!g_state)\n";
  os << "    return 0u;\n";
  os << "  return g_state->max_n;\n";
  os << "}\n";
  os.flush();

  return writeBundleFile(kernel.func.getOperation(), bundleDir,
                         "kernel_launch.c",
                         [&](llvm::raw_ostream &fileOS) { fileOS << source; });
}

static void appendManifestPublicParameter(llvm::raw_ostream &os,
                                          llvm::StringRef name,
                                          llvm::StringRef cType,
                                          llvm::StringRef role,
                                          bool trailingComma) {
  os << "      {\"name\": ";
  appendJSONEscapedString(os, name);
  os << ", \"c_type\": ";
  appendJSONEscapedString(os, cType);
  os << ", \"role\": ";
  appendJSONEscapedString(os, role);
  os << "}";
  if (trailingComma)
    os << ",";
  os << "\n";
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

static void appendManifestLaunchABI(llvm::raw_ostream &os,
                                    const LaunchABIModel &launchABI) {
  os << "  \"launch_abi\": {\n";
  os << "    \"public_name\": ";
  appendJSONEscapedString(os, launchABI.publicName);
  os << ",\n";
  os << "    \"tail_policy\": ";
  appendJSONEscapedString(os, launchABI.tailPolicy);
  os << ",\n";
  os << "    \"uniform_words_per_qpu\": "
     << launchABI.uniformWordsPerQPU << ",\n";
  os << "    \"arg_count\": " << launchABI.arguments.size() << ",\n";
  os << "    \"args\": [\n";
  for (size_t i = 0; i != launchABI.arguments.size(); ++i) {
    appendManifestLaunchABIArgument(os, launchABI.arguments[i],
                                    i + 1 != launchABI.arguments.size());
  }
  os << "    ],\n";
  os << "    \"builtins\": [\n";
  for (size_t i = 0; i != launchABI.builtins.size(); ++i) {
    appendManifestLaunchABIBuiltin(os, launchABI.builtins[i],
                                   i + 1 != launchABI.builtins.size());
  }
  os << "    ],\n";
  os << "    \"public_api\": {\n";
  os << "      \"function_name\": ";
  appendJSONEscapedString(os, launchABI.publicName);
  os << ",\n";
  os << "      \"return_type\": \"int\",\n";
  os << "      \"parameters\": [\n";
  appendManifestPublicParameter(os, "rt", "struct vc4_runtime *", "runtime",
                                !launchABI.arguments.empty());
  for (size_t i = 0; i != launchABI.arguments.size(); ++i) {
    const LaunchABIArgumentModel &arg = launchABI.arguments[i];
    appendManifestPublicParameter(os, arg.name, arg.cType, "kernel_arg",
                                  i + 1 != launchABI.arguments.size());
  }
  os << "      ]\n";
  os << "    }\n";
  os << "  }";
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
                           os << "    \"kernel_launch.h\"";
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
  if (failed(writeLauncherSource(kernels.front(), bundleDir)))
    return failure();
  if (failed(writeLauncherHeader(kernels.front(), bundleDir)))
    return failure();
  if (failed(writeManifest(vc4Module, kernels, bundleDir)))
    return failure();

  return success();
}
