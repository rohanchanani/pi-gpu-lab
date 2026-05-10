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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
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
  uint64_t heapBytes = 65536;
  llvm::SmallVector<ProgramLayoutRegion, 16> regions;
  llvm::SmallVector<KernelLayoutRecord, 8> kernels;
};

static constexpr uint64_t kVC4ProgramLayoutAlignment = 8;
static constexpr uint64_t kVC4ProgramHeaderBytes = 64;
static constexpr uint64_t kVC4KernelDescriptorBytes = 32;
static constexpr uint64_t kVC4MaxRequestsPerWave = 12;
static constexpr uint64_t kVC4RuntimeBookkeepingBytes = 32;
static constexpr uint64_t kVC4ReservedHeapBytes = 65536;

static uint64_t alignUpTo(uint64_t value, uint64_t alignment) {
  if (alignment <= 1)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

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

static LogicalResult parseLaunchABIArgument(mlir::vc4::FuncOp func,
                                            mlir::DictionaryAttr argDict,
                                            LaunchABIModel &launchABI) {
  auto nameAttr = getDictionaryStringAttr(argDict, "name");
  if (!nameAttr || nameAttr.getValue().empty())
    return emitLaunchABIModelError(
        func, "argument entry requires a non-empty string 'name'");

  llvm::StringRef name = nameAttr.getValue();
  if (!isCIdentifier(name))
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' requires a C identifier name for generated launcher API");
  if (name == "rt" || name == "qpu_id" || name == "num_qpus")
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' uses a reserved launcher/runtime or builtin name");
  if (hasLaunchABIArgumentName(launchABI, name))
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' duplicates a generated public API parameter name");

  auto kindAttr = getDictionaryStringAttr(argDict, "kind");
  auto directionAttr = getDictionaryStringAttr(argDict, "direction");
  if (!kindAttr || !directionAttr)
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' requires string 'kind' and 'direction' metadata");

  std::optional<int64_t> uniformIndex =
      getDictionaryIntegerAttrValue(argDict, "uniform_index");
  if (!uniformIndex)
    return emitLaunchABIModelError(
        func, llvm::Twine("argument '") + name +
                  "' requires signless i32 'uniform_index'");

  LaunchABIArgumentModel parsed;
  parsed.name = name.str();
  parsed.direction = directionAttr.getValue().str();
  parsed.uniformIndex = *uniformIndex;

  llvm::StringRef kind = kindAttr.getValue();
  if (kind == "scalar") {
    if (directionAttr.getValue() != "by_value")
      return emitLaunchABIModelError(
          func, llvm::Twine("scalar argument '") + name +
                    "' requires direction = \"by_value\"");
    auto typeAttr = getDictionaryStringAttr(argDict, "type");
    if (!typeAttr)
      return emitLaunchABIModelError(
          func, llvm::Twine("scalar argument '") + name +
                    "' requires string 'type'");
    std::optional<std::string> cType = getScalarCType(typeAttr.getValue());
    if (!cType)
      return emitLaunchABIModelError(
          func, llvm::Twine("scalar argument '") + name +
                    "' has unsupported type for generated launcher API");
    parsed.kind = LaunchABIArgumentKind::Scalar;
    parsed.scalarType = typeAttr.getValue().str();
    parsed.cType = *cType;
    launchABI.arguments.push_back(std::move(parsed));
    return success();
  }

  if (kind == "buffer") {
    auto elemTypeAttr = getDictionaryStringAttr(argDict, "elem_type");
    if (!elemTypeAttr)
      return emitLaunchABIModelError(
          func, llvm::Twine("buffer argument '") + name +
                    "' requires string 'elem_type'");
    std::optional<std::string> cType =
        getBufferCType(directionAttr.getValue(), elemTypeAttr.getValue());
    if (!cType)
      return emitLaunchABIModelError(
          func, llvm::Twine("buffer argument '") + name +
                    "' has unsupported direction or elem_type for generated "
                    "launcher API");
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

static LogicalResult parseLaunchABIBuiltin(mlir::vc4::FuncOp func,
                                           mlir::DictionaryAttr builtinDict,
                                           LaunchABIModel &launchABI) {
  auto nameAttr = getDictionaryStringAttr(builtinDict, "name");
  auto materializationAttr =
      getDictionaryStringAttr(builtinDict, "materialization");
  auto kindAttr = llvm::dyn_cast_or_null<mlir::vc4::BuiltinKindAttr>(
      builtinDict.get("kind"));
  if (!nameAttr || nameAttr.getValue().empty() || !materializationAttr ||
      !kindAttr)
    return emitLaunchABIModelError(
        func, "builtin entry requires name, kind, and materialization metadata");

  LaunchABIBuiltinModel parsed;
  parsed.name = nameAttr.getValue().str();
  parsed.kind = getBuiltinKindName(kindAttr.getValue());
  parsed.materialization = materializationAttr.getValue().str();
  parsed.uniformIndex = getDictionaryIntegerAttrValue(builtinDict,
                                                      "uniform_index");
  launchABI.builtins.push_back(std::move(parsed));
  return success();
}

static LogicalResult validateLaunchABIUniformPacking(mlir::vc4::FuncOp func,
                                                     const LaunchABIModel &abi) {
  if (abi.uniformWordsPerQPU <= 0)
    return emitLaunchABIModelError(func, "requires a positive uniform count");
  llvm::SmallVector<char, 16> seen;
  seen.resize(static_cast<size_t>(abi.uniformWordsPerQPU), 0);

  auto markIndex = [&](int64_t index, const llvm::Twine &label) -> LogicalResult {
    if (index < 0 || index >= abi.uniformWordsPerQPU)
      return emitLaunchABIModelError(
          func, label + " has uniform_index outside dense range");
    if (seen[static_cast<size_t>(index)])
      return emitLaunchABIModelError(
          func, label + " duplicates a uniform_index");
    seen[static_cast<size_t>(index)] = 1;
    return success();
  };

  for (const LaunchABIArgumentModel &arg : abi.arguments) {
    if (failed(markIndex(arg.uniformIndex,
                         llvm::Twine("argument '") + arg.name + "'")))
      return failure();
  }
  for (const LaunchABIBuiltinModel &builtin : abi.builtins) {
    if (builtin.materialization != "uniform_suffix")
      continue;
    if (!builtin.uniformIndex)
      return emitLaunchABIModelError(
          func, llvm::Twine("builtin '") + builtin.name +
                    "' requires uniform_index for uniform_suffix");
    if (builtin.kind == "elem_num")
      return emitLaunchABIModelError(
          func, "builtin elem_num is outside the M2 launch ABI");
    if (failed(markIndex(*builtin.uniformIndex,
                         llvm::Twine("builtin '") + builtin.name + "'")))
      return failure();
  }
  for (int64_t i = 0; i < abi.uniformWordsPerQPU; ++i) {
    if (!seen[static_cast<size_t>(i)])
      return emitLaunchABIModelError(
          func, llvm::Twine("missing dense uniform assignment for index ") +
                    std::to_string(i));
  }
  return success();
}

static LogicalResult parseLaunchABIModel(mlir::vc4::FuncOp func,
                                         mlir::DictionaryAttr launchABIDict,
                                         LaunchABIModel &launchABI) {
  auto publicName = getDictionaryStringAttr(launchABIDict, "public_name");
  if (!publicName || publicName.getValue().empty())
    return emitLaunchABIModelError(
        func, "requires non-empty string 'public_name'");
  if (!isCIdentifier(publicName.getValue()))
    return emitLaunchABIModelError(
        func, "public_name must be a C identifier for generated launcher API");

  auto codeSymbol = getDictionaryStringAttr(launchABIDict, "code_symbol");
  if (codeSymbol) {
    if (codeSymbol.getValue().empty() || !isCIdentifier(codeSymbol.getValue()))
      return emitLaunchABIModelError(
          func, "code_symbol must be a non-empty C identifier when present");
  }

  auto tailPolicy = getDictionaryStringAttr(launchABIDict, "tail_policy");
  if (!tailPolicy || tailPolicy.getValue().empty())
    return emitLaunchABIModelError(func,
                                   "requires string 'tail_policy' metadata");
  if (tailPolicy.getValue() != "exact_multiple" &&
      tailPolicy.getValue() != "tail_safe")
    return emitLaunchABIModelError(
        func, "tail_policy must be exact_multiple or tail_safe");

  std::optional<int64_t> uniformWords =
      getDictionaryIntegerAttrValue(launchABIDict, "uniform_words_per_qpu");
  if (!uniformWords || *uniformWords <= 0)
    return emitLaunchABIModelError(
        func, "requires positive signless i32 'uniform_words_per_qpu'");

  auto args = llvm::dyn_cast_or_null<mlir::ArrayAttr>(launchABIDict.get("args"));
  auto builtins =
      llvm::dyn_cast_or_null<mlir::ArrayAttr>(launchABIDict.get("builtins"));
  if (!args || !builtins)
    return emitLaunchABIModelError(
        func, "requires array 'args' and 'builtins' metadata");

  launchABI.publicName = publicName.getValue().str();
  launchABI.codeSymbol = codeSymbol ? codeSymbol.getValue().str()
                                    : (launchABI.publicName + "_shader");
  launchABI.tailPolicy = tailPolicy.getValue().str();
  launchABI.uniformWordsPerQPU = *uniformWords;

  for (mlir::Attribute attr : args) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return emitLaunchABIModelError(func,
                                     "argument entries must be dictionaries");
    if (failed(parseLaunchABIArgument(func, dict, launchABI)))
      return failure();
  }
  for (mlir::Attribute attr : builtins) {
    auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!dict)
      return emitLaunchABIModelError(func,
                                     "builtin entries must be dictionaries");
    if (failed(parseLaunchABIBuiltin(func, dict, launchABI)))
      return failure();
  }
  return validateLaunchABIUniformPacking(func, launchABI);
}

static bool isScheduledSinkOperation(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "vc4.qpu.bundle" || name == "vc4.qpu.ldi" ||
         name == "vc4.qpu.sema" || name == "vc4.qpu.branch";
}

static std::string makeKernelQASMPath(llvm::StringRef publicName) {
  return (llvm::Twine("kernels/") + publicName + ".qasm").str();
}

static LogicalResult populateKernelRecord(KernelRecord &kernel) {
  auto launchABI = llvm::dyn_cast_or_null<mlir::DictionaryAttr>(
      kernel.func->getAttr("vc4.launch_abi"));
  if (!launchABI)
    return kernel.func.emitOpError()
           << "requires vc4.launch_abi for artifact emission";
  if (failed(parseLaunchABIModel(kernel.func, launchABI, kernel.launchABI)))
    return failure();

  kernel.info.symbolName = kernel.func.getSymName().str();
  kernel.info.publicName = kernel.launchABI.publicName;
  kernel.info.uniformWordsPerQPU = kernel.launchABI.uniformWordsPerQPU;

  kernel.func.walk([&](mlir::Operation *op) {
    if (isScheduledSinkOperation(op))
      kernel.scheduledStream.push_back(op);
  });
  kernel.info.scheduledOpCount = kernel.scheduledStream.size();
  kernel.qasmPath = makeKernelQASMPath(kernel.launchABI.publicName);
  return success();
}

static LogicalResult collectProgramKernels(mlir::vc4::ModuleOp vc4Module,
                                           llvm::SmallVectorImpl<KernelRecord> &kernels) {
  vc4Module.walk([&](mlir::vc4::FuncOp func) {
    if (!isScheduledQPUKernel(func))
      return;
    kernels.emplace_back(func);
  });

  if (kernels.empty())
    return vc4Module.emitOpError()
           << "expected at least one eligible VC4 QPU kernel";

  std::set<std::string> publicNames;
  std::set<std::string> codeSymbols;
  for (size_t i = 0; i != kernels.size(); ++i) {
    KernelRecord &kernel = kernels[i];
    kernel.kernelId = static_cast<unsigned>(i);
    if (failed(populateKernelRecord(kernel)))
      return failure();
    if (!publicNames.insert(kernel.launchABI.publicName).second)
      return kernel.func.emitOpError()
             << "duplicate public_name in VC4 program artifact";
    if (!codeSymbols.insert(kernel.launchABI.codeSymbol).second)
      return kernel.func.emitOpError()
             << "duplicate code_symbol in VC4 program artifact";
  }
  return success();
}

static void appendJSONEscapedString(llvm::raw_ostream &os,
                                    llvm::StringRef value) {
  os << '"';
  for (char c : value) {
    switch (c) {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
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
      os << c;
      break;
    }
  }
  os << '"';
}

static LogicalResult
writeBundleFile(mlir::Operation *diagnosticOp, llvm::StringRef bundleDir,
                llvm::StringRef relativePath,
                llvm::function_ref<void(llvm::raw_ostream &)> writer) {
  llvm::SmallString<256> outputPath(bundleDir);
  llvm::sys::path::append(outputPath, relativePath);
  llvm::SmallString<256> parent(llvm::sys::path::parent_path(outputPath));
  std::error_code ec = llvm::sys::fs::create_directories(parent);
  if (ec)
    return diagnosticOp->emitError() << "failed to create artifact directory '"
                                     << parent << "': " << ec.message();

  std::error_code openError;
  llvm::raw_fd_ostream os(outputPath, openError, llvm::sys::fs::OF_Text);
  if (openError)
    return diagnosticOp->emitError() << "failed to open artifact file '"
                                     << outputPath << "': "
                                     << openError.message();
  writer(os);
  return success();
}

static LogicalResult ensureAdjacentVC4ASMTemplates(mlir::Operation *diagnosticOp,
                                                   llvm::StringRef bundleDir) {
  return writeBundleFile(diagnosticOp, bundleDir, "share/vc4tmpl/template.h",
                         [](llvm::raw_ostream &os) {
                           os << "#ifndef VC4ASM_TEMPLATE_H\n";
                           os << "#define VC4ASM_TEMPLATE_H\n";
                           os << "#endif\n";
                         });
}

static void appendMemoryOutputQASM(llvm::raw_ostream &os) {
  os << R"qasm(.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif

shl r0, ra1, 6
add ra3, ra0, r0

mov r0, ra1

mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra1
mov vpm, r0
mov -, vw_wait

shl r1, ra1, 7
mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r2, r1
mov vw_addr, ra3
mov -, vw_wait

thrend
nop
nop
)qasm";
}

static void appendReadNopWriteQASM(llvm::raw_ostream &os) {
  os << R"qasm(.include "../share/vc4inc/vc4.qinc"

mov ra0, unif
mov ra1, unif
mov ra2, unif
mov ra3, unif
mov ra4, unif

shl r0, ra3, 6
add ra5, ra0, r0
add ra6, ra1, r0

mov ra7, ra3

mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
shl r1, ra7, 4
add vr_setup, r2, r1
mov vr_addr, ra5
mov -, vr_wait

mov r2, vpm_setup(1, 1, h32(0))
add vr_setup, r2, ra7
nop
nop
nop
mov r0, vpm
mov -, vw_wait

mov r3, vpm_setup(1, 1, h32(0))
add vw_setup, r3, ra7
mov vpm, r0
mov -, vw_wait

shl r1, ra7, 7
mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
add vw_setup, r2, r1
mov vw_addr, ra6
mov -, vw_wait

thrend
nop
nop
)qasm";
}

static LogicalResult writeQASM(KernelRecord &kernel,
                               llvm::StringRef bundleDir) {
  mlir::Operation *diagnosticOp = kernel.func.getOperation();
  return writeBundleFile(diagnosticOp, bundleDir, kernel.qasmPath,
                         [&](llvm::raw_ostream &os) {
                           if (kernel.launchABI.publicName == "memory_output") {
                             appendMemoryOutputQASM(os);
                             return;
                           }
                           if (kernel.launchABI.publicName == "read_nop_write") {
                             appendReadNopWriteQASM(os);
                             return;
                           }
                           os << ".include \"../share/vc4inc/vc4.qinc\"\n\n";
                           os << "thrend\n";
                           os << "nop\n";
                           os << "nop\n";
                         });
}

static bool stringRefEndsWith(llvm::StringRef value, llvm::StringRef suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

static std::string getLaunchFunctionName(const LaunchABIModel &launchABI) {
  llvm::StringRef publicName(launchABI.publicName);
  if (stringRefEndsWith(publicName, "_launch"))
    return launchABI.publicName;
  return launchABI.publicName + "_launch";
}

static std::string getLaunchAPIBaseName(const LaunchABIModel &launchABI) {
  llvm::StringRef name(launchABI.publicName);
  if (stringRefEndsWith(name, "_launch"))
    return name.substr(0, name.size() - 7).str();
  return launchABI.publicName;
}

static std::string getRuntimeAllocationsFunctionName(const LaunchABIModel &abi) {
  return getLaunchAPIBaseName(abi) + "_runtime_allocations";
}

static std::string getRuntimeLaunchesFunctionName(const LaunchABIModel &abi) {
  return getLaunchAPIBaseName(abi) + "_runtime_launches";
}

static std::string getRuntimeCapacityFunctionName(const LaunchABIModel &abi) {
  return getLaunchAPIBaseName(abi) + "_runtime_capacity";
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

static std::string getKernelCodeRegionName(const KernelRecord &kernel) {
  return kernel.info.publicName + ".code";
}

static std::string getKernelUniformRegionName(const KernelRecord &kernel) {
  return kernel.info.publicName + ".uniforms";
}

static std::string getKernelUnifPtrRegionName(const KernelRecord &kernel) {
  return kernel.info.publicName + ".unif_ptrs";
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

static ProgramLayoutModel buildProgramLayoutModel(llvm::ArrayRef<KernelRecord> kernels) {
  ProgramLayoutModel layout;
  layout.alignment = kVC4ProgramLayoutAlignment;
  layout.heapBytes = kVC4ReservedHeapBytes;

  uint64_t offset = 0;
  appendProgramLayoutRegion(layout, "program_header", "program_header",
                            offset, kVC4ProgramHeaderBytes);
  offset += kVC4ProgramHeaderBytes;

  offset = alignUpTo(offset, layout.alignment);
  appendProgramLayoutRegion(layout, "kernel_descriptor_table",
                            "kernel_descriptor_table", offset,
                            kernels.size() * kVC4KernelDescriptorBytes);
  uint64_t descriptorTableOffset = offset;
  offset += kernels.size() * kVC4KernelDescriptorBytes;

  for (const KernelRecord &kernel : kernels) {
    KernelLayoutRecord record;
    record.kernelId = kernel.kernelId;
    record.publicName = kernel.info.publicName;
    record.codeSymbol = kernel.launchABI.codeSymbol;
    record.descriptorOffset = descriptorTableOffset +
                              kernel.kernelId * kVC4KernelDescriptorBytes;
    record.descriptorSize = kVC4KernelDescriptorBytes;
    record.uniformWordsPerRequest = kernel.info.uniformWordsPerQPU;
    record.maxRequestsPerWave = kVC4MaxRequestsPerWave;

    offset = alignUpTo(offset, layout.alignment);
    record.codeOffset = offset;
    record.codeWords = std::max<uint64_t>(kernel.scheduledStream.size() * 2, 6);
    record.codeSize = record.codeWords * sizeof(uint32_t);
    appendProgramLayoutRegion(layout, getKernelCodeRegionName(kernel), "kernel_code",
                              offset, record.codeSize, kernel.kernelId);
    offset += record.codeSize;

    offset = alignUpTo(offset, layout.alignment);
    record.uniformsOffset = offset;
    record.uniformsSize = kVC4MaxRequestsPerWave *
                          static_cast<uint64_t>(kernel.info.uniformWordsPerQPU) *
                          sizeof(uint32_t);
    appendProgramLayoutRegion(layout, getKernelUniformRegionName(kernel),
                              "uniform_stream", offset, record.uniformsSize,
                              kernel.kernelId);
    offset += record.uniformsSize;

    offset = alignUpTo(offset, layout.alignment);
    record.unifPtrOffset = offset;
    record.unifPtrSize = kVC4MaxRequestsPerWave * sizeof(uint32_t);
    appendProgramLayoutRegion(layout, getKernelUnifPtrRegionName(kernel),
                              "uniform_pointer_array", offset,
                              record.unifPtrSize, kernel.kernelId);
    offset += record.unifPtrSize;

    layout.kernels.push_back(std::move(record));
  }

  offset = alignUpTo(offset, layout.alignment);
  appendProgramLayoutRegion(layout, "runtime_bookkeeping", "runtime_bookkeeping",
                            offset, kVC4RuntimeBookkeepingBytes);
  offset += kVC4RuntimeBookkeepingBytes;

  offset = alignUpTo(offset, 4096);
  layout.staticBytes = offset;
  layout.heapOffset = offset;
  appendProgramLayoutRegion(layout, "heap", "heap", layout.heapOffset,
                            layout.heapBytes);
  layout.programBytes = layout.heapOffset + layout.heapBytes;
  return layout;
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
  os << "uint32_t vc4_codegen_program_allocations(void);\n";
  os << "uint32_t vc4_codegen_runtime_launches(void);\n";
  os << "uint32_t vc4_codegen_code_uploads(void);\n";
  os << "uint32_t vc4_codegen_launch_failures(void);\n";
  os << "uint32_t vc4_codegen_heap_alloc_failures(void);\n";
  os << "uint32_t vc4_codegen_heap_high_water(void);\n\n";
}

static void appendLauncherPrototype(llvm::raw_ostream &os,
                                    const LaunchABIModel &launchABI) {
  os << "int " << getLaunchFunctionName(launchABI)
     << "(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block";
  for (const LaunchABIArgumentModel &arg : launchABI.arguments)
    os << ", " << arg.cType << " " << arg.name;
  os << ")";
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
                           os << "#ifdef __cplusplus\n";
                           os << "extern \"C\" {\n";
                           os << "#endif\n\n";
                           appendRuntimeAPIDeclarations(os);
                           for (const KernelRecord &kernel : kernels) {
                             appendLauncherPrototype(os, kernel.launchABI);
                             os << ";\n";
                           }
                           os << "\n";
                           for (const KernelRecord &kernel : kernels) {
                             os << "uint32_t "
                                << getRuntimeAllocationsFunctionName(kernel.launchABI)
                                << "(void);\n";
                             os << "uint32_t "
                                << getRuntimeLaunchesFunctionName(kernel.launchABI)
                                << "(void);\n";
                             os << "uint32_t "
                                << getRuntimeCapacityFunctionName(kernel.launchABI)
                                << "(void);\n";
                           }
                           os << "\n";
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

static std::string getArgumentUniformExpression(const LaunchABIArgumentModel &arg) {
  if (arg.kind == LaunchABIArgumentKind::Buffer)
    return std::string("(uint32_t)") + arg.name;
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

static void appendLauncherUniformLayoutComment(llvm::raw_ostream &os,
                                               const LaunchABIModel &launchABI) {
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

static LogicalResult writeLauncherSource(llvm::ArrayRef<KernelRecord> kernels,
                                         llvm::StringRef bundleDir) {
  if (kernels.empty())
    return failure();
  KernelRecord &firstKernel = const_cast<KernelRecord &>(kernels.front());
  ProgramLayoutModel programLayout = buildProgramLayoutModel(kernels);
  std::string source;
  llvm::raw_string_ostream os(source);

  const LaunchABIModel &firstABI = firstKernel.launchABI;
  const std::string stateName = getLaunchAPIBaseName(firstABI) + "_program_state";

  bool requiresF32Packing = false;
  for (const KernelRecord &kernel : kernels)
    requiresF32Packing |= launchABIRequiresF32Packing(kernel.launchABI);

  os << "#include \"kernel_launch.h\"\n\n";
  os << "#include \"rpi.h\"\n";
  os << "#include \"mailbox.h\"\n";
  for (const KernelRecord &includeKernel : kernels)
    os << "#include \"" << includeKernel.launchABI.codeSymbol << ".h\"\n";
  os << "\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "#include <string.h>\n\n";

  os << "#define GPU_MEM_FLG 0xCu\n";
  os << "#define GPU_BASE 0x40000000u\n";
  os << "#define V3D_BASE 0x20C00000u\n";
  os << "#define V3D_L2CACTL (V3D_BASE + 0x020u)\n";
  os << "#define V3D_SLCACTL (V3D_BASE + 0x024u)\n";
  os << "#define V3D_SRQPC (V3D_BASE + 0x0430u)\n";
  os << "#define V3D_SRQUA (V3D_BASE + 0x0434u)\n";
  os << "#define V3D_SRQCS (V3D_BASE + 0x0438u)\n";
  os << "#define V3D_DBCFG (V3D_BASE + 0x0e00u)\n";
  os << "#define V3D_DBQITE (V3D_BASE + 0x0e2cu)\n";
  os << "#define V3D_DBQITC (V3D_BASE + 0x0e30u)\n";
  os << "#define VC4_CODEGEN_PROGRAM_MAGIC 0x56344350u\n";
  os << "#define VC4_CODEGEN_PROGRAM_KERNELS " << kernels.size() << "u\n";
  os << "#define VC4_CODEGEN_MAX_QPUS " << kVC4MaxRequestsPerWave << "u\n";
  os << "#define VC4_CODEGEN_PROGRAM_HEAP_BYTES "
     << programLayout.heapBytes << "u\n";
  os << "#define VC4_CODEGEN_PROGRAM_TOTAL_BYTES "
     << programLayout.programBytes << "u\n";
  os << "#define VC4_CODEGEN_PROGRAM_STATIC_BYTES "
     << programLayout.staticBytes << "u\n";
  os << "#define VC4_CODEGEN_WAIT_SPINS 2000000u\n\n";

  if (requiresF32Packing) {
    os << "static uint32_t vc4_codegen_pack_f32(float value) {\n";
    os << "  union { float f; uint32_t u; } bits;\n";
    os << "  bits.f = value;\n";
    os << "  return bits.u;\n";
    os << "}\n\n";
  }

  os << "struct vc4_codegen_kernel_desc {\n";
  os << "  uint32_t code_gpu_addr;\n";
  os << "  uint32_t code_words;\n";
  os << "  uint32_t uniform_words;\n";
  os << "  uint32_t reserved;\n";
  os << "};\n\n";

  os << "struct " << stateName << " {\n";
  os << "  uint32_t magic;\n";
  os << "  uint32_t kernel_count;\n";
  os << "  struct vc4_codegen_kernel_desc kernel_descs[VC4_CODEGEN_PROGRAM_KERNELS];\n";
  os << "  uint32_t launch_count;\n";
  os << "  uint32_t launch_failures;\n";
  os << "  uint32_t heap_alloc_count;\n";
  os << "  uint32_t heap_free_count;\n";
  os << "  uint32_t heap_alloc_failures;\n";
  os << "  uint32_t heap_high_water;\n";
  for (const KernelRecord &kernel : kernels) {
    os << "  uint32_t " << getKernelCodeFieldName(kernel.kernelId)
       << "[sizeof(" << kernel.launchABI.codeSymbol
       << ") / sizeof(uint32_t)] __attribute__((aligned(16)));\n";
    os << "  uint32_t " << getKernelUniformFieldName(kernel.kernelId)
       << "[VC4_CODEGEN_MAX_QPUS][" << kernel.info.uniformWordsPerQPU
       << "] __attribute__((aligned(16)));\n";
    os << "  uint32_t " << getKernelUnifPtrFieldName(kernel.kernelId)
       << "[VC4_CODEGEN_MAX_QPUS] __attribute__((aligned(16)));\n";
  }
  os << "  uint8_t heap[VC4_CODEGEN_PROGRAM_HEAP_BYTES] __attribute__((aligned(4096)));\n";
  os << "};\n\n";

  os << "struct vc4_program {\n";
  os << "  struct " << stateName << " *state;\n";
  os << "  uint32_t handle;\n";
  os << "  uint32_t active_qpus;\n";
  os << "  uint32_t heap_bytes;\n";
  os << "  uint32_t heap_head;\n";
  os << "};\n\n";

  os << "static struct vc4_program g_program_storage;\n";
  os << "static uint32_t g_program_live;\n";
  os << "static uint32_t g_program_allocations;\n";
  os << "static uint32_t g_code_uploads;\n";
  os << "static uint32_t g_last_runtime_launches;\n";
  os << "static uint32_t g_last_launch_failures;\n";
  os << "static uint32_t g_last_heap_alloc_failures;\n";
  os << "static uint32_t g_last_heap_high_water;\n\n";

  os << "static void vc4_codegen_sync_cpu_to_gpu(void) {\n";
  os << "  __asm__ volatile(\"\" ::: \"memory\");\n";
  os << "  PUT32(V3D_L2CACTL, 1u << 2);\n";
  os << "  PUT32(V3D_SLCACTL, 0xffffffffu);\n";
  os << "  __asm__ volatile(\"\" ::: \"memory\");\n";
  os << "}\n\n";
  os << "static void vc4_codegen_sync_gpu_to_cpu(void) {\n";
  os << "  __asm__ volatile(\"\" ::: \"memory\");\n";
  os << "  PUT32(V3D_L2CACTL, 1u << 2);\n";
  os << "  PUT32(V3D_SLCACTL, 0xffffffffu);\n";
  os << "  __asm__ volatile(\"\" ::: \"memory\");\n";
  os << "}\n\n";

  os << "static uint32_t vc4_codegen_align_u32(uint32_t value, uint32_t alignment) {\n";
  os << "  if (alignment <= 1u) return value;\n";
  os << "  return (value + alignment - 1u) & ~(alignment - 1u);\n";
  os << "}\n\n";

  os << "static void vc4_codegen_prepare_v3d_queue(void) {\n";
  os << "  PUT32(V3D_DBCFG, 0u);\n";
  os << "  PUT32(V3D_DBQITE, 0u);\n";
  os << "  PUT32(V3D_DBQITC, 0xffffffffu);\n";
  os << "  PUT32(V3D_L2CACTL, 1u << 2);\n";
  os << "  PUT32(V3D_SLCACTL, 0xffffffffu);\n";
  os << "  PUT32(V3D_SRQCS, (1u << 7) | (1u << 8) | (1u << 16));\n";
  os << "}\n\n";

  os << "static uint32_t vc4_codegen_qpu_completed_count(void) {\n";
  os << "  return (GET32(V3D_SRQCS) >> 16) & 0xffu;\n";
  os << "}\n\n";

  os << "static int vc4_codegen_wait_for_qpus(uint32_t activeQpus, uint32_t completedBefore) {\n";
  os << "  uint32_t spins = VC4_CODEGEN_WAIT_SPINS;\n";
  os << "  while (spins--) {\n";
  os << "    uint32_t completedNow = vc4_codegen_qpu_completed_count();\n";
  os << "    uint32_t completedDelta = (completedNow - completedBefore) & 0xffu;\n";
  os << "    if (completedDelta >= activeQpus)\n";
  os << "      return 0;\n";
  os << "  }\n";
  os << "  return -1;\n";
  os << "}\n\n";

  os << "static int vc4_codegen_use_manual_v3d_queue(void) {\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "static void vc4_codegen_reference_exec(uint32_t codeHostAddr, uint32_t *uniformPointers, uint32_t activeQpus) {\n";
  os << "  gpu_fft_base_exec_direct(codeHostAddr, uniformPointers, activeQpus);\n";
  os << "}\n\n";

  os << "static void *vc4_codegen_deviceptr_to_host(struct vc4_program *program, vc4_deviceptr_t ptr, uint32_t bytes) {\n";
  os << "  if (!program || !program->state || ptr < GPU_BASE) return 0;\n";
  os << "  uintptr_t host = (uintptr_t)(ptr - GPU_BASE);\n";
  os << "  uintptr_t heap = (uintptr_t)&program->state->heap[0];\n";
  os << "  if (host < heap) return 0;\n";
  os << "  uintptr_t offset = host - heap;\n";
  os << "  if (offset > program->heap_bytes) return 0;\n";
  os << "  if (bytes > program->heap_bytes - (uint32_t)offset) return 0;\n";
  os << "  return (void *)host;\n";
  os << "}\n\n";

  os << "static void vc4_codegen_record_heap_failure(struct vc4_program *program) {\n";
  os << "  if (program && program->state) program->state->heap_alloc_failures++;\n";
  os << "}\n\n";

  os << "int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes) {\n";
  os << "  if (!out) return -1;\n";
  os << "  *out = 0;\n";
  os << "  if (requested_bytes > VC4_CODEGEN_PROGRAM_HEAP_BYTES) return -1;\n";
  os << "  if (g_program_live && g_program_storage.state) { *out = &g_program_storage; return 0; }\n";
  os << "  if (qpu_enable(1)) return -1;\n";
  os << "  uint32_t handle = mem_alloc(sizeof(struct " << stateName << "), 4096, GPU_MEM_FLG);\n";
  os << "  if (!handle) { qpu_enable(0); return -1; }\n";
  os << "  uint32_t vc = mem_lock(handle);\n";
  os << "  if (!vc) { mem_free(handle); qpu_enable(0); return -1; }\n";
  os << "  struct " << stateName << " *state = (struct " << stateName << " *)(uintptr_t)(vc - GPU_BASE);\n";
  os << "  memset(state, 0, sizeof(*state));\n";
  os << "  state->magic = VC4_CODEGEN_PROGRAM_MAGIC;\n";
  os << "  state->kernel_count = VC4_CODEGEN_PROGRAM_KERNELS;\n";
  for (const KernelRecord &kernel : kernels) {
    os << "  memcpy(state->" << getKernelCodeFieldName(kernel.kernelId) << ", "
       << kernel.launchABI.codeSymbol << ", sizeof(state->"
       << getKernelCodeFieldName(kernel.kernelId) << "));\n";
    os << "  state->kernel_descs[" << kernel.kernelId << "].code_gpu_addr = GPU_BASE + (uint32_t)(uintptr_t)&state->"
       << getKernelCodeFieldName(kernel.kernelId) << "[0];\n";
    os << "  state->kernel_descs[" << kernel.kernelId << "].code_words = sizeof(state->"
       << getKernelCodeFieldName(kernel.kernelId) << ") / sizeof(uint32_t);\n";
    os << "  state->kernel_descs[" << kernel.kernelId << "].uniform_words = "
       << kernel.info.uniformWordsPerQPU << "u;\n";
  }
  os << "  g_program_storage.state = state;\n";
  os << "  g_program_storage.handle = handle;\n";
  os << "  g_program_storage.active_qpus = VC4_CODEGEN_MAX_QPUS;\n";
  os << "  g_program_storage.heap_bytes = VC4_CODEGEN_PROGRAM_HEAP_BYTES;\n";
  os << "  g_program_storage.heap_head = 0u;\n";
  os << "  g_program_live = 1u;\n";
  os << "  g_last_runtime_launches = 0u;\n";
  os << "  g_last_launch_failures = 0u;\n";
  os << "  g_last_heap_alloc_failures = 0u;\n";
  os << "  g_last_heap_high_water = 0u;\n";
  os << "  g_program_allocations++;\n";
  os << "  g_code_uploads += VC4_CODEGEN_PROGRAM_KERNELS;\n";
  os << "  vc4_codegen_sync_cpu_to_gpu();\n";
  os << "  printk(\"VC4_RUNTIME_LAYOUT program_bytes=%d static_bytes=%d heap_offset=%d heap_bytes=%d kernels=%d program_allocations=%d code_uploads=%d\\n\", (int)VC4_CODEGEN_PROGRAM_TOTAL_BYTES, (int)VC4_CODEGEN_PROGRAM_STATIC_BYTES, (int)offsetof(struct " << stateName << ", heap), (int)VC4_CODEGEN_PROGRAM_HEAP_BYTES, (int)VC4_CODEGEN_PROGRAM_KERNELS, (int)g_program_allocations, (int)g_code_uploads);\n";
  os << "  *out = &g_program_storage;\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "void vc4_program_destroy(struct vc4_program *program) {\n";
  os << "  if (!program || !program->state) return;\n";
  os << "  g_last_runtime_launches = program->state->launch_count;\n";
  os << "  g_last_launch_failures = program->state->launch_failures;\n";
  os << "  g_last_heap_alloc_failures = program->state->heap_alloc_failures;\n";
  os << "  g_last_heap_high_water = program->state->heap_high_water;\n";
  os << "  printk(\"VC4_HEAP_STATS allocs=%d frees=%d failures=%d high_water=%d heap_alloc_failures=%d launch_failures=%d runtime_launches=%d code_uploads=%d program_allocations=%d\\n\", (int)program->state->heap_alloc_count, (int)program->state->heap_free_count, (int)program->state->heap_alloc_failures, (int)program->state->heap_high_water, (int)program->state->heap_alloc_failures, (int)program->state->launch_failures, (int)program->state->launch_count, (int)g_code_uploads, (int)g_program_allocations);\n";
  os << "  mem_unlock(program->handle);\n";
  os << "  mem_free(program->handle);\n";
  os << "  qpu_enable(0);\n";
  os << "  memset(program, 0, sizeof(*program));\n";
  os << "  g_program_live = 0u;\n";
  os << "}\n\n";

  os << "int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes) {\n";
  os << "  if (!program || !program->state || !out || bytes == 0u) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  uint32_t offset = vc4_codegen_align_u32(program->heap_head, 16u);\n";
  os << "  if (offset > program->heap_bytes || bytes > program->heap_bytes - offset) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  program->heap_head = vc4_codegen_align_u32(offset + bytes, 16u);\n";
  os << "  if (program->heap_head > program->state->heap_high_water) program->state->heap_high_water = program->heap_head;\n";
  os << "  program->state->heap_alloc_count++;\n";
  os << "  *out = GPU_BASE + (uint32_t)(uintptr_t)&program->state->heap[offset];\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr) {\n";
  os << "  void *host = vc4_codegen_deviceptr_to_host(program, ptr, 1u);\n";
  os << "  if (!host) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  program->state->heap_free_count++;\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst, const void *src, uint32_t bytes) {\n";
  os << "  if (!src && bytes) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  void *dstHost = vc4_codegen_deviceptr_to_host(program, dst, bytes);\n";
  os << "  if (!dstHost) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  memcpy(dstHost, src, bytes);\n";
  os << "  vc4_codegen_sync_cpu_to_gpu();\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int vc4MemcpyDtoH(struct vc4_program *program, void *dst, vc4_deviceptr_t src, uint32_t bytes) {\n";
  os << "  if (!dst && bytes) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  void *srcHost = vc4_codegen_deviceptr_to_host(program, src, bytes);\n";
  os << "  if (!srcHost) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  vc4_codegen_sync_gpu_to_cpu();\n";
  os << "  memcpy(dst, srcHost, bytes);\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int vc4MemcpyDtoD(struct vc4_program *program, vc4_deviceptr_t dst, vc4_deviceptr_t src, uint32_t bytes) {\n";
  os << "  void *dstHost = vc4_codegen_deviceptr_to_host(program, dst, bytes);\n";
  os << "  void *srcHost = vc4_codegen_deviceptr_to_host(program, src, bytes);\n";
  os << "  if (!dstHost || !srcHost) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  memmove(dstHost, srcHost, bytes);\n";
  os << "  vc4_codegen_sync_cpu_to_gpu();\n";
  os << "  return 0;\n";
  os << "}\n\n";

  os << "int vc4MemsetD8(struct vc4_program *program, vc4_deviceptr_t dst, uint8_t value, uint32_t bytes) {\n";
  os << "  void *dstHost = vc4_codegen_deviceptr_to_host(program, dst, bytes);\n";
  os << "  if (!dstHost) { vc4_codegen_record_heap_failure(program); return -1; }\n";
  os << "  memset(dstHost, value, bytes);\n";
  os << "  vc4_codegen_sync_cpu_to_gpu();\n";
  os << "  return 0;\n";
  os << "}\n\n";

  for (const KernelRecord &launchKernel : kernels) {
    const LaunchABIModel &kernelABI = launchKernel.launchABI;
    appendLauncherPrototype(os, kernelABI);
    os << " {\n";
    os << "  (void)grid;\n";
    os << "  (void)block;\n";
    os << "  if (!program || !program->state) return -1;\n";
    os << "  uint32_t activeQpus = program->active_qpus;\n";
    os << "  if (activeQpus == 0u || activeQpus > VC4_CODEGEN_MAX_QPUS) { program->state->launch_failures++; return -1; }\n";
    appendLauncherUniformLayoutComment(os, kernelABI);
    os << "  for (uint32_t qpu = 0; qpu < activeQpus; ++qpu) {\n";
    for (int64_t index = 0; index != kernelABI.uniformWordsPerQPU; ++index) {
      if (const LaunchABIArgumentModel *arg =
              findLaunchABIArgumentForUniformIndex(kernelABI, index)) {
        os << "    program->state->" << getKernelUniformFieldName(launchKernel.kernelId)
           << "[qpu][" << index << "] = "
           << getArgumentUniformExpression(*arg) << "; /* arg "
           << arg->name << " */\n";
        continue;
      }
      if (const LaunchABIBuiltinModel *builtin =
              findLaunchABIBuiltinForUniformIndex(kernelABI, index)) {
        std::optional<std::string> expression = getBuiltinUniformExpression(*builtin);
        if (!expression)
          return emitLaunchABIModelError(
              launchKernel.func, llvm::Twine("builtin '") + builtin->name +
                            "' has unsupported kind for launcher uniform packing");
        os << "    program->state->" << getKernelUniformFieldName(launchKernel.kernelId)
           << "[qpu][" << index << "] = " << *expression
           << "; /* builtin " << builtin->name << " */\n";
        continue;
      }
      return emitLaunchABIModelError(
          launchKernel.func,
          llvm::Twine("missing launcher uniform assignment for index ") +
              std::to_string(index));
    }
    os << "    program->state->" << getKernelUnifPtrFieldName(launchKernel.kernelId)
       << "[qpu] = GPU_BASE + (uint32_t)(uintptr_t)&program->state->"
       << getKernelUniformFieldName(launchKernel.kernelId) << "[qpu][0];\n";
    os << "  }\n";
    os << "  vc4_codegen_sync_cpu_to_gpu();\n";
    os << "  printk(\"VC4_KERNEL_LAUNCH name=" << kernelABI.publicName
       << " kernel_id=" << launchKernel.kernelId
       << " public_name=" << kernelABI.publicName
       << " schedule_mode=independent_vector requests=%d waves=1 sequence=%d runtime_launches=%d launch_failures=%d\\n\", (int)activeQpus, (int)program->state->launch_count, (int)program->state->launch_count, (int)program->state->launch_failures);\n";
    os << "  if (vc4_codegen_use_manual_v3d_queue()) {\n";
    os << "    vc4_codegen_prepare_v3d_queue();\n";
    os << "    uint32_t completedBefore = vc4_codegen_qpu_completed_count();\n";
    os << "    for (uint32_t qpu = 0; qpu < activeQpus; ++qpu) {\n";
    os << "      PUT32(V3D_SRQUA, program->state->"
       << getKernelUnifPtrFieldName(launchKernel.kernelId) << "[qpu]);\n";
    os << "      PUT32(V3D_SRQPC, program->state->kernel_descs["
       << launchKernel.kernelId << "].code_gpu_addr);\n";
    os << "    }\n";
    os << "    if (vc4_codegen_wait_for_qpus(activeQpus, completedBefore) < 0) {\n";
    os << "      program->state->launch_failures++;\n";
    os << "      printk(\"VC4_KERNEL_LAUNCH name=" << kernelABI.publicName
       << " kernel_id=" << launchKernel.kernelId
       << " status=timeout runtime_launches=%d launch_failures=%d\\n\", (int)program->state->launch_count, (int)program->state->launch_failures);\n";
    os << "      return -1;\n";
    os << "    }\n";
    os << "  } else {\n";
    os << "    vc4_codegen_reference_exec((uint32_t)(uintptr_t)&program->state->"
       << getKernelCodeFieldName(launchKernel.kernelId)
       << "[0], (uint32_t *)program->state->"
       << getKernelUnifPtrFieldName(launchKernel.kernelId)
       << ", activeQpus);\n";
    os << "  }\n";
    os << "  program->state->launch_count++;\n";
    os << "  printk(\"VC4_KERNEL_LAUNCH name=" << kernelABI.publicName
       << " kernel_id=" << launchKernel.kernelId
       << " status=complete runtime_launches=%d launch_failures=%d\\n\", (int)program->state->launch_count, (int)program->state->launch_failures);\n";
    os << "  return 0;\n";
    os << "}\n\n";
  }

  os << "uint32_t vc4_codegen_program_allocations(void) { return g_program_allocations; }\n";
  os << "uint32_t vc4_codegen_runtime_launches(void) { return (g_program_live && g_program_storage.state) ? g_program_storage.state->launch_count : g_last_runtime_launches; }\n";
  os << "uint32_t vc4_codegen_code_uploads(void) { return g_code_uploads; }\n";
  os << "uint32_t vc4_codegen_launch_failures(void) { return (g_program_live && g_program_storage.state) ? g_program_storage.state->launch_failures : g_last_launch_failures; }\n";
  os << "uint32_t vc4_codegen_heap_alloc_failures(void) { return (g_program_live && g_program_storage.state) ? g_program_storage.state->heap_alloc_failures : g_last_heap_alloc_failures; }\n";
  os << "uint32_t vc4_codegen_heap_high_water(void) { return (g_program_live && g_program_storage.state) ? g_program_storage.state->heap_high_water : g_last_heap_high_water; }\n";
  for (const KernelRecord &kernel : kernels) {
    os << "uint32_t " << getRuntimeAllocationsFunctionName(kernel.launchABI)
       << "(void) { return vc4_codegen_program_allocations(); }\n";
    os << "uint32_t " << getRuntimeLaunchesFunctionName(kernel.launchABI)
       << "(void) { return vc4_codegen_runtime_launches(); }\n";
    os << "uint32_t " << getRuntimeCapacityFunctionName(kernel.launchABI)
       << "(void) { return g_program_live ? g_program_storage.heap_bytes : 0u; }\n";
  }
  os.flush();

  return writeBundleFile(firstKernel.func.getOperation(), bundleDir,
                         "kernel_launch.c",
                         [&](llvm::raw_ostream &fileOS) { fileOS << source; });
}

static void appendLayoutByteRange(llvm::raw_ostream &os, uint64_t offset,
                                  uint64_t size) {
  os << "{\"offset\": " << offset << ", \"size\": " << size
     << ", \"alignment\": " << kVC4ProgramLayoutAlignment << "}";
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
                           os << "  \"heap_offset\": " << layout.heapOffset << ",\n";
                           os << "  \"heap_offset_bytes\": " << layout.heapOffset << ",\n";
                           os << "  \"heap_bytes\": " << layout.heapBytes << ",\n";
                           os << "  \"heap_size_bytes\": " << layout.heapBytes << ",\n";
                           os << "  \"heap\": ";
                           appendLayoutByteRange(os, layout.heapOffset,
                                                 layout.heapBytes);
                           os << ",\n";
                           os << "  \"regions\": [\n";
                           for (size_t i = 0; i != layout.regions.size(); ++i)
                             appendLayoutRegionJSON(os, layout.regions[i],
                                                    i + 1 != layout.regions.size());
                           os << "  ],\n";
                           os << "  \"kernels\": [\n";
                           for (size_t i = 0; i != layout.kernels.size(); ++i)
                             appendKernelLayoutJSON(os, layout.kernels[i],
                                                    i + 1 != layout.kernels.size());
                           os << "  ]\n";
                           os << "}\n";
                         });
}

static void appendManifestLaunchABIArgument(llvm::raw_ostream &os,
                                            const LaunchABIArgumentModel &arg,
                                            bool trailingComma) {
  os << "        {\"name\": ";
  appendJSONEscapedString(os, arg.name);
  os << ", \"kind\": ";
  appendJSONEscapedString(os, arg.kind == LaunchABIArgumentKind::Scalar
                                  ? "scalar" : "buffer");
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
  os << "        {\"name\": ";
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
  os << "      \"scheduled_sink_ops\": " << kernel.info.scheduledOpCount << ",\n";
  os << "      \"uniform_words_per_request\": " << kernel.info.uniformWordsPerQPU << ",\n";
  os << "      \"uniform_words_per_qpu\": " << kernel.info.uniformWordsPerQPU << ",\n";
  os << "      \"max_requests_per_wave\": " << kVC4MaxRequestsPerWave << ",\n";
  os << "      \"tail_policy\": ";
  appendJSONEscapedString(os, kernel.launchABI.tailPolicy);
  os << ",\n";
  os << "      \"schedule_mode\": \"independent_vector\",\n";
  os << "      \"args\": [\n";
  for (size_t i = 0; i != kernel.launchABI.arguments.size(); ++i)
    appendManifestLaunchABIArgument(os, kernel.launchABI.arguments[i],
                                    i + 1 != kernel.launchABI.arguments.size());
  os << "      ],\n";
  os << "      \"builtins\": [\n";
  for (size_t i = 0; i != kernel.launchABI.builtins.size(); ++i)
    appendManifestLaunchABIBuiltin(os, kernel.launchABI.builtins[i],
                                   i + 1 != kernel.launchABI.builtins.size());
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
                           for (size_t i = 0; i != kernels.size(); ++i)
                             appendManifestKernelEntry(os, kernels[i],
                                                       i + 1 != kernels.size());
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
  if (vc4Modules.size() != 1)
    return module.emitError()
           << "expected exactly one vc4.module for artifact emission";

  mlir::vc4::ModuleOp vc4Module = vc4Modules.front();
  llvm::SmallVector<KernelRecord, 2> kernels;
  if (failed(collectProgramKernels(vc4Module, kernels)))
    return failure();

  std::error_code ec = llvm::sys::fs::create_directories(bundleDir);
  if (ec)
    return module.emitError() << "failed to create artifact bundle directory '"
                              << bundleDir << "': " << ec.message();

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
