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
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <system_error>

using namespace mlir;

namespace {

struct KernelRecord {
  explicit KernelRecord(mlir::vc4::FuncOp func) : func(func) {}

  mlir::vc4::FuncOp func;
  mlir::vc4::VC4ArtifactKernelInfo info;
};

static bool isScheduledQPUKernelWithLaunchABI(mlir::vc4::FuncOp func) {
  if (func.isExternal())
    return false;

  std::optional<mlir::vc4::ExecutionDomain> domain = func.getDomain();
  std::optional<mlir::vc4::FunctionForm> form = func.getForm();
  if (!domain || *domain != mlir::vc4::ExecutionDomain::qpu || !form ||
      *form != mlir::vc4::FunctionForm::scheduled)
    return false;

  if (!func->hasAttr("kernel"))
    return false;

  return static_cast<bool>(
      func->getAttrOfType<mlir::DictionaryAttr>("vc4.launch_abi"));
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

static LogicalResult appendScheduledSinkOp(mlir::Operation *op,
                                           unsigned &scheduledOpCount);

static LogicalResult appendBranchDelaySlotSinkOps(
    mlir::vc4::QPUBranchOp branch, unsigned &scheduledOpCount) {
  if (!branch.getDelaySlots().hasOneBlock()) {
    return branch.emitOpError()
           << "requires exactly one delay-slot block for artifact emission";
  }

  for (mlir::Operation &delaySlotOp : branch.getDelaySlots().front()) {
    if (mlir::failed(appendScheduledSinkOp(&delaySlotOp, scheduledOpCount)))
      return failure();
  }
  return success();
}

static LogicalResult appendScheduledSinkOp(mlir::Operation *op,
                                           unsigned &scheduledOpCount) {
  if (llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPULDIOp,
                mlir::vc4::QPUSemaOp>(op)) {
    ++scheduledOpCount;
    return success();
  }

  if (auto branch = llvm::dyn_cast<mlir::vc4::QPUBranchOp>(op)) {
    ++scheduledOpCount;
    return appendBranchDelaySlotSinkOps(branch, scheduledOpCount);
  }

  return op->emitOpError()
         << "is not a supported final scheduled VC4 QPU sink op for "
            "artifact emission";
}

static LogicalResult populateScheduledSinkInfo(KernelRecord &kernel) {
  if (!kernel.func.getBody().hasOneBlock()) {
    return kernel.func.emitOpError()
           << "requires a single top-level block for artifact emission";
  }

  unsigned scheduledOpCount = 0;
  for (mlir::Operation &op : kernel.func.getBody().front()) {
    if (mlir::failed(appendScheduledSinkOp(&op, scheduledOpCount)))
      return failure();
  }

  kernel.info.scheduledOpCount = scheduledOpCount;
  return success();
}

static LogicalResult populateLaunchABIInfo(KernelRecord &kernel) {
  auto launchABI =
      kernel.func->getAttrOfType<mlir::DictionaryAttr>("vc4.launch_abi");
  if (!launchABI) {
    return kernel.func.emitOpError()
           << "requires a dictionary \"vc4.launch_abi\" attribute";
  }

  auto publicName =
      llvm::dyn_cast_or_null<mlir::StringAttr>(launchABI.get("public_name"));
  if (!publicName || publicName.getValue().empty()) {
    return kernel.func.emitOpError()
           << "requires \"vc4.launch_abi\" public_name for artifact emission";
  }

  if (!isCIdentifier(publicName.getValue())) {
    return kernel.func.emitOpError()
           << "requires \"vc4.launch_abi\" public_name to be a C identifier "
              "for the skeleton launcher";
  }

  auto uniformWords = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
      launchABI.get("uniform_words_per_qpu"));
  if (!uniformWords || uniformWords.getInt() <= 0) {
    return kernel.func.emitOpError()
           << "requires positive \"vc4.launch_abi\" uniform_words_per_qpu";
  }

  kernel.info.symbolName = kernel.func.getSymName().str();
  kernel.info.publicName = publicName.getValue().str();
  kernel.info.uniformWordsPerQPU = uniformWords.getInt();
  return success();
}

static LogicalResult
collectSingleKernel(mlir::ModuleOp module,
                    llvm::SmallVectorImpl<KernelRecord> &kernels) {
  module.walk([&](mlir::vc4::FuncOp func) {
    if (isScheduledQPUKernelWithLaunchABI(func))
      kernels.emplace_back(func);
  });

  if (kernels.size() != 1) {
    return module.emitError()
           << "expected exactly one eligible VC4 QPU kernel with domain = "
              "#vc4.execution_domain<qpu>, form = "
              "#vc4.function_form<scheduled>, kernel marker, and "
              "\"vc4.launch_abi\"";
  }

  if (failed(populateLaunchABIInfo(kernels.front())))
    return failure();
  if (failed(populateScheduledSinkInfo(kernels.front())))
    return failure();
  return success();
}

static LogicalResult writeBundleFile(
    mlir::Operation *diagOp, llvm::StringRef bundleDir, llvm::StringRef fileName,
    llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  llvm::SmallString<256> path(bundleDir);
  llvm::sys::path::append(path, fileName);

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

static LogicalResult writeQASM(KernelRecord &kernel,
                               llvm::StringRef bundleDir) {
  return writeBundleFile(kernel.func.getOperation(), bundleDir, "kernel.qasm",
                         [&](llvm::raw_ostream &os) {
                           os << "# vc4-codegen skeleton qasm\n";
                           os << "# kernel: " << kernel.info.symbolName << "\n";
                           os << "# public_name: " << kernel.info.publicName
                              << "\n";
                           os << "# scheduled_sink_ops: "
                              << kernel.info.scheduledOpCount << "\n";
                         });
}

static LogicalResult writeLauncherHeader(KernelRecord &kernel,
                                         llvm::StringRef bundleDir) {
  return writeBundleFile(kernel.func.getOperation(), bundleDir,
                         "kernel_launch.h", [&](llvm::raw_ostream &os) {
                           os << "#ifndef VC4_CODEGEN_KERNEL_LAUNCH_H\n";
                           os << "#define VC4_CODEGEN_KERNEL_LAUNCH_H\n\n";
                           os << "#ifdef __cplusplus\n";
                           os << "extern \"C\" {\n";
                           os << "#endif\n\n";
                           os << "int " << kernel.info.publicName << "(void);\n\n";
                           os << "#ifdef __cplusplus\n";
                           os << "}\n";
                           os << "#endif\n\n";
                           os << "#endif // VC4_CODEGEN_KERNEL_LAUNCH_H\n";
                         });
}

static LogicalResult writeLauncherSource(KernelRecord &kernel,
                                         llvm::StringRef bundleDir) {
  return writeBundleFile(kernel.func.getOperation(), bundleDir,
                         "kernel_launch.c", [&](llvm::raw_ostream &os) {
                           os << "#include \"kernel_launch.h\"\n\n";
                           os << "int " << kernel.info.publicName << "(void) {\n";
                           os << "  return 0;\n";
                           os << "}\n";
                         });
}

static LogicalResult writeManifest(KernelRecord &kernel,
                                   llvm::StringRef bundleDir) {
  return writeBundleFile(kernel.func.getOperation(), bundleDir, "manifest.json",
                         [&](llvm::raw_ostream &os) {
                           os << "{\n";
                           os << "  \"kind\": \"vc4-codegen-artifact-bundle-v0\",\n";
                           os << "  \"bundle_format\": \"vc4-codegen-artifact-bundle-v0\",\n";
                           os << "  \"kernel\": ";
                           appendJSONEscapedString(os, kernel.info.symbolName);
                           os << ",\n";
                           os << "  \"symbol_name\": ";
                           appendJSONEscapedString(os, kernel.info.symbolName);
                           os << ",\n";
                           os << "  \"public_name\": ";
                           appendJSONEscapedString(os, kernel.info.publicName);
                           os << ",\n";
                           os << "  \"scheduled_sink_ops\": "
                              << kernel.info.scheduledOpCount << ",\n";
                           os << "  \"uniform_words_per_qpu\": "
                              << kernel.info.uniformWordsPerQPU << ",\n";
                           os << "  \"kernel_info\": {\n";
                           os << "    \"symbol_name\": ";
                           appendJSONEscapedString(os, kernel.info.symbolName);
                           os << ",\n";
                           os << "    \"public_name\": ";
                           appendJSONEscapedString(os, kernel.info.publicName);
                           os << ",\n";
                           os << "    \"uniform_words_per_qpu\": "
                              << kernel.info.uniformWordsPerQPU << ",\n";
                           os << "    \"scheduled_sink_ops\": "
                              << kernel.info.scheduledOpCount << "\n";
                           os << "  },\n";
                           os << "  \"artifacts\": [\n";
                           os << "    \"kernel.qasm\",\n";
                           os << "    \"kernel_launch.c\",\n";
                           os << "    \"kernel_launch.h\"\n";
                           os << "  ]\n";
                           os << "}\n";
                         });
}

} // namespace

LogicalResult mlir::vc4::emitVC4ArtifactBundle(mlir::ModuleOp module,
                                               llvm::StringRef bundleDir) {
  llvm::SmallVector<KernelRecord, 1> kernels;
  if (failed(collectSingleKernel(module, kernels)))
    return failure();
  KernelRecord &kernel = kernels.front();

  std::error_code ec = llvm::sys::fs::create_directories(bundleDir);
  if (ec) {
    return module.emitError() << "failed to create artifact bundle directory '"
                              << bundleDir << "': " << ec.message();
  }

  if (failed(writeQASM(kernel, bundleDir)))
    return failure();
  if (failed(writeLauncherSource(kernel, bundleDir)))
    return failure();
  if (failed(writeLauncherHeader(kernel, bundleDir)))
    return failure();
  if (failed(writeManifest(kernel, bundleDir)))
    return failure();

  return success();
}
