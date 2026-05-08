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
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <optional>
#include <system_error>

using namespace mlir;

namespace {

struct KernelRecord {
  explicit KernelRecord(mlir::vc4::FuncOp func) : func(func) {}

  mlir::vc4::FuncOp func;
  mlir::vc4::VC4ArtifactKernelInfo info;
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
collectSingleKernel(mlir::vc4::ModuleOp vc4Module,
                    llvm::SmallVectorImpl<KernelRecord> &kernels) {
  vc4Module.walk([&](mlir::vc4::FuncOp func) {
    if (isScheduledQPUKernel(func))
      kernels.emplace_back(func);
  });

  if (kernels.size() != 1) {
    return vc4Module.emitOpError()
           << "expected exactly one eligible VC4 QPU kernel: exactly one "
              "kernel qpu scheduled vc4.func for artifact emission";
  }

  if (failed(populateLaunchABIInfo(kernels.front())))
    return failure();
  if (failed(populateScheduledSinkInfo(kernels.front())))
    return failure();
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

static bool isMinimalQASMNopBundle(mlir::vc4::QPUBundleOp bundle,
                                      mlir::vc4::QPUSignal expectedSignal) {
  return bundle && bundle.getSig() == expectedSignal &&
         bundle.getOpAdd() == mlir::vc4::AddOpcode::nop &&
         bundle.getOpMul() == mlir::vc4::MulOpcode::nop;
}

static LogicalResult verifyMinimalThreadEndQASM(KernelRecord &kernel) {
  llvm::ArrayRef<mlir::Operation *> stream = kernel.scheduledStream;
  if (stream.size() != 3) {
    return kernel.func.emitOpError()
           << "is not directly emittable by the minimal qasm emitter: "
              "expected exactly three vc4.qpu.bundle nop slots encoding "
              "thrend, nop, nop";
  }

  if (!isMinimalQASMNopBundle(
          llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[0]),
          mlir::vc4::QPUSignal::thrend) ||
      !isMinimalQASMNopBundle(
          llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[1]),
          mlir::vc4::QPUSignal::none) ||
      !isMinimalQASMNopBundle(
          llvm::dyn_cast<mlir::vc4::QPUBundleOp>(stream[2]),
          mlir::vc4::QPUSignal::none)) {
    return kernel.func.emitOpError()
           << "is not directly emittable by the minimal qasm emitter: "
              "expected exactly three vc4.qpu.bundle nop slots encoding "
              "thrend, nop, nop";
  }

  return success();
}

static LogicalResult writeQASM(KernelRecord &kernel,
                               llvm::StringRef bundleDir) {
  if (failed(verifyMinimalThreadEndQASM(kernel)))
    return failure();

  return writeBundleFile(kernel.func.getOperation(), bundleDir, "kernel.qasm",
                         [&](llvm::raw_ostream &os) {
                           os << "thrend\n";
                           os << "nop\n";
                           os << "nop\n";
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
  llvm::SmallVector<mlir::vc4::ModuleOp, 1> vc4Modules;
  module.walk([&](mlir::vc4::ModuleOp vc4Module) {
    vc4Modules.push_back(vc4Module);
  });
  if (vc4Modules.size() != 1) {
    return module.emitError()
           << "expected exactly one vc4.module for artifact emission";
  }

  llvm::SmallVector<KernelRecord, 1> kernels;
  if (failed(collectSingleKernel(vc4Modules.front(), kernels)))
    return failure();
  KernelRecord &kernel = kernels.front();

  std::error_code ec = llvm::sys::fs::create_directories(bundleDir);
  if (ec) {
    return module.emitError() << "failed to create artifact bundle directory '"
                              << bundleDir << "': " << ec.message();
  }

  if (failed(ensureAdjacentVC4ASMTemplates(module.getOperation(), bundleDir)))
    return failure();

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
