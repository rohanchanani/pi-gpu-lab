//===- EmitBundle.cpp - VC4 artifact bundle emission ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Target/VC4/EmitBundle.h"

#include "vc4/Dialect/VC4/IR/VC4Ops.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <system_error>

using namespace mlir;

namespace {

static const char kLaunchAbiAttrName[] = "vc4.launch_abi";

static bool isScheduledQPUDomain(mlir::vc4::FuncOp func) {
 std::optional<mlir::vc4::ExecutionDomain> domain = func.getDomain();
 std::optional<mlir::vc4::FunctionForm> form = func.getForm();
 return domain && *domain == mlir::vc4::ExecutionDomain::qpu && form &&
 *form == mlir::vc4::FunctionForm::scheduled;
}

static bool isEligibleScheduledQPUKernel(mlir::vc4::FuncOp func) {
 if (func.isExternal())
 return false;
 if (!func->hasAttr("kernel"))
 return false;
 if (!isScheduledQPUDomain(func))
 return false;
 return static_cast<bool>(
 func->getAttrOfType<mlir::DictionaryAttr>(kLaunchAbiAttrName));
}

static bool isAllowedFinalScheduledOp(mlir::Operation *op) {
 return llvm::isa<mlir::vc4::QPUBundleOp, mlir::vc4::QPUBranchOp,
 mlir::vc4::QPULDIOp, mlir::vc4::QPUSemaOp>(op);
}

static LogicalResult
collectExactlyOneEligibleKernel(mlir::ModuleOp module,
 llvm::SmallVectorImpl<mlir::vc4::FuncOp>
 &kernels) {
 module.walk([&](mlir::vc4::FuncOp func) {
 if (isEligibleScheduledQPUKernel(func))
 kernels.push_back(func);
 });

 if (kernels.size() == 1)
 return success();

 module.emitError() << "vc4-codegen requires exactly one eligible kernel "
 "(non-external vc4.func with kernel marker, "
 "domain = #vc4.execution_domain<qpu>, "
 "form = #vc4.function_form<scheduled>, and "
 "'vc4.launch_abi'); found "
 << kernels.size();
 return failure();
}

static LogicalResult verifyFinalScheduledSinkBody(mlir::vc4::FuncOp func) {
 if (!func.getBody().hasOneBlock()) {
 return func.emitOpError()
 << "is not directly emittable by vc4-codegen: scheduled QPU "
 "kernel body must have exactly one top-level block";
 }

 mlir::Operation *unsupportedOp = nullptr;
 func.walk([&](mlir::Operation *op) {
 if (op == func.getOperation())
 return mlir::WalkResult::advance();
 if (isAllowedFinalScheduledOp(op))
 return mlir::WalkResult::advance();
 unsupportedOp = op;
 return mlir::WalkResult::interrupt();
 });

 if (!unsupportedOp)
 return success();

 return unsupportedOp->emitOpError()
 << "is not directly emittable by vc4-codegen: expected only final "
 "scheduled QPU sink operations (vc4.qpu.bundle, vc4.qpu.ldi, "
 "vc4.qpu.sema, or vc4.qpu.branch)";
}

static llvm::StringRef getLaunchPublicName(mlir::vc4::FuncOp kernel) {
 auto launchAbi =
 kernel->getAttrOfType<mlir::DictionaryAttr>(kLaunchAbiAttrName);
 if (!launchAbi)
 return kernel.getSymName();

 auto publicName =
 llvm::dyn_cast_or_null<mlir::StringAttr>(launchAbi.get("public_name"));
 if (!publicName)
 return kernel.getSymName();

 return publicName.getValue();
}

static bool isCIdentifierHead(char c) {
 return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

static bool isCIdentifierTail(char c) {
 return isCIdentifierHead(c) || (c >= '0' && c <= '9');
}

static std::string sanitizeCIdentifier(llvm::StringRef name) {
 std::string result;
 result.reserve(name.size() + 1);

 for (char c : name) {
 if (result.empty()) {
 result.push_back(isCIdentifierHead(c) ? c : '_');
 if (c >= '0' && c <= '9')
 result.push_back(c);
 continue;
 }
 result.push_back(isCIdentifierTail(c) ? c : '_');
 }

 if (result.empty())
 return "vc4_kernel_launch";
 return result;
}

static void writeJSONString(llvm::raw_ostream &os, llvm::StringRef value) {
 static const char hex[] = "0123456789abcdef";
 os << '"';
 for (char c : value) {
 unsigned char uc = static_cast<unsigned char>(c);
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
 if (uc < 0x20) {
 os << "\\u00";
 os << hex[(uc >> 4) & 0xf] << hex[uc & 0xf];
 } else {
 os << c;
 }
 break;
 }
 }
 os << '"';
}

static llvm::SmallString<128> getBundleFilePath(llvm::StringRef bundleDir,
 llvm::StringRef fileName) {
 llvm::SmallString<128> path(bundleDir);
 llvm::sys::path::append(path, fileName);
 return path;
}

static LogicalResult emitOpenError(mlir::ModuleOp module,
 llvm::StringRef fileName,
 const std::error_code &ec) {
 return module.emitError() << "failed to open '" << fileName
 << "' for writing: " << ec.message();
}

static LogicalResult writeKernelQASM(mlir::ModuleOp module,
 mlir::vc4::FuncOp kernel,
 llvm::StringRef bundleDir) {
 llvm::SmallString<128> path = getBundleFilePath(bundleDir, "kernel.qasm");
 std::error_code ec;
 llvm::raw_fd_ostream os(path.str(), ec, llvm::sys::fs::OF_Text);
 if (ec)
 return emitOpenError(module, path.str(), ec);

 os << "; vc4-codegen placeholder qasm\n";
 os << "; kernel: " << kernel.getSymName() << "\n";
 os << "; real QPU instruction emission is implemented by later slices\n";
 return success();
}

static LogicalResult writeKernelLaunchHeader(mlir::ModuleOp module,
 mlir::vc4::FuncOp kernel,
 llvm::StringRef bundleDir) {
 llvm::SmallString<128> path =
 getBundleFilePath(bundleDir, "kernel_launch.h");
 std::error_code ec;
 llvm::raw_fd_ostream os(path.str(), ec, llvm::sys::fs::OF_Text);
 if (ec)
 return emitOpenError(module, path.str(), ec);

 std::string publicCName = sanitizeCIdentifier(getLaunchPublicName(kernel));

 os << "#ifndef VC4_GENERATED_KERNEL_LAUNCH_H\n";
 os << "#define VC4_GENERATED_KERNEL_LAUNCH_H\n\n";
 os << "#ifdef __cplusplus\n";
 os << "extern \"C\" {\n";
 os << "#endif\n\n";
 os << "int " << publicCName << "(void);\n\n";
 os << "#ifdef __cplusplus\n";
 os << "}\n";
 os << "#endif\n\n";
 os << "#endif /* VC4_GENERATED_KERNEL_LAUNCH_H */\n";
 return success();
}

static LogicalResult writeKernelLaunchC(mlir::ModuleOp module,
 mlir::vc4::FuncOp kernel,
 llvm::StringRef bundleDir) {
 llvm::SmallString<128> path =
 getBundleFilePath(bundleDir, "kernel_launch.c");
 std::error_code ec;
 llvm::raw_fd_ostream os(path.str(), ec, llvm::sys::fs::OF_Text);
 if (ec)
 return emitOpenError(module, path.str(), ec);

 std::string publicCName = sanitizeCIdentifier(getLaunchPublicName(kernel));

 os << "#include \"kernel_launch.h\"\n\n";
 os << "int " << publicCName << "(void) {\n";
 os << " return 0;\n";
 os << "}\n";
 return success();
}

static LogicalResult writeManifest(mlir::ModuleOp module,
 mlir::vc4::FuncOp kernel,
 llvm::StringRef bundleDir) {
 llvm::SmallString<128> path = getBundleFilePath(bundleDir, "manifest.json");
 std::error_code ec;
 llvm::raw_fd_ostream os(path.str(), ec, llvm::sys::fs::OF_Text);
 if (ec)
 return emitOpenError(module, path.str(), ec);

 os << "{\n";
 os << " \"schema_version\": 1,\n";
 os << " \"tool\": \"vc4-codegen\",\n";
 os << " \"kernel\": ";
 writeJSONString(os, kernel.getSymName());
 os << ",\n";
 os << " \"public_name\": ";
 writeJSONString(os, getLaunchPublicName(kernel));
 os << ",\n";
 os << " \"artifacts\": {\n";
 os << " \"qasm\": \"kernel.qasm\",\n";
 os << " \"launcher_c\": \"kernel_launch.c\",\n";
 os << " \"launcher_h\": \"kernel_launch.h\"\n";
 os << " },\n";
 os << " \"placeholder\": true\n";
 os << "}\n";
 return success();
}

} // namespace

LogicalResult mlir::vc4::emitVC4Bundle(mlir::ModuleOp module,
 const EmitBundleOptions &options) {
 if (options.emitBundleDir.empty())
 return module.emitError() << "--emit-bundle directory must not be empty";

 llvm::SmallVector<mlir::vc4::FuncOp, 2> kernels;
 if (failed(collectExactlyOneEligibleKernel(module, kernels)))
 return failure();

 mlir::vc4::FuncOp kernel = kernels.front();
 if (failed(verifyFinalScheduledSinkBody(kernel)))
 return failure();

 if (std::error_code ec =
 llvm::sys::fs::create_directories(options.emitBundleDir)) {
 return module.emitError()
 << "failed to create --emit-bundle directory '"
 << options.emitBundleDir << "': " << ec.message();
 }

 if (failed(writeKernelQASM(module, kernel, options.emitBundleDir)))
 return failure();
 if (failed(writeKernelLaunchHeader(module, kernel, options.emitBundleDir)))
 return failure();
 if (failed(writeKernelLaunchC(module, kernel, options.emitBundleDir)))
 return failure();
 if (failed(writeManifest(module, kernel, options.emitBundleDir)))
 return failure();

 return success();
}
