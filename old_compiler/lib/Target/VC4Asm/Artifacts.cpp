#include "vc4/Target/VC4Asm/VC4Asm.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "vc4/Target/KernelModel.h"

namespace mlir::vc4 {

static LogicalResult writeFile(llvm::StringRef path,
                               llvm::function_ref<LogicalResult(llvm::raw_ostream &)> writer) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return failure();
  return writer(os);
}

static LogicalResult copyFile(llvm::StringRef sourcePath,
                              llvm::StringRef destinationPath) {
  auto buffer = llvm::MemoryBuffer::getFile(sourcePath);
  if (!buffer)
    return failure();

  std::error_code ec;
  llvm::raw_fd_ostream os(destinationPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return failure();
  os << (*buffer)->getBuffer();
  return success();
}

LogicalResult emitVC4Artifacts(Operation *op, llvm::StringRef outputDir) {
  FailureOr<LoweredKernelArtifactModel> model = extractLoweredKernelArtifactModel(op);
  if (failed(model))
    return failure();

  std::error_code ec = llvm::sys::fs::create_directories(outputDir);
  if (ec)
    return failure();

  llvm::SmallString<128> qasmPath(outputDir);
  llvm::sys::path::append(qasmPath, "kernel.qasm");
  llvm::SmallString<128> sourcePath(outputDir);
  llvm::sys::path::append(sourcePath, "kernel_launch.c");
  llvm::SmallString<128> headerPath(outputDir);
  llvm::sys::path::append(headerPath, "kernel_launch.h");
  llvm::SmallString<128> supportDir(outputDir);
  llvm::sys::path::append(supportDir, "share", "vc4inc");
  llvm::SmallString<128> supportPath(supportDir);
  llvm::sys::path::append(supportPath, "vc4.qinc");
  llvm::SmallString<128> sourceSupportPath(VC4_SOURCE_DIR);
  llvm::sys::path::append(sourceSupportPath, "reference", "share", "vc4inc",
                          "vc4.qinc");

  if (failed(writeFile(qasmPath, [&](llvm::raw_ostream &os) { return emitVC4QASM(*model, os); })))
    return failure();
  if (failed(writeFile(sourcePath, [&](llvm::raw_ostream &os) { return emitVC4LauncherSource(*model, os); })))
    return failure();
  if (failed(writeFile(headerPath, [&](llvm::raw_ostream &os) { return emitVC4LauncherHeader(*model, os); })))
    return failure();
  ec = llvm::sys::fs::create_directories(supportDir);
  if (ec)
    return failure();
  if (failed(copyFile(sourceSupportPath, supportPath)))
    return failure();
  return success();
}

} // namespace mlir::vc4
