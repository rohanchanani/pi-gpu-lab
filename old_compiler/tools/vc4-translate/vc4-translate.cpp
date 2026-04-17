#include "llvm/ADT/StringRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "vc4/Target/KernelModel.h"
#include "vc4/Target/VC4Asm/VC4Asm.h"

namespace {

static llvm::StringRef getArtifactDirArg(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg.consume_front("--emit-vc4-artifacts="))
      return argv[i] + strlen("--emit-vc4-artifacts=");
  }
  return {};
}

static llvm::StringRef getInputPathArg(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg.starts_with("-"))
      continue;
    return arg;
  }
  return "-";
}

static int emitArtifactsMain(int argc, char **argv, llvm::StringRef outputDir) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::vc4::VC4Dialect>();
  mlir::MLIRContext context(registry);

  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(getInputPathArg(argc, argv));
  if (!fileOrErr)
    return 1;

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::Operation *> op =
      mlir::parseSourceFile<mlir::Operation *>(sourceMgr, &context);
  if (!op)
    return 1;
  return failed(mlir::vc4::emitVC4Artifacts(*op, outputDir)) ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  mlir::vc4::registerToVC4KernelModelTranslation();
  mlir::vc4::registerToVC4AsmTranslation();
  mlir::vc4::registerToVC4QASMTranslation();
  mlir::vc4::registerToVC4LauncherHeaderTranslation();
  mlir::vc4::registerToVC4LauncherSourceTranslation();

  if (llvm::StringRef outputDir = getArtifactDirArg(argc, argv); !outputDir.empty())
    return emitArtifactsMain(argc, argv, outputDir);

  return failed(mlir::mlirTranslateMain(argc, argv, "VC4 translation driver\n"));
}
