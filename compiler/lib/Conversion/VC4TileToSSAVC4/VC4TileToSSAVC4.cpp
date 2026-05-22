//===- VC4TileToSSAVC4.cpp - VC4Tile to SSAVC4 lowering ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.h"
#include "vc4/Dialect/VC4/IR/VC4Ops.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileAttrs.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <cctype>
#include <memory>
#include <optional>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kVC4TileKernelOpName("vc4tile.kernel");
constexpr llvm::StringLiteral kVC4TileReturnOpName("vc4tile.return");
constexpr llvm::StringLiteral kSSAVC4ModuleOpName("ssavc4.module");
constexpr llvm::StringLiteral kSSAVC4FuncOpName("ssavc4.func");
constexpr llvm::StringLiteral kSSAVC4ThreadEndOpName("ssavc4.thread_end");
static bool hasName(Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static StringAttr getSymbolNameAttr(Operation *op) {
  return op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

static bool isVC4TileOp(Operation *op) {
  return op->getName().getStringRef().starts_with("vc4tile.");
}

static std::string makeCIdentifier(llvm::StringRef value) {
  std::string result;
  result.reserve(value.size() + 8);
  for (char c : value) {
    unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc) || c == '_')
      result.push_back(c);
    else
      result.push_back('_');
  }
  if (result.empty() || !(std::isalpha(static_cast<unsigned char>(result[0])) ||
                          result[0] == '_'))
    result.insert(result.begin(), '_');
  return result;
}

static llvm::StringRef getPublicName(Operation *kernel) {
  if (auto attr = kernel->getAttrOfType<StringAttr>("public_name"))
    return attr.getValue();
  if (StringAttr symName = getSymbolNameAttr(kernel))
    return symName.getValue();
  return "vc4tile_kernel";
}

static std::optional<int64_t> getI32Attr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

static bool getBoolAttr(Operation *op, StringRef name, bool fallback = false) {
  if (auto attr = op->getAttrOfType<BoolAttr>(name))
    return attr.getValue();
  return fallback;
}

static llvm::StringRef getScheduleModeString(Operation *kernel) {
  auto attr = kernel->getAttrOfType<mlir::vc4tile::ScheduleModeAttr>(
      "schedule_mode");
  if (!attr)
    return "independent_vector";

  switch (attr.getValue()) {
  case mlir::vc4tile::ScheduleMode::independent_vector:
    return "independent_vector";
  case mlir::vc4tile::ScheduleMode::cooperative_block:
    return "cooperative_block";
  }
  return "independent_vector";
}

static bool isCooperativeKernel(Operation *kernel) {
  return getScheduleModeString(kernel) == "cooperative_block";
}

static DictionaryAttr getDictionaryAttr(Operation *op, StringRef name) {
  return op->getAttrOfType<DictionaryAttr>(name);
}

static DictionaryAttr getCarriedLaunchABI(Operation *kernel) {
  if (DictionaryAttr attr = getDictionaryAttr(kernel, "vc4.launch_abi"))
    return attr;
  return getDictionaryAttr(kernel, "launch_abi");
}

static DictionaryAttr getCarriedResource(Operation *kernel) {
  if (DictionaryAttr attr = getDictionaryAttr(kernel, "vc4.resource"))
    return attr;
  return getDictionaryAttr(kernel, "resource");
}

static DictionaryAttr buildLaunchABI(Operation *kernel, OpBuilder &builder) {
  if (DictionaryAttr carried = getCarriedLaunchABI(kernel))
    return carried;

  MLIRContext *ctx = builder.getContext();
  llvm::StringRef publicName = getPublicName(kernel);
  std::string codeSymbol = makeCIdentifier(publicName) + "_shader";
  StringAttr symName = getSymbolNameAttr(kernel);

  // The existing scheduled-VC4 verifier requires launch ABI uniform slots to be
  // described by dense arg/builtin indices.  The M4-03 return-only skeleton does
  // not consume launch uniforms, so use a harmless runtime launch-geometry
  // builtin rather than inventing a user-visible argument or weakening the
  // lower-half verifier contract.
  Attribute numQPUsKind = mlir::vc4::BuiltinKindAttr::get(
      ctx, mlir::vc4::BuiltinKind::num_qpus);
  Attribute numQPUsBuiltin = builder.getDictionaryAttr({
      builder.getNamedAttr("name", builder.getStringAttr("num_qpus")),
      builder.getNamedAttr("kind", numQPUsKind),
      builder.getNamedAttr("materialization",
                           builder.getStringAttr("uniform_suffix")),
      builder.getNamedAttr("uniform_index", builder.getI32IntegerAttr(0)),
  });

  return builder.getDictionaryAttr({
      builder.getNamedAttr("public_name", builder.getStringAttr(publicName)),
      builder.getNamedAttr("symbol_name",
                           symName ? symName : builder.getStringAttr(publicName)),
      builder.getNamedAttr("code_symbol", builder.getStringAttr(codeSymbol)),
      builder.getNamedAttr("uniform_words_per_qpu",
                           builder.getI32IntegerAttr(1)),
      builder.getNamedAttr("args", builder.getArrayAttr({})),
      builder.getNamedAttr("builtins", builder.getArrayAttr({numQPUsBuiltin})),
      builder.getNamedAttr("tail_policy", builder.getStringAttr("exact_multiple")),
  });
}

static DictionaryAttr buildResource(Operation *kernel, OpBuilder &builder) {
  if (DictionaryAttr carried = getCarriedResource(kernel))
    return carried;

  const bool cooperative = isCooperativeKernel(kernel);
  const bool usesShared = getBoolAttr(kernel, "uses_shared_vpm");
  const bool usesBarrier = getBoolAttr(kernel, "uses_barrier");
  const int64_t warpsPerBlock =
      getI32Attr(kernel, "warps_per_block_max").value_or(cooperative ? 12 : 1);
  const int64_t vpmRows =
      getI32Attr(kernel, "vpm_rows_per_block").value_or(0);
  const int64_t vpmBytes =
      getI32Attr(kernel, "vpm_bytes_per_block").value_or(0);
  const int64_t semaphores =
      getI32Attr(kernel, "semaphores_per_block").value_or(usesBarrier ? 4 : 0);
  const bool fullResidency =
      getBoolAttr(kernel, "require_full_block_residency", usesBarrier);

  return builder.getDictionaryAttr({
      builder.getNamedAttr("schedule_mode",
                           builder.getStringAttr(getScheduleModeString(kernel))),
      builder.getNamedAttr("warps_per_block_max",
                           builder.getI32IntegerAttr(warpsPerBlock)),
      builder.getNamedAttr("uses_shared_vpm", builder.getBoolAttr(usesShared)),
      builder.getNamedAttr("uses_barrier", builder.getBoolAttr(usesBarrier)),
      builder.getNamedAttr("require_full_block_residency",
                           builder.getBoolAttr(fullResidency)),
      builder.getNamedAttr("semaphores_per_block",
                           builder.getI32IntegerAttr(semaphores)),
      builder.getNamedAttr("vpm_rows_per_block",
                           builder.getI32IntegerAttr(vpmRows)),
      builder.getNamedAttr("vpm_bytes_per_block",
                           builder.getI32IntegerAttr(vpmBytes)),
      builder.getNamedAttr("shared_vpm_bytes",
                           builder.getI32IntegerAttr(vpmBytes)),
      builder.getNamedAttr("user_shared_vpm_rows_per_block",
                           builder.getI32IntegerAttr(vpmRows)),
  });
}

static LogicalResult verifyMinimalKernelBody(Operation *kernel) {
  if (kernel->getNumRegions() != 1 || kernel->getRegion(0).empty())
    return kernel->emitOpError("requires one non-empty body region");
  if (!llvm::hasSingleElement(kernel->getRegion(0))) {
    return kernel->emitOpError(
        "supports only a single-block body in this M4 lowering slice");
  }

  bool sawReturn = false;
  for (Operation &nested : kernel->getRegion(0).front()) {
    if (hasName(&nested, kVC4TileReturnOpName)) {
      sawReturn = true;
      continue;
    }
    return nested.emitOpError(
        "cannot be lowered by the M4 lowering skeleton; only "
        "vc4tile.return-only kernels are supported in this slice");
  }

  if (!sawReturn)
    return kernel->emitOpError("requires a vc4tile.return terminator");
  return success();
}

static Operation *createSSAVC4Module(Operation *sourceKernel,
                                     OpBuilder &builder) {
  OperationState state(sourceKernel->getLoc(), kSSAVC4ModuleOpName);
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr("vc4tile_lowered"));
  state.addRegion();
  Operation *module = builder.create(state);
  module->getRegion(0).push_back(new Block());
  return module;
}

static void addAttrIfPresent(Operation *source, OperationState &state,
                             StringRef name) {
  if (Attribute attr = source->getAttr(name))
    state.addAttribute(name, attr);
}

static Operation *createSSAVC4Func(Operation *kernel, Operation *ssavc4Module,
                                   OpBuilder &moduleBuilder) {
  MLIRContext *ctx = moduleBuilder.getContext();
  OperationState state(kernel->getLoc(), kSSAVC4FuncOpName);

  StringAttr symName = getSymbolNameAttr(kernel);
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     symName ? symName : moduleBuilder.getStringAttr("kernel"));
  state.addAttribute("function_type",
                     TypeAttr::get(FunctionType::get(ctx, {}, {})));
  state.addAttribute("kernel", moduleBuilder.getUnitAttr());
  state.addAttribute("threading", mlir::vc4::ThreadingModeAttr::get(
                                    ctx, mlir::vc4::ThreadingMode::single));
  state.addAttribute("vc4.launch_abi", buildLaunchABI(kernel, moduleBuilder));
  state.addAttribute("vc4.resource", buildResource(kernel, moduleBuilder));

  addAttrIfPresent(kernel, state, "arg_attrs");
  addAttrIfPresent(kernel, state, "res_attrs");

  state.addRegion();
  moduleBuilder.setInsertionPointToEnd(&ssavc4Module->getRegion(0).front());
  Operation *func = moduleBuilder.create(state);
  func->getRegion(0).push_back(new Block());

  OpBuilder bodyBuilder(ctx);
  bodyBuilder.setInsertionPointToEnd(&func->getRegion(0).front());
  OperationState threadEndState(kernel->getLoc(), kSSAVC4ThreadEndOpName);
  bodyBuilder.create(threadEndState);
  return func;
}

struct ConvertVC4TileToSSAVC4Pass
    : public PassWrapper<ConvertVC4TileToSSAVC4Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVC4TileToSSAVC4Pass)

  StringRef getArgument() const override { return "convert-vc4tile-to-ssavc4"; }
  StringRef getDescription() const override {
    return "Lower minimal VC4Tile kernels and metadata to SSAVC4";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::vc4::VC4Dialect, mlir::ssavc4::SSAVC4Dialect,
                    mlir::vc4tile::VC4TileDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 4> kernels;
    for (Operation &op : module.getBody()->getOperations()) {
      if (hasName(&op, kVC4TileKernelOpName))
        kernels.push_back(&op);
      else if (isVC4TileOp(&op)) {
        op.emitOpError(
            "is not legal at top level for --convert-vc4tile-to-ssavc4; "
            "expected vc4tile.kernel");
        signalPassFailure();
        return;
      }
    }

    if (kernels.empty()) {
      // Be idempotent for already-lowered or unrelated inputs: if this pass is
      // re-run on a saved SSAVC4 candidate, preserve the module and do not
      // fabricate VC4Tile or SSAVC4 operations.
      return;
    }

    for (Operation *kernel : kernels) {
      if (failed(verifyMinimalKernelBody(kernel))) {
        signalPassFailure();
        return;
      }
    }

    OpBuilder builder(module.getContext());
    builder.setInsertionPoint(kernels.front());
    Operation *ssavc4Module = createSSAVC4Module(kernels.front(), builder);

    for (Operation *kernel : kernels)
      createSSAVC4Func(kernel, ssavc4Module, builder);

    for (Operation *kernel : kernels)
      kernel->erase();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vc4::createConvertVC4TileToSSAVC4Pass() {
  return std::make_unique<ConvertVC4TileToSSAVC4Pass>();
}

void mlir::vc4::registerConvertVC4TileToSSAVC4Pass() {
  // This translation unit is linked into vc4-opt for M4 slice 03, and the
  // file-scope PassRegistration below installs --convert-vc4tile-to-ssavc4.
}

static PassRegistration<ConvertVC4TileToSSAVC4Pass>
    registerVC4TileToSSAVC4Pass;
