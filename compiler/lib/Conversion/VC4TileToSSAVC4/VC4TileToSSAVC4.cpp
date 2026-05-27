//===- VC4TileToSSAVC4.cpp - VC4Tile to SSAVC4 lowering ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Attrs.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.h"
#include "vc4/Dialect/VC4/IR/VC4Ops.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileAttrs.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileDialect.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

#include <cctype>
#include <iterator>
#include <memory>
#include <optional>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kVC4TileKernelOpName("vc4tile.kernel");
constexpr llvm::StringLiteral kVC4TileReturnOpName("vc4tile.return");
constexpr llvm::StringLiteral kVC4TileProgramIdOpName("vc4tile.program_id");
constexpr llvm::StringLiteral kVC4TileBlockIdOpName("vc4tile.block_id");
constexpr llvm::StringLiteral kVC4TileWarpIdOpName("vc4tile.warp_id");
constexpr llvm::StringLiteral kVC4TileLaneIdOpName("vc4tile.lane_id");
constexpr llvm::StringLiteral kVC4TileLaneRangeOpName("vc4tile.lane_range");
constexpr llvm::StringLiteral kVC4TileThreadIdOpName("vc4tile.thread_id");
constexpr llvm::StringLiteral kVC4TileMaskAllOpName("vc4tile.mask_all");
constexpr llvm::StringLiteral kVC4TileTailMaskOpName("vc4tile.tail_mask");
constexpr llvm::StringLiteral kVC4TileMaskedLoadGlobalOpName(
    "vc4tile.masked_load_global");
constexpr llvm::StringLiteral kVC4TileMaskedStoreGlobalOpName(
    "vc4tile.masked_store_global");
constexpr llvm::StringLiteral kVC4TileRotateOpName("vc4tile.rotate");
constexpr llvm::StringLiteral kVC4TileReduceOpName("vc4tile.reduce");
constexpr llvm::StringLiteral kVC4TileSharedAllocOpName("vc4tile.shared_alloc");
constexpr llvm::StringLiteral kVC4TileSharedLoadOpName("vc4tile.shared_load");
constexpr llvm::StringLiteral kVC4TileSharedStoreOpName("vc4tile.shared_store");
constexpr llvm::StringLiteral kVC4TileBarrierOpName("vc4tile.barrier");
constexpr llvm::StringLiteral kVC4TileTileDescriptorOpName(
    "vc4tile.tile_descriptor");
constexpr llvm::StringLiteral kVC4TileTileLoadOpName("vc4tile.tile_load");
constexpr llvm::StringLiteral kVC4TileTileStoreOpName("vc4tile.tile_store");
constexpr llvm::StringLiteral kVC4TileCopyTileOpName("vc4tile.copy_tile");
constexpr llvm::StringLiteral kVC4TileTileViewOpName("vc4tile.tile_view");
constexpr llvm::StringLiteral kVC4TileTileSubviewOpName("vc4tile.tile_subview");
constexpr llvm::StringLiteral kVC4TileTransposeViewOpName("vc4tile.transpose_view");
constexpr llvm::StringLiteral kVC4TileSharedTileAllocOpName(
    "vc4tile.shared_tile_alloc");
constexpr llvm::StringLiteral kVC4TileSurfacePlaceholderOpName(
    "vc4tile.surface_placeholder");

constexpr llvm::StringLiteral kArithConstantOpName("arith.constant");
constexpr llvm::StringLiteral kArithAddIOpName("arith.addi");
constexpr llvm::StringLiteral kArithSubIOpName("arith.subi");
constexpr llvm::StringLiteral kArithMulIOpName("arith.muli");
constexpr llvm::StringLiteral kArithShLIOpName("arith.shli");
constexpr llvm::StringLiteral kArithShRUIOpName("arith.shrui");
constexpr llvm::StringLiteral kArithShRSIOpName("arith.shrsi");
constexpr llvm::StringLiteral kArithAndIOpName("arith.andi");
constexpr llvm::StringLiteral kArithOrIOpName("arith.ori");
constexpr llvm::StringLiteral kArithXOrIOpName("arith.xori");
constexpr llvm::StringLiteral kArithCmpIOpName("arith.cmpi");
constexpr llvm::StringLiteral kArithIndexCastOpName("arith.index_cast");
constexpr llvm::StringLiteral kArithIndexCastUIOpName("arith.index_castui");
constexpr llvm::StringLiteral kArithIndexCastSIOpName("arith.index_castsi");
constexpr llvm::StringLiteral kVectorSplatOpName("vector.broadcast");
constexpr llvm::StringLiteral kCFBranchOpName("cf.br");
constexpr llvm::StringLiteral kCFCondBranchOpName("cf.cond_br");

constexpr llvm::StringLiteral kSSAVC4ModuleOpName("ssavc4.module");
constexpr llvm::StringLiteral kSSAVC4FuncOpName("ssavc4.func");
constexpr llvm::StringLiteral kSSAVC4ThreadEndOpName("ssavc4.thread_end");
constexpr llvm::StringLiteral kSSAVC4LoadImmOpName("ssavc4.load_imm");
constexpr llvm::StringLiteral kSSAVC4ElementNumberOpName(
    "ssavc4.element_number");
constexpr llvm::StringLiteral kSSAVC4UniformReadOpName("ssavc4.uniform.read");
constexpr llvm::StringLiteral kSSAVC4SplatOpName("ssavc4.splat");
constexpr llvm::StringLiteral kSSAVC4ALUAddOpName("ssavc4.alu.add");
constexpr llvm::StringLiteral kSSAVC4ALUMulOpName("ssavc4.alu.mul");
constexpr llvm::StringLiteral kSSAVC4MakeFlagsOpName("ssavc4.make_flags");
constexpr llvm::StringLiteral kSSAVC4TMURequestOpName("ssavc4.tmu.request");
constexpr llvm::StringLiteral kSSAVC4TMUReadOpName("ssavc4.tmu.read");
constexpr llvm::StringLiteral kSSAVC4VDWStoreOpName("ssavc4.vdw.store");
constexpr llvm::StringLiteral kSSAVC4VPMReadOpName("ssavc4.vpm.read");
constexpr llvm::StringLiteral kSSAVC4VPMWriteOpName("ssavc4.vpm.write");
constexpr llvm::StringLiteral kSSAVC4RotateOpName("ssavc4.rotate");
constexpr llvm::StringLiteral kSSAVC4BranchOpName("ssavc4.br");
constexpr llvm::StringLiteral kSSAVC4BarrierOpName("ssavc4.barrier");
constexpr llvm::StringLiteral kSSAVC4CondBranchOpName("ssavc4.cond_br");

static bool hasName(Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static StringAttr getSymbolNameAttr(Operation *op) {
  return op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

static bool isVC4TileOp(Operation *op) {
  return op->getName().getStringRef().starts_with("vc4tile.");
}

static bool isVC4TileSurfaceOp(Operation *op) {
  return hasName(op, kVC4TileSurfacePlaceholderOpName) ||
         hasName(op, kVC4TileTileDescriptorOpName) ||
         hasName(op, kVC4TileTileLoadOpName) ||
         hasName(op, kVC4TileTileStoreOpName) ||
         hasName(op, kVC4TileCopyTileOpName) ||
         hasName(op, kVC4TileTileViewOpName) ||
         hasName(op, kVC4TileTileSubviewOpName) ||
         hasName(op, kVC4TileTransposeViewOpName) ||
         hasName(op, kVC4TileSharedTileAllocOpName);
}

static LogicalResult emitSurfaceOpOrderingError(Operation *op,
                                                StringRef beforePass) {
  return op->emitOpError()
         << "is a surface operation; run --canonicalize-vc4tile-surface and "
            "--plan-vc4tile-copies before "
         << beforePass;
}

static Operation *getParentVC4TileKernel(Operation *op) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (hasName(cur, kVC4TileKernelOpName))
      return cur;
  }
  return nullptr;
}

static bool isVC4TileKernel(Operation *op) {
  return hasName(op, kVC4TileKernelOpName);
}

static bool containsIndexType(Type type) {
  if (!type)
    return false;
  if (type.isIndex())
    return true;
  if (auto shaped = llvm::dyn_cast<ShapedType>(type))
    return containsIndexType(shaped.getElementType());
  if (auto tuple = llvm::dyn_cast<TupleType>(type)) {
    for (Type element : tuple.getTypes()) {
      if (containsIndexType(element))
        return true;
    }
    return false;
  }
  if (auto function = llvm::dyn_cast<FunctionType>(type)) {
    for (Type input : function.getInputs()) {
      if (containsIndexType(input))
        return true;
    }
    for (Type result : function.getResults()) {
      if (containsIndexType(result))
        return true;
    }
  }
  return false;
}

static bool isSupportedCoreType(Type type) {
  if (!type || containsIndexType(type))
    return false;
  if (type.isInteger(1) || type.isSignlessInteger(32) || type.isF32())
    return true;
  if (mlir::vc4tile::isVC4TileSharedTileType(type))
    return true;
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 ||
      vectorType.getDimSize(0) != 16)
    return false;
  Type element = vectorType.getElementType();
  return element.isInteger(1) || element.isSignlessInteger(32) ||
         element.isF32();
}

static bool isSupportedCoreFunctionType(FunctionType type) {
  for (Type input : type.getInputs()) {
    if (!isSupportedCoreType(input))
      return false;
  }
  for (Type result : type.getResults()) {
    if (!isSupportedCoreType(result))
      return false;
  }
  return true;
}

static bool attrContainsIndexType(Attribute attr) {
  if (!attr)
    return false;
  if (auto typeAttr = llvm::dyn_cast<TypeAttr>(attr))
    return containsIndexType(typeAttr.getValue());
  if (auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr)) {
    for (Attribute element : arrayAttr) {
      if (attrContainsIndexType(element))
        return true;
    }
    return false;
  }
  if (auto dictAttr = llvm::dyn_cast<DictionaryAttr>(attr)) {
    for (NamedAttribute named : dictAttr) {
      if (attrContainsIndexType(named.getValue()))
        return true;
    }
  }
  return false;
}

static bool isProducerDialectNamespace(StringRef dialectNamespace) {
  return dialectNamespace == "affine" || dialectNamespace == "gpu" ||
         dialectNamespace == "triton" || dialectNamespace == "tt" ||
         dialectNamespace == "ttg" || dialectNamespace == "nvgpu" ||
         dialectNamespace == "iree" || dialectNamespace == "stablehlo" ||
         dialectNamespace == "tosa" || dialectNamespace == "linalg" ||
         dialectNamespace == "tensor" || dialectNamespace == "memref" ||
         dialectNamespace == "spirv" || dialectNamespace == "nvvm" ||
         dialectNamespace == "rocdl";
}

static StringRef getDialectNamespace(Operation *op) {
  return op->getName().getDialectNamespace();
}

static bool isAllowedCoreTerminator(Operation *op) {
  return hasName(op, kVC4TileReturnOpName) || hasName(op, kCFBranchOpName) ||
         hasName(op, kCFCondBranchOpName);
}

static bool isAllowedCoreVC4TileOp(Operation *op) {
  return hasName(op, kVC4TileProgramIdOpName) ||
         hasName(op, kVC4TileBlockIdOpName) ||
         hasName(op, kVC4TileWarpIdOpName) ||
         hasName(op, kVC4TileLaneIdOpName) ||
         hasName(op, kVC4TileLaneRangeOpName) ||
         hasName(op, kVC4TileThreadIdOpName) ||
         hasName(op, kVC4TileMaskAllOpName) ||
         hasName(op, kVC4TileTailMaskOpName) ||
         hasName(op, kVC4TileMaskedLoadGlobalOpName) ||
         hasName(op, kVC4TileMaskedStoreGlobalOpName) ||
         hasName(op, kVC4TileRotateOpName) ||
         hasName(op, kVC4TileReduceOpName) ||
         hasName(op, kVC4TileSharedAllocOpName) ||
         hasName(op, kVC4TileSharedLoadOpName) ||
         hasName(op, kVC4TileSharedStoreOpName) ||
         hasName(op, kVC4TileBarrierOpName);
}

static bool isAllowedCoreArithOp(Operation *op) {
  return hasName(op, kArithConstantOpName) ||
         hasName(op, kArithAddIOpName) || hasName(op, kArithSubIOpName) ||
         hasName(op, kArithMulIOpName) || hasName(op, kArithShLIOpName) ||
         hasName(op, kArithShRUIOpName) || hasName(op, kArithShRSIOpName) ||
         hasName(op, kArithAndIOpName) || hasName(op, kArithOrIOpName) ||
         hasName(op, kArithXOrIOpName) || hasName(op, kArithCmpIOpName);
}

static bool isAllowedCoreOp(Operation *op) {
  if (isVC4TileKernel(op))
    return true;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return isAllowedCoreTerminator(op);
  if (getDialectNamespace(op) == "vc4tile")
    return isAllowedCoreVC4TileOp(op);
  if (getDialectNamespace(op) == "arith")
    return isAllowedCoreArithOp(op);
  if (getDialectNamespace(op) == "vector")
    return hasName(op, kVectorSplatOpName);
  return false;
}

static LogicalResult emitUnsupportedCoreType(Location loc, Type type) {
  if (containsIndexType(type))
    return emitError(loc) << "vc4tile core type must not contain index: "
                          << type;
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    return emitError(loc)
           << "unsupported vector type in vc4tile core; expected vector width "
              "16 with i1/i32/f32 element, got "
           << vectorType;
  }
  return emitError(loc) << "unsupported scalar type in vc4tile core: " << type;
}

static LogicalResult verifyCoreValueType(Value value, Operation *anchor) {
  Location loc = value.getLoc();
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(value))
    loc = blockArg.getLoc();
  if (isSupportedCoreType(value.getType()))
    return success();
  (void)anchor;
  return emitUnsupportedCoreType(loc, value.getType());
}

static LogicalResult verifyKernelFunctionTypeAttr(Operation *kernel) {
  auto typeAttr = kernel->getAttrOfType<TypeAttr>("function_type");
  if (!typeAttr)
    return success();
  auto functionType = llvm::dyn_cast<FunctionType>(typeAttr.getValue());
  if (!functionType)
    return kernel->emitOpError("function_type must be a function type");
  if (containsIndexType(functionType))
    return kernel->emitOpError(
        "vc4tile core function_type must not contain index");
  if (!isSupportedCoreFunctionType(functionType))
    return kernel->emitOpError(
        "vc4tile core function_type contains unsupported scalar or vector type; "
        "vectors must have width 16");
  return success();
}

static LogicalResult verifyVC4TileCoreKernel(Operation *kernel) {
  if (!isVC4TileKernel(kernel))
    return success();
  if (failed(verifyKernelFunctionTypeAttr(kernel)))
    return failure();

  bool sawError = false;
  kernel->walk([&](Operation *op) {
    if (op == kernel)
      return WalkResult::advance();

    StringRef dialect = getDialectNamespace(op);
    if (dialect == "scf") {
      op->emitOpError("is not allowed in vc4tile core; raw scf must be "
                      "legalized before lowering");
      sawError = true;
      return WalkResult::interrupt();
    }
    if (isProducerDialectNamespace(dialect)) {
      op->emitOpError()
          << "is a producer dialect operation and is not allowed in vc4tile "
             "core";
      sawError = true;
      return WalkResult::interrupt();
    }
    if (op->hasTrait<OpTrait::IsTerminator>() &&
        !isAllowedCoreTerminator(op)) {
      op->emitOpError("is not an allowed vc4tile core terminator; expected "
                      "vc4tile.return, cf.br, or cf.cond_br");
      sawError = true;
      return WalkResult::interrupt();
    }
    if (isVC4TileSurfaceOp(op)) {
      (void)emitSurfaceOpOrderingError(op, "--verify-vc4tile-core");
      sawError = true;
      return WalkResult::interrupt();
    }
    if (!isAllowedCoreOp(op)) {
      op->emitOpError("is not explicitly allowed in lowering-ready vc4tile "
                      "core");
      sawError = true;
      return WalkResult::interrupt();
    }
    if (auto condBranch = llvm::dyn_cast<mlir::cf::CondBranchOp>(op)) {
      if (!condBranch.getCondition().getType().isInteger(1)) {
        op->emitOpError("cf.cond_br condition in vc4tile core must be scalar "
                        "i1");
        sawError = true;
        return WalkResult::interrupt();
      }
    }
    for (Value operand : op->getOperands()) {
      if (failed(verifyCoreValueType(operand, op))) {
        sawError = true;
        return WalkResult::interrupt();
      }
    }
    for (Value result : op->getResults()) {
      if (failed(verifyCoreValueType(result, op))) {
        sawError = true;
        return WalkResult::interrupt();
      }
    }
    for (NamedAttribute attr : op->getAttrs()) {
      if (attrContainsIndexType(attr.getValue())) {
        op->emitOpError("vc4tile core attribute must not contain index");
        sawError = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (sawError)
    return failure();

  for (Block &block : kernel->getRegion(0)) {
    for (BlockArgument arg : block.getArguments()) {
      if (failed(verifyCoreValueType(arg, kernel)))
        return failure();
    }
  }
  return success();
}

static LogicalResult verifyVC4TileCore(ModuleOp module) {
  for (Operation &op : module.getBody()->getOperations()) {
    if (isVC4TileKernel(&op) && failed(verifyVC4TileCoreKernel(&op)))
      return failure();
  }
  return success();
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

static bool kernelContains(Operation *kernel, llvm::StringRef opName) {
  bool found = false;
  kernel->walk([&](Operation *op) {
    if (hasName(op, opName)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool dictionaryHasStringName(DictionaryAttr dict, StringRef name) {
  if (!dict)
    return false;
  if (auto nameAttr = dict.getAs<StringAttr>("name"))
    return nameAttr.getValue() == name;
  auto legacyNameAttr = dict.getAs<StringAttr>("abi_name");
  return legacyNameAttr && legacyNameAttr.getValue() == name;
}

static StringAttr getABINameAttr(DictionaryAttr dict) {
  if (!dict)
    return {};
  if (auto nameAttr = dict.getAs<StringAttr>("name"))
    return nameAttr;
  return dict.getAs<StringAttr>("abi_name");
}

static std::optional<int64_t> getUniformIndex(DictionaryAttr dict) {
  if (!dict)
    return std::nullopt;
  auto attr = dict.getAs<IntegerAttr>("uniform_index");
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static int64_t getMaxUniformIndex(ArrayAttr entries) {
  int64_t maxIndex = -1;
  if (!entries)
    return maxIndex;
  for (Attribute attr : entries) {
    auto dict = llvm::dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      continue;
    if (std::optional<int64_t> index = getUniformIndex(dict))
      maxIndex = std::max(maxIndex, *index);
  }
  return maxIndex;
}

static bool arrayHasEntryNamed(ArrayAttr entries, StringRef name) {
  if (!entries)
    return false;
  for (Attribute attr : entries) {
    if (dictionaryHasStringName(llvm::dyn_cast<DictionaryAttr>(attr), name))
      return true;
  }
  return false;
}

static Attribute getBuiltinKindAttr(OpBuilder &builder, StringRef name) {
  MLIRContext *ctx = builder.getContext();
  if (name == "logical_request")
    return mlir::vc4::BuiltinKindAttr::get(ctx,
                                           mlir::vc4::BuiltinKind::logical_request);
  if (name == "total_requests")
    return mlir::vc4::BuiltinKindAttr::get(ctx,
                                           mlir::vc4::BuiltinKind::total_requests);
  if (name == "logical_block_id")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::logical_block_id);
  if (name == "logical_warp_id")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::logical_warp_id);
  if (name == "warps_per_block")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::warps_per_block);
  if (name == "vpm_base_row")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::vpm_base_row);
  llvm_unreachable("unhandled VC4 launch builtin name");
}

static DictionaryAttr buildBuiltinABIEntry(OpBuilder &builder, StringRef name,
                                           int64_t uniformIndex) {
  return builder.getDictionaryAttr({
      builder.getNamedAttr("name", builder.getStringAttr(name)),
      builder.getNamedAttr("kind", getBuiltinKindAttr(builder, name)),
      builder.getNamedAttr("materialization",
                           builder.getStringAttr("uniform_suffix")),
      builder.getNamedAttr("uniform_index",
                           builder.getI32IntegerAttr(uniformIndex)),
  });
}

static StringAttr inferABITypeForBlockArgument(OpBuilder &builder,
                                               Type type) {
  if (type.isF32())
    return builder.getStringAttr("f32");
  return builder.getStringAttr("u32");
}

static DictionaryAttr buildDefaultArgABIEntry(OpBuilder &builder,
                                              BlockArgument arg,
                                              unsigned argIndex,
                                              int64_t uniformIndex) {
  std::string name = (Twine("arg") + Twine(argIndex)).str();
  return builder.getDictionaryAttr({
      builder.getNamedAttr("name", builder.getStringAttr(name)),
      builder.getNamedAttr("kind", builder.getStringAttr("scalar")),
      builder.getNamedAttr("direction", builder.getStringAttr("by_value")),
      builder.getNamedAttr("type", inferABITypeForBlockArgument(builder,
                                                                 arg.getType())),
      builder.getNamedAttr("uniform_index",
                           builder.getI32IntegerAttr(uniformIndex)),
  });
}

static DictionaryAttr buildArgABIEntry(OpBuilder &builder, Operation *kernel,
                                       BlockArgument arg, unsigned argIndex,
                                       int64_t uniformIndex) {
  auto argAttrs = kernel->getAttrOfType<ArrayAttr>("arg_attrs");
  DictionaryAttr source;
  if (argAttrs && argIndex < argAttrs.size())
    source = llvm::dyn_cast<DictionaryAttr>(argAttrs[argIndex]);
  if (!source)
    return buildDefaultArgABIEntry(builder, arg, argIndex, uniformIndex);

  auto kindAttr = source.getAs<StringAttr>("kind");
  const bool sourceIsBuffer = kindAttr && kindAttr.getValue() == "buffer";

  SmallVector<NamedAttribute, 8> attrs;
  bool sawName = false;
  for (NamedAttribute named : source) {
    StringRef key = named.getName().getValue();
    if (key == "uniform_index")
      continue;
    if (sourceIsBuffer && key == "type")
      continue;
    if (key == "abi_name") {
      if (!source.get("name")) {
        attrs.push_back(builder.getNamedAttr("name", named.getValue()));
        sawName = true;
      }
      continue;
    }
    if (key == "name")
      sawName = true;
    attrs.push_back(named);
  }

  if (!sawName)
    attrs.push_back(builder.getNamedAttr(
        "name", builder.getStringAttr((Twine("arg") + Twine(argIndex)).str())));
  if (!source.get("kind"))
    attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr("scalar")));
  if (!source.get("direction"))
    attrs.push_back(builder.getNamedAttr("direction",
                                         builder.getStringAttr("by_value")));
  if (!sourceIsBuffer && !source.get("type") && !source.get("elem_type"))
    attrs.push_back(builder.getNamedAttr(
        "type", inferABITypeForBlockArgument(builder, arg.getType())));
  attrs.push_back(builder.getNamedAttr("uniform_index",
                                       builder.getI32IntegerAttr(uniformIndex)));
  return builder.getDictionaryAttr(attrs);
}

static ArrayAttr buildFormalArgABIEntries(Operation *kernel,
                                          OpBuilder &builder,
                                          int64_t &nextUniformIndex) {
  SmallVector<Attribute, 8> args;
  if (kernel->getNumRegions() == 0 || kernel->getRegion(0).empty())
    return builder.getArrayAttr(args);

  Block &entry = kernel->getRegion(0).front();
  args.reserve(entry.getNumArguments());
  for (BlockArgument arg : entry.getArguments())
    args.push_back(buildArgABIEntry(builder, kernel, arg, arg.getArgNumber(),
                                    nextUniformIndex++));
  return builder.getArrayAttr(args);
}

static DictionaryAttr normalizeAndCompleteLaunchABI(Operation *kernel,
                                                    DictionaryAttr carried,
                                                    OpBuilder &builder) {
  SmallVector<NamedAttribute, 8> attrs;
  llvm::StringRef publicName = getPublicName(kernel);
  std::string codeSymbol = makeCIdentifier(publicName) + "_shader";
  StringAttr symName = getSymbolNameAttr(kernel);
  bool sawPublicName = false;
  bool sawSymbolName = false;
  bool sawCodeSymbol = false;
  bool sawTailPolicy = false;

  ArrayAttr args;
  ArrayAttr builtins;
  if (carried) {
    for (NamedAttribute attr : carried) {
      StringRef key = attr.getName().getValue();
      if (key == "args") {
        args = llvm::dyn_cast<ArrayAttr>(attr.getValue());
        continue;
      }
      if (key == "builtins") {
        builtins = llvm::dyn_cast<ArrayAttr>(attr.getValue());
        continue;
      }
      if (key == "uniform_words_per_qpu")
        continue;
      if (key == "public_name")
        sawPublicName = true;
      if (key == "symbol_name")
        sawSymbolName = true;
      if (key == "code_symbol")
        sawCodeSymbol = true;
      if (key == "tail_policy") {
        sawTailPolicy = true;
        if (auto tailPolicy = llvm::dyn_cast<StringAttr>(attr.getValue())) {
          if (tailPolicy.getValue() == "tail") {
            attrs.push_back(builder.getNamedAttr(
                "tail_policy", builder.getStringAttr("tail_safe")));
            continue;
          }
        }
      }
      attrs.push_back(attr);
    }
  }

  if (!sawPublicName)
    attrs.push_back(builder.getNamedAttr("public_name",
                                         builder.getStringAttr(publicName)));
  if (!sawSymbolName)
    attrs.push_back(builder.getNamedAttr(
        "symbol_name", symName ? symName : builder.getStringAttr(publicName)));
  if (!sawCodeSymbol)
    attrs.push_back(builder.getNamedAttr("code_symbol",
                                         builder.getStringAttr(codeSymbol)));
  if (!sawTailPolicy)
    attrs.push_back(builder.getNamedAttr("tail_policy",
                                         builder.getStringAttr("exact_multiple")));

  int64_t nextUniformIndex = std::max<int64_t>(0, getMaxUniformIndex(args) + 1);
  if (!args)
    args = buildFormalArgABIEntries(kernel, builder, nextUniformIndex);
  else
    nextUniformIndex = std::max<int64_t>(nextUniformIndex,
                                         getMaxUniformIndex(args) + 1);

  SmallVector<Attribute, 8> builtinEntries;
  if (builtins)
    builtinEntries.append(builtins.begin(), builtins.end());
  nextUniformIndex = std::max<int64_t>(nextUniformIndex,
                                       getMaxUniformIndex(builtins) + 1);

  auto appendBuiltinIfMissing = [&](StringRef name) {
    ArrayAttr current = builder.getArrayAttr(builtinEntries);
    if (arrayHasEntryNamed(current, name))
      return;
    builtinEntries.push_back(buildBuiltinABIEntry(builder, name,
                                                  nextUniformIndex++));
  };

  const bool needsBarrier = getBoolAttr(kernel, "uses_barrier") ||
                            kernelContains(kernel, kVC4TileBarrierOpName);
  const bool needsLogicalRequest =
      kernelContains(kernel, kVC4TileProgramIdOpName) ||
      (!isCooperativeKernel(kernel) && kernelContains(kernel, kVC4TileThreadIdOpName));
  if (needsLogicalRequest) {
    appendBuiltinIfMissing("logical_request");
    appendBuiltinIfMissing("total_requests");
  }
  if (kernelContains(kernel, kVC4TileBlockIdOpName))
    appendBuiltinIfMissing("logical_block_id");
  if (kernelContains(kernel, kVC4TileWarpIdOpName) ||
      (isCooperativeKernel(kernel) && kernelContains(kernel, kVC4TileThreadIdOpName)) ||
      needsBarrier)
    appendBuiltinIfMissing("logical_warp_id");
  if (isCooperativeKernel(kernel))
    appendBuiltinIfMissing("warps_per_block");
  if (kernelContains(kernel, kVC4TileSharedAllocOpName) ||
      kernelContains(kernel, kVC4TileSharedLoadOpName) ||
      kernelContains(kernel, kVC4TileSharedStoreOpName))
    appendBuiltinIfMissing("vpm_base_row");

  ArrayAttr completedBuiltins = builder.getArrayAttr(builtinEntries);
  int64_t maxUniformIndex = std::max(getMaxUniformIndex(args),
                                     getMaxUniformIndex(completedBuiltins));
  int64_t uniformWords = std::max<int64_t>(0, maxUniformIndex + 1);

  attrs.push_back(builder.getNamedAttr("uniform_words_per_qpu",
                                       builder.getI32IntegerAttr(uniformWords)));
  attrs.push_back(builder.getNamedAttr("args", args));
  attrs.push_back(builder.getNamedAttr("builtins", completedBuiltins));
  return builder.getDictionaryAttr(attrs);
}

static DictionaryAttr buildLaunchABI(Operation *kernel, OpBuilder &builder) {
  return normalizeAndCompleteLaunchABI(kernel, getCarriedLaunchABI(kernel),
                                       builder);
}

static std::optional<int64_t> lookupLaunchArgUniformIndex(Operation *kernel,
                                                          OpBuilder &builder,
                                                          unsigned argIndex) {
  DictionaryAttr abi = buildLaunchABI(kernel, builder);
  auto args = abi.getAs<ArrayAttr>("args");
  if (!args || argIndex >= args.size())
    return std::nullopt;
  return getUniformIndex(llvm::dyn_cast<DictionaryAttr>(args[argIndex]));
}

static std::optional<int64_t> lookupLaunchBuiltinUniformIndex(Operation *kernel,
                                                              OpBuilder &builder,
                                                              StringRef name) {
  DictionaryAttr abi = buildLaunchABI(kernel, builder);
  auto builtins = abi.getAs<ArrayAttr>("builtins");
  if (!builtins)
    return std::nullopt;
  for (Attribute attr : builtins) {
    auto dict = llvm::dyn_cast<DictionaryAttr>(attr);
    if (!dictionaryHasStringName(dict, name))
      continue;
    return getUniformIndex(dict);
  }
  return std::nullopt;
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

static bool isSupportedKernelTerminator(Operation *op) {
  return hasName(op, kVC4TileReturnOpName) || hasName(op, kCFBranchOpName) ||
         hasName(op, kCFCondBranchOpName);
}

static LogicalResult verifySupportedKernelShape(Operation *kernel) {
  if (kernel->getNumRegions() != 1 || kernel->getRegion(0).empty())
    return kernel->emitOpError("requires one non-empty body region");

  bool sawReturn = false;
  for (Block &block : kernel->getRegion(0)) {
    if (block.empty())
      return kernel->emitOpError("does not support empty body blocks");

    Operation *terminator = block.getTerminator();
    if (!terminator || !isSupportedKernelTerminator(terminator)) {
      return kernel->emitOpError(
          "supports only vc4tile.return, cf.br, or cf.cond_br block terminators in this M4 lowering slice");
    }
    if (hasName(terminator, kVC4TileReturnOpName))
      sawReturn = true;
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

  // VC4Tile formal argument metadata is lowered into vc4.launch_abi.args[].
  // Do not copy the source-level arg_attrs/res_attrs onto the zero-argument
  // SSAVC4 function; the downstream scheduled vc4 verifier treats those as
  // MLIR function argument attributes and requires them to match the actual
  // function argument count.

  state.addRegion();
  moduleBuilder.setInsertionPointToEnd(&ssavc4Module->getRegion(0).front());
  return moduleBuilder.create(state);
}

static VectorType getVector16I32Type(OpBuilder &builder) {
  return VectorType::get({16}, builder.getI32Type());
}

static bool isVector16I32(Type type) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 &&
         vectorType.getDimSize(0) == 16 &&
         vectorType.getElementType().isSignlessInteger(32);
}

static bool isVector16I1(Type type) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 &&
         vectorType.getDimSize(0) == 16 && vectorType.getElementType().isInteger(1);
}

static bool isVector16F32(Type type) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 &&
         vectorType.getDimSize(0) == 16 && vectorType.getElementType().isF32();
}

static bool isVector16Data(Type type) {
  return isVector16I32(type) || isVector16F32(type);
}

static bool isSupportedI32Carrier(Type type) {
  return type.isSignlessInteger(32) || isVector16I32(type);
}

static Value createSSAVC4OpWithResult(OpBuilder &builder, Location loc,
                                      llvm::StringRef name,
                                      ArrayRef<Value> operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      Type resultType) {
  OperationState state(loc, name);
  state.addOperands(operands);
  for (NamedAttribute attr : attrs)
    state.addAttribute(attr.getName(), attr.getValue());
  state.addTypes(resultType);
  Operation *created = builder.create(state);
  return created->getResult(0);
}

static Operation *createSSAVC4Op(OpBuilder &builder, Location loc,
                                 llvm::StringRef name,
                                 ArrayRef<Value> operands,
                                 ArrayRef<NamedAttribute> attrs) {
  OperationState state(loc, name);
  state.addOperands(operands);
  for (NamedAttribute attr : attrs)
    state.addAttribute(attr.getName(), attr.getValue());
  return builder.create(state);
}

static Value createLoadImm(OpBuilder &builder, Location loc, Type resultType,
                           Attribute value) {
  Attribute mode = mlir::vc4::LoadImmModeAttr::get(
      builder.getContext(), mlir::vc4::LoadImmMode::splat32);
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4LoadImmOpName, {},
      {builder.getNamedAttr("mode", mode),
       builder.getNamedAttr("value", value)},
      resultType);
}

static Value createLoadImmI32(OpBuilder &builder, Location loc, Type resultType,
                              int64_t value) {
  return createLoadImm(builder, loc, resultType,
                       builder.getI32IntegerAttr(value));
}

static Value createElementNumber(OpBuilder &builder, Location loc) {
  OperationState state(loc, kSSAVC4ElementNumberOpName);
  state.addTypes(getVector16I32Type(builder));
  Operation *created = builder.create(state);
  return created->getResult(0);
}

static Value createUniformRead(OpBuilder &builder, Location loc, Type resultType,
                               int64_t index) {
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4UniformReadOpName, {},
      {builder.getNamedAttr("index", builder.getI32IntegerAttr(index))},
      resultType);
}

static Value createEntryUniformRead(OpBuilder &builder, Location loc,
                                    Type resultType, int64_t index) {
  OpBuilder::InsertionGuard guard(builder);
  Region *region = builder.getInsertionBlock()->getParent();
  Block &entry = region->front();
  Operation *insertBefore = nullptr;
  for (Operation &op : entry) {
    if (hasName(&op, kSSAVC4UniformReadOpName))
      continue;
    insertBefore = &op;
    break;
  }

  if (insertBefore)
    builder.setInsertionPoint(insertBefore);
  else
    builder.setInsertionPointToEnd(&entry);
  return createUniformRead(builder, loc, resultType, index);
}

static Value createSplat(OpBuilder &builder, Location loc, Value input,
                         Type resultType) {
  return createSSAVC4OpWithResult(builder, loc, kSSAVC4SplatOpName, {input}, {},
                                  resultType);
}

static Value createALUAdd(OpBuilder &builder, Location loc,
                          ArrayRef<Value> operands,
                          mlir::vc4::AddOpcode opcode, Type resultType) {
  Attribute opcodeAttr =
      mlir::vc4::AddOpcodeAttr::get(builder.getContext(), opcode);
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4ALUAddOpName, operands,
      {builder.getNamedAttr("opcode", opcodeAttr)}, resultType);
}

static Value createALUMul(OpBuilder &builder, Location loc,
                          ArrayRef<Value> operands,
                          mlir::vc4::MulOpcode opcode, Type resultType) {
  Attribute opcodeAttr =
      mlir::vc4::MulOpcodeAttr::get(builder.getContext(), opcode);
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4ALUMulOpName, operands,
      {builder.getNamedAttr("opcode", opcodeAttr)}, resultType);
}

static Value createMakeFlags(OpBuilder &builder, Location loc,
                             ArrayRef<Value> operands,
                             mlir::ssavc4::FlagKind kind) {
  Attribute kindAttr =
      mlir::ssavc4::FlagKindAttr::get(builder.getContext(), kind);
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4MakeFlagsOpName, operands,
      {builder.getNamedAttr("kind", kindAttr)},
      mlir::ssavc4::FlagsType::get(builder.getContext()));
}

static Value createRotate(OpBuilder &builder, Location loc, Value input,
                          int64_t amount, Type resultType) {
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4RotateOpName, {input},
      {builder.getNamedAttr("amount", builder.getI32IntegerAttr(amount))},
      resultType);
}

static Type getAsyncTokenType(OpBuilder &builder) {
  return mlir::ssavc4::AsyncTokenType::get(builder.getContext());
}

static Value createTMURequest(OpBuilder &builder, Location loc, Value address) {
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4TMURequestOpName, {address},
      {builder.getNamedAttr("unit", builder.getStringAttr("tmu0")),
       builder.getNamedAttr("mode", builder.getStringAttr("direct"))},
      getAsyncTokenType(builder));
}

static Value createTMURead(OpBuilder &builder, Location loc, Value token,
                           Type resultType) {
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4TMUReadOpName, {token},
      {builder.getNamedAttr("unit", builder.getStringAttr("tmu0")),
       builder.getNamedAttr("part", builder.getStringAttr("raw32"))},
      resultType);
}

static StringRef getVPMOrientation(Operation *op, StringRef fallback) {
  auto layout = op->getAttrOfType<mlir::vc4tile::VPMLayoutAttr>("layout");
  if (!layout)
    return fallback;
  switch (layout.getValue()) {
  case mlir::vc4tile::VPMLayout::row_major:
    return "horizontal";
  case mlir::vc4tile::VPMLayout::column_major:
    return "vertical";
  }
  return fallback;
}

static bool isAllLanesMask(Value value) {
  return hasName(value.getDefiningOp(), kVC4TileMaskAllOpName);
}

struct SharedVPMAllocationState {
  int64_t nextRowOffset = 0;
};

static Value createVPMRowAddress(OpBuilder &builder, Location loc,
                                 Value baseRow, Value localRow) {
  SmallVector<Value, 2> operands{baseRow, localRow};
  return createALUAdd(builder, loc, operands, mlir::vc4::AddOpcode::add,
                      builder.getI32Type());
}


static Value lookupMappedValue(Operation *user, Value source,
                               llvm::DenseMap<Value, Value> &valueMap) {
  auto it = valueMap.find(source);
  if (it != valueMap.end())
    return it->second;
  user->emitOpError("uses a value that has not been lowered to SSAVC4");
  return Value();
}

static std::optional<mlir::vc4::AddOpcode>
getIntegerAddOpcode(Operation *op) {
  if (hasName(op, kArithAddIOpName))
    return mlir::vc4::AddOpcode::add;
  if (hasName(op, kArithSubIOpName))
    return mlir::vc4::AddOpcode::sub;
  if (hasName(op, kArithShLIOpName))
    return mlir::vc4::AddOpcode::shl;
  if (hasName(op, kArithShRUIOpName))
    return mlir::vc4::AddOpcode::shr;
  if (hasName(op, kArithShRSIOpName))
    return mlir::vc4::AddOpcode::asr;
  if (hasName(op, kArithAndIOpName))
    return mlir::vc4::AddOpcode::bit_and;
  if (hasName(op, kArithOrIOpName))
    return mlir::vc4::AddOpcode::bit_or;
  if (hasName(op, kArithXOrIOpName))
    return mlir::vc4::AddOpcode::bit_xor;
  return std::nullopt;
}

static LogicalResult lowerConstant(Operation *op, OpBuilder &builder,
                                   llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  Attribute value = op->getAttr("value");
  if (!value)
    return op->emitOpError("requires a value attribute");

  Type resultType = op->getResult(0).getType();
  if (!isSupportedI32Carrier(resultType))
    return op->emitOpError(
        "unsupported arith.constant type for VC4Tile M4 independent lowering");

  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(value)) {
    value = builder.getI32IntegerAttr(intAttr.getInt());
  } else if (auto denseAttr = llvm::dyn_cast<DenseIntElementsAttr>(value)) {
    if (!denseAttr.isSplat())
      return op->emitOpError(
          "currently lowers only splat dense integer vector constants");
    value = builder.getI32IntegerAttr(
        denseAttr.getSplatValue<llvm::APInt>().getSExtValue());
  } else {
    return op->emitOpError(
        "unsupported arith.constant value for VC4Tile M4 independent lowering");
  }

  valueMap[op->getResult(0)] =
      createLoadImm(builder, op->getLoc(), resultType, value);
  return success();
}

static LogicalResult lowerVectorSplat(Operation *op, OpBuilder &builder,
                                      llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return op->emitOpError("expected one operand and one result");
  Value input = lookupMappedValue(op, op->getOperand(0), valueMap);
  if (!input)
    return failure();
  valueMap[op->getResult(0)] =
      createSplat(builder, op->getLoc(), input, op->getResult(0).getType());
  return success();
}

static LogicalResult lowerIntegerALU(Operation *op, OpBuilder &builder,
                                     llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("expected two operands and one result");
  std::optional<mlir::vc4::AddOpcode> opcode = getIntegerAddOpcode(op);
  if (!opcode)
    return op->emitOpError("has no VC4 ADD-pipe opcode mapping");
  Value lhs = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value rhs = lookupMappedValue(op, op->getOperand(1), valueMap);
  if (!lhs || !rhs)
    return failure();
  SmallVector<Value, 2> operands{lhs, rhs};
  valueMap[op->getResult(0)] =
      createALUAdd(builder, op->getLoc(), operands, *opcode,
                   op->getResult(0).getType());
  return success();
}

static LogicalResult lowerIntegerMul(Operation *op, OpBuilder &builder,
                                     llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("expected two operands and one result");
  Value lhs = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value rhs = lookupMappedValue(op, op->getOperand(1), valueMap);
  if (!lhs || !rhs)
    return failure();
  SmallVector<Value, 2> operands{lhs, rhs};
  valueMap[op->getResult(0)] =
      createALUMul(builder, op->getLoc(), operands,
                   mlir::vc4::MulOpcode::mul24,
                   op->getResult(0).getType());
  return success();
}

static LogicalResult lowerCmpI(Operation *op, OpBuilder &builder,
                               llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("expected two operands and one result");
  auto cmp = llvm::cast<arith::CmpIOp>(op);
  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::eq:
  case arith::CmpIPredicate::ne:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::uge:
    break;
  default:
    return op->emitOpError(
        "unsupported arith.cmpi predicate for VC4Tile-to-SSAVC4 lowering; "
        "supported predicates are eq, ne, ult, and uge");
  }
  Value lhs = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value rhs = lookupMappedValue(op, op->getOperand(1), valueMap);
  if (!lhs || !rhs)
    return failure();
  SmallVector<Value, 2> operands{lhs, rhs};
  valueMap[op->getResult(0)] =
      createMakeFlags(builder, op->getLoc(), operands,
                      mlir::ssavc4::FlagKind::compare);
  return success();
}

static Value getOrCreateRuntimeBuiltinUniformRead(
    Operation *op, OpBuilder &builder, llvm::StringMap<Value> &builtinValueMap,
    StringRef builtinName, Type resultType) {
  (void)resultType;
  auto cached = builtinValueMap.find(builtinName);
  if (cached != builtinValueMap.end())
    return cached->second;

  Operation *kernel = getParentVC4TileKernel(op);
  if (!kernel) {
    op->emitOpError("must be nested in vc4tile.kernel");
    return Value();
  }

  DictionaryAttr abi = buildLaunchABI(kernel, builder);
  auto builtins = abi.getAs<ArrayAttr>("builtins");
  if (!builtins) {
    op->emitOpError() << "could not find launch ABI builtin '" << builtinName
                      << "'";
    return Value();
  }

  std::optional<int64_t> targetIndex;
  for (Attribute attr : builtins) {
    auto dict = llvm::dyn_cast<DictionaryAttr>(attr);
    if (!dictionaryHasStringName(dict, builtinName))
      continue;
    targetIndex = getUniformIndex(dict);
    break;
  }
  if (!targetIndex) {
    op->emitOpError() << "could not find launch ABI builtin '" << builtinName
                      << "'";
    return Value();
  }

  Value requested;
  for (int64_t index = 0; index <= *targetIndex; ++index) {
    for (Attribute attr : builtins) {
      auto dict = llvm::dyn_cast<DictionaryAttr>(attr);
      if (!dict)
        continue;
      std::optional<int64_t> uniformIndex = getUniformIndex(dict);
      if (!uniformIndex || *uniformIndex != index)
        continue;
      StringAttr nameAttr = getABINameAttr(dict);
      if (!nameAttr || nameAttr.getValue().empty())
        continue;

      auto alreadyRead = builtinValueMap.find(nameAttr.getValue());
      Value read = alreadyRead == builtinValueMap.end()
                       ? createEntryUniformRead(builder, op->getLoc(),
                                                builder.getI32Type(), index)
                       : alreadyRead->second;
      if (alreadyRead == builtinValueMap.end())
        builtinValueMap[nameAttr.getValue()] = read;
      if (nameAttr.getValue() == builtinName)
        requested = read;
    }
  }

  if (!requested) {
    op->emitOpError() << "could not materialize launch ABI builtin '"
                      << builtinName << "'";
    return Value();
  }
  return requested;
}

static LogicalResult lowerRuntimeBuiltinId(Operation *op, OpBuilder &builder,
                                             llvm::DenseMap<Value, Value> &valueMap,
                                             llvm::StringMap<Value> &builtinValueMap,
                                             StringRef builtinName,
                                             StringRef opDescription) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  Type resultType = op->getResult(0).getType();
  if (!resultType.isSignlessInteger(32)) {
    return op->emitOpError() << "currently lowers only i32 " << opDescription
                             << " results";
  }

  Value read = getOrCreateRuntimeBuiltinUniformRead(
      op, builder, builtinValueMap, builtinName, resultType);
  if (!read)
    return failure();
  valueMap[op->getResult(0)] = read;
  return success();
}

static LogicalResult lowerProgramId(Operation *op, OpBuilder &builder,
                                    llvm::DenseMap<Value, Value> &valueMap,
                                    llvm::StringMap<Value> &builtinValueMap) {
  return lowerRuntimeBuiltinId(op, builder, valueMap, builtinValueMap,
                               "logical_request", "program_id");
}

static LogicalResult lowerBlockId(Operation *op, OpBuilder &builder,
                                  llvm::DenseMap<Value, Value> &valueMap,
                                  llvm::StringMap<Value> &builtinValueMap) {
  return lowerRuntimeBuiltinId(op, builder, valueMap, builtinValueMap,
                               "logical_block_id", "block_id");
}

static LogicalResult lowerWarpId(Operation *op, OpBuilder &builder,
                                 llvm::DenseMap<Value, Value> &valueMap,
                                 llvm::StringMap<Value> &builtinValueMap) {
  return lowerRuntimeBuiltinId(op, builder, valueMap, builtinValueMap,
                               "logical_warp_id", "warp_id");
}

static LogicalResult lowerLaneRange(Operation *op, OpBuilder &builder,
                                    llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  if (!isVector16I32(op->getResult(0).getType()))
    return op->emitOpError("currently lowers only vector<16xi32> lane_range results");
  valueMap[op->getResult(0)] = createElementNumber(builder, op->getLoc());
  return success();
}

static LogicalResult lowerLaneId(Operation *op, OpBuilder &builder,
                                 llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  if (!op->getResult(0).getType().isSignlessInteger(32) &&
      !isVector16I32(op->getResult(0).getType()))
    return op->emitOpError("currently lowers only i32 or vector<16xi32> lane_id results");
  // VC4's lane identity is represented by ELEMENT_NUMBER.  This is a 16-lane
  // vector value; scalar lane_id uses are accepted only when the value is not
  // subsequently consumed by a scalar-only op in this early lowering slice.
  valueMap[op->getResult(0)] = createElementNumber(builder, op->getLoc());
  return success();
}

static LogicalResult lowerThreadId(Operation *op, OpBuilder &builder,
                                   llvm::DenseMap<Value, Value> &valueMap,
                                   llvm::StringMap<Value> &builtinValueMap) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  Type resultType = op->getResult(0).getType();
  if (!isVector16I32(resultType))
    return op->emitOpError("currently lowers only vector<16xi32> thread_id results");

  Operation *kernel = getParentVC4TileKernel(op);
  if (!kernel)
    return op->emitOpError("must be nested in vc4tile.kernel");
  StringRef baseBuiltin = isCooperativeKernel(kernel) ? "logical_warp_id"
                                                      : "logical_request";
  Type i32Type = builder.getI32Type();
  Value logicalWarpOrRequest = getOrCreateRuntimeBuiltinUniformRead(
      op, builder, builtinValueMap, baseBuiltin, i32Type);
  if (!logicalWarpOrRequest)
    return failure();
  Value shiftFour = createLoadImmI32(builder, op->getLoc(), i32Type, 4);
  SmallVector<Value, 2> shiftOperands{logicalWarpOrRequest, shiftFour};
  Value base = createALUAdd(builder, op->getLoc(), shiftOperands,
                            mlir::vc4::AddOpcode::shl, i32Type);
  Value baseVec = createSplat(builder, op->getLoc(), base, resultType);
  Value lane = createElementNumber(builder, op->getLoc());
  SmallVector<Value, 2> addOperands{baseVec, lane};
  valueMap[op->getResult(0)] = createALUAdd(
      builder, op->getLoc(), addOperands, mlir::vc4::AddOpcode::add,
      resultType);
  return success();
}

static LogicalResult lowerMaskAll(Operation *op, OpBuilder &builder,
                                  llvm::DenseMap<Value, Value> &valueMap) {
  (void)builder;
  (void)valueMap;
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  if (!isVector16I1(op->getResult(0).getType()))
    return op->emitOpError("currently lowers only vector<16xi1> mask_all results");

  // VC4Tile mask values are semantic operands for M4 global memory and
  // reduction operations.  The current SSAVC4 flags value is only legal when
  // consumed exactly once by ssavc4.cond_br, so do not materialize mask_all as
  // ssavc4.make_flags unless a future non-memory mask consumer requires it.
  return success();
}

static LogicalResult lowerTailMask(Operation *op, OpBuilder &builder,
                                   llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("expected base, limit, and one result");
  if (!isVector16I1(op->getResult(0).getType()))
    return op->emitOpError("currently lowers only vector<16xi1> tail_mask results");

  Value base = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value limit = lookupMappedValue(op, op->getOperand(1), valueMap);
  if (!base || !limit)
    return failure();

  Type vecType = getVector16I32Type(builder);
  Value lanes = createElementNumber(builder, op->getLoc());
  Value baseVec = createSplat(builder, op->getLoc(), base, vecType);
  SmallVector<Value, 2> addOperands{baseVec, lanes};
  Value absoluteIndex = createALUAdd(builder, op->getLoc(), addOperands,
                                     mlir::vc4::AddOpcode::add, vecType);
  Value limitVec = createSplat(builder, op->getLoc(), limit, vecType);
  SmallVector<Value, 2> compareOperands{absoluteIndex, limitVec};
  valueMap[op->getResult(0)] =
      createMakeFlags(builder, op->getLoc(), compareOperands,
                      mlir::ssavc4::FlagKind::compare);
  return success();
}

static bool isDirectLaneRangeValue(Value value) {
  Operation *def = value.getDefiningOp();
  return hasName(def, kVC4TileLaneRangeOpName);
}

static bool isMaskedGlobalMemoryOp(Operation *op) {
  return hasName(op, kVC4TileMaskedLoadGlobalOpName) ||
         hasName(op, kVC4TileMaskedStoreGlobalOpName);
}

static bool valueHasOnlyMaskedGlobalMemoryUsers(Value value) {
  for (OpOperand &use : value.getUses()) {
    if (!isMaskedGlobalMemoryOp(use.getOwner()))
      return false;
  }
  return true;
}

static void eraseLoweredFlagMaskIfMemoryOnly(
    Value sourceMask, llvm::DenseMap<Value, Value> &valueMap) {
  if (!valueHasOnlyMaskedGlobalMemoryUsers(sourceMask))
    return;
  auto it = valueMap.find(sourceMask);
  if (it == valueMap.end())
    return;
  Operation *def = it->second.getDefiningOp();
  if (hasName(def, kSSAVC4MakeFlagsOpName) && def->use_empty()) {
    def->erase();
    valueMap.erase(it);
  }
}


static LogicalResult appendStoreActiveLaneOperand(Operation *op,
                                                  OpBuilder &builder,
                                                  llvm::DenseMap<Value, Value> &valueMap,
                                                  SmallVectorImpl<Value> &operands,
                                                  int64_t &activeLanesAttr) {
  Value sourceMask = op->getOperand(3);
  Operation *maskDef = sourceMask.getDefiningOp();

  if (hasName(maskDef, kVC4TileMaskAllOpName)) {
    activeLanesAttr = 16;
    eraseLoweredFlagMaskIfMemoryOnly(sourceMask, valueMap);
    return success();
  }

  if (hasName(maskDef, kVC4TileTailMaskOpName)) {
    if (maskDef->getNumOperands() != 2)
      return maskDef->emitOpError("expected base and limit operands");

    Value base = lookupMappedValue(maskDef, maskDef->getOperand(0), valueMap);
    Value limit = lookupMappedValue(maskDef, maskDef->getOperand(1), valueMap);
    if (!base || !limit)
      return failure();
    SmallVector<Value, 2> activeLaneOperands{limit, base};
    Value tailSpan = createALUAdd(builder, op->getLoc(), activeLaneOperands,
                                  mlir::vc4::AddOpcode::sub,
                                  builder.getI32Type());
    Value laneWidth =
        createLoadImmI32(builder, op->getLoc(), builder.getI32Type(), 16);
    SmallVector<Value, 2> clampOperands{tailSpan, laneWidth};
    Value activeLanes = createALUAdd(builder, op->getLoc(), clampOperands,
                                     mlir::vc4::AddOpcode::min,
                                     builder.getI32Type());
    operands.push_back(activeLanes);
    activeLanesAttr = 16;
    eraseLoweredFlagMaskIfMemoryOnly(sourceMask, valueMap);
    return success();
  }

  return op->emitOpError(
      "currently supports only vc4tile.mask_all or vc4tile.tail_mask masks for coalesced VDW stores");
}

static LogicalResult verifyMaskedLoadShape(Operation *op) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return op->emitOpError("expected base, offsets, mask, and one result");
  if (!op->getOperand(0).getType().isSignlessInteger(32))
    return op->emitOpError("requires an i32 scalar global base address");
  if (!isVector16I32(op->getOperand(1).getType()))
    return op->emitOpError("requires vector<16xi32> offsets");
  if (!isVector16I1(op->getOperand(2).getType()))
    return op->emitOpError("requires a vector<16xi1> mask");
  if (!isVector16Data(op->getResult(0).getType()))
    return op->emitOpError("requires a vector<16xi32> or vector<16xf32> result");

  auto elemBytes = op->getAttrOfType<IntegerAttr>("elem_bytes");
  if (!elemBytes || elemBytes.getInt() != 4)
    return op->emitOpError("supports only elem_bytes = 4 for TMU loads");

  auto offsetUnit = op->getAttrOfType<mlir::vc4tile::OffsetUnitAttr>("offset_unit");
  if (!offsetUnit)
    return op->emitOpError("requires an offset_unit attribute");
  if (offsetUnit.getValue() != mlir::vc4tile::OffsetUnit::element)
    return op->emitOpError("supports only element offsets for coalesced TMU loads");

  Operation *offsetDef = op->getOperand(1).getDefiningOp();
  if (!hasName(offsetDef, kVC4TileLaneRangeOpName)) {
    return op->emitOpError(
        "requires offsets to be the direct vc4tile.lane_range value for the M4 coalesced TMU subset");
  }

  auto access = op->getAttrOfType<mlir::vc4tile::MemoryAccessAttr>("access");
  if (access && access.getValue() != mlir::vc4tile::MemoryAccess::coalesced)
    return op->emitOpError("supports only access = #vc4tile.memory_access<coalesced> for TMU loads");

  auto memorySpace = op->getAttrOfType<mlir::vc4tile::MemorySpaceAttr>("memory_space");
  if (memorySpace && memorySpace.getValue() != mlir::vc4tile::MemorySpace::global)
    return op->emitOpError("requires memory_space = #vc4tile.memory_space<global>");

  Operation *maskDef = op->getOperand(2).getDefiningOp();
  if (!hasName(maskDef, kVC4TileMaskAllOpName) &&
      !hasName(maskDef, kVC4TileTailMaskOpName)) {
    return op->emitOpError(
        "currently supports only vc4tile.mask_all or vc4tile.tail_mask masks for TMU loads");
  }

  return success();
}

static LogicalResult lowerMaskedLoadGlobal(Operation *op, OpBuilder &builder,
                                           llvm::DenseMap<Value, Value> &valueMap) {
  if (failed(verifyMaskedLoadShape(op)))
    return failure();

  Value base = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value offsets = lookupMappedValue(op, op->getOperand(1), valueMap);
  if (!base || !offsets)
    return failure();

  Type addressType = getVector16I32Type(builder);
  Value baseVec = createSplat(builder, op->getLoc(), base, addressType);
  Value byteOffsets = offsets;

  auto offsetUnit =
      op->getAttrOfType<mlir::vc4tile::OffsetUnitAttr>("offset_unit");
  if (offsetUnit.getValue() == mlir::vc4tile::OffsetUnit::element) {
    Value four = createLoadImmI32(builder, op->getLoc(), addressType, 4);
    SmallVector<Value, 2> mulOperands{offsets, four};
    byteOffsets = createALUMul(builder, op->getLoc(), mulOperands,
                               mlir::vc4::MulOpcode::mul24, addressType);
  }

  SmallVector<Value, 2> addressOperands{baseVec, byteOffsets};
  Value address = createALUAdd(builder, op->getLoc(), addressOperands,
                               mlir::vc4::AddOpcode::add, addressType);
  Value token = createTMURequest(builder, op->getLoc(), address);
  valueMap[op->getResult(0)] =
      createTMURead(builder, op->getLoc(), token, op->getResult(0).getType());
  eraseLoweredFlagMaskIfMemoryOnly(op->getOperand(2), valueMap);
  return success();
}

static LogicalResult verifyCoalescedStoreShape(Operation *op) {
  if (op->getNumOperands() != 4)
    return op->emitOpError("expected base, offsets, value, and mask operands");
  if (!op->getOperand(0).getType().isSignlessInteger(32))
    return op->emitOpError("requires an i32 scalar global base address");
  if (!isVector16I32(op->getOperand(1).getType()))
    return op->emitOpError("requires vector<16xi32> offsets");
  if (!isVector16Data(op->getOperand(2).getType()))
    return op->emitOpError("requires a vector<16xi32> or vector<16xf32> value");

  auto elemBytes = op->getAttrOfType<IntegerAttr>("elem_bytes");
  if (!elemBytes || elemBytes.getInt() != 4)
    return op->emitOpError("supports only elem_bytes = 4 for VDW stores");

  auto offsetUnit = op->getAttrOfType<mlir::vc4tile::OffsetUnitAttr>("offset_unit");
  if (!offsetUnit || offsetUnit.getValue() != mlir::vc4tile::OffsetUnit::element) {
    return op->emitOpError(
        "currently supports only element offsets for coalesced VDW stores");
  }

  auto memorySpace = op->getAttrOfType<mlir::vc4tile::MemorySpaceAttr>("memory_space");
  if (memorySpace && memorySpace.getValue() != mlir::vc4tile::MemorySpace::global)
    return op->emitOpError("requires memory_space = #vc4tile.memory_space<global>");

  auto access = op->getAttrOfType<mlir::vc4tile::MemoryAccessAttr>("access");
  if (access && access.getValue() == mlir::vc4tile::MemoryAccess::generic) {
    return op->emitOpError(
        "generic per-lane store access is outside the M4 hardware-lowered contract");
  }
  if (access && access.getValue() != mlir::vc4tile::MemoryAccess::coalesced &&
      access.getValue() != mlir::vc4tile::MemoryAccess::affine_contiguous) {
    return op->emitOpError(
        "requires coalesced or affine_contiguous store access");
  }

  if (!isDirectLaneRangeValue(op->getOperand(1))) {
    return op->emitOpError(
        "requires offsets to be the direct vc4tile.lane_range value for the M4 coalesced VDW subset");
  }

  return success();
}

static LogicalResult lowerMaskedStoreGlobal(Operation *op, OpBuilder &builder,
                                            llvm::DenseMap<Value, Value> &valueMap) {
  if (failed(verifyCoalescedStoreShape(op)))
    return failure();

  Value base = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value value = lookupMappedValue(op, op->getOperand(2), valueMap);
  if (!base || !value)
    return failure();

  SmallVector<Value, 4> operands{base, value};
  int64_t activeLanesAttr = 16;
  if (failed(appendStoreActiveLaneOperand(op, builder, valueMap, operands,
                                          activeLanesAttr)))
    return failure();

  // VDW global-store lowering stages the outgoing vector through one VPM row
  // before kicking the VDW DMA path.  In ordinary global-store kernels row 0 is
  // harmless and matches the earlier M4 fixtures.  In shared-VPM kernels,
  // however, row 0 can hold user shared tile data; clobbering it between
  // column reads turns a transpose back into an untransposed row copy.  Use the
  // top user-visible VPM row as scratch when shared VPM is active, matching the
  // existing SSAVC4 shared-transpose fixture convention.
  Operation *kernel = getParentVC4TileKernel(op);
  int64_t stagingRow = 0;
  if (kernel && (getBoolAttr(kernel, "uses_shared_vpm") ||
                 kernelContains(kernel, kVC4TileSharedAllocOpName) ||
                 kernelContains(kernel, kVC4TileSharedLoadOpName) ||
                 kernelContains(kernel, kVC4TileSharedStoreOpName)))
    stagingRow = 63;

  SmallVector<int32_t, 4> operandSegmentSizes{
      1, 1, static_cast<int32_t>(operands.size() == 3), 0};
  createSSAVC4Op(builder, op->getLoc(), kSSAVC4VDWStoreOpName, operands,
                 {builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4)),
                  builder.getNamedAttr("active_lanes",
                                       builder.getI32IntegerAttr(activeLanesAttr)),
                  builder.getNamedAttr("vpm_row",
                                       builder.getI32IntegerAttr(stagingRow)),
                  builder.getNamedAttr("serialize", builder.getStringAttr("mutex")),
                  builder.getNamedAttr(
                      "operandSegmentSizes",
                      builder.getDenseI32ArrayAttr(operandSegmentSizes))});
  return success();
}

static LogicalResult lowerRotate(Operation *op, OpBuilder &builder,
                                 llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return op->emitOpError("expected one input operand and one result");

  Type inputType = op->getOperand(0).getType();
  Type resultType = op->getResult(0).getType();
  if (!isVector16Data(inputType) || inputType != resultType)
    return op->emitOpError(
        "requires matching vector<16xi32> or vector<16xf32> input/result types");

  auto amountAttr = op->getAttrOfType<IntegerAttr>("amount");
  if (!amountAttr)
    return op->emitOpError("requires an i32 amount attribute");
  int64_t amount = amountAttr.getInt();
  if (amount < 0 || amount > 15)
    return op->emitOpError("amount must be in range [0, 15]");

  Value input = lookupMappedValue(op, op->getOperand(0), valueMap);
  if (!input)
    return failure();
  valueMap[op->getResult(0)] =
      createRotate(builder, op->getLoc(), input, amount, resultType);
  return success();
}

static bool maskIsAllLanes(Operation *op, unsigned operandIndex) {
  if (operandIndex >= op->getNumOperands())
    return false;
  return hasName(op->getOperand(operandIndex).getDefiningOp(),
                 kVC4TileMaskAllOpName);
}

static LogicalResult lowerReduce(Operation *op, OpBuilder &builder,
                                 llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("expected input, mask, and one result");

  Type inputType = op->getOperand(0).getType();
  Type resultType = op->getResult(0).getType();
  if (!isVector16Data(inputType) || inputType != resultType)
    return op->emitOpError(
        "requires matching vector<16xi32> or vector<16xf32> input/result types");
  if (!isVector16I1(op->getOperand(1).getType()))
    return op->emitOpError("requires a vector<16xi1> mask");
  if (!maskIsAllLanes(op, 1))
    return op->emitOpError(
        "currently lowers only vc4tile.mask_all masks for warp reductions");

  auto kind = op->getAttrOfType<mlir::vc4tile::ReduceKindAttr>("kind");
  if (!kind)
    return op->emitOpError("requires a reduce kind attribute");
  if (kind.getValue() != mlir::vc4tile::ReduceKind::add)
    return op->emitOpError(
        "currently lowers only add reductions through SSAVC4 rotate/add");

  Value acc = lookupMappedValue(op, op->getOperand(0), valueMap);
  if (!acc)
    return failure();

  mlir::vc4::AddOpcode addOpcode = isVector16F32(inputType)
                                      ? mlir::vc4::AddOpcode::fadd
                                      : mlir::vc4::AddOpcode::add;
  for (int64_t amount : {8, 4, 2, 1}) {
    Value rotated = createRotate(builder, op->getLoc(), acc, amount, resultType);
    SmallVector<Value, 2> operands{acc, rotated};
    acc = createALUAdd(builder, op->getLoc(), operands, addOpcode, resultType);
  }

  valueMap[op->getResult(0)] = acc;
  return success();
}

static LogicalResult lowerSharedAlloc(Operation *op, OpBuilder &builder,
                                      llvm::DenseMap<Value, Value> &valueMap,
                                      llvm::StringMap<Value> &builtinValueMap,
                                      SharedVPMAllocationState &sharedState) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one shared tile result");

  auto rowsAttr = op->getAttrOfType<IntegerAttr>("rows");
  if (!rowsAttr || rowsAttr.getInt() <= 0)
    return op->emitOpError("requires a positive rows attribute");

  Operation *kernel = getParentVC4TileKernel(op);
  if (!kernel)
    return op->emitOpError("must be nested in vc4tile.kernel");
  int64_t declaredRows = getI32Attr(kernel, "vpm_rows_per_block").value_or(0);
  int64_t rowOffset = sharedState.nextRowOffset;
  int64_t newEnd = rowOffset + rowsAttr.getInt();
  if (declaredRows > 0 && newEnd > declaredRows) {
    return op->emitOpError()
           << "shared_alloc rows exceed parent kernel vpm_rows_per_block";
  }
  sharedState.nextRowOffset = newEnd;

  Value baseRow = getOrCreateRuntimeBuiltinUniformRead(
      op, builder, builtinValueMap, "vpm_base_row", builder.getI32Type());
  if (!baseRow)
    return failure();

  if (rowOffset == 0) {
    valueMap[op->getResult(0)] = baseRow;
    return success();
  }

  Value offset = createLoadImmI32(builder, op->getLoc(), builder.getI32Type(),
                                 rowOffset);
  SmallVector<Value, 2> operands{baseRow, offset};
  valueMap[op->getResult(0)] = createALUAdd(
      builder, op->getLoc(), operands, mlir::vc4::AddOpcode::add,
      builder.getI32Type());
  return success();
}

static LogicalResult lowerSharedStore(Operation *op, OpBuilder &builder,
                                      llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 4)
    return op->emitOpError("expected handle, row, value, and mask operands");
  if (!isAllLanesMask(op->getOperand(3)))
    return op->emitOpError(
        "currently lowers only vc4tile.mask_all masks for shared VPM stores");

  auto elemBytes = op->getAttrOfType<IntegerAttr>("elem_bytes");
  if (!elemBytes || elemBytes.getInt() != 4)
    return op->emitOpError("supports only elem_bytes = 4 for shared VPM stores");
  if (!isVector16Data(op->getOperand(2).getType()))
    return op->emitOpError("requires a vector<16xi32> or vector<16xf32> value");

  Value baseRow = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value localRow = lookupMappedValue(op, op->getOperand(1), valueMap);
  Value value = lookupMappedValue(op, op->getOperand(2), valueMap);
  if (!baseRow || !localRow || !value)
    return failure();

  Value row = createVPMRowAddress(builder, op->getLoc(), baseRow, localRow);
  createSSAVC4Op(builder, op->getLoc(), kSSAVC4VPMWriteOpName, {row, value},
                 {builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4)),
                  builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16)),
                  builder.getNamedAttr("orientation", builder.getStringAttr(
                      getVPMOrientation(op, "horizontal"))),
                  builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))});
  return success();
}

static LogicalResult lowerSharedLoad(Operation *op, OpBuilder &builder,
                                     llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return op->emitOpError("expected handle, row, mask, and one result");
  if (!isAllLanesMask(op->getOperand(2)))
    return op->emitOpError(
        "currently lowers only vc4tile.mask_all masks for shared VPM loads");

  auto elemBytes = op->getAttrOfType<IntegerAttr>("elem_bytes");
  if (!elemBytes || elemBytes.getInt() != 4)
    return op->emitOpError("supports only elem_bytes = 4 for shared VPM loads");
  if (!isVector16Data(op->getResult(0).getType()))
    return op->emitOpError(
        "requires a vector<16xi32> or vector<16xf32> result");

  Value baseRow = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value localRow = lookupMappedValue(op, op->getOperand(1), valueMap);
  if (!baseRow || !localRow)
    return failure();

  Value row = createVPMRowAddress(builder, op->getLoc(), baseRow, localRow);
  valueMap[op->getResult(0)] = createSSAVC4OpWithResult(
      builder, op->getLoc(), kSSAVC4VPMReadOpName, {row},
      {builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4)),
       builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16)),
       builder.getNamedAttr("orientation", builder.getStringAttr(
           getVPMOrientation(op, "horizontal"))),
       builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))},
      op->getResult(0).getType());
  return success();
}

static LogicalResult lowerBarrier(Operation *op, OpBuilder &builder) {
  Operation *kernel = getParentVC4TileKernel(op);
  if (!kernel)
    return op->emitOpError("must be nested in vc4tile.kernel");
  if (!isCooperativeKernel(kernel)) {
    return op->emitOpError(
        "requires parent vc4tile.kernel schedule_mode = #vc4tile.schedule_mode<cooperative_block>");
  }
  if (!getBoolAttr(kernel, "uses_barrier")) {
    return op->emitOpError(
        "requires parent vc4tile.kernel to set uses_barrier = true");
  }
  if (!getBoolAttr(kernel, "require_full_block_residency")) {
    return op->emitOpError(
        "requires parent vc4tile.kernel to set require_full_block_residency = true");
  }
  if (getI32Attr(kernel, "semaphores_per_block").value_or(0) != 4)
    return op->emitOpError("requires parent semaphores_per_block = 4");

  createSSAVC4Op(builder, op->getLoc(), kSSAVC4BarrierOpName, {},
                 {builder.getNamedAttr("arrive_offset",
                                       builder.getI32IntegerAttr(0)),
                  builder.getNamedAttr("go_offset",
                                       builder.getI32IntegerAttr(1)),
                  builder.getNamedAttr("depart_offset",
                                       builder.getI32IntegerAttr(2)),
                  builder.getNamedAttr("reset_offset",
                                       builder.getI32IntegerAttr(3))});
  return success();
}

static LogicalResult lowerBodyOp(Operation *op, OpBuilder &builder,
                                 llvm::DenseMap<Value, Value> &valueMap,
                                 llvm::StringMap<Value> &builtinValueMap,
                                 SharedVPMAllocationState &sharedState) {
  if (hasName(op, kArithConstantOpName))
    return lowerConstant(op, builder, valueMap);
  if (hasName(op, kVectorSplatOpName))
    return lowerVectorSplat(op, builder, valueMap);
  if (hasName(op, kArithMulIOpName))
    return lowerIntegerMul(op, builder, valueMap);
  if (getIntegerAddOpcode(op))
    return lowerIntegerALU(op, builder, valueMap);
  if (hasName(op, kArithCmpIOpName))
    return lowerCmpI(op, builder, valueMap);
  if (hasName(op, kVC4TileProgramIdOpName))
    return lowerProgramId(op, builder, valueMap, builtinValueMap);
  if (hasName(op, kVC4TileBlockIdOpName))
    return lowerBlockId(op, builder, valueMap, builtinValueMap);
  if (hasName(op, kVC4TileWarpIdOpName))
    return lowerWarpId(op, builder, valueMap, builtinValueMap);
  if (hasName(op, kVC4TileLaneRangeOpName))
    return lowerLaneRange(op, builder, valueMap);
  if (hasName(op, kVC4TileLaneIdOpName))
    return lowerLaneId(op, builder, valueMap);
  if (hasName(op, kVC4TileThreadIdOpName))
    return lowerThreadId(op, builder, valueMap, builtinValueMap);
  if (hasName(op, kVC4TileMaskAllOpName))
    return lowerMaskAll(op, builder, valueMap);
  if (hasName(op, kVC4TileTailMaskOpName))
    return lowerTailMask(op, builder, valueMap);
  if (hasName(op, kVC4TileMaskedLoadGlobalOpName))
    return lowerMaskedLoadGlobal(op, builder, valueMap);
  if (hasName(op, kVC4TileMaskedStoreGlobalOpName))
    return lowerMaskedStoreGlobal(op, builder, valueMap);
  if (hasName(op, kVC4TileRotateOpName))
    return lowerRotate(op, builder, valueMap);
  if (hasName(op, kVC4TileReduceOpName))
    return lowerReduce(op, builder, valueMap);
  if (hasName(op, kVC4TileSharedAllocOpName))
    return lowerSharedAlloc(op, builder, valueMap, builtinValueMap, sharedState);
  if (hasName(op, kVC4TileSharedStoreOpName))
    return lowerSharedStore(op, builder, valueMap);
  if (hasName(op, kVC4TileSharedLoadOpName))
    return lowerSharedLoad(op, builder, valueMap);
  if (hasName(op, kVC4TileBarrierOpName))
    return lowerBarrier(op, builder);

  return op->emitOpError(
      "is not supported yet by --convert-vc4tile-to-ssavc4 in this M4 slice");
}

static Operation *createSSAVC4Branch(OpBuilder &builder, Location loc,
                                          Block *target,
                                          ArrayRef<Value> targetOperands) {
  OperationState state(loc, kSSAVC4BranchOpName);
  state.addOperands(targetOperands);
  state.addSuccessors(target);
  return builder.create(state);
}

static Operation *createSSAVC4CondBranch(OpBuilder &builder, Location loc,
                                         Value flags,
                                         mlir::vc4::BranchCond cond,
                                         Block *trueDest,
                                         ArrayRef<Value> trueDestOperands,
                                         Block *falseDest,
                                         ArrayRef<Value> falseDestOperands) {
  OperationState state(loc, kSSAVC4CondBranchOpName);
  SmallVector<Value, 1> flagOperands{flags};
  state.addOperands(flagOperands);
  state.addOperands(trueDestOperands);
  state.addOperands(falseDestOperands);
  state.addSuccessors(trueDest);
  state.addSuccessors(falseDest);
  state.addAttribute(
      "cond", mlir::vc4::BranchCondAttr::get(builder.getContext(), cond));
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(trueDestOperands.size()),
           static_cast<int32_t>(falseDestOperands.size())}));
  return builder.create(state);
}

static FailureOr<mlir::vc4::BranchCond>
getBranchCondForArithCmpI(Value condition) {
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return failure();

  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return mlir::vc4::BranchCond::any_z_set;
  case arith::CmpIPredicate::ne:
    return mlir::vc4::BranchCond::any_z_clear;
  case arith::CmpIPredicate::ult:
    return mlir::vc4::BranchCond::any_c_set;
  case arith::CmpIPredicate::uge:
    return mlir::vc4::BranchCond::any_c_clear;
  default:
    return failure();
  }
}

static Block *createSSAVC4FalseFallthroughBlock(
    OpBuilder &builder, Location loc, Block *falseDest,
    ArrayRef<Value> falseDestOperands) {
  Block *currentBlock = builder.getInsertionBlock();
  if (!currentBlock || !currentBlock->getParent())
    return nullptr;

  SmallVector<Type, 4> argTypes;
  SmallVector<Location, 4> argLocs;
  argTypes.reserve(falseDestOperands.size());
  argLocs.reserve(falseDestOperands.size());
  for (Value operand : falseDestOperands) {
    argTypes.push_back(operand.getType());
    argLocs.push_back(loc);
  }

  Region *parentRegion = currentBlock->getParent();
  Block *fallthrough = builder.createBlock(
      parentRegion, std::next(currentBlock->getIterator()), argTypes, argLocs);

  SmallVector<Value, 4> forwardedOperands;
  forwardedOperands.reserve(fallthrough->getNumArguments());
  for (BlockArgument arg : fallthrough->getArguments())
    forwardedOperands.push_back(arg);
  createSSAVC4Branch(builder, loc, falseDest, forwardedOperands);

  builder.setInsertionPointToEnd(currentBlock);
  return fallthrough;
}

static unsigned countVC4TileReturnTerminators(Operation *kernel) {
  unsigned count = 0;
  kernel->walk([&](Operation *op) {
    if (hasName(op, kVC4TileReturnOpName))
      ++count;
  });
  return count;
}

static Block *createSSAVC4ReturnBlock(Operation *func, Location loc) {
  Region &region = func->getRegion(0);
  Block *returnBlock = new Block();
  region.push_back(returnBlock);

  OpBuilder builder(func->getContext());
  builder.setInsertionPointToEnd(returnBlock);
  OperationState threadEndState(loc, kSSAVC4ThreadEndOpName);
  builder.create(threadEndState);
  return returnBlock;
}

static LogicalResult lowerBranchTerminator(Operation *op, OpBuilder &builder,
                                           llvm::DenseMap<Value, Value> &valueMap,
                                           llvm::DenseMap<Block *, Block *> &blockMap,
                                           Block *returnBlock) {
  if (hasName(op, kVC4TileReturnOpName)) {
    if (returnBlock) {
      createSSAVC4Branch(builder, op->getLoc(), returnBlock, {});
      return success();
    }

    OperationState threadEndState(op->getLoc(), kSSAVC4ThreadEndOpName);
    builder.create(threadEndState);
    return success();
  }

  if (auto branch = llvm::dyn_cast<mlir::cf::BranchOp>(op)) {
    SmallVector<Value, 4> operands;
    operands.reserve(branch.getDestOperands().size());
    for (Value operand : branch.getDestOperands()) {
      Value mapped = lookupMappedValue(op, operand, valueMap);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    auto targetIt = blockMap.find(branch.getDest());
    if (targetIt == blockMap.end())
      return op->emitOpError("branches to a block outside the lowered kernel");
    createSSAVC4Branch(builder, op->getLoc(), targetIt->second, operands);
    return success();
  }

  if (auto condBranch = llvm::dyn_cast<mlir::cf::CondBranchOp>(op)) {
    if (!condBranch.getCondition().getType().isInteger(1)) {
      return op->emitOpError(
          "requires a uniform i1 condition; vector masks must stay as masks");
    }

    FailureOr<mlir::vc4::BranchCond> branchCond =
        getBranchCondForArithCmpI(condBranch.getCondition());
    if (failed(branchCond)) {
      return op->emitOpError(
          "condition must be produced by arith.cmpi with a supported "
          "predicate for VC4Tile-to-SSAVC4 lowering");
    }

    Value flags = lookupMappedValue(op, condBranch.getCondition(), valueMap);
    if (!flags)
      return failure();
    if (!llvm::isa<mlir::ssavc4::FlagsType>(flags.getType())) {
      return op->emitOpError(
          "condition must be produced by a lowerable uniform comparison and lower to !ssavc4.flags");
    }

    SmallVector<Value, 4> trueOperands;
    trueOperands.reserve(condBranch.getTrueDestOperands().size());
    for (Value operand : condBranch.getTrueDestOperands()) {
      Value mapped = lookupMappedValue(op, operand, valueMap);
      if (!mapped)
        return failure();
      trueOperands.push_back(mapped);
    }

    SmallVector<Value, 4> falseOperands;
    falseOperands.reserve(condBranch.getFalseDestOperands().size());
    for (Value operand : condBranch.getFalseDestOperands()) {
      Value mapped = lookupMappedValue(op, operand, valueMap);
      if (!mapped)
        return failure();
      falseOperands.push_back(mapped);
    }

    auto trueIt = blockMap.find(condBranch.getTrueDest());
    auto falseIt = blockMap.find(condBranch.getFalseDest());
    if (trueIt == blockMap.end() || falseIt == blockMap.end())
      return op->emitOpError("branches to a block outside the lowered kernel");
    Block *falseFallthrough = createSSAVC4FalseFallthroughBlock(
        builder, op->getLoc(), falseIt->second, falseOperands);
    if (!falseFallthrough)
      return op->emitOpError("internal error: unable to create false fallthrough block");
    createSSAVC4CondBranch(builder, op->getLoc(), flags, *branchCond,
                           trueIt->second, trueOperands, falseFallthrough,
                           falseOperands);
    return success();
  }

  return op->emitOpError("unsupported VC4Tile control-flow terminator");
}

static bool condBranchHasNoSuccessorOperands(mlir::cf::CondBranchOp branch) {
  return branch.getTrueDestOperands().empty() &&
         branch.getFalseDestOperands().empty();
}

static void appendSourceBlockInFalseFallthroughOrder(
    Block *sourceBlock, SmallVectorImpl<Block *> &orderedBlocks,
    llvm::DenseMap<Block *, bool> &alreadyOrdered) {
  if (!sourceBlock || alreadyOrdered.lookup(sourceBlock))
    return;

  alreadyOrdered[sourceBlock] = true;
  orderedBlocks.push_back(sourceBlock);

  auto condBranch =
      llvm::dyn_cast_or_null<mlir::cf::CondBranchOp>(sourceBlock->getTerminator());
  if (!condBranch || !condBranchHasNoSuccessorOperands(condBranch))
    return;

  appendSourceBlockInFalseFallthroughOrder(condBranch.getFalseDest(),
                                           orderedBlocks, alreadyOrdered);
  appendSourceBlockInFalseFallthroughOrder(condBranch.getTrueDest(),
                                           orderedBlocks, alreadyOrdered);
}

static SmallVector<Block *, 8> computeSSAVC4BlockOrder(Operation *kernel) {
  SmallVector<Block *, 8> orderedBlocks;
  llvm::DenseMap<Block *, bool> alreadyOrdered;
  Region &sourceRegion = kernel->getRegion(0);

  if (!sourceRegion.empty())
    appendSourceBlockInFalseFallthroughOrder(&sourceRegion.front(), orderedBlocks,
                                             alreadyOrdered);

  for (Block &sourceBlock : sourceRegion)
    appendSourceBlockInFalseFallthroughOrder(&sourceBlock, orderedBlocks,
                                             alreadyOrdered);
  return orderedBlocks;
}

static LogicalResult createSSAVC4Blocks(Operation *kernel, Operation *func,
                                        llvm::DenseMap<Value, Value> &valueMap,
                                        llvm::DenseMap<Block *, Block *> &blockMap) {
  Region &destRegion = func->getRegion(0);
  Block *entryBlock = &kernel->getRegion(0).front();

  for (Block *sourceBlock : computeSSAVC4BlockOrder(kernel)) {
    Block *destBlock = new Block();
    destRegion.push_back(destBlock);
    blockMap[sourceBlock] = destBlock;

    // VC4Tile entry block arguments are user/caller ABI inputs.  They lower to
    // explicit ssavc4.uniform.read operations in the entry block, not to
    // SSAVC4 block arguments.  Non-entry block arguments remain real CFG block
    // arguments and lower through SSAVC4 successor operands.
    if (sourceBlock == entryBlock)
      continue;

    for (BlockArgument arg : sourceBlock->getArguments()) {
      BlockArgument loweredArg = destBlock->addArgument(arg.getType(), arg.getLoc());
      valueMap[arg] = loweredArg;
    }
  }
  return success();
}

static LogicalResult mapKernelFormalArguments(Operation *kernel,
                                              OpBuilder &builder,
                                              llvm::DenseMap<Value, Value> &valueMap) {
  if (kernel->getRegion(0).empty())
    return success();

  Block &entry = kernel->getRegion(0).front();
  for (BlockArgument arg : entry.getArguments()) {
    std::optional<int64_t> uniformIndex =
        lookupLaunchArgUniformIndex(kernel, builder, arg.getArgNumber());
    if (!uniformIndex) {
      return kernel->emitOpError()
             << "missing vc4.launch_abi.args[] entry for kernel formal argument "
             << arg.getArgNumber();
    }
    if (!arg.getType().isSignlessInteger(32) && !arg.getType().isF32()) {
      return kernel->emitOpError()
             << "currently lowers only i32 or f32 scalar kernel formal arguments";
    }
    valueMap[arg] = createUniformRead(builder, arg.getLoc(), arg.getType(),
                                      *uniformIndex);
  }
  return success();
}

static LogicalResult mapLaunchBuiltins(Operation *kernel, OpBuilder &builder,
                                             llvm::StringMap<Value> &builtinValueMap) {
  (void)builtinValueMap;
  DictionaryAttr abi = buildLaunchABI(kernel, builder);
  auto builtins = abi.getAs<ArrayAttr>("builtins");
  if (!builtins)
    return success();

  // Runtime builtins are launch ABI metadata.  Only VC4Tile operations that
  // actually need a builtin SSA value should materialize an ssavc4.uniform.read
  // through getOrCreateRuntimeBuiltinUniformRead().  Eagerly reading all
  // metadata builtins would consume uniform stream entries for metadata-only
  // users such as barriers and would also violate the no-argument minimal ABI
  // canary.
  for (Attribute attr : builtins) {
    auto dict = llvm::dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return kernel->emitOpError("vc4.launch_abi.builtins entries must be dictionaries");

    StringAttr nameAttr = getABINameAttr(dict);
    if (!nameAttr || nameAttr.getValue().empty())
      return kernel->emitOpError("vc4.launch_abi.builtins entries require a name");
    std::optional<int64_t> uniformIndex = getUniformIndex(dict);
    if (!uniformIndex)
      return kernel->emitOpError() << "vc4.launch_abi builtin '"
                                   << nameAttr.getValue()
                                   << "' requires uniform_index";
  }
  return success();
}

static LogicalResult lowerKernelBody(Operation *kernel, Operation *func) {
  OpBuilder bodyBuilder(func->getContext());
  llvm::DenseMap<Value, Value> valueMap;
  llvm::StringMap<Value> builtinValueMap;
  llvm::DenseMap<Block *, Block *> blockMap;
  SharedVPMAllocationState sharedState;

  if (failed(createSSAVC4Blocks(kernel, func, valueMap, blockMap)))
    return failure();

  Block *returnBlock = nullptr;
  if (countVC4TileReturnTerminators(kernel) > 1)
    returnBlock = createSSAVC4ReturnBlock(func, kernel->getLoc());

  bodyBuilder.setInsertionPointToStart(blockMap.lookup(&kernel->getRegion(0).front()));
  if (failed(mapKernelFormalArguments(kernel, bodyBuilder, valueMap)))
    return failure();
  if (failed(mapLaunchBuiltins(kernel, bodyBuilder, builtinValueMap)))
    return failure();

  bool sawReturn = false;
  for (Block &sourceBlock : kernel->getRegion(0)) {
    Block *destBlock = blockMap.lookup(&sourceBlock);
    if (!destBlock)
      return kernel->emitOpError("internal error: missing lowered block");
    bodyBuilder.setInsertionPointToEnd(destBlock);

    for (Operation &nested : sourceBlock) {
      if (nested.hasTrait<OpTrait::IsTerminator>()) {
        if (hasName(&nested, kVC4TileReturnOpName))
          sawReturn = true;
        if (failed(lowerBranchTerminator(&nested, bodyBuilder, valueMap,
                                         blockMap, returnBlock)))
          return failure();
        continue;
      }

      if (failed(lowerBodyOp(&nested, bodyBuilder, valueMap, builtinValueMap,
                               sharedState)))
        return failure();
    }
  }

  if (!sawReturn)
    return kernel->emitOpError("requires a vc4tile.return terminator");
  return success();
}

static bool isIndexCastOp(Operation *op) {
  return hasName(op, kArithIndexCastOpName) ||
         hasName(op, kArithIndexCastUIOpName) ||
         hasName(op, kArithIndexCastSIOpName);
}

static Operation *createCFBranch(OpBuilder &builder, Location loc,
                                 Block *target,
                                 ArrayRef<Value> targetOperands) {
  OperationState state(loc, kCFBranchOpName);
  state.addOperands(targetOperands);
  state.addSuccessors(target);
  return builder.create(state);
}

static Operation *createCFCondBranch(OpBuilder &builder, Location loc,
                                     Value condition, Block *trueDest,
                                     ArrayRef<Value> trueOperands,
                                     Block *falseDest,
                                     ArrayRef<Value> falseOperands) {
  OperationState state(loc, kCFCondBranchOpName);
  state.addOperands(condition);
  state.addOperands(trueOperands);
  state.addOperands(falseOperands);
  state.addSuccessors(trueDest);
  state.addSuccessors(falseDest);
  state.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(trueOperands.size()),
           static_cast<int32_t>(falseOperands.size())}));
  return builder.create(state);
}

static LogicalResult moveSCFYieldRegionToBlock(OpBuilder &builder,
                                                Region &sourceRegion,
                                                Block *destBlock,
                                                Block *mergeBlock) {
  if (sourceRegion.empty())
    return failure();
  Block &sourceBlock = sourceRegion.front();
  Operation *terminator = sourceBlock.getTerminator();
  if (!terminator || !hasName(terminator, "scf.yield"))
    return emitError(sourceRegion.getLoc(),
                     "unsupported scf: expected scf.yield terminator");

  while (!sourceBlock.empty() && &sourceBlock.front() != terminator)
    sourceBlock.front().moveBefore(destBlock, destBlock->end());

  SmallVector<Value, 4> yieldedOperands(terminator->getOperands());
  builder.setInsertionPointToEnd(destBlock);
  createCFBranch(builder, terminator->getLoc(), mergeBlock, yieldedOperands);
  return success();
}

static LogicalResult legalizeSCFIfOp(mlir::scf::IfOp ifOp) {
  Operation *op = ifOp.getOperation();
  if (!getParentVC4TileKernel(op))
    return success();
  if (!ifOp.getCondition().getType().isInteger(1))
    return op->emitOpError("unsupported scf.if condition for vc4tile core; "
                           "expected scalar i1");
  for (Type resultType : op->getResultTypes()) {
    if (!isSupportedCoreType(resultType))
      return emitUnsupportedCoreType(op->getLoc(), resultType);
  }
  if (op->getNumResults() != 0 && ifOp.getElseRegion().empty())
    return op->emitOpError(
        "unsupported scf.if with results: else region is required");

  Block *headerBlock = op->getBlock();
  Region *parentRegion = headerBlock->getParent();
  Block *mergeBlock = headerBlock->splitBlock(op->getIterator());

  SmallVector<Location, 4> resultLocs;
  for (Value result : op->getResults())
    resultLocs.push_back(result.getLoc());
  SmallVector<BlockArgument, 4> mergeArgs;
  mergeArgs.reserve(op->getNumResults());
  for (auto [type, loc] : llvm::zip(op->getResultTypes(), resultLocs))
    mergeArgs.push_back(mergeBlock->addArgument(type, loc));
  for (auto [result, arg] : llvm::zip(op->getResults(), mergeArgs))
    result.replaceAllUsesWith(arg);

  Block *thenBlock = new Block();
  parentRegion->getBlocks().insert(mergeBlock->getIterator(), thenBlock);
  Block *elseBlock = new Block();
  parentRegion->getBlocks().insert(mergeBlock->getIterator(), elseBlock);

  OpBuilder builder(op->getContext());
  builder.setInsertionPointToEnd(headerBlock);
  createCFCondBranch(builder, op->getLoc(), ifOp.getCondition(), thenBlock, {},
                     elseBlock, {});

  if (failed(moveSCFYieldRegionToBlock(builder, ifOp.getThenRegion(),
                                       thenBlock, mergeBlock)))
    return failure();
  if (!ifOp.getElseRegion().empty()) {
    if (failed(moveSCFYieldRegionToBlock(builder, ifOp.getElseRegion(),
                                         elseBlock, mergeBlock)))
      return failure();
  } else {
    builder.setInsertionPointToEnd(elseBlock);
    createCFBranch(builder, op->getLoc(), mergeBlock, {});
  }

  op->erase();
  return success();
}

static std::optional<int64_t> getIndexConstant(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, kArithConstantOpName))
    return std::nullopt;
  if (!value.getType().isIndex())
    return std::nullopt;
  auto attr = def->getAttrOfType<IntegerAttr>("value");
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static FailureOr<Value> materializeIndexBoundAsI32(Value bound,
                                                   OpBuilder &builder,
                                                   Location loc) {
  if (bound.getType().isSignlessInteger(32))
    return bound;
  if (!bound.getType().isIndex())
    return emitError(loc) << "unsupported scf.for bound type for vc4tile core; "
                          << "expected index or i32";

  if (std::optional<int64_t> constant = getIndexConstant(bound))
    if (*constant < 0)
      return emitError(loc)
             << "unsupported scf.for negative constant bound for vc4tile core";

  if (std::optional<int64_t> constant = getIndexConstant(bound))
    return arith::ConstantIntOp::create(builder, loc, *constant, 32)
        .getResult();

  Operation *def = bound.getDefiningOp();
  if (isIndexCastOp(def) && def->getNumOperands() == 1 &&
      def->getOperand(0).getType().isSignlessInteger(32))
    return def->getOperand(0);

  return emitError(loc) << "unsupported scf.for index bound for vc4tile core; "
                        << "bounds must be index constants or i32 index_cast "
                           "values";
}

static void eraseDefIfUnused(Value value) {
  Operation *def = value.getDefiningOp();
  if (def && def->use_empty())
    def->erase();
}

static LogicalResult replaceForInductionUses(mlir::scf::ForOp forOp,
                                             Value replacementIV) {
  BlockArgument oldIV = llvm::cast<BlockArgument>(forOp.getInductionVar());
  SmallVector<Operation *, 4> deadCasts;
  for (Operation *user : llvm::make_early_inc_range(oldIV.getUsers())) {
    if (!isIndexCastOp(user) || user->getNumResults() != 1 ||
        !user->getResult(0).getType().isSignlessInteger(32)) {
      return user->emitOpError(
          "unsupported scf.for induction variable use for vc4tile core; "
          "index uses must go through arith.index_cast to i32");
    }
    user->getResult(0).replaceAllUsesWith(replacementIV);
    deadCasts.push_back(user);
  }
  for (Operation *cast : deadCasts)
    cast->erase();
  if (!oldIV.use_empty()) {
    Operation *owner = (*oldIV.getUsers().begin());
    return owner->emitOpError(
        "unsupported scf.for induction variable index use for vc4tile core");
  }
  return success();
}

static LogicalResult legalizeSCFForOp(mlir::scf::ForOp forOp) {
  Operation *op = forOp.getOperation();
  if (!getParentVC4TileKernel(op))
    return success();

  std::optional<int64_t> step = getIndexConstant(forOp.getStep());
  if (!step || *step <= 0)
    return op->emitOpError(
        "unsupported scf.for for vc4tile core: step must be a positive static "
        "integer constant");

  for (Type resultType : op->getResultTypes()) {
    if (!isSupportedCoreType(resultType))
      return emitUnsupportedCoreType(op->getLoc(), resultType);
  }
  for (Value initArg : forOp.getInitArgs()) {
    if (!isSupportedCoreType(initArg.getType()))
      return emitUnsupportedCoreType(initArg.getLoc(), initArg.getType());
  }

  Block *preheaderBlock = op->getBlock();
  Region *parentRegion = preheaderBlock->getParent();
  Block *afterBlock = preheaderBlock->splitBlock(op->getIterator());

  OpBuilder builder(op->getContext());
  builder.setInsertionPointToEnd(preheaderBlock);
  FailureOr<Value> maybeLower =
      materializeIndexBoundAsI32(forOp.getLowerBound(), builder, op->getLoc());
  if (failed(maybeLower))
    return failure();
  FailureOr<Value> maybeUpper =
      materializeIndexBoundAsI32(forOp.getUpperBound(), builder, op->getLoc());
  if (failed(maybeUpper))
    return failure();
  Value lowerI32 = *maybeLower;
  Value upperI32 = *maybeUpper;

  SmallVector<Value, 4> initialOperands;
  initialOperands.push_back(lowerI32);
  initialOperands.append(forOp.getInitArgs().begin(), forOp.getInitArgs().end());

  SmallVector<Type, 4> loopArgTypes;
  SmallVector<Location, 4> loopArgLocs;
  loopArgTypes.push_back(builder.getI32Type());
  loopArgLocs.push_back(forOp.getInductionVar().getLoc());
  for (Value initArg : forOp.getInitArgs()) {
    loopArgTypes.push_back(initArg.getType());
    loopArgLocs.push_back(initArg.getLoc());
  }

  Block *condBlock = new Block();
  parentRegion->getBlocks().insert(afterBlock->getIterator(), condBlock);
  condBlock->addArguments(loopArgTypes, loopArgLocs);
  Block *bodyBlock = new Block();
  parentRegion->getBlocks().insert(afterBlock->getIterator(), bodyBlock);
  bodyBlock->addArguments(loopArgTypes, loopArgLocs);

  SmallVector<BlockArgument, 4> afterArgs;
  afterArgs.reserve(op->getNumResults());
  for (Value result : op->getResults())
    afterArgs.push_back(afterBlock->addArgument(result.getType(), result.getLoc()));
  for (auto [result, arg] : llvm::zip(op->getResults(), afterArgs))
    result.replaceAllUsesWith(arg);

  createCFBranch(builder, op->getLoc(), condBlock, initialOperands);

  builder.setInsertionPointToEnd(condBlock);
  Value condIV = condBlock->getArgument(0);
  Value done = arith::CmpIOp::create(builder, op->getLoc(),
                                     arith::CmpIPredicate::ult, condIV,
                                     upperI32);
  SmallVector<Value, 4> condLoopOperands(condBlock->args_begin(),
                                         condBlock->args_end());
  SmallVector<Value, 4> exitOperands;
  for (BlockArgument arg : llvm::drop_begin(condBlock->getArguments()))
    exitOperands.push_back(arg);
  createCFCondBranch(builder, op->getLoc(), done, bodyBlock, condLoopOperands,
                     afterBlock, exitOperands);

  Block &sourceBody = forOp.getRegion().front();
  Operation *yield = sourceBody.getTerminator();
  if (!yield || !hasName(yield, "scf.yield"))
    return op->emitOpError("unsupported scf.for: expected scf.yield terminator");
  if (yield->getNumOperands() != forOp.getInitArgs().size())
    return yield->emitOpError("unsupported scf.for yield operand count");

  Value bodyIV = bodyBlock->getArgument(0);
  if (failed(replaceForInductionUses(forOp, bodyIV)))
    return failure();
  for (auto [oldArg, newArg] :
       llvm::zip(llvm::drop_begin(sourceBody.getArguments()),
                 llvm::drop_begin(bodyBlock->getArguments())))
    oldArg.replaceAllUsesWith(newArg);

  while (!sourceBody.empty() && &sourceBody.front() != yield)
    sourceBody.front().moveBefore(bodyBlock, bodyBlock->end());

  SmallVector<Value, 4> yieldedOperands(yield->getOperands());
  builder.setInsertionPointToEnd(bodyBlock);
  Value stepI32 =
      arith::ConstantIntOp::create(builder, op->getLoc(), *step, 32);
  Value nextIV = arith::AddIOp::create(builder, op->getLoc(), bodyIV, stepI32);
  SmallVector<Value, 4> backedgeOperands;
  backedgeOperands.push_back(nextIV);
  backedgeOperands.append(yieldedOperands.begin(), yieldedOperands.end());
  createCFBranch(builder, yield->getLoc(), condBlock, backedgeOperands);

  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value stepValue = forOp.getStep();
  op->erase();
  eraseDefIfUnused(stepValue);
  eraseDefIfUnused(upperBound);
  eraseDefIfUnused(lowerBound);
  return success();
}

static Operation *findFirstSCFOpInKernel(Operation *kernel) {
  auto findInRegion = [&](Region &region, auto &&findInRegionRef)
      -> Operation * {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (getDialectNamespace(&op) == "scf")
          return &op;
        for (Region &nested : op.getRegions()) {
          if (Operation *found = findInRegionRef(nested, findInRegionRef))
            return found;
        }
      }
    }
    return nullptr;
  };
  for (Region &region : kernel->getRegions()) {
    if (Operation *found = findInRegion(region, findInRegion))
      return found;
  }
  return nullptr;
}

static LogicalResult legalizeVC4TileKernelSCF(Operation *kernel) {
  while (Operation *scfOp = findFirstSCFOpInKernel(kernel)) {
    if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(scfOp)) {
      if (failed(legalizeSCFIfOp(ifOp)))
        return failure();
      continue;
    }
    if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(scfOp)) {
      if (failed(legalizeSCFForOp(forOp)))
        return failure();
      continue;
    }
    return scfOp->emitOpError(
        "unsupported scf operation in vc4tile surface; only scf.if and "
        "scf.for are legalizable by --legalize-vc4tile-core-cfg");
  }
  return success();
}

static LogicalResult rejectSurfaceOpsInKernels(ModuleOp module,
                                               StringRef beforePass) {
  for (Operation &op : module.getBody()->getOperations()) {
    if (!isVC4TileKernel(&op))
      continue;
    Operation *surfaceOp = nullptr;
    op.walk([&](Operation *nested) {
      if (nested == &op)
        return WalkResult::advance();
      if (!isVC4TileSurfaceOp(nested))
        return WalkResult::advance();
      surfaceOp = nested;
      return WalkResult::interrupt();
    });
    if (surfaceOp)
      return emitSurfaceOpOrderingError(surfaceOp, beforePass);
  }
  return success();
}

static LogicalResult rejectRawSCFInKernels(ModuleOp module,
                                           StringRef diagnosticSuffix) {
  for (Operation &op : module.getBody()->getOperations()) {
    if (!isVC4TileKernel(&op))
      continue;
    Operation *scfOp = findFirstSCFOpInKernel(&op);
    if (!scfOp)
      continue;
    scfOp->emitOpError() << diagnosticSuffix;
    return failure();
  }
  return success();
}

struct CanonicalizeVC4TileSurfacePass
    : public PassWrapper<CanonicalizeVC4TileSurfacePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeVC4TileSurfacePass)

  StringRef getArgument() const override { return "canonicalize-vc4tile-surface"; }
  StringRef getDescription() const override {
    return "Erase temporary VC4Tile surface sentinels and prepare ergonomic surface IR for copy planning";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    mlir::vc4tile::VC4TileDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 8> placeholders;
    module.walk([&](Operation *op) {
      if (hasName(op, kVC4TileSurfacePlaceholderOpName))
        placeholders.push_back(op);
    });
    for (Operation *op : placeholders)
      op->erase();
  }
};


static Value createVC4TileLaneRangeCore(OpBuilder &builder, Location loc) {
  OperationState state(loc, kVC4TileLaneRangeOpName);
  state.addTypes(getVector16I32Type(builder));
  return builder.create(state)->getResult(0);
}

static bool isI32Scalar(Value value) {
  return value && value.getType().isSignlessInteger(32);
}

static FailureOr<Value> createAdjustedGlobalBaseForTileCopy(OpBuilder &builder,
                                                            Operation *op,
                                                            Value base,
                                                            Value offset) {
  if (!isI32Scalar(base) || !isI32Scalar(offset))
    return op->emitOpError("M5 copy planner v1 requires i32 base and offset operands");

  if (auto constOp = offset.getDefiningOp<arith::ConstantIntOp>()) {
    if (constOp.value() == 0)
      return base;
  }

  Value four = arith::ConstantIntOp::create(builder, op->getLoc(), 4, 32);
  Value byteOffset = arith::MulIOp::create(builder, op->getLoc(), offset, four);
  return arith::AddIOp::create(builder, op->getLoc(), base, byteOffset)
      .getResult();
}

static NamedAttribute namedAttr(OpBuilder &builder, StringRef name,
                                Attribute attr) {
  return builder.getNamedAttr(name, attr);
}

static Attribute getVC4TileMemorySpaceAttr(OpBuilder &builder,
                                           mlir::vc4tile::MemorySpace space) {
  return mlir::vc4tile::MemorySpaceAttr::get(builder.getContext(), space);
}

static Attribute getVC4TileOffsetUnitAttr(OpBuilder &builder,
                                          mlir::vc4tile::OffsetUnit unit) {
  return mlir::vc4tile::OffsetUnitAttr::get(builder.getContext(), unit);
}

static Attribute getVC4TileMemoryAccessAttr(OpBuilder &builder,
                                            mlir::vc4tile::MemoryAccess access) {
  return mlir::vc4tile::MemoryAccessAttr::get(builder.getContext(), access);
}

static Operation *createVC4TileCoreOp(OpBuilder &builder, Location loc,
                                      StringRef name, ArrayRef<Value> operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      TypeRange resultTypes = TypeRange{}) {
  OperationState state(loc, name);
  state.addOperands(operands);
  for (NamedAttribute attr : attrs)
    state.addAttribute(attr.getName(), attr.getValue());
  state.addTypes(resultTypes);
  return builder.create(state);
}

static FailureOr<Value> planGlobalRegisterTileLoad(Operation *op,
                                                   OpBuilder &builder) {
  auto memorySpace = op->getAttrOfType<mlir::vc4tile::MemorySpaceAttr>("memory_space");
  if (!memorySpace || memorySpace.getValue() != mlir::vc4tile::MemorySpace::global)
    return op->emitOpError("copy planner v1 supports tile_load only for global->register; use copy_tile for shared paths");
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return op->emitOpError("expected base, offset, mask and one result");
  if (!isVector16Data(op->getResult(0).getType()))
    return op->emitOpError("copy planner v1 requires tile_load result to be vector<16xi32> or vector<16xf32>");

  FailureOr<Value> adjustedBase = createAdjustedGlobalBaseForTileCopy(
      builder, op, op->getOperand(0), op->getOperand(1));
  if (failed(adjustedBase))
    return failure();
  Value lanes = createVC4TileLaneRangeCore(builder, op->getLoc());
  Operation *load = createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileMaskedLoadGlobalOpName,
      {*adjustedBase, lanes, op->getOperand(2)},
      {namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
       namedAttr(builder, "offset_unit",
                 getVC4TileOffsetUnitAttr(builder,
                                          mlir::vc4tile::OffsetUnit::element)),
       namedAttr(builder, "memory_space",
                 getVC4TileMemorySpaceAttr(builder,
                                           mlir::vc4tile::MemorySpace::global)),
       namedAttr(builder, "access",
                 getVC4TileMemoryAccessAttr(builder,
                                            mlir::vc4tile::MemoryAccess::coalesced))},
      op->getResultTypes());
  return load->getResult(0);
}

static LogicalResult planRegisterGlobalTileStore(Operation *op,
                                                 OpBuilder &builder) {
  auto memorySpace = op->getAttrOfType<mlir::vc4tile::MemorySpaceAttr>("memory_space");
  if (!memorySpace || memorySpace.getValue() != mlir::vc4tile::MemorySpace::global)
    return op->emitOpError("copy planner v1 supports tile_store only for register->global; use copy_tile for shared paths");
  if (op->getNumOperands() != 4)
    return op->emitOpError("expected tile, base, offset, and mask operands");
  if (!isVector16Data(op->getOperand(0).getType()))
    return op->emitOpError("copy planner v1 requires tile_store input to be vector<16xi32> or vector<16xf32>");

  FailureOr<Value> adjustedBase = createAdjustedGlobalBaseForTileCopy(
      builder, op, op->getOperand(1), op->getOperand(2));
  if (failed(adjustedBase))
    return failure();
  Value lanes = createVC4TileLaneRangeCore(builder, op->getLoc());
  createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileMaskedStoreGlobalOpName,
      {*adjustedBase, lanes, op->getOperand(0), op->getOperand(3)},
      {namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
       namedAttr(builder, "offset_unit",
                 getVC4TileOffsetUnitAttr(builder,
                                          mlir::vc4tile::OffsetUnit::element)),
       namedAttr(builder, "memory_space",
                 getVC4TileMemorySpaceAttr(builder,
                                           mlir::vc4tile::MemorySpace::global)),
       namedAttr(builder, "access",
                 getVC4TileMemoryAccessAttr(
                     builder, mlir::vc4tile::MemoryAccess::affine_contiguous))});
  return success();
}

static Attribute getVPMLayoutFromTileLayout(OpBuilder &builder,
                                            mlir::vc4tile::LayoutAttr layout,
                                            bool defaultColumn = false) {
  mlir::vc4tile::VPMLayout vpmLayout = defaultColumn
                                          ? mlir::vc4tile::VPMLayout::column_major
                                          : mlir::vc4tile::VPMLayout::row_major;
  if (layout) {
    switch (layout.getValue()) {
    case mlir::vc4tile::Layout::vpm_col:
    case mlir::vc4tile::Layout::col_major:
    case mlir::vc4tile::Layout::transposed_view:
      vpmLayout = mlir::vc4tile::VPMLayout::column_major;
      break;
    case mlir::vc4tile::Layout::row_major:
    case mlir::vc4tile::Layout::affine_2d:
    case mlir::vc4tile::Layout::vpm_row:
      vpmLayout = mlir::vc4tile::VPMLayout::row_major;
      break;
    }
  }
  return mlir::vc4tile::VPMLayoutAttr::get(builder.getContext(), vpmLayout);
}

static LogicalResult planSharedTileAlloc(Operation *op, OpBuilder &builder) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  if (!mlir::vc4tile::isVC4TileSharedTileType(op->getResult(0).getType())) {
    return op->emitOpError(
        "copy planner v1 requires shared_tile_alloc result type !vc4tile.shared_tile");
  }
  auto rows = op->getAttrOfType<IntegerAttr>("rows");
  auto elemBytes = op->getAttrOfType<IntegerAttr>("elem_bytes");
  if (!rows || !elemBytes)
    return op->emitOpError("requires rows and elem_bytes attributes");
  Operation *alloc = createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileSharedAllocOpName, {},
      {namedAttr(builder, "rows", rows),
       namedAttr(builder, "elem_bytes", elemBytes),
       namedAttr(builder, "memory_space",
                 getVC4TileMemorySpaceAttr(builder,
                                           mlir::vc4tile::MemorySpace::shared_vpm)),
       namedAttr(builder, "layout",
                 getVPMLayoutFromTileLayout(
                     builder,
                     op->getAttrOfType<mlir::vc4tile::LayoutAttr>("layout")))},
      op->getResultTypes());
  op->getResult(0).replaceAllUsesWith(alloc->getResult(0));
  return success();
}

static LogicalResult planRegisterSharedCopy(Operation *op, OpBuilder &builder) {
  if (op->getNumOperands() != 4 || op->getNumResults() != 0) {
    return op->emitOpError(
        "register->shared_vpm copy_tile expects operands (value, shared_tile, row, mask) and no results");
  }
  Value value = op->getOperand(0);
  Value handle = op->getOperand(1);
  Value row = op->getOperand(2);
  Value mask = op->getOperand(3);
  if (!isVector16Data(value.getType()) ||
      !mlir::vc4tile::isVC4TileSharedTileType(handle.getType()) ||
      !isI32Scalar(row) || !isVector16I1(mask.getType())) {
    return op->emitOpError(
        "register->shared_vpm copy_tile requires vector value, !vc4tile.shared_tile handle, i32 row, and vector<16xi1> mask");
  }
  createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileSharedStoreOpName,
      {handle, row, value, mask},
      {namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
       namedAttr(builder, "memory_space",
                 getVC4TileMemorySpaceAttr(builder,
                                           mlir::vc4tile::MemorySpace::shared_vpm)),
       namedAttr(builder, "layout",
                 getVPMLayoutFromTileLayout(
                     builder,
                     op->getAttrOfType<mlir::vc4tile::LayoutAttr>("dst_layout")))});
  return success();
}

static LogicalResult planSharedRegisterCopy(Operation *op, OpBuilder &builder) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
    return op->emitOpError(
        "shared_vpm->register copy_tile expects operands (shared_tile, row, mask) and one result");
  }
  Value handle = op->getOperand(0);
  Value row = op->getOperand(1);
  Value mask = op->getOperand(2);
  if (!mlir::vc4tile::isVC4TileSharedTileType(handle.getType()) ||
      !isI32Scalar(row) || !isVector16I1(mask.getType()) ||
      !isVector16Data(op->getResult(0).getType())) {
    return op->emitOpError(
        "shared_vpm->register copy_tile requires !vc4tile.shared_tile handle, i32 row, vector<16xi1> mask, and vector result");
  }
  Operation *load = createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileSharedLoadOpName, {handle, row, mask},
      {namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
       namedAttr(builder, "memory_space",
                 getVC4TileMemorySpaceAttr(builder,
                                           mlir::vc4tile::MemorySpace::shared_vpm)),
       namedAttr(builder, "layout",
                 getVPMLayoutFromTileLayout(
                     builder,
                     op->getAttrOfType<mlir::vc4tile::LayoutAttr>("src_layout")))},
      op->getResultTypes());
  op->getResult(0).replaceAllUsesWith(load->getResult(0));
  return success();
}

static LogicalResult planCopyTile(Operation *op, OpBuilder &builder) {
  auto src = op->getAttrOfType<mlir::vc4tile::MemorySpaceAttr>("src_space");
  auto dst = op->getAttrOfType<mlir::vc4tile::MemorySpaceAttr>("dst_space");
  if (!src || !dst)
    return op->emitOpError("copy_tile requires src_space and dst_space attributes");

  if (src.getValue() == mlir::vc4tile::MemorySpace::register_space &&
      dst.getValue() == mlir::vc4tile::MemorySpace::shared_vpm)
    return planRegisterSharedCopy(op, builder);
  if (src.getValue() == mlir::vc4tile::MemorySpace::shared_vpm &&
      dst.getValue() == mlir::vc4tile::MemorySpace::register_space)
    return planSharedRegisterCopy(op, builder);

  return op->emitOpError()
         << "copy planner v1 cannot plan requested copy path; supported paths are register->shared_vpm and shared_vpm->register in copy_tile, plus tile_load/tile_store for global/register";
}

static LogicalResult foldViewOp(Operation *op) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return op->emitOpError("view folding expects one source and one result");
  if (op->getOperand(0).getType() != op->getResult(0).getType())
    return op->emitOpError("view folding requires source and result types to match");
  op->getResult(0).replaceAllUsesWith(op->getOperand(0));
  return success();
}

static LogicalResult planOneVC4TileSurfaceOp(Operation *op) {
  if (hasName(op, kVC4TileSurfacePlaceholderOpName) ||
      hasName(op, kVC4TileTileDescriptorOpName))
    return success();

  OpBuilder builder(op);
  if (hasName(op, kVC4TileTileLoadOpName)) {
    FailureOr<Value> result = planGlobalRegisterTileLoad(op, builder);
    if (failed(result))
      return failure();
    op->getResult(0).replaceAllUsesWith(*result);
    return success();
  }
  if (hasName(op, kVC4TileTileStoreOpName))
    return planRegisterGlobalTileStore(op, builder);
  if (hasName(op, kVC4TileSharedTileAllocOpName))
    return planSharedTileAlloc(op, builder);
  if (hasName(op, kVC4TileCopyTileOpName))
    return planCopyTile(op, builder);
  if (hasName(op, kVC4TileTileViewOpName) ||
      hasName(op, kVC4TileTileSubviewOpName) ||
      hasName(op, kVC4TileTransposeViewOpName))
    return foldViewOp(op);

  return op->emitOpError("unrecognized VC4Tile surface operation");
}

struct PlanVC4TileCopiesPass
    : public PassWrapper<PlanVC4TileCopiesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanVC4TileCopiesPass)

  StringRef getArgument() const override { return "plan-vc4tile-copies"; }
  StringRef getDescription() const override {
    return "Plan ergonomic VC4Tile copy operations into lowering-ready VC4Tile core operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    mlir::vc4tile::VC4TileDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 16> surfaceOps;
    module.walk([&](Operation *op) {
      if (op != module.getOperation() && isVC4TileSurfaceOp(op))
        surfaceOps.push_back(op);
    });

    for (Operation *op : surfaceOps) {
      if (!op->getBlock())
        continue;
      if (failed(planOneVC4TileSurfaceOp(op))) {
        signalPassFailure();
        return;
      }
      if (op->use_empty())
        op->erase();
      else if (isVC4TileSurfaceOp(op)) {
        op->emitOpError("copy planner v1 could not erase all uses of surface operation");
        signalPassFailure();
        return;
      }
    }
  }
};

struct VerifyVC4TileCorePass
    : public PassWrapper<VerifyVC4TileCorePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyVC4TileCorePass)

  StringRef getArgument() const override { return "verify-vc4tile-core"; }
  StringRef getDescription() const override {
    return "Verify lowering-ready VC4Tile CFG core with no raw SCF or index leakage";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::vector::VectorDialect,
                    mlir::vc4tile::VC4TileDialect>();
  }

  void runOnOperation() override {
    if (failed(verifyVC4TileCore(getOperation())))
      signalPassFailure();
  }
};

struct LegalizeVC4TileCoreCFGPass
    : public PassWrapper<LegalizeVC4TileCoreCFGPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeVC4TileCoreCFGPass)

  StringRef getArgument() const override { return "legalize-vc4tile-core-cfg"; }
  StringRef getDescription() const override {
    return "Canonicalize constrained VC4Tile surface SCF into lowering-ready VC4Tile CFG core";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    mlir::vc4tile::VC4TileDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (Operation &op : module.getBody()->getOperations()) {
      if (!isVC4TileKernel(&op))
        continue;
      if (failed(legalizeVC4TileKernelSCF(&op))) {
        signalPassFailure();
        return;
      }
    }
    if (failed(verifyVC4TileCore(module)))
      signalPassFailure();
  }
};

struct ConvertVC4TileToSSAVC4Pass
    : public PassWrapper<ConvertVC4TileToSSAVC4Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVC4TileToSSAVC4Pass)

  StringRef getArgument() const override { return "convert-vc4tile-to-ssavc4"; }
  StringRef getDescription() const override {
    return "Lower supported VC4Tile kernels, formal ABI args, runtime IDs, masks, global loads/stores, shared VPM loads/stores, barriers, rotate/reduce, uniform control flow, block arguments, and metadata to SSAVC4";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::vector::VectorDialect,
                    mlir::vc4::VC4Dialect,
                    mlir::ssavc4::SSAVC4Dialect,
                    mlir::vc4tile::VC4TileDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(rejectSurfaceOpsInKernels(module,
                                         "--convert-vc4tile-to-ssavc4"))) {
      signalPassFailure();
      return;
    }
    if (failed(rejectRawSCFInKernels(
            module,
            " must be legalized with --legalize-vc4tile-core-cfg before "
            "--convert-vc4tile-to-ssavc4; expected lowering-ready VC4Tile "
            "core"))) {
      signalPassFailure();
      return;
    }

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
      if (failed(verifySupportedKernelShape(kernel))) {
        signalPassFailure();
        return;
      }
    }

    OpBuilder builder(module.getContext());
    builder.setInsertionPoint(kernels.front());
    Operation *ssavc4Module = createSSAVC4Module(kernels.front(), builder);

    SmallVector<Operation *, 4> funcs;
    for (Operation *kernel : kernels)
      funcs.push_back(createSSAVC4Func(kernel, ssavc4Module, builder));

    for (auto [kernel, func] : llvm::zip(kernels, funcs)) {
      if (failed(lowerKernelBody(kernel, func))) {
        signalPassFailure();
        return;
      }
    }

    for (Operation *kernel : kernels)
      kernel->erase();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vc4::createCanonicalizeVC4TileSurfacePass() {
  return std::make_unique<CanonicalizeVC4TileSurfacePass>();
}

std::unique_ptr<Pass> mlir::vc4::createPlanVC4TileCopiesPass() {
  return std::make_unique<PlanVC4TileCopiesPass>();
}

std::unique_ptr<Pass> mlir::vc4::createVerifyVC4TileCorePass() {
  return std::make_unique<VerifyVC4TileCorePass>();
}

std::unique_ptr<Pass> mlir::vc4::createLegalizeVC4TileCoreCFGPass() {
  return std::make_unique<LegalizeVC4TileCoreCFGPass>();
}

std::unique_ptr<Pass> mlir::vc4::createConvertVC4TileToSSAVC4Pass() {
  return std::make_unique<ConvertVC4TileToSSAVC4Pass>();
}

void mlir::vc4::registerConvertVC4TileToSSAVC4Pass() {
  // This translation unit is linked into vc4-opt for M4/M5 staging, and the
  // file-scope PassRegistration objects below install the pass flags.
}

static PassRegistration<CanonicalizeVC4TileSurfacePass>
    registerCanonicalizeVC4TileSurfacePass;
static PassRegistration<PlanVC4TileCopiesPass> registerPlanVC4TileCopiesPass;
static PassRegistration<VerifyVC4TileCorePass> registerVerifyVC4TileCorePass;
static PassRegistration<LegalizeVC4TileCoreCFGPass>
    registerLegalizeVC4TileCoreCFGPass;
static PassRegistration<ConvertVC4TileToSSAVC4Pass>
    registerVC4TileToSSAVC4Pass;
