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
#include "llvm/ADT/APFloat.h"
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
constexpr llvm::StringLiteral kVC4TileTileRectMaskOpName(
    "vc4tile.tile_rect_mask");
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
constexpr llvm::StringLiteral kVC4TileVDRLoadTileOpName("vc4tile.vdr_load_tile");
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
constexpr llvm::StringLiteral kVC4TileTileFillOpName("vc4tile.tile_fill");
constexpr llvm::StringLiteral kVC4TileTileBroadcastOpName(
    "vc4tile.tile_broadcast");
constexpr llvm::StringLiteral kVC4TileTileAddOpName("vc4tile.tile_add");
constexpr llvm::StringLiteral kVC4TileTileSubOpName("vc4tile.tile_sub");
constexpr llvm::StringLiteral kVC4TileTileMulOpName("vc4tile.tile_mul");
constexpr llvm::StringLiteral kVC4TileTileSelectOpName("vc4tile.tile_select");
constexpr llvm::StringLiteral kVC4TileTileDotOpName("vc4tile.tile_dot");
constexpr llvm::StringLiteral kVC4TileTileContractOpName("vc4tile.tile_contract");
constexpr llvm::StringLiteral kVC4TileTileMatmulOpName("vc4tile.tile_matmul");
constexpr llvm::StringLiteral kVC4TileTileReduceOpName("vc4tile.tile_reduce");
constexpr llvm::StringLiteral kVC4TileRowReduceOpName("vc4tile.row_reduce");
constexpr llvm::StringLiteral kVC4TileWarpReduceOpName("vc4tile.warp_reduce");
constexpr llvm::StringLiteral kVC4TileBlockReduceOpName("vc4tile.block_reduce");
constexpr llvm::StringLiteral kVC4TileSurfacePlaceholderOpName(
    "vc4tile.surface_placeholder");

constexpr llvm::StringLiteral kArithConstantOpName("arith.constant");
constexpr llvm::StringLiteral kArithAddIOpName("arith.addi");
constexpr llvm::StringLiteral kArithSubIOpName("arith.subi");
constexpr llvm::StringLiteral kArithMulIOpName("arith.muli");
constexpr llvm::StringLiteral kArithAddFOpName("arith.addf");
constexpr llvm::StringLiteral kArithSubFOpName("arith.subf");
constexpr llvm::StringLiteral kArithMulFOpName("arith.mulf");
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
constexpr llvm::StringLiteral kSSAVC4CondSelectOpName("ssavc4.cond_select");
constexpr llvm::StringLiteral kSSAVC4TMURequestOpName("ssavc4.tmu.request");
constexpr llvm::StringLiteral kSSAVC4TMUReadOpName("ssavc4.tmu.read");
constexpr llvm::StringLiteral kSSAVC4VDRLoadOpName("ssavc4.vdr.load");
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
         hasName(op, kVC4TileSharedTileAllocOpName) ||
         hasName(op, kVC4TileTileFillOpName) ||
         hasName(op, kVC4TileTileBroadcastOpName) ||
         hasName(op, kVC4TileTileAddOpName) ||
         hasName(op, kVC4TileTileSubOpName) ||
         hasName(op, kVC4TileTileMulOpName) ||
         hasName(op, kVC4TileTileSelectOpName) ||
         hasName(op, kVC4TileTileReduceOpName) ||
         hasName(op, kVC4TileRowReduceOpName) ||
         hasName(op, kVC4TileWarpReduceOpName) ||
         hasName(op, kVC4TileBlockReduceOpName) ||
         hasName(op, kVC4TileTileDotOpName) ||
         hasName(op, kVC4TileTileContractOpName) ||
         hasName(op, kVC4TileTileMatmulOpName);
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
         hasName(op, kVC4TileTileRectMaskOpName) ||
         hasName(op, kVC4TileMaskedLoadGlobalOpName) ||
         hasName(op, kVC4TileMaskedStoreGlobalOpName) ||
         hasName(op, kVC4TileRotateOpName) ||
         hasName(op, kVC4TileReduceOpName) ||
         hasName(op, kVC4TileSharedAllocOpName) ||
         hasName(op, kVC4TileSharedLoadOpName) ||
         hasName(op, kVC4TileSharedStoreOpName) ||
         hasName(op, kVC4TileVDRLoadTileOpName) ||
         hasName(op, kVC4TileBarrierOpName);
}

static bool isAllowedCoreArithOp(Operation *op) {
  return hasName(op, kArithConstantOpName) ||
         hasName(op, kArithAddIOpName) || hasName(op, kArithSubIOpName) ||
         hasName(op, kArithMulIOpName) || hasName(op, kArithAddFOpName) ||
         hasName(op, kArithSubFOpName) || hasName(op, kArithMulFOpName) ||
         hasName(op, kArithShLIOpName) || hasName(op, kArithShRUIOpName) ||
         hasName(op, kArithShRSIOpName) || hasName(op, kArithAndIOpName) ||
         hasName(op, kArithOrIOpName) ||
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


static DictionaryAttr getResourceIntent(Operation *kernel) {
  return getDictionaryAttr(kernel, "resource_intent");
}

static std::optional<bool> getResourceIntentBool(Operation *kernel,
                                                 StringRef name) {
  DictionaryAttr intent = getResourceIntent(kernel);
  if (!intent)
    return std::nullopt;
  auto attr = intent.getAs<BoolAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

static std::optional<int64_t> getResourceIntentI32(Operation *kernel,
                                                   StringRef name) {
  DictionaryAttr intent = getResourceIntent(kernel);
  if (!intent)
    return std::nullopt;
  auto attr = intent.getAs<IntegerAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool getBoolAttrOrResourceIntent(Operation *kernel, StringRef attrName,
                                        bool fallback = false) {
  if (auto attr = kernel->getAttrOfType<BoolAttr>(attrName))
    return attr.getValue();
  if (std::optional<bool> value = getResourceIntentBool(kernel, attrName))
    return *value;
  return fallback;
}

static std::optional<int64_t> getI32AttrOrResourceIntent(Operation *kernel,
                                                         StringRef attrName,
                                                         StringRef intentName) {
  if (auto attr = kernel->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return getResourceIntentI32(kernel, intentName);
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
      kernelContains(kernel, kVC4TileSharedStoreOpName) ||
      kernelContains(kernel, kVC4TileVDRLoadTileOpName))
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
  const bool usesShared =
      getBoolAttrOrResourceIntent(kernel, "uses_shared_vpm");
  const bool usesBarrier =
      getBoolAttrOrResourceIntent(kernel, "uses_barrier");
  const int64_t warpsPerBlock = getI32AttrOrResourceIntent(
      kernel, "warps_per_block_max", "warps_per_block").value_or(
      cooperative ? 12 : 1);
  const int64_t vpmRows = getI32AttrOrResourceIntent(
      kernel, "vpm_rows_per_block", "vpm_rows").value_or(0);
  const int64_t vpmBytes = getI32AttrOrResourceIntent(
      kernel, "vpm_bytes_per_block", "vpm_bytes").value_or(0);
  const int64_t semaphores = getI32AttrOrResourceIntent(
      kernel, "semaphores_per_block", "semaphores").value_or(
      usesBarrier ? 4 : 0);
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

static VectorType getVector16DataType(OpBuilder &builder, Type elementType) {
  if (elementType && elementType.isF32())
    return VectorType::get({16}, builder.getF32Type());
  return getVector16I32Type(builder);
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
  return type.isSignlessInteger(32) || type.isF32() ||
         isVector16I32(type) || isVector16F32(type);
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

static void appendTileSemanticMetadataAttrs(
    Operation *source, OpBuilder &builder,
    SmallVectorImpl<NamedAttribute> &attrs);
static void copyTileSemanticMetadataAttrs(Operation *source,
                                          Operation *target);

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

static Value createLoadImmPerElemU2(OpBuilder &builder, Location loc,
                                    Type resultType,
                                    ArrayRef<int32_t> values) {
  Attribute mode = mlir::vc4::LoadImmModeAttr::get(
      builder.getContext(), mlir::vc4::LoadImmMode::per_elem_u2);
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4LoadImmOpName, {},
      {builder.getNamedAttr("mode", mode),
       builder.getNamedAttr("values", builder.getDenseI32ArrayAttr(values))},
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

static Value createCondSelect(OpBuilder &builder, Location loc, Value flags,
                              Value trueValue, Value falseValue,
                              mlir::vc4::Cond cond, Type resultType) {
  Attribute condAttr = mlir::vc4::CondAttr::get(builder.getContext(), cond);
  return createSSAVC4OpWithResult(
      builder, loc, kSSAVC4CondSelectOpName,
      {flags, trueValue, falseValue},
      {builder.getNamedAttr("cond", condAttr)}, resultType);
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

static std::optional<mlir::vc4::AddOpcode>
getFloatAddOpcode(Operation *op) {
  if (hasName(op, kArithAddFOpName))
    return mlir::vc4::AddOpcode::fadd;
  if (hasName(op, kArithSubFOpName))
    return mlir::vc4::AddOpcode::fsub;
  return std::nullopt;
}

static LogicalResult lowerFloatALU(Operation *op, OpBuilder &builder,
                                   llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("expected two operands and one result");
  std::optional<mlir::vc4::AddOpcode> opcode = getFloatAddOpcode(op);
  if (!opcode)
    return op->emitOpError("has no VC4 float ADD-pipe opcode mapping");
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

static LogicalResult lowerFloatMul(Operation *op, OpBuilder &builder,
                                   llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("expected two operands and one result");
  Value lhs = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value rhs = lookupMappedValue(op, op->getOperand(1), valueMap);
  if (!lhs || !rhs)
    return failure();
  SmallVector<Value, 2> operands{lhs, rhs};
  valueMap[op->getResult(0)] =
      createALUMul(builder, op->getLoc(), operands, mlir::vc4::MulOpcode::fmul,
                   op->getResult(0).getType());
  return success();
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
  } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(value)) {
    if (!floatAttr.getType().isF32())
      return op->emitOpError(
          "unsupported arith.constant float width for VC4Tile lowering");
    value = FloatAttr::get(builder.getF32Type(), floatAttr.getValue());
  } else if (auto denseAttr = llvm::dyn_cast<DenseIntElementsAttr>(value)) {
    if (denseAttr.isSplat()) {
      value = builder.getI32IntegerAttr(
          denseAttr.getSplatValue<llvm::APInt>().getSExtValue());
    } else {
      auto vectorType = llvm::dyn_cast<VectorType>(resultType);
      if (!vectorType || vectorType.getRank() != 1 ||
          vectorType.getDimSize(0) != 16 ||
          !vectorType.getElementType().isSignlessInteger(32))
        return op->emitOpError(
            "currently lowers only vector<16xi32> non-splat dense integer constants");
      SmallVector<int32_t, 16> values;
      values.reserve(16);
      for (const llvm::APInt &lane : denseAttr.getValues<llvm::APInt>()) {
        int64_t laneValue = lane.getSExtValue();
        if (laneValue < 0 || laneValue > 3)
          return op->emitOpError(
              "non-splat dense integer vector constants must use lane values "
              "in range [0, 3]");
        values.push_back(static_cast<int32_t>(laneValue));
      }
      valueMap[op->getResult(0)] =
          createLoadImmPerElemU2(builder, op->getLoc(), resultType, values);
      return success();
    }
  } else if (auto denseFloatAttr =
                 llvm::dyn_cast<DenseFPElementsAttr>(value)) {
    if (!denseFloatAttr.isSplat())
      return op->emitOpError(
          "currently lowers only splat dense float vector constants");
    llvm::APFloat splat = denseFloatAttr.getSplatValue<llvm::APFloat>();
    if (!denseFloatAttr.getElementType().isF32())
      return op->emitOpError(
          "unsupported arith.constant float vector width for VC4Tile lowering");
    value = FloatAttr::get(builder.getF32Type(), splat);
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
  Type resultType = op->getResult(0).getType();
  SmallVector<Value, 2> operands{lhs, rhs};
  Value rawProduct =
      createALUMul(builder, op->getLoc(), operands, mlir::vc4::MulOpcode::mul24,
                   resultType);

  // VC4's integer multiply is a 24-bit unsigned operation. M5 integer tile
  // carriers are sign-extended i32 values whose dynamic payloads fit the
  // signed 24-bit QPU multiply domain, so repair the raw product to signed
  // two's-complement semantics:
  //   a*b = mul24(a,b) - (a < 0 ? b << 24 : 0) - (b < 0 ? a << 24 : 0).
  Value shift31 = createLoadImmI32(builder, op->getLoc(), resultType, 31);
  Value shift24 = createLoadImmI32(builder, op->getLoc(), resultType, 24);
  Value lhsSign =
      createALUAdd(builder, op->getLoc(), {lhs, shift31},
                   mlir::vc4::AddOpcode::asr, resultType);
  Value rhsSign =
      createALUAdd(builder, op->getLoc(), {rhs, shift31},
                   mlir::vc4::AddOpcode::asr, resultType);
  Value lhsHigh =
      createALUAdd(builder, op->getLoc(), {lhs, shift24},
                   mlir::vc4::AddOpcode::shl, resultType);
  Value rhsHigh =
      createALUAdd(builder, op->getLoc(), {rhs, shift24},
                   mlir::vc4::AddOpcode::shl, resultType);
  Value lhsCorrection =
      createALUAdd(builder, op->getLoc(), {rhsHigh, lhsSign},
                   mlir::vc4::AddOpcode::bit_and, resultType);
  Value rhsCorrection =
      createALUAdd(builder, op->getLoc(), {lhsHigh, rhsSign},
                   mlir::vc4::AddOpcode::bit_and, resultType);
  Value withoutLhsCorrection =
      createALUAdd(builder, op->getLoc(), {rawProduct, lhsCorrection},
                   mlir::vc4::AddOpcode::sub, resultType);
  valueMap[op->getResult(0)] =
      createALUAdd(builder, op->getLoc(),
                   {withoutLhsCorrection, rhsCorrection},
                   mlir::vc4::AddOpcode::sub, resultType);
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

static bool isAffineLaneRangeValue(Value value) {
  if (isDirectLaneRangeValue(value))
    return true;

  Operation *def = value.getDefiningOp();
  if (!hasName(def, kArithMulIOpName) || def->getNumOperands() != 2)
    return false;
  return isDirectLaneRangeValue(def->getOperand(0)) ||
         isDirectLaneRangeValue(def->getOperand(1));
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

static LogicalResult lowerTileRectMask(Operation *op) {
  if (op->getNumOperands() != 0 || op->getNumResults() != 1)
    return op->emitOpError("tile_rect_mask expects no operands and one result");
  if (!isVector16I1(op->getResult(0).getType()))
    return op->emitOpError(
        "currently lowers only vector<16xi1> tile_rect_mask results");
  if (!valueHasOnlyMaskedGlobalMemoryUsers(op->getResult(0)))
    return op->emitOpError(
        "tile_rect_mask currently lowers only for masked global memory users");
  return success();
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

  if (!isAffineLaneRangeValue(op->getOperand(1))) {
    return op->emitOpError(
        "requires offsets to be vc4tile.lane_range or lane_range scaled by a positive constant for the M5 affine TMU subset");
  }

  auto access = op->getAttrOfType<mlir::vc4tile::MemoryAccessAttr>("access");
  if (access && access.getValue() != mlir::vc4tile::MemoryAccess::coalesced &&
      access.getValue() != mlir::vc4tile::MemoryAccess::affine_contiguous)
    return op->emitOpError(
        "supports only coalesced or affine_contiguous TMU loads");

  auto memorySpace = op->getAttrOfType<mlir::vc4tile::MemorySpaceAttr>("memory_space");
  if (memorySpace && memorySpace.getValue() != mlir::vc4tile::MemorySpace::global)
    return op->emitOpError("requires memory_space = #vc4tile.memory_space<global>");

  Operation *maskDef = op->getOperand(2).getDefiningOp();
  if (!hasName(maskDef, kVC4TileMaskAllOpName) &&
      !hasName(maskDef, kVC4TileTailMaskOpName) &&
      !hasName(maskDef, kVC4TileTileRectMaskOpName)) {
    return op->emitOpError(
        "currently supports only vc4tile.mask_all, vc4tile.tail_mask, or vc4tile.tile_rect_mask masks for TMU loads");
  }

  return success();
}

static std::optional<SmallVector<int32_t, 16>>
getTileRectMaskLanes(Operation *maskDef) {
  if (!hasName(maskDef, kVC4TileTileRectMaskOpName))
    return std::nullopt;

  auto rowsAttr = maskDef->getAttrOfType<IntegerAttr>("active_rows");
  auto colsAttr = maskDef->getAttrOfType<IntegerAttr>("active_cols");
  auto layout = maskDef->getAttrOfType<mlir::vc4tile::LayoutAttr>("layout");
  auto shape = maskDef->getAttrOfType<ArrayAttr>("shape");
  if (!rowsAttr || !colsAttr || !layout || !shape || shape.size() != 2 ||
      layout.getValue() != mlir::vc4tile::Layout::row_major)
    return std::nullopt;
  auto shapeRows = llvm::dyn_cast<IntegerAttr>(shape[0]);
  auto shapeCols = llvm::dyn_cast<IntegerAttr>(shape[1]);
  if (!shapeRows || !shapeCols || shapeRows.getInt() != 4 ||
      shapeCols.getInt() != 4)
    return std::nullopt;

  int64_t activeRows = rowsAttr.getInt();
  int64_t activeCols = colsAttr.getInt();
  if (activeRows < 1 || activeRows > 4 || activeCols < 1 || activeCols > 4)
    return std::nullopt;

  SmallVector<int32_t, 16> lanes;
  lanes.reserve(16);
  for (int64_t lane = 0; lane < 16; ++lane) {
    int64_t row = lane / 4;
    int64_t col = lane % 4;
    lanes.push_back((row < activeRows && col < activeCols) ? 1 : 0);
  }
  return lanes;
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
  Value activeMask;
  Operation *maskDef = op->getOperand(2).getDefiningOp();
  if (std::optional<SmallVector<int32_t, 16>> maskLanes =
          getTileRectMaskLanes(maskDef)) {
    if (!isVector16I32(op->getResult(0).getType()))
      return op->emitOpError(
          "tile_rect_mask TMU loads currently support only vector<16xi32> results");
    activeMask = createLoadImmPerElemU2(builder, op->getLoc(), addressType,
                                        *maskLanes);
  }

  auto offsetUnit =
      op->getAttrOfType<mlir::vc4tile::OffsetUnitAttr>("offset_unit");
  if (offsetUnit.getValue() == mlir::vc4tile::OffsetUnit::element) {
    Value four = createLoadImmI32(builder, op->getLoc(), addressType, 4);
    SmallVector<Value, 2> mulOperands{offsets, four};
    byteOffsets = createALUMul(builder, op->getLoc(), mulOperands,
                               mlir::vc4::MulOpcode::mul24, addressType);
  }

  if (activeMask) {
    SmallVector<Value, 2> maskedOffsetOperands{byteOffsets, activeMask};
    byteOffsets = createALUMul(builder, op->getLoc(), maskedOffsetOperands,
                               mlir::vc4::MulOpcode::mul24, addressType);
  }

  SmallVector<Value, 2> addressOperands{baseVec, byteOffsets};
  Value address = createALUAdd(builder, op->getLoc(), addressOperands,
                               mlir::vc4::AddOpcode::add, addressType);
  Value token = createTMURequest(builder, op->getLoc(), address);
  copyTileSemanticMetadataAttrs(op, token.getDefiningOp());
  Value loaded =
      createTMURead(builder, op->getLoc(), token, op->getResult(0).getType());
  if (activeMask) {
    Value zero = createLoadImmI32(builder, op->getLoc(),
                                  op->getResult(0).getType(), 0);
    Value flags = createMakeFlags(builder, op->getLoc(), {activeMask},
                                  mlir::ssavc4::FlagKind::zero_test);
    loaded = createCondSelect(builder, op->getLoc(), flags, loaded, zero,
                              mlir::vc4::Cond::zc,
                              op->getResult(0).getType());
  }
  valueMap[op->getResult(0)] = loaded;
  copyTileSemanticMetadataAttrs(op, valueMap[op->getResult(0)].getDefiningOp());
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
                 kernelContains(kernel, kVC4TileSharedStoreOpName) ||
                 kernelContains(kernel, kVC4TileVDRLoadTileOpName)))
    stagingRow = 63;

  SmallVector<int32_t, 4> operandSegmentSizes{
      1, 1, static_cast<int32_t>(operands.size() == 3), 0};
  SmallVector<NamedAttribute, 8> attrs{
      builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4)),
      builder.getNamedAttr("active_lanes",
                           builder.getI32IntegerAttr(activeLanesAttr)),
      builder.getNamedAttr("vpm_row", builder.getI32IntegerAttr(stagingRow)),
      builder.getNamedAttr("serialize", builder.getStringAttr("mutex")),
      builder.getNamedAttr("operandSegmentSizes",
                           builder.getDenseI32ArrayAttr(operandSegmentSizes))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  createSSAVC4Op(builder, op->getLoc(), kSSAVC4VDWStoreOpName, operands,
                 attrs);
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
  Operation *write = createSSAVC4Op(
      builder, op->getLoc(), kSSAVC4VPMWriteOpName, {row, value},
      {builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4)),
       builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16)),
       builder.getNamedAttr("orientation",
                            builder.getStringAttr(
                                getVPMOrientation(op, "horizontal"))),
       builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))});
  copyTileSemanticMetadataAttrs(op, write);
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
       builder.getNamedAttr("orientation",
                            builder.getStringAttr(
                                getVPMOrientation(op, "horizontal"))),
       builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))},
      op->getResult(0).getType());
  copyTileSemanticMetadataAttrs(op, valueMap[op->getResult(0)].getDefiningOp());
  return success();
}


static LogicalResult lowerVDRLoadTile(Operation *op, OpBuilder &builder,
                                      llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2)
    return op->emitOpError("expected global base and shared tile operands");
  Value address = lookupMappedValue(op, op->getOperand(0), valueMap);
  Value shared = lookupMappedValue(op, op->getOperand(1), valueMap);
  if (!address || !shared)
    return failure();

  auto elemBytes = op->getAttrOfType<IntegerAttr>("elem_bytes");
  auto rowLen = op->getAttrOfType<IntegerAttr>("row_len");
  auto nrows = op->getAttrOfType<IntegerAttr>("nrows");
  auto pitch = op->getAttrOfType<IntegerAttr>("memory_pitch_bytes");
  auto baseCol = op->getAttrOfType<IntegerAttr>("vpm_base_col");
  auto vpitch = op->getAttrOfType<IntegerAttr>("vpitch");
  if (!elemBytes || !rowLen || !nrows || !pitch || !baseCol || !vpitch)
    return op->emitOpError("requires complete VDR load planning attributes");

  Value vpmBaseRow = shared;
  auto baseRow = op->getAttrOfType<IntegerAttr>("vpm_base_row");
  if (baseRow && baseRow.getInt() != 0) {
    Value localRow = createLoadImmI32(builder, op->getLoc(), builder.getI32Type(),
                                     baseRow.getInt());
    SmallVector<Value, 2> operands{shared, localRow};
    vpmBaseRow = createALUAdd(builder, op->getLoc(), operands,
                              mlir::vc4::AddOpcode::add,
                              builder.getI32Type());
  }

  StringRef orientation = "horizontal";
  if (auto layout = op->getAttrOfType<mlir::vc4tile::LayoutAttr>("layout")) {
    if (layout.getValue() == mlir::vc4tile::Layout::vpm_col ||
        layout.getValue() == mlir::vc4tile::Layout::col_major ||
        layout.getValue() == mlir::vc4tile::Layout::transposed_view)
      orientation = "vertical";
  }
  SmallVector<NamedAttribute, 12> attrs{
      builder.getNamedAttr("elem_bytes", elemBytes),
      builder.getNamedAttr("row_len", rowLen),
      builder.getNamedAttr("nrows", nrows),
      builder.getNamedAttr("memory_pitch_bytes", pitch),
      builder.getNamedAttr("vpm_base_col", baseCol),
      builder.getNamedAttr("orientation", builder.getStringAttr(orientation)),
      builder.getNamedAttr("vpitch", vpitch),
      builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  createSSAVC4Op(builder, op->getLoc(), kSSAVC4VDRLoadOpName,
                 {address, vpmBaseRow}, attrs);
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
  if (getFloatAddOpcode(op))
    return lowerFloatALU(op, builder, valueMap);
  if (hasName(op, kArithMulFOpName))
    return lowerFloatMul(op, builder, valueMap);
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
  if (hasName(op, kVC4TileTileRectMaskOpName))
    return lowerTileRectMask(op);
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
  if (hasName(op, kVC4TileVDRLoadTileOpName))
    return lowerVDRLoadTile(op, builder, valueMap);
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


static bool isVC4TileTileComputeOp(Operation *op) {
  return hasName(op, kVC4TileTileFillOpName) ||
         hasName(op, kVC4TileTileBroadcastOpName) ||
         hasName(op, kVC4TileTileAddOpName) ||
         hasName(op, kVC4TileTileSubOpName) ||
         hasName(op, kVC4TileTileMulOpName) ||
         hasName(op, kVC4TileTileSelectOpName) ||
         hasName(op, kVC4TileTileReduceOpName) ||
         hasName(op, kVC4TileRowReduceOpName) ||
         hasName(op, kVC4TileWarpReduceOpName) ||
         hasName(op, kVC4TileBlockReduceOpName) ||
         hasName(op, kVC4TileTileDotOpName) ||
         hasName(op, kVC4TileTileContractOpName) ||
         hasName(op, kVC4TileTileMatmulOpName);
}

static std::optional<Type> getVector16DataElementType(Type type) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 ||
      vectorType.getDimSize(0) != 16)
    return std::nullopt;
  Type elementType = vectorType.getElementType();
  if (!elementType.isSignlessInteger(32) && !elementType.isF32())
    return std::nullopt;
  return elementType;
}

static Value createArithConstantFromTileFillValue(OpBuilder &builder,
                                                  Operation *op,
                                                  Attribute value,
                                                  Type elementType) {
  if (elementType.isSignlessInteger(32)) {
    auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
    if (!intAttr) {
      op->emitOpError("tile_fill i32 value must be an integer attribute");
      return Value();
    }
    TypedAttr typed = builder.getIntegerAttr(elementType, intAttr.getInt());
    return arith::ConstantOp::create(builder, op->getLoc(), elementType, typed)
        .getResult();
  }

  if (elementType.isF32()) {
    auto floatAttr = llvm::dyn_cast<FloatAttr>(value);
    if (!floatAttr) {
      op->emitOpError("tile_fill f32 value must be a float attribute");
      return Value();
    }
    TypedAttr typed = FloatAttr::get(elementType, floatAttr.getValue());
    return arith::ConstantOp::create(builder, op->getLoc(), elementType, typed)
        .getResult();
  }

  op->emitOpError("tile_fill supports only i32 and f32 element types");
  return Value();
}

static LogicalResult canonicalizeTileFillOp(Operation *op,
                                            OpBuilder &builder) {
  if (op->getNumOperands() != 0 || op->getNumResults() != 1)
    return op->emitOpError("tile_fill expects no operands and one result");
  std::optional<Type> elementType =
      getVector16DataElementType(op->getResult(0).getType());
  if (!elementType)
    return op->emitOpError(
        "tile_fill result must be vector<16xi32> or vector<16xf32>");
  Attribute value = op->getAttr("value");
  if (!value)
    return op->emitOpError("tile_fill requires a value attribute");

  Value scalar = createArithConstantFromTileFillValue(builder, op, value,
                                                      *elementType);
  if (!scalar)
    return failure();
  Value result = mlir::vector::BroadcastOp::create(
                     builder, op->getLoc(), op->getResult(0).getType(), scalar)
                     .getResult();
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

static LogicalResult canonicalizeTileBroadcastOp(Operation *op,
                                                 OpBuilder &builder) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return op->emitOpError("tile_broadcast expects one operand and one result");
  std::optional<Type> elementType =
      getVector16DataElementType(op->getResult(0).getType());
  if (!elementType)
    return op->emitOpError(
        "tile_broadcast result must be vector<16xi32> or vector<16xf32>");
  if (op->getOperand(0).getType() != *elementType) {
    return op->emitOpError(
        "tile_broadcast operand type must match result element type");
  }

  Value result = mlir::vector::BroadcastOp::create(
                     builder, op->getLoc(), op->getResult(0).getType(),
                     op->getOperand(0))
                     .getResult();
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

static LogicalResult canonicalizeTileBinaryComputeOp(Operation *op,
                                                     OpBuilder &builder) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("tile binary compute expects two operands and one result");
  Type resultType = op->getResult(0).getType();
  if (op->getOperand(0).getType() != resultType ||
      op->getOperand(1).getType() != resultType) {
    return op->emitOpError(
        "tile binary operands and result must have identical types");
  }
  std::optional<Type> elementType = getVector16DataElementType(resultType);
  if (!elementType)
    return op->emitOpError(
        "tile binary result must be vector<16xi32> or vector<16xf32>");

  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Value result;
  if (elementType->isSignlessInteger(32)) {
    if (hasName(op, kVC4TileTileAddOpName))
      result = arith::AddIOp::create(builder, op->getLoc(), lhs, rhs)
                   .getResult();
    else if (hasName(op, kVC4TileTileSubOpName))
      result = arith::SubIOp::create(builder, op->getLoc(), lhs, rhs)
                   .getResult();
    else if (hasName(op, kVC4TileTileMulOpName))
      result = arith::MulIOp::create(builder, op->getLoc(), lhs, rhs)
                   .getResult();
  } else if (elementType->isF32()) {
    if (hasName(op, kVC4TileTileAddOpName))
      result = arith::AddFOp::create(builder, op->getLoc(), lhs, rhs)
                   .getResult();
    else if (hasName(op, kVC4TileTileSubOpName))
      result = arith::SubFOp::create(builder, op->getLoc(), lhs, rhs)
                   .getResult();
    else if (hasName(op, kVC4TileTileMulOpName))
      result = arith::MulFOp::create(builder, op->getLoc(), lhs, rhs)
                   .getResult();
  }
  if (!result)
    return op->emitOpError("unsupported tile binary compute operation");
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return success();
}

static bool tileSelectTailMaskUseIsPredicatedStore(Operation *selectOp,
                                                   Operation *user,
                                                   Value mask) {
  (void)selectOp;
  if (hasName(user, kVC4TileTileStoreOpName) && user->getNumOperands() == 4 &&
      user->getOperand(3) == mask)
    return true;
  if (hasName(user, kVC4TileMaskedStoreGlobalOpName) &&
      user->getNumOperands() == 4 && user->getOperand(3) == mask)
    return true;
  return false;
}

static LogicalResult canonicalizeTileSelectOp(Operation *op,
                                              OpBuilder &builder) {
  (void)builder;
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return op->emitOpError("tile_select expects mask, true_value, false_value, and one result");
  if (!isVector16I1(op->getOperand(0).getType()))
    return op->emitOpError("tile_select mask must be vector<16xi1>");
  Type resultType = op->getResult(0).getType();
  if (op->getOperand(1).getType() != resultType ||
      op->getOperand(2).getType() != resultType)
    return op->emitOpError(
        "tile_select true_value, false_value, and result must have identical types");
  if (!getVector16DataElementType(resultType))
    return op->emitOpError(
        "tile_select result must be vector<16xi32> or vector<16xf32>");

  Value mask = op->getOperand(0);
  Operation *maskDef = mask.getDefiningOp();
  if (hasName(maskDef, kVC4TileMaskAllOpName)) {
    op->getResult(0).replaceAllUsesWith(op->getOperand(1));
    op->erase();
    return success();
  }

  if (hasName(maskDef, kVC4TileTailMaskOpName)) {
    for (Operation *user : llvm::make_early_inc_range(op->getResult(0).getUsers())) {
      if (!tileSelectTailMaskUseIsPredicatedStore(op, user, mask)) {
        return op->emitOpError(
            "tile_select with tail_predicated mask requires every use to be "
            "a tile/global store using the same mask in M5");
      }
    }
    op->getResult(0).replaceAllUsesWith(op->getOperand(1));
    op->erase();
    return success();
  }

  return op->emitOpError(
      "tile_select currently supports only vc4tile.mask_all or "
      "vc4tile.tail_mask masks in M5");
}

static bool isMaskAllValue(Value mask) {
  return hasName(mask.getDefiningOp(), kVC4TileMaskAllOpName);
}

static LogicalResult verifySurfaceReductionForCanonicalization(Operation *op,
                                                               bool requireAxis) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("tile reduction expects input, mask, and one result");
  Type inputType = op->getOperand(0).getType();
  Type resultType = op->getResult(0).getType();
  if (!getVector16DataElementType(inputType) || inputType != resultType)
    return op->emitOpError(
        "tile reduction requires matching vector<16xi32> or vector<16xf32> input/result types");
  if (!isVector16I1(op->getOperand(1).getType()))
    return op->emitOpError("tile reduction mask must be vector<16xi1>");
  if (!isMaskAllValue(op->getOperand(1)))
    return op->emitOpError(
        "tile reductions currently support only vc4tile.mask_all masks in M5");
  auto kind = op->getAttrOfType<mlir::vc4tile::ReduceKindAttr>("kind");
  if (!kind)
    return op->emitOpError("tile reduction requires a kind attribute");
  if (kind.getValue() != mlir::vc4tile::ReduceKind::add)
    return op->emitOpError(
        "tile reductions currently support only kind = #vc4tile.reduce_kind<add> in M5");
  if (requireAxis && !op->getAttrOfType<IntegerAttr>("axis"))
    return op->emitOpError("tile_reduce requires an axis attribute");
  return success();
}

static Operation *createCoreReduceFromSurface(OpBuilder &builder,
                                              Operation *op, Value input,
                                              Value mask) {
  OperationState state(op->getLoc(), kVC4TileReduceOpName);
  state.addOperands({input, mask});
  state.addAttribute("kind", op->getAttr("kind"));
  state.addTypes(op->getResultTypes());
  return builder.create(state);
}

static Operation *createCoreAddReduce(OpBuilder &builder, Operation *op,
                                      Value input, Value mask,
                                      TypeRange resultTypes) {
  OperationState state(op->getLoc(), kVC4TileReduceOpName);
  state.addOperands({input, mask});
  state.addAttribute(
      "kind", mlir::vc4tile::ReduceKindAttr::get(
                  builder.getContext(), mlir::vc4tile::ReduceKind::add));
  state.addTypes(resultTypes);
  return builder.create(state);
}

static LogicalResult verifyContractionVectorInputsForCanonicalization(
    Operation *op, unsigned expectedOperands) {
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return op->emitOpError("tile contraction expects the declared operands and one result");
  Type resultType = op->getResult(0).getType();
  if (!getVector16DataElementType(resultType))
    return op->emitOpError(
        "tile contraction result must be vector<16xi32> or vector<16xf32>");
  for (unsigned i = 0; i + 1 < expectedOperands; ++i) {
    if (op->getOperand(i).getType() != resultType)
      return op->emitOpError(
          "tile contraction data operands and result must have identical vector types");
  }
  if (!isVector16I1(op->getOperand(expectedOperands - 1).getType()))
    return op->emitOpError("tile contraction mask must be vector<16xi1>");
  if (!isMaskAllValue(op->getOperand(expectedOperands - 1)))
    return op->emitOpError(
        "tile contractions currently support only vc4tile.mask_all masks in M5");
  auto kAttr = op->getAttrOfType<IntegerAttr>("k");
  if (!kAttr || kAttr.getInt() < 1 || kAttr.getInt() > 16)
    return op->emitOpError("tile contraction k must be in range [1, 16]");
  return success();
}

static Value createElementwiseProduct(OpBuilder &builder, Operation *op,
                                      Value lhs, Value rhs, Type resultType) {
  std::optional<Type> elementType = getVector16DataElementType(resultType);
  if (!elementType)
    return Value();
  if (elementType->isSignlessInteger(32))
    return arith::MulIOp::create(builder, op->getLoc(), lhs, rhs).getResult();
  if (elementType->isF32())
    return arith::MulFOp::create(builder, op->getLoc(), lhs, rhs).getResult();
  return Value();
}

static Value createElementwiseAccumulation(OpBuilder &builder, Operation *op,
                                           Value lhs, Value rhs,
                                           Type resultType) {
  std::optional<Type> elementType = getVector16DataElementType(resultType);
  if (!elementType)
    return Value();
  if (elementType->isSignlessInteger(32))
    return arith::AddIOp::create(builder, op->getLoc(), lhs, rhs).getResult();
  if (elementType->isF32())
    return arith::AddFOp::create(builder, op->getLoc(), lhs, rhs).getResult();
  return Value();
}

static Value createCoreRotate(OpBuilder &builder, Operation *op, Value input,
                              int64_t amount, Type resultType) {
  amount %= 16;
  if (amount == 0)
    return input;
  OperationState state(op->getLoc(), kVC4TileRotateOpName);
  state.addOperands(input);
  state.addAttribute("amount", builder.getI32IntegerAttr(amount));
  state.addTypes(resultType);
  return builder.create(state)->getResult(0);
}

static Value createVector16I32MaskConstant(OpBuilder &builder, Location loc,
                                           VectorType vectorType,
                                           ArrayRef<int32_t> lanes) {
  SmallVector<llvm::APInt, 16> values;
  values.reserve(lanes.size());
  for (int32_t lane : lanes)
    values.emplace_back(/*numBits=*/32, lane);
  auto attr = DenseIntElementsAttr::get(vectorType, values);
  return arith::ConstantOp::create(builder, loc, vectorType, attr).getResult();
}

static Value createGatheredI32Lanes(
    OpBuilder &builder, Operation *op, Value input, Type resultType,
    llvm::function_ref<unsigned(unsigned)> sourceLaneForOutputLane) {
  auto vectorType = llvm::dyn_cast<VectorType>(resultType);
  if (!vectorType || vectorType.getRank() != 1 ||
      vectorType.getDimSize(0) != 16 ||
      !vectorType.getElementType().isSignlessInteger(32))
    return Value();

  Value gathered;
  for (unsigned amount = 0; amount < 16; ++amount) {
    SmallVector<int32_t, 16> lanes(16, 0);
    bool any = false;
    for (unsigned lane = 0; lane < 16; ++lane) {
      unsigned sourceLane = sourceLaneForOutputLane(lane);
      if (((sourceLane + 16 - lane) & 15) == amount) {
        lanes[lane] = 1;
        any = true;
      }
    }
    if (!any)
      continue;
    Value rotated = createCoreRotate(builder, op, input, amount, resultType);
    Value mask = createVector16I32MaskConstant(builder, op->getLoc(), vectorType,
                                              lanes);
    Value selected = createElementwiseProduct(builder, op, rotated, mask,
                                              resultType);
    if (!selected)
      return Value();
    gathered = gathered ? createElementwiseAccumulation(builder, op, gathered,
                                                        selected, resultType)
                        : selected;
    if (!gathered)
      return Value();
  }
  return gathered;
}

static bool hasContractionDimsAttr(Operation *op) {
  auto dims = op->getAttrOfType<ArrayAttr>("contracting_dims");
  if (!dims || dims.size() != 2)
    return false;
  auto lhsDims = llvm::dyn_cast<ArrayAttr>(dims[0]);
  auto rhsDims = llvm::dyn_cast<ArrayAttr>(dims[1]);
  if (!lhsDims || !rhsDims || lhsDims.size() != 1 || rhsDims.size() != 1)
    return false;
  auto lhsDim = llvm::dyn_cast<IntegerAttr>(lhsDims[0]);
  auto rhsDim = llvm::dyn_cast<IntegerAttr>(rhsDims[0]);
  return lhsDim && rhsDim && lhsDim.getInt() == 1 && rhsDim.getInt() == 0;
}

static bool hasIteratorTypesAttr(Operation *op) {
  auto iterators = op->getAttrOfType<ArrayAttr>("iterator_types");
  if (!iterators || iterators.size() != 3)
    return false;
  constexpr llvm::StringLiteral expected[3] = {"parallel", "parallel",
                                               "reduction"};
  for (auto [index, attr] : llvm::enumerate(iterators)) {
    auto stringAttr = llvm::dyn_cast<StringAttr>(attr);
    if (!stringAttr || stringAttr.getValue() != expected[index])
      return false;
  }
  return true;
}

static LogicalResult canonicalizeTileDotOp(Operation *op, OpBuilder &builder) {
  if (failed(verifyContractionVectorInputsForCanonicalization(op,
                                                              /*expectedOperands=*/3)))
    return failure();
  Type resultType = op->getResult(0).getType();
  Value product = createElementwiseProduct(builder, op, op->getOperand(0),
                                           op->getOperand(1), resultType);
  if (!product)
    return op->emitOpError("unsupported tile_dot element type");
  Operation *reduced = createCoreAddReduce(builder, op, product, op->getOperand(2),
                                           op->getResultTypes());
  op->getResult(0).replaceAllUsesWith(reduced->getResult(0));
  op->erase();
  return success();
}

static LogicalResult canonicalizeTileContractOrMatmulOp(Operation *op,
                                                        OpBuilder &builder) {
  if (failed(verifyContractionVectorInputsForCanonicalization(op,
                                                              /*expectedOperands=*/4)))
    return failure();
  Type resultType = op->getResult(0).getType();
  auto vectorType = llvm::dyn_cast<VectorType>(resultType);
  if (!vectorType || !vectorType.getElementType().isSignlessInteger(32))
    return op->emitOpError(
        "tile_contract/tile_matmul currently lower only i32 4x4x4 matrix forms");
  auto mAttr = op->getAttrOfType<IntegerAttr>("m");
  auto nAttr = op->getAttrOfType<IntegerAttr>("n");
  auto kAttr = op->getAttrOfType<IntegerAttr>("k");
  if (!mAttr || !nAttr || !kAttr || mAttr.getInt() != 4 ||
      nAttr.getInt() != 4 || kAttr.getInt() != 4)
    return op->emitOpError(
        "tile_contract/tile_matmul require m = 4, n = 4, k = 4 for the "
        "current single-vector matrix carrier");
  if (!hasContractionDimsAttr(op))
    return op->emitOpError(
        "tile_contract/tile_matmul require contracting_dims = [[1], [0]]");
  if (!hasIteratorTypesAttr(op))
    return op->emitOpError(
        "tile_contract/tile_matmul require iterator_types = [\"parallel\", "
        "\"parallel\", \"reduction\"]");
  auto lhsLayout = op->getAttrOfType<mlir::vc4tile::LayoutAttr>("lhs_layout");
  auto rhsLayout = op->getAttrOfType<mlir::vc4tile::LayoutAttr>("rhs_layout");
  auto accLayout = op->getAttrOfType<mlir::vc4tile::LayoutAttr>("acc_layout");
  if (!lhsLayout || !rhsLayout || !accLayout ||
      lhsLayout.getValue() != mlir::vc4tile::Layout::row_major ||
      accLayout.getValue() != mlir::vc4tile::Layout::row_major)
    return op->emitOpError(
        "tile_contract/tile_matmul require row_major lhs and acc layouts in M5");
  bool rhsRowMajor = rhsLayout.getValue() == mlir::vc4tile::Layout::row_major;
  bool rhsTransposed =
      rhsLayout.getValue() == mlir::vc4tile::Layout::col_major ||
      rhsLayout.getValue() == mlir::vc4tile::Layout::transposed_view;
  if (!rhsRowMajor && !rhsTransposed)
    return op->emitOpError(
        "tile_contract/tile_matmul require row_major, col_major, or "
        "transposed_view rhs layout in M5");
  int64_t activeK = 4;
  if (auto activeKAttr = op->getAttrOfType<IntegerAttr>("active_k")) {
    activeK = activeKAttr.getInt();
    if (activeK < 1 || activeK > 4)
      return op->emitOpError(
          "tile_contract/tile_matmul active_k must be in range [1, 4]");
  }

  Value update;
  for (unsigned kk = 0; kk < static_cast<unsigned>(activeK); ++kk) {
    Value lhsK = createGatheredI32Lanes(
        builder, op, op->getOperand(0), resultType, [kk](unsigned lane) {
          unsigned row = lane / 4;
          return row * 4 + kk;
        });
    Value rhsK = createGatheredI32Lanes(
        builder, op, op->getOperand(1), resultType,
        [kk, rhsRowMajor](unsigned lane) {
          unsigned col = lane & 3;
          return rhsRowMajor ? kk * 4 + col : col * 4 + kk;
        });
    if (!lhsK || !rhsK)
      return op->emitOpError("failed to materialize 4x4x4 lane gathers");
    Value product = createElementwiseProduct(builder, op, lhsK, rhsK,
                                             resultType);
    if (!product)
      return op->emitOpError("unsupported tile contraction element type");
    update = update ? createElementwiseAccumulation(builder, op, update,
                                                    product, resultType)
                    : product;
    if (!update)
      return op->emitOpError("unsupported tile contraction accumulation type");
  }

  Value accumulated = createElementwiseAccumulation(builder, op, op->getOperand(2),
                                                    update, resultType);
  if (!accumulated)
    return op->emitOpError("unsupported tile contraction accumulation type");
  op->getResult(0).replaceAllUsesWith(accumulated);
  op->erase();
  return success();
}

static Operation *createCoreBarrier(OpBuilder &builder, Location loc) {
  OperationState state(loc, kVC4TileBarrierOpName);
  state.addAttribute(
      "scope", mlir::vc4tile::BarrierScopeAttr::get(
                   builder.getContext(), mlir::vc4tile::BarrierScope::block));
  return builder.create(state);
}

static Operation *createCoreSharedAlloc(OpBuilder &builder, Operation *op) {
  OperationState state(op->getLoc(), kVC4TileSharedAllocOpName);
  state.addAttribute("rows", builder.getI32IntegerAttr(1));
  state.addAttribute("elem_bytes", builder.getI32IntegerAttr(4));
  state.addAttribute(
      "memory_space", mlir::vc4tile::MemorySpaceAttr::get(
                          builder.getContext(),
                          mlir::vc4tile::MemorySpace::shared_vpm));
  state.addAttribute(
      "layout", mlir::vc4tile::VPMLayoutAttr::get(
                    builder.getContext(), mlir::vc4tile::VPMLayout::row_major));
  state.addTypes(mlir::vc4tile::SharedTileType::get(builder.getContext()));
  return builder.create(state);
}

static Operation *createCoreSharedStore(OpBuilder &builder, Operation *op,
                                        Value shared, Value row, Value value,
                                        Value mask) {
  OperationState state(op->getLoc(), kVC4TileSharedStoreOpName);
  state.addOperands({shared, row, value, mask});
  state.addAttribute("elem_bytes", builder.getI32IntegerAttr(4));
  state.addAttribute(
      "memory_space", mlir::vc4tile::MemorySpaceAttr::get(
                          builder.getContext(),
                          mlir::vc4tile::MemorySpace::shared_vpm));
  state.addAttribute(
      "layout", mlir::vc4tile::VPMLayoutAttr::get(
                    builder.getContext(), mlir::vc4tile::VPMLayout::row_major));
  return builder.create(state);
}

static Operation *createCoreSharedLoad(OpBuilder &builder, Operation *op,
                                       Value shared, Value row, Value mask,
                                       TypeRange resultTypes) {
  OperationState state(op->getLoc(), kVC4TileSharedLoadOpName);
  state.addOperands({shared, row, mask});
  state.addAttribute("elem_bytes", builder.getI32IntegerAttr(4));
  state.addAttribute(
      "memory_space", mlir::vc4tile::MemorySpaceAttr::get(
                          builder.getContext(),
                          mlir::vc4tile::MemorySpace::shared_vpm));
  state.addAttribute(
      "layout", mlir::vc4tile::VPMLayoutAttr::get(
                    builder.getContext(), mlir::vc4tile::VPMLayout::row_major));
  state.addTypes(resultTypes);
  return builder.create(state);
}

static LogicalResult canonicalizeTileOrWarpReduceOp(Operation *op,
                                                    OpBuilder &builder) {
  if (failed(verifySurfaceReductionForCanonicalization(
          op, /*requireAxis=*/hasName(op, kVC4TileTileReduceOpName))))
    return failure();
  Operation *reduce = createCoreReduceFromSurface(builder, op, op->getOperand(0),
                                                 op->getOperand(1));
  op->getResult(0).replaceAllUsesWith(reduce->getResult(0));
  op->erase();
  return success();
}

static LogicalResult canonicalizeBlockReduceOp(Operation *op,
                                               OpBuilder &builder) {
  if (failed(verifySurfaceReductionForCanonicalization(op,
                                                       /*requireAxis=*/false)))
    return failure();
  Value mask = op->getOperand(1);
  Value zero = arith::ConstantIntOp::create(builder, op->getLoc(), 0, 32);
  Operation *shared = createCoreSharedAlloc(builder, op);
  createCoreSharedStore(builder, op, shared->getResult(0), zero, op->getOperand(0),
                        mask);
  createCoreBarrier(builder, op->getLoc());
  Operation *loaded = createCoreSharedLoad(builder, op, shared->getResult(0), zero,
                                          mask, op->getResultTypes());
  Operation *reduce = createCoreReduceFromSurface(builder, op,
                                                 loaded->getResult(0), mask);
  createCoreBarrier(builder, op->getLoc());
  op->getResult(0).replaceAllUsesWith(reduce->getResult(0));
  op->erase();
  return success();
}

static LogicalResult canonicalizeOneTileComputeOp(Operation *op) {
  OpBuilder builder(op);
  if (hasName(op, kVC4TileTileFillOpName))
    return canonicalizeTileFillOp(op, builder);
  if (hasName(op, kVC4TileTileBroadcastOpName))
    return canonicalizeTileBroadcastOp(op, builder);
  if (hasName(op, kVC4TileTileAddOpName) ||
      hasName(op, kVC4TileTileSubOpName) ||
      hasName(op, kVC4TileTileMulOpName))
    return canonicalizeTileBinaryComputeOp(op, builder);
  if (hasName(op, kVC4TileTileSelectOpName))
    return canonicalizeTileSelectOp(op, builder);
  if (hasName(op, kVC4TileTileReduceOpName) ||
      hasName(op, kVC4TileRowReduceOpName) ||
      hasName(op, kVC4TileWarpReduceOpName))
    return canonicalizeTileOrWarpReduceOp(op, builder);
  if (hasName(op, kVC4TileBlockReduceOpName))
    return canonicalizeBlockReduceOp(op, builder);
  if (hasName(op, kVC4TileTileDotOpName))
    return canonicalizeTileDotOp(op, builder);
  if (hasName(op, kVC4TileTileContractOpName) ||
      hasName(op, kVC4TileTileMatmulOpName))
    return canonicalizeTileContractOrMatmulOp(op, builder);
  return op->emitOpError("unrecognized VC4Tile tile compute operation");
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
    SmallVector<Operation *, 8> laneStrideSurfaceOps;
    SmallVector<Operation *, 16> tileComputeOps;
    SmallVector<Operation *, 4> coreLegalizeMarkers;
    module.walk([&](Operation *op) {
      if (hasName(op, kVC4TileSurfacePlaceholderOpName)) {
        placeholders.push_back(op);
        return;
      }
      if (hasName(op, kVC4TileKernelOpName) &&
          op->getAttr("requires_core_legalize"))
        coreLegalizeMarkers.push_back(op);
      if (isVC4TileTileComputeOp(op))
        tileComputeOps.push_back(op);
      if ((hasName(op, kVC4TileTileLoadOpName) ||
           hasName(op, kVC4TileTileStoreOpName)) &&
          (op->getAttr("lane_stride") || op->getAttr("stride")))
        laneStrideSurfaceOps.push_back(op);
    });

    for (Operation *op : coreLegalizeMarkers)
      op->removeAttr("requires_core_legalize");

    Builder builder(module.getContext());
    for (Operation *op : laneStrideSurfaceOps) {
      IntegerAttr strideAttr = op->getAttrOfType<IntegerAttr>("lane_stride");
      if (!strideAttr)
        strideAttr = op->getAttrOfType<IntegerAttr>("stride");
      if (strideAttr && !op->getAttr("strides"))
        op->setAttr("strides", builder.getArrayAttr({strideAttr}));
      op->removeAttr("lane_stride");
      op->removeAttr("stride");
    }

    for (Operation *op : tileComputeOps) {
      if (!op->getBlock())
        continue;
      if (failed(canonicalizeOneTileComputeOp(op))) {
        signalPassFailure();
        return;
      }
    }

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

static FailureOr<int64_t> getTileLaneStride(Operation *op) {
  IntegerAttr strideAttr = op->getAttrOfType<IntegerAttr>("lane_stride");
  if (!strideAttr)
    strideAttr = op->getAttrOfType<IntegerAttr>("stride");
  if (strideAttr) {
    int64_t stride = strideAttr.getInt();
    if (stride <= 0)
      return op->emitOpError("lane_stride must be a positive element stride");
    return stride;
  }

  if (auto strides = op->getAttrOfType<ArrayAttr>("strides")) {
    if (strides.empty())
      return op->emitOpError("strides must contain at least one element stride");
    auto last = llvm::dyn_cast<IntegerAttr>(strides.getValue().back());
    if (!last)
      return op->emitOpError("strides entries must be integer attributes");
    int64_t stride = last.getInt();
    if (stride <= 0)
      return op->emitOpError("strides entries must be positive element strides");
    return stride;
  }

  return 1;
}

static Value createLaneOffsetsForTileCopy(OpBuilder &builder, Operation *op,
                                          int64_t laneStride) {
  Value lanes = createVC4TileLaneRangeCore(builder, op->getLoc());
  if (laneStride == 1)
    return lanes;

  Value strideScalar =
      arith::ConstantIntOp::create(builder, op->getLoc(), laneStride, 32);
  Value strideVector = mlir::vector::BroadcastOp::create(
      builder, op->getLoc(), getVector16I32Type(builder), strideScalar)
                           .getResult();
  return arith::MulIOp::create(builder, op->getLoc(), lanes, strideVector)
      .getResult();
}

static NamedAttribute namedAttr(OpBuilder &builder, StringRef name,
                                Attribute attr) {
  return builder.getNamedAttr(name, attr);
}


static void appendTileSemanticMetadataAttrs(Operation *source,
                                            OpBuilder &builder,
                                            SmallVectorImpl<NamedAttribute> &attrs) {
  for (StringRef name : {"role", "boundary", "copy_stage", "reuse_hint"}) {
    if (Attribute attr = source->getAttr(name))
      attrs.push_back(namedAttr(builder, name, attr));
  }
}

static void copyTileSemanticMetadataAttrs(Operation *source, Operation *target) {
  for (StringRef name : {"role", "boundary", "copy_stage", "reuse_hint"}) {
    if (Attribute attr = source->getAttr(name))
      target->setAttr(name, attr);
  }
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
  FailureOr<int64_t> laneStride = getTileLaneStride(op);
  if (failed(laneStride))
    return failure();
  Value laneOffsets = createLaneOffsetsForTileCopy(builder, op, *laneStride);
  mlir::vc4tile::MemoryAccess access =
      *laneStride == 1 ? mlir::vc4tile::MemoryAccess::coalesced
                       : mlir::vc4tile::MemoryAccess::affine_contiguous;
  SmallVector<NamedAttribute, 8> attrs{
      namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
      namedAttr(builder, "offset_unit",
                getVC4TileOffsetUnitAttr(builder,
                                         mlir::vc4tile::OffsetUnit::element)),
      namedAttr(builder, "memory_space",
                getVC4TileMemorySpaceAttr(builder,
                                          mlir::vc4tile::MemorySpace::global)),
      namedAttr(builder, "access", getVC4TileMemoryAccessAttr(builder, access))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  Operation *load = createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileMaskedLoadGlobalOpName,
      {*adjustedBase, laneOffsets, op->getOperand(2)}, attrs,
      op->getResultTypes());
  return load->getResult(0);
}

static LogicalResult planSharedGlobalTileStore(Operation *op, OpBuilder &builder);

static LogicalResult planRegisterGlobalTileStore(Operation *op,
                                                 OpBuilder &builder) {
  auto memorySpace = op->getAttrOfType<mlir::vc4tile::MemorySpaceAttr>("memory_space");
  if (!memorySpace || memorySpace.getValue() != mlir::vc4tile::MemorySpace::global)
    return op->emitOpError("copy planner v1 supports tile_store only for register->global; use copy_tile for shared paths");
  if (op->getNumOperands() != 4)
    return op->emitOpError("expected tile, base, offset, and mask operands");
  if (mlir::vc4tile::isVC4TileSharedTileType(op->getOperand(0).getType()))
    return planSharedGlobalTileStore(op, builder);
  if (!isVector16Data(op->getOperand(0).getType()))
    return op->emitOpError("copy planner v1 requires tile_store input to be vector<16xi32>, vector<16xf32>, or !vc4tile.shared_tile");

  FailureOr<int64_t> laneStride = getTileLaneStride(op);
  if (failed(laneStride))
    return failure();
  if (*laneStride != 1)
    return op->emitOpError(
        "copy planner v1 supports affine lane strides for tile_load; tile_store affine strides require later VDW stride support");

  FailureOr<Value> adjustedBase = createAdjustedGlobalBaseForTileCopy(
      builder, op, op->getOperand(1), op->getOperand(2));
  if (failed(adjustedBase))
    return failure();
  Value lanes = createLaneOffsetsForTileCopy(builder, op, *laneStride);
  SmallVector<NamedAttribute, 8> attrs{
      namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
      namedAttr(builder, "offset_unit",
                getVC4TileOffsetUnitAttr(builder,
                                         mlir::vc4tile::OffsetUnit::element)),
      namedAttr(builder, "memory_space",
                getVC4TileMemorySpaceAttr(builder,
                                          mlir::vc4tile::MemorySpace::global)),
      namedAttr(builder, "access",
                getVC4TileMemoryAccessAttr(
                    builder, mlir::vc4tile::MemoryAccess::affine_contiguous))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileMaskedStoreGlobalOpName,
      {*adjustedBase, lanes, op->getOperand(0), op->getOperand(3)}, attrs);
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
  SmallVector<NamedAttribute, 8> attrs{
      namedAttr(builder, "rows", rows),
      namedAttr(builder, "elem_bytes", elemBytes),
      namedAttr(builder, "memory_space",
                getVC4TileMemorySpaceAttr(
                    builder, mlir::vc4tile::MemorySpace::shared_vpm)),
      namedAttr(builder, "layout",
                getVPMLayoutFromTileLayout(
                    builder,
                    op->getAttrOfType<mlir::vc4tile::LayoutAttr>("layout")))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  Operation *alloc = createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileSharedAllocOpName, {}, attrs,
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
  SmallVector<NamedAttribute, 8> attrs{
      namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
      namedAttr(builder, "memory_space",
                getVC4TileMemorySpaceAttr(
                    builder, mlir::vc4tile::MemorySpace::shared_vpm)),
      namedAttr(builder, "layout",
                getVPMLayoutFromTileLayout(
                    builder,
                    op->getAttrOfType<mlir::vc4tile::LayoutAttr>("dst_layout")))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  createVC4TileCoreOp(builder, op->getLoc(), kVC4TileSharedStoreOpName,
                      {handle, row, value, mask}, attrs);
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
  SmallVector<NamedAttribute, 8> attrs{
      namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
      namedAttr(builder, "memory_space",
                getVC4TileMemorySpaceAttr(
                    builder, mlir::vc4tile::MemorySpace::shared_vpm)),
      namedAttr(builder, "layout",
                getVPMLayoutFromTileLayout(
                    builder,
                    op->getAttrOfType<mlir::vc4tile::LayoutAttr>("src_layout")))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  Operation *load = createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileSharedLoadOpName, {handle, row, mask},
      attrs, op->getResultTypes());
  op->getResult(0).replaceAllUsesWith(load->getResult(0));
  return success();
}


static FailureOr<SmallVector<int64_t, 2>> getTileShape2D(Operation *op) {
  auto shapeAttr = op->getAttrOfType<ArrayAttr>("shape");
  if (!shapeAttr || shapeAttr.empty() || shapeAttr.size() > 2)
    return op->emitOpError("M5 shared VPM copy planning requires rank-1 or rank-2 static shape");
  SmallVector<int64_t, 2> shape;
  for (Attribute attr : shapeAttr) {
    auto intAttr = llvm::dyn_cast<IntegerAttr>(attr);
    if (!intAttr || intAttr.getInt() <= 0)
      return op->emitOpError("shape entries must be positive integers");
    shape.push_back(intAttr.getInt());
  }
  if (shape.size() == 1)
    shape.insert(shape.begin(), 1);
  if (shape[0] > 16 || shape[1] > 16)
    return op->emitOpError("M5 VPM copy planning supports at most 16x16 32-bit tiles");
  return shape;
}

static IntegerAttr getOptionalI32AttrOr(Operation *op, OpBuilder &builder,
                                        StringRef name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr;
  return builder.getI32IntegerAttr(fallback);
}

static Value createI32ConstantForPlan(OpBuilder &builder, Location loc,
                                      int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 32).getResult();
}

static Value addI32Constant(OpBuilder &builder, Location loc, Value base,
                            int64_t value) {
  Value constant = createI32ConstantForPlan(builder, loc, value);
  return arith::AddIOp::create(builder, loc, base, constant).getResult();
}

static Attribute getVPMTileLoadLayoutAttr(OpBuilder &builder,
                                          mlir::vc4tile::LayoutAttr layout) {
  return getVPMLayoutFromTileLayout(builder, layout,
                                    /*defaultColumn=*/false);
}

static LogicalResult planGlobalSharedCopy(Operation *op, OpBuilder &builder) {
  if (op->getNumOperands() < 3 || op->getNumOperands() > 4 || op->getNumResults() != 0) {
    return op->emitOpError(
        "global->shared_vpm copy_tile expects operands (base, shared_tile, offset[, mask]) and no results");
  }
  Value base = op->getOperand(0);
  Value shared = op->getOperand(1);
  Value offset = op->getOperand(2);
  if (!isI32Scalar(base) || !mlir::vc4tile::isVC4TileSharedTileType(shared.getType()) ||
      !isI32Scalar(offset)) {
    return op->emitOpError(
        "global->shared_vpm copy_tile requires i32 base, !vc4tile.shared_tile handle, and i32 element offset");
  }
  if (op->getNumOperands() == 4 && !isVector16I1(op->getOperand(3).getType()))
    return op->emitOpError("optional global->shared_vpm copy_tile mask must be vector<16xi1>");
  FailureOr<SmallVector<int64_t, 2>> shape = getTileShape2D(op);
  if (failed(shape))
    return failure();
  int64_t nrows = (*shape)[0];
  int64_t rowLen = (*shape)[1];
  FailureOr<Value> adjustedBase = createAdjustedGlobalBaseForTileCopy(builder, op, base, offset);
  if (failed(adjustedBase))
    return failure();
  auto dstLayout = op->getAttrOfType<mlir::vc4tile::LayoutAttr>("dst_layout");
  if (dstLayout && dstLayout.getValue() != mlir::vc4tile::Layout::vpm_row &&
      dstLayout.getValue() != mlir::vc4tile::Layout::row_major &&
      dstLayout.getValue() != mlir::vc4tile::Layout::affine_2d) {
    return op->emitOpError(
        "global->shared_vpm VDR copy requires row-major or vpm_row destination layout in M5");
  }
  int64_t memoryPitch = getOptionalI32AttrOr(op, builder, "memory_pitch_bytes", rowLen * 4).getInt();
  SmallVector<NamedAttribute, 10> attrs{
      namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
      namedAttr(builder, "row_len", builder.getI32IntegerAttr(rowLen)),
      namedAttr(builder, "nrows", builder.getI32IntegerAttr(nrows)),
      namedAttr(builder, "memory_pitch_bytes",
                builder.getI32IntegerAttr(memoryPitch)),
      namedAttr(builder, "vpm_base_row",
                getOptionalI32AttrOr(op, builder, "vpm_base_row", 0)),
      namedAttr(builder, "vpm_base_col",
                getOptionalI32AttrOr(op, builder, "vpm_base_col", 0)),
      namedAttr(builder, "vpitch",
                getOptionalI32AttrOr(op, builder, "vpitch", rowLen)),
      namedAttr(builder, "layout",
                dstLayout ? dstLayout
                          : mlir::vc4tile::LayoutAttr::get(
                                builder.getContext(),
                                mlir::vc4tile::Layout::vpm_row)),
      namedAttr(builder, "serialize", builder.getStringAttr("mutex"))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  createVC4TileCoreOp(builder, op->getLoc(), kVC4TileVDRLoadTileOpName,
                      {*adjustedBase, shared}, attrs);
  return success();
}

static FailureOr<Value> createSharedLoadForPlan(Operation *op, OpBuilder &builder,
                                                Value shared, Value row, Value mask,
                                                mlir::vc4tile::LayoutAttr layout,
                                                Type resultType) {
  if (!mlir::vc4tile::isVC4TileSharedTileType(shared.getType()) ||
      !isI32Scalar(row) || !isVector16I1(mask.getType())) {
    return op->emitOpError(
        "shared_vpm copy/store planning requires !vc4tile.shared_tile, i32 row, and vector<16xi1> mask");
  }
  SmallVector<NamedAttribute, 8> attrs{
      namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
      namedAttr(builder, "memory_space",
                getVC4TileMemorySpaceAttr(
                    builder, mlir::vc4tile::MemorySpace::shared_vpm)),
      namedAttr(builder, "layout", getVPMTileLoadLayoutAttr(builder, layout))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  Operation *load = createVC4TileCoreOp(
      builder, op->getLoc(), kVC4TileSharedLoadOpName, {shared, row, mask},
      attrs, resultType);
  return load->getResult(0);
}

static LogicalResult createGlobalStoreForPlan(Operation *op, OpBuilder &builder,
                                              Value value, Value base,
                                              Value offset, Value mask) {
  FailureOr<Value> adjustedBase = createAdjustedGlobalBaseForTileCopy(builder, op, base, offset);
  if (failed(adjustedBase))
    return failure();
  Value lanes = createLaneOffsetsForTileCopy(builder, op, 1);
  SmallVector<NamedAttribute, 8> attrs{
      namedAttr(builder, "elem_bytes", builder.getI32IntegerAttr(4)),
      namedAttr(builder, "offset_unit",
                getVC4TileOffsetUnitAttr(builder,
                                         mlir::vc4tile::OffsetUnit::element)),
      namedAttr(builder, "memory_space",
                getVC4TileMemorySpaceAttr(builder,
                                          mlir::vc4tile::MemorySpace::global)),
      namedAttr(builder, "access",
                getVC4TileMemoryAccessAttr(
                    builder, mlir::vc4tile::MemoryAccess::affine_contiguous))};
  appendTileSemanticMetadataAttrs(op, builder, attrs);
  createVC4TileCoreOp(builder, op->getLoc(), kVC4TileMaskedStoreGlobalOpName,
                      {*adjustedBase, lanes, value, mask}, attrs);
  return success();
}

static LogicalResult planSharedRowsToGlobal(Operation *op, OpBuilder &builder,
                                            Value shared, Value base,
                                            Value offset, Value mask,
                                            Value rowBase,
                                            mlir::vc4tile::LayoutAttr layout,
                                            Type elementType) {
  if (!mlir::vc4tile::isVC4TileSharedTileType(shared.getType()) ||
      !isI32Scalar(base) || !isI32Scalar(offset) ||
      !isI32Scalar(rowBase) || !isVector16I1(mask.getType())) {
    return op->emitOpError(
        "shared_vpm->global copy requires !vc4tile.shared_tile plus i32 base/offset/row and vector<16xi1> mask");
  }
  FailureOr<SmallVector<int64_t, 2>> shape = getTileShape2D(op);
  if (failed(shape))
    return failure();
  int64_t rows = (*shape)[0];
  int64_t cols = (*shape)[1];
  if (rows < 1 || rows > 16 || cols != 16)
    return op->emitOpError(
        "shared_vpm->global copy supports static [1,16] through [16,16] 32-bit tiles in M5");

  Type resultType = getVector16DataType(builder, elementType);
  for (int64_t row = 0; row < rows; ++row) {
    Value localRow = row == 0 ? rowBase : addI32Constant(builder, op->getLoc(), rowBase, row);
    FailureOr<Value> loaded =
        createSharedLoadForPlan(op, builder, shared, localRow, mask, layout, resultType);
    if (failed(loaded))
      return failure();
    Value rowOffset = row == 0 ? offset : addI32Constant(builder, op->getLoc(), offset, row * cols);
    if (failed(createGlobalStoreForPlan(op, builder, *loaded, base, rowOffset, mask)))
      return failure();
  }
  return success();
}

static LogicalResult planSharedGlobalTileStore(Operation *op, OpBuilder &builder) {
  if (op->getNumOperands() != 4)
    return op->emitOpError("shared_vpm tile_store expects tile, base, offset, and mask operands");
  Value shared = op->getOperand(0);
  Value base = op->getOperand(1);
  Value offset = op->getOperand(2);
  Value mask = op->getOperand(3);
  int64_t sharedRow = getOptionalI32AttrOr(op, builder, "shared_row", 0).getInt();
  if (sharedRow < 0 || sharedRow > 63)
    return op->emitOpError("shared_row must be in range [0, 63]");
  Value rowBase = createI32ConstantForPlan(builder, op->getLoc(), sharedRow);
  auto elementTypeAttr = op->getAttrOfType<TypeAttr>("element_type");
  Type elementType = elementTypeAttr ? elementTypeAttr.getValue() : builder.getI32Type();
  auto layout = op->getAttrOfType<mlir::vc4tile::LayoutAttr>("source_layout");
  if (!layout)
    layout = op->getAttrOfType<mlir::vc4tile::LayoutAttr>("layout");
  return planSharedRowsToGlobal(op, builder, shared, base, offset, mask, rowBase,
                                layout, elementType);
}

static LogicalResult planSharedGlobalCopy(Operation *op, OpBuilder &builder) {
  if ((op->getNumOperands() != 4 && op->getNumOperands() != 5) || op->getNumResults() != 0) {
    return op->emitOpError(
        "shared_vpm->global copy_tile expects operands (shared_tile, base, offset, mask) or (shared_tile, row, base, offset, mask) and no results");
  }
  Value shared = op->getOperand(0);
  Value rowBase = op->getNumOperands() == 5 ? op->getOperand(1)
      : createI32ConstantForPlan(builder, op->getLoc(), 0);
  unsigned baseIndex = op->getNumOperands() == 5 ? 2 : 1;
  Value base = op->getOperand(baseIndex);
  Value offset = op->getOperand(baseIndex + 1);
  Value mask = op->getOperand(baseIndex + 2);
  auto elementTypeAttr = op->getAttrOfType<TypeAttr>("element_type");
  Type elementType = elementTypeAttr ? elementTypeAttr.getValue() : builder.getI32Type();
  auto srcLayout = op->getAttrOfType<mlir::vc4tile::LayoutAttr>("src_layout");
  return planSharedRowsToGlobal(op, builder, shared, base, offset, mask, rowBase,
                                srcLayout, elementType);
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
  if (src.getValue() == mlir::vc4tile::MemorySpace::global &&
      dst.getValue() == mlir::vc4tile::MemorySpace::shared_vpm)
    return planGlobalSharedCopy(op, builder);
  if (src.getValue() == mlir::vc4tile::MemorySpace::shared_vpm &&
      dst.getValue() == mlir::vc4tile::MemorySpace::global)
    return planSharedGlobalCopy(op, builder);

  return op->emitOpError()
         << "copy planner v1 cannot plan requested copy path; supported paths are register->shared_vpm, shared_vpm->register, global->shared_vpm VDR, shared_vpm->global, plus tile_load/tile_store for global/register";
}

static LogicalResult foldViewOp(Operation *op) {
  OpBuilder builder(op);
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return op->emitOpError("view folding expects one source and one result");
  if (op->getOperand(0).getType() != op->getResult(0).getType())
    return op->emitOpError("view folding requires source and result types to match");
  if (hasName(op, kVC4TileTransposeViewOpName)) {
    Attribute transposed = mlir::vc4tile::LayoutAttr::get(
        builder.getContext(), mlir::vc4tile::Layout::transposed_view);
    for (Operation *user : llvm::make_early_inc_range(op->getResult(0).getUsers())) {
      if (hasName(user, kVC4TileTileStoreOpName) &&
          user->getNumOperands() > 0 && user->getOperand(0) == op->getResult(0) &&
          !user->getAttr("source_layout"))
        user->setAttr("source_layout", transposed);
      if (hasName(user, kVC4TileCopyTileOpName) &&
          user->getNumOperands() > 0 && user->getOperand(0) == op->getResult(0) &&
          !user->getAttr("src_layout"))
        user->setAttr("src_layout", transposed);
    }
  }
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
  if (hasName(op, kVC4TileTileStoreOpName)) {
    if (op->getNumOperands() == 4 &&
        mlir::vc4tile::isVC4TileSharedTileType(op->getOperand(0).getType()))
      return planSharedGlobalTileStore(op, builder);
    return planRegisterGlobalTileStore(op, builder);
  }
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
