//===- VC4ValueToVC4Kernel.cpp - VC4 value to VC4Kernel lowering ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Phase 5 lowered the first executable standard value-layer slice into the
// locked VC4Kernel target-kernel planning dialect.  Phase 8 extends that slice
// with standard cf.br/cf.cond_br multi-block control flow while preserving a
// crisp boundary between:
//
//   * the broad value-surface contract, which is allowed to contain staged
//     fixed vectors, rank-2 shapes, subword storage, reductions, contracts,
//     scf/cf, and future TTIR-importable patterns; and
//   * the executable subset, which is vector<16> i32/f32 elementwise code over
//     rank-1 contiguous #vc4value.global memrefs plus Phase 8 V1 cf control
//     flow.
//
// Staged value-surface features are not rejected by the value-surface verifier;
// they are rejected here with precise diagnostics until the corresponding
// executable phases are implemented and hardware-proven.
//
//===----------------------------------------------------------------------===//

#include "vc4/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.h"

#include "vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelOps.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.h"
#include "vc4/Dialect/VC4Value/IR/VC4ValueAttrs.h"
#include "vc4/Dialect/VC4Value/IR/VC4ValueDialect.h"
#include "vc4/Dialect/VC4Value/IR/VC4ValueOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kKernelAttr("vc4value.kernel");
constexpr llvm::StringLiteral kGridRankAttr("vc4value.grid_rank");
constexpr llvm::StringLiteral kArgNameAttr("vc4value.arg_name");
constexpr llvm::StringLiteral kDirectionAttr("vc4value.direction");
constexpr llvm::StringLiteral kScalarRoleAttr("vc4value.scalar_role");
constexpr llvm::StringLiteral kFPDomainAttr("vc4value.fp_domain");
constexpr llvm::StringLiteral kI32MulPolicyAttr("vc4value.i32_mul_policy");

constexpr llvm::StringLiteral kVC4KernelOpName("vc4kernel.kernel");
constexpr llvm::StringLiteral kReturnOpName("vc4kernel.return");
constexpr llvm::StringLiteral kProgramIdOpName("vc4kernel.program_id");
constexpr llvm::StringLiteral kNumProgramsOpName("vc4kernel.num_programs");
constexpr llvm::StringLiteral kLaneRangeOpName("vc4kernel.lane_range");
constexpr llvm::StringLiteral kPredFullOpName("vc4kernel.pred.full");
constexpr llvm::StringLiteral kPredEmptyOpName("vc4kernel.pred.empty");
constexpr llvm::StringLiteral kPredTailOpName("vc4kernel.pred.tail");
constexpr llvm::StringLiteral kSplatOpName("vc4kernel.splat");
constexpr llvm::StringLiteral kFragmentConstOpName("vc4kernel.fragment_const");
constexpr llvm::StringLiteral kFragmentALUAddOpName("vc4kernel.fragment_alu.add");
constexpr llvm::StringLiteral kFragmentALUMulOpName("vc4kernel.fragment_alu.mul");
constexpr llvm::StringLiteral kFragmentCmpOpName("vc4kernel.fragment_cmp");
constexpr llvm::StringLiteral kFragmentSelectOpName("vc4kernel.fragment_select");
constexpr llvm::StringLiteral kTMULoadFragmentOpName("vc4kernel.tmu_load_fragment");
constexpr llvm::StringLiteral kVDWStoreFragmentOpName("vc4kernel.vdw_store_fragment");

static bool hasName(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static bool hasStringAttr(Operation *op, StringRef name, StringRef expected) {
  auto attr = llvm::dyn_cast_or_null<StringAttr>(op->getAttr(name));
  return attr && attr.getValue() == expected;
}

static std::optional<int64_t> getI32Attr(Operation *op, StringRef name) {
  auto attr = llvm::dyn_cast_or_null<IntegerAttr>(op->getAttr(name));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool isVectorOf(Type type, unsigned width,
                       llvm::function_ref<bool(Type)> elementPred) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  return vectorType && !vectorType.isScalable() && vectorType.getRank() == 1 &&
         vectorType.getDimSize(0) == static_cast<int64_t>(width) &&
         elementPred(vectorType.getElementType());
}

static bool isVector16I1(Type type) {
  return isVectorOf(type, 16, [](Type elem) { return elem.isInteger(1); });
}

static bool isVector16Index(Type type) {
  return isVectorOf(type, 16, [](Type elem) { return elem.isIndex(); });
}

static bool isVector16I32(Type type) {
  return isVectorOf(type, 16,
                    [](Type elem) { return elem.isSignlessInteger(32); });
}

static bool isVector16F32(Type type) {
  return isVectorOf(type, 16, [](Type elem) { return elem.isF32(); });
}

static bool isVector16I32OrIndex(Type type) {
  return isVector16I32(type) || isVector16Index(type);
}

static bool isVector16Data(Type type) {
  return isVector16I32(type) || isVector16F32(type);
}

static bool isScalarI32OrIndex(Type type) {
  return type.isSignlessInteger(32) || type.isIndex();
}

static bool isScalarF32(Type type) { return type.isF32(); }

static bool isRank1IdentityGlobalMemref(Type type) {
  auto memrefType = llvm::dyn_cast<MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 1)
    return false;
  Attribute space = memrefType.getMemorySpace();
  if (!llvm::isa_and_nonnull<mlir::vc4value::GlobalMemorySpaceAttr>(space))
    return false;
  if (!memrefType.getElementType().isSignlessInteger(32) &&
      !memrefType.getElementType().isF32())
    return false;
  return memrefType.getLayout().isIdentity();
}

static bool isRank1IdentityTransferMap(Operation *op) {
  auto read = llvm::dyn_cast<vector::TransferReadOp>(op);
  if (!read)
    return false;
  AffineMap map = read.getPermutationMap();
  return map.getNumDims() == 1 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 1 && map.isMinorIdentity();
}

static bool isRank1IdentityTransferWriteMap(Operation *op) {
  auto write = llvm::dyn_cast<vector::TransferWriteOp>(op);
  if (!write)
    return false;
  AffineMap map = write.getPermutationMap();
  return map.getNumDims() == 1 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 1 && map.isMinorIdentity();
}

static StringRef getElementTypeName(Type type) {
  if (type.isF32())
    return "f32";
  if (type.isSignlessInteger(32))
    return "i32";
  if (type.isSignlessInteger(16))
    return "i16";
  if (type.isSignlessInteger(8))
    return "i8";
  if (type.isF16())
    return "f16";
  return "unknown";
}

static Type lowerValueType(Type type, MLIRContext *ctx) {
  Builder builder(ctx);
  if (type.isIndex())
    return builder.getI32Type();
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    if (isVector16Index(type))
      return VectorType::get({16}, builder.getI32Type());
    if (isVector16I32(type) || isVector16F32(type))
      return type;
  }
  return type;
}

static Type getVector16I32(MLIRContext *ctx) {
  return VectorType::get({16}, IntegerType::get(ctx, 32));
}

static Type getPred16(MLIRContext *ctx) {
  return mlir::vc4kernel::PredType::get(ctx, 16);
}

static Operation *createOp(OpBuilder &builder, Location loc, StringRef name,
                           ValueRange operands, ArrayRef<NamedAttribute> attrs,
                           TypeRange resultTypes = {}) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  for (NamedAttribute attr : attrs)
    state.addAttribute(attr.getName(), attr.getValue());
  return builder.create(state);
}

static Value createOpWithResult(OpBuilder &builder, Location loc,
                                StringRef name, ValueRange operands,
                                ArrayRef<NamedAttribute> attrs,
                                Type resultType) {
  Operation *op = createOp(builder, loc, name, operands, attrs, resultType);
  return op->getResult(0);
}

static Value createI32Constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(value));
}

static Value createF32Constant(OpBuilder &builder, Location loc, APFloat value) {
  return builder.create<arith::ConstantOp>(
      loc, builder.getF32Type(), FloatAttr::get(builder.getF32Type(), value));
}

static Value createFragmentConst(OpBuilder &builder, Location loc, Type type,
                                 Attribute value) {
  return createOpWithResult(builder, loc, kFragmentConstOpName, {},
                            {builder.getNamedAttr("value", value)}, type);
}

static Value createPredFull(OpBuilder &builder, Location loc) {
  return createOpWithResult(builder, loc, kPredFullOpName, {}, {},
                            getPred16(builder.getContext()));
}

static Value createPredEmpty(OpBuilder &builder, Location loc) {
  return createOpWithResult(builder, loc, kPredEmptyOpName, {}, {},
                            getPred16(builder.getContext()));
}

static Value createPredTail(OpBuilder &builder, Location loc, Value base,
                            Value limit) {
  return createOpWithResult(builder, loc, kPredTailOpName, {base, limit}, {},
                            getPred16(builder.getContext()));
}

static Value createSplat(OpBuilder &builder, Location loc, Value scalar,
                         Type resultType) {
  return createOpWithResult(builder, loc, kSplatOpName, scalar, {}, resultType);
}

static Value createAddPipe(OpBuilder &builder, Location loc, ValueRange inputs,
                           mlir::vc4kernel::AddALUOpcode opcode,
                           Type resultType) {
  return createOpWithResult(
      builder, loc, kFragmentALUAddOpName, inputs,
      {builder.getNamedAttr(
          "opcode", mlir::vc4kernel::AddALUOpcodeAttr::get(
                        builder.getContext(), opcode))},
      resultType);
}

static Value createMulPipe(OpBuilder &builder, Location loc, ValueRange inputs,
                           mlir::vc4kernel::MulALUOpcode opcode,
                           Type resultType) {
  return createOpWithResult(
      builder, loc, kFragmentALUMulOpName, inputs,
      {builder.getNamedAttr(
          "opcode", mlir::vc4kernel::MulALUOpcodeAttr::get(
                        builder.getContext(), opcode))},
      resultType);
}

static Value createFragmentSelect(OpBuilder &builder, Location loc, Value pred,
                                  Value trueValue, Value falseValue,
                                  Type resultType) {
  return createOpWithResult(builder, loc, kFragmentSelectOpName,
                            {pred, trueValue, falseValue}, {}, resultType);
}

static DenseElementsAttr getI32VectorDenseAttr(OpBuilder &builder,
                                               ArrayRef<int64_t> values) {
  SmallVector<APInt, 16> elements;
  elements.reserve(values.size());
  for (int64_t value : values)
    elements.push_back(APInt(32, static_cast<uint64_t>(value), true));
  return DenseIntElementsAttr::get(
      llvm::cast<ShapedType>(getVector16I32(builder.getContext())), elements);
}

static Value createLaneByteOffsets(OpBuilder &builder, Location loc,
                                   int64_t elemBytes) {
  SmallVector<int64_t, 16> bytes;
  for (int64_t lane = 0; lane < 16; ++lane)
    bytes.push_back(lane * elemBytes);
  return createFragmentConst(builder, loc, getVector16I32(builder.getContext()),
                             getI32VectorDenseAttr(builder, bytes));
}

static Value createPoisonByteOffsets(OpBuilder &builder, Location loc,
                                     int64_t elemBytes) {
  // Keep inactive TMU lanes far away from real test buffers.  The TMU lowering
  // still receives an explicit safe_offset; the poison vector makes accidental
  // pre-policy uses obvious in static IR and hardware triage artifacts.
  constexpr int64_t kPoisonBase = 0x10000000;
  SmallVector<int64_t, 16> bytes;
  for (int64_t lane = 0; lane < 16; ++lane)
    bytes.push_back(kPoisonBase + lane * elemBytes);
  return createFragmentConst(builder, loc, getVector16I32(builder.getContext()),
                             getI32VectorDenseAttr(builder, bytes));
}

static std::optional<int64_t> getIntegerConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  if (auto integer = llvm::dyn_cast<IntegerAttr>(constant.getValue()))
    return integer.getInt();
  return std::nullopt;
}

static bool isZeroConstant(Value value) {
  if (auto integer = getIntegerConstant(value))
    return *integer == 0;
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto fp = llvm::dyn_cast<FloatAttr>(constant.getValue()))
    return fp.getValue().isZero();
  return false;
}

static LogicalResult emitStagedDiagnostic(Operation *op, Twine detail) {
  return op->emitOpError()
         << detail << " is not Phase 5 lowerable; staged value-surface feature. "
         << "READY_FOR_TRITON remains NO";
}

static LogicalResult emitPhase5Diagnostic(Operation *op, Twine detail) {
  return op->emitOpError() << detail << "; not Phase 5 lowerable. "
                           << "READY_FOR_TRITON remains NO";
}

static LogicalResult emitRawSCFDiagnostic(Operation *op) {
  return op->emitOpError()
         << "raw scf operation cannot lower directly to VC4Kernel; run "
         << "explicit upstream --convert-scf-to-cf before "
         << "--convert-vc4-value-to-vc4kernel. READY_FOR_TRITON remains NO";
}

struct LoweringState {
  explicit LoweringState(Operation *sourceKernel) : sourceKernel(sourceKernel) {}

  Operation *sourceKernel = nullptr;
  DenseMap<Block *, Block *> blocks;
  DenseMap<Value, Value> values;
  DenseMap<Value, Value> predicates;
};

class KernelLowerer {
public:
  KernelLowerer(ModuleOp module, func::FuncOp func)
      : module(module), func(func), ctx(module.getContext()), builder(ctx),
        state(func.getOperation()) {}

  LogicalResult lower() {
    if (failed(verifyKernelShape()))
      return failure();

    Operation *kernel = createKernelSkeleton();
    if (!kernel)
      return failure();

    if (failed(createTargetBlocks(kernel)))
      return failure();
    if (failed(mapFunctionArguments()))
      return failure();
    if (failed(mapBlockArguments()))
      return failure();

    for (Block &sourceBlock : func.getBody()) {
      Block *targetBlock = lookupTargetBlock(func.getOperation(), &sourceBlock);
      if (!targetBlock)
        return failure();
      if (failed(lowerBlockBody(sourceBlock, *targetBlock)))
        return failure();
    }

    func.erase();
    return success();
  }

private:
  LogicalResult verifyKernelShape() {
    if (!func->hasAttr(kKernelAttr))
      return func.emitOpError("expected vc4value.kernel for Phase 5 lowering");
    auto gridRank = llvm::dyn_cast_or_null<IntegerAttr>(func->getAttr(kGridRankAttr));
    if (!gridRank || gridRank.getInt() != 1)
      return emitStagedDiagnostic(func.getOperation(),
                                  "grid_rank other than 1");
    if (!func.getFunctionType().getResults().empty())
      return func.emitOpError("Phase 8 value kernels must have no function results");
    return success();
  }

  Operation *createKernelSkeleton() {
    SmallVector<Type, 8> loweredArgTypes;
    SmallVector<Location, 8> argLocs;
    SmallVector<Attribute, 8> argAttrs;

    Block &entry = func.getBody().front();
    for (BlockArgument arg : entry.getArguments()) {
      FailureOr<Type> loweredType = lowerPublicArgType(arg);
      if (failed(loweredType))
        return nullptr;
      loweredArgTypes.push_back(*loweredType);
      argLocs.push_back(arg.getLoc());
      FailureOr<DictionaryAttr> abi = buildArgABI(arg);
      if (failed(abi))
        return nullptr;
      argAttrs.push_back(*abi);
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(func);

    OperationState opState(func.getLoc(), kVC4KernelOpName);
    StringAttr sym = func.getSymNameAttr();
    opState.addAttribute(SymbolTable::getSymbolAttrName(), sym);
    opState.addAttribute("function_type",
                         TypeAttr::get(FunctionType::get(ctx, loweredArgTypes, {})));
    opState.addAttribute("public_name", builder.getStringAttr(sym.getValue()));
    opState.addAttribute(
        "schedule_mode",
        mlir::vc4kernel::ScheduleModeAttr::get(
            ctx, mlir::vc4kernel::ScheduleMode::independent_vector));
    opState.addAttribute("warps_per_block", builder.getI32IntegerAttr(1));
    opState.addAttribute("arg_attrs", builder.getArrayAttr(argAttrs));
    opState.addRegion();

    Operation *kernel = builder.create(opState);
    Region &body = kernel->getRegion(0);
    auto *block = new Block();
    body.push_back(block);
    block->addArguments(loweredArgTypes, argLocs);
    return kernel;
  }

  LogicalResult createTargetBlocks(Operation *kernel) {
    Region &body = kernel->getRegion(0);
    Block &sourceEntry = func.getBody().front();
    Block &targetEntry = body.front();
    state.blocks[&sourceEntry] = &targetEntry;

    for (Block &sourceBlock : llvm::drop_begin(func.getBody())) {
      auto *targetBlock = new Block();
      SmallVector<Type, 4> loweredArgTypes;
      SmallVector<Location, 4> argLocs;
      for (BlockArgument arg : sourceBlock.getArguments()) {
        FailureOr<Type> loweredType = lowerBlockArgType(arg);
        if (failed(loweredType))
          return failure();
        loweredArgTypes.push_back(*loweredType);
        argLocs.push_back(arg.getLoc());
      }
      targetBlock->addArguments(loweredArgTypes, argLocs);
      body.push_back(targetBlock);
      state.blocks[&sourceBlock] = targetBlock;
    }
    return success();
  }

  FailureOr<Type> lowerPublicArgType(BlockArgument arg) {
    Type type = arg.getType();
    if (auto memrefType = llvm::dyn_cast<MemRefType>(type)) {
      if (!isRank1IdentityGlobalMemref(type)) {
        InFlightDiagnostic diag =
            arg.getOwner()->getParentOp()->emitOpError();
        diag << "expected rank-1 contiguous i32/f32 #vc4value.global memref "
             << "for Phase 5 argument " << arg.getArgNumber()
             << "; not Phase 5 lowerable; staged value-surface feature. "
             << "READY_FOR_TRITON remains NO";
        return failure();
      }
      return builder.getI32Type();
    }
    if (type.isIndex() || type.isSignlessInteger(32))
      return builder.getI32Type();
    if (type.isF32())
      return type;
    InFlightDiagnostic diag = arg.getOwner()->getParentOp()->emitOpError();
    diag << "public kernel argument type " << type
         << " is not Phase 5 lowerable; staged value-surface feature. "
         << "READY_FOR_TRITON remains NO";
    return failure();
  }

  FailureOr<Type> lowerBlockArgType(BlockArgument arg) {
    Type type = arg.getType();
    if (type.isIndex() || type.isSignlessInteger(32))
      return builder.getI32Type();
    if (type.isInteger(1))
      return builder.getI1Type();
    if (type.isF32())
      return builder.getF32Type();
    if (isVector16I1(type))
      return getPred16(ctx);
    if (isVector16Index(type))
      return getVector16I32(ctx);
    if (isVector16I32(type) || isVector16F32(type))
      return type;

    InFlightDiagnostic diag = arg.getOwner()->getParentOp()->emitOpError();
    diag << "block argument type " << type
         << " is not Phase 8 value control-flow lowerable. "
         << "READY_FOR_TRITON remains NO";
    return failure();
  }

  FailureOr<DictionaryAttr> buildArgABI(BlockArgument arg) {
    SmallVector<NamedAttribute, 8> attrs;
    unsigned index = arg.getArgNumber();
    StringAttr name = func.getArgAttrOfType<StringAttr>(index, kArgNameAttr);
    if (!name)
      name = builder.getStringAttr((Twine("arg") + Twine(index)).str());
    attrs.push_back(builder.getNamedAttr("name", name));

    Type type = arg.getType();
    if (auto memrefType = llvm::dyn_cast<MemRefType>(type)) {
      StringAttr direction =
          func.getArgAttrOfType<StringAttr>(index, kDirectionAttr);
      if (!direction) {
        (void)func.emitOpError("memref arguments require vc4value.direction");
        return failure();
      }
      attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr("buffer")));
      attrs.push_back(builder.getNamedAttr("direction", direction));
      attrs.push_back(builder.getNamedAttr(
          "elem_type", builder.getStringAttr(getElementTypeName(memrefType.getElementType()))));
      return builder.getDictionaryAttr(attrs);
    }

    attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr("scalar")));
    attrs.push_back(builder.getNamedAttr("direction", builder.getStringAttr("by_value")));
    StringRef typeName = type.isF32() ? "f32" : "i32";
    attrs.push_back(builder.getNamedAttr("type", builder.getStringAttr(typeName)));
    if (StringAttr role = func.getArgAttrOfType<StringAttr>(index, kScalarRoleAttr))
      attrs.push_back(builder.getNamedAttr("role", role));
    return builder.getDictionaryAttr(attrs);
  }

  LogicalResult mapFunctionArguments() {
    Block &sourceEntry = func.getBody().front();
    Block *targetEntry = lookupTargetBlock(func.getOperation(), &sourceEntry);
    if (!targetEntry)
      return failure();
    for (auto [sourceArg, targetArg] : llvm::zip(sourceEntry.getArguments(),
                                                 targetEntry->getArguments())) {
      state.values[sourceArg] = targetArg;
    }
    return success();
  }

  LogicalResult mapBlockArguments() {
    for (Block &sourceBlock : llvm::drop_begin(func.getBody())) {
      Block *targetBlock = lookupTargetBlock(func.getOperation(), &sourceBlock);
      if (!targetBlock)
        return failure();
      for (auto [sourceArg, targetArg] : llvm::zip(sourceBlock.getArguments(),
                                                   targetBlock->getArguments()))
        mapLoweredValue(sourceArg, targetArg);
    }
    return success();
  }

  void mapLoweredValue(Value source, Value target) {
    if (isVector16I1(source.getType()))
      state.predicates[source] = target;
    else
      state.values[source] = target;
  }

  Block *lookupTargetBlock(Operation *op, Block *block) {
    auto it = state.blocks.find(block);
    if (it != state.blocks.end())
      return it->second;
    op->emitOpError("successor block has no Phase 8 lowering plan");
    return nullptr;
  }

  Value lookupValue(Operation *op, Value value) {
    auto it = state.values.find(value);
    if (it != state.values.end())
      return it->second;
    op->emitOpError("normal SSA operand has no Phase 5 lowering plan");
    return {};
  }

  Value lookupPredicate(Operation *op, Value value) {
    auto it = state.predicates.find(value);
    if (it != state.predicates.end())
      return it->second;
    op->emitOpError("predicate/mask operand has no Phase 5 lowering plan");
    return {};
  }

  Value lookupBranchOperand(Operation *op, Value value) {
    if (isVector16I1(value.getType()))
      return lookupPredicate(op, value);
    return lookupValue(op, value);
  }

  LogicalResult lowerBlockBody(Block &sourceBlock, Block &targetBlock) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&targetBlock);

    for (Operation &op : sourceBlock.getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        return lowerTerminator(&op);
      if (failed(lowerOperation(&op)))
        return failure();
    }

    return sourceBlock.getParentOp()->emitOpError()
           << "value block has no terminator; not Phase 8 lowerable. "
           << "READY_FOR_TRITON remains NO";
  }

  LogicalResult lowerTerminator(Operation *op) {
    if (auto ret = llvm::dyn_cast<func::ReturnOp>(op))
      return lowerReturn(ret);
    if (auto branch = llvm::dyn_cast<cf::BranchOp>(op))
      return lowerBranch(branch);
    if (auto condBranch = llvm::dyn_cast<cf::CondBranchOp>(op))
      return lowerCondBranch(condBranch);
    if (op->getName().getDialectNamespace() == "scf")
      return emitRawSCFDiagnostic(op);
    return emitPhase5Diagnostic(op, Twine("terminator '") +
                                        op->getName().getStringRef() + "'");
  }

  LogicalResult lowerReturn(func::ReturnOp ret) {
    if (ret.getNumOperands() != 0)
      return ret.emitOpError("Phase 8 value kernels must return void");
    createOp(builder, ret.getLoc(), kReturnOpName, {}, {});
    return success();
  }

  LogicalResult lowerSuccessorOperands(Operation *op, OperandRange operands,
                                       Block *sourceSuccessor,
                                       SmallVectorImpl<Value> &lowered) {
    Block *targetSuccessor = lookupTargetBlock(op, sourceSuccessor);
    if (!targetSuccessor)
      return failure();
    if (operands.size() != targetSuccessor->getNumArguments()) {
      return op->emitOpError()
             << "successor operand count " << operands.size()
             << " does not match lowered target block argument count "
             << targetSuccessor->getNumArguments()
             << "; not Phase 8 lowerable. READY_FOR_TRITON remains NO";
    }

    for (auto [operand, targetArg] :
         llvm::zip(operands, targetSuccessor->getArguments())) {
      Value mapped = lookupBranchOperand(op, operand);
      if (!mapped)
        return failure();
      if (mapped.getType() != targetArg.getType()) {
        return op->emitOpError()
               << "successor operand type " << mapped.getType()
               << " does not match lowered target block argument type "
               << targetArg.getType()
               << "; not Phase 8 lowerable. READY_FOR_TRITON remains NO";
      }
      lowered.push_back(mapped);
    }
    return success();
  }

  LogicalResult lowerBranch(cf::BranchOp branch) {
    Block *target = lookupTargetBlock(branch.getOperation(), branch.getDest());
    if (!target)
      return failure();
    SmallVector<Value, 4> operands;
    if (failed(lowerSuccessorOperands(branch.getOperation(),
                                      branch.getDestOperands(),
                                      branch.getDest(), operands)))
      return failure();
    builder.create<cf::BranchOp>(branch.getLoc(), target, operands);
    return success();
  }

  LogicalResult lowerCondBranch(cf::CondBranchOp branch) {
    if (!branch.getCondition().getType().isInteger(1))
      return branch.emitOpError()
             << "condition must be scalar i1 for Phase 8 value control flow; "
             << "READY_FOR_TRITON remains NO";
    Value condition = lookupValue(branch.getOperation(), branch.getCondition());
    if (!condition)
      return failure();
    if (!condition.getType().isInteger(1))
      return branch.emitOpError()
             << "condition must lower to scalar i1 for Phase 8 value control flow; "
             << "READY_FOR_TRITON remains NO";

    Block *trueTarget =
        lookupTargetBlock(branch.getOperation(), branch.getTrueDest());
    Block *falseTarget =
        lookupTargetBlock(branch.getOperation(), branch.getFalseDest());
    if (!trueTarget || !falseTarget)
      return failure();

    SmallVector<Value, 4> trueOperands;
    SmallVector<Value, 4> falseOperands;
    if (failed(lowerSuccessorOperands(branch.getOperation(),
                                      branch.getTrueDestOperands(),
                                      branch.getTrueDest(), trueOperands)) ||
        failed(lowerSuccessorOperands(branch.getOperation(),
                                      branch.getFalseDestOperands(),
                                      branch.getFalseDest(), falseOperands)))
      return failure();

    builder.create<cf::CondBranchOp>(branch.getLoc(), condition, trueTarget,
                                     trueOperands, falseTarget, falseOperands);
    return success();
  }

  LogicalResult lowerOperation(Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "vc4value.program_id")
      return lowerProgramId(op);
    if (name == "vc4value.num_programs")
      return lowerNumPrograms(op);
    if (name == "vector.step")
      return lowerVectorStep(op);
    if (name == "vector.splat" || name == "vector.broadcast")
      return lowerVectorSplatLike(op);
    if (name == "vector.create_mask")
      return lowerCreateMask(op);
    if (name == "vector.transfer_read")
      return lowerTransferRead(op);
    if (name == "vector.transfer_write")
      return lowerTransferWrite(op);
    if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op))
      return lowerConstant(constant);
    if (llvm::isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
                  arith::AddFOp, arith::SubFOp, arith::MulFOp>(op))
      return lowerBinaryArith(op);
    if (auto cmpi = llvm::dyn_cast<arith::CmpIOp>(op))
      return lowerCmpI(cmpi);
    if (auto cmpf = llvm::dyn_cast<arith::CmpFOp>(op))
      return lowerCmpF(cmpf);
    if (auto select = llvm::dyn_cast<arith::SelectOp>(op))
      return lowerSelect(select);
    if (name == "arith.index_cast")
      return lowerIndexCast(op);

    if (op->getName().getDialectNamespace() == "scf")
      return emitRawSCFDiagnostic(op);
    if (op->getName().getDialectNamespace() == "cf")
      return emitStagedDiagnostic(op, "non-terminator control-flow operation");
    if (op->getName().getDialectNamespace() == "math")
      return emitStagedDiagnostic(op, "math dialect operation");
    if (op->getName().getDialectNamespace() == "vector")
      return emitStagedDiagnostic(op, Twine("vector operation '") + name + "'");
    if (op->getName().getDialectNamespace() == "memref")
      return emitStagedDiagnostic(op, "memref operation");
    return emitPhase5Diagnostic(op, Twine("operation '") + name + "'");
  }

  LogicalResult lowerProgramId(Operation *op) {
    int64_t axis = getI32Attr(op, "axis").value_or(0);
    if (axis != 0)
      return emitStagedDiagnostic(op, "program_id axis other than 0");
    Value result = createOpWithResult(
        builder, op->getLoc(), kProgramIdOpName, {},
        {builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis))},
        builder.getI32Type());
    state.values[op->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerNumPrograms(Operation *op) {
    int64_t axis = getI32Attr(op, "axis").value_or(0);
    if (axis != 0)
      return emitStagedDiagnostic(op, "num_programs axis other than 0");
    Value result = createOpWithResult(
        builder, op->getLoc(), kNumProgramsOpName, {},
        {builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis))},
        builder.getI32Type());
    state.values[op->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerVectorStep(Operation *op) {
    if (op->getNumResults() != 1 || !isVector16Index(op->getResult(0).getType()))
      return emitStagedDiagnostic(op, "vector.step shape other than vector<16xindex>");
    Value lanes = createOpWithResult(builder, op->getLoc(), kLaneRangeOpName,
                                     {}, {}, getVector16I32(ctx));
    state.values[op->getResult(0)] = lanes;
    return success();
  }

  LogicalResult lowerVectorSplatLike(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "vector splat/broadcast arity");
    Type resultType = op->getResult(0).getType();
    if (!isVector16I32(resultType) && !isVector16Index(resultType) &&
        !isVector16F32(resultType))
      return emitStagedDiagnostic(op, "non-vector<16> splat/broadcast");
    Value scalar = lookupValue(op, op->getOperand(0));
    if (!scalar)
      return failure();
    Type loweredResultType = lowerValueType(resultType, ctx);
    if (!scalar.getType().isSignlessInteger(32) && !scalar.getType().isF32())
      return emitPhase5Diagnostic(op, "splat scalar type");
    Value result = createSplat(builder, op->getLoc(), scalar, loweredResultType);
    state.values[op->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerCreateMask(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1 ||
        !isVector16I1(op->getResult(0).getType()))
      return emitStagedDiagnostic(op, "vector.create_mask shape other than vector<16xi1>");
    Value limit = lookupValue(op, op->getOperand(0));
    if (!limit)
      return failure();
    Value zero = createI32Constant(builder, op->getLoc(), 0);
    Value sixteen = createI32Constant(builder, op->getLoc(), 16);
    Value belowZero = builder.create<arith::CmpIOp>(
        op->getLoc(), arith::CmpIPredicate::slt, limit, zero);
    Value nonNegative = builder.create<arith::SelectOp>(
        op->getLoc(), belowZero, zero, limit);
    Value aboveSixteen = builder.create<arith::CmpIOp>(
        op->getLoc(), arith::CmpIPredicate::sgt, nonNegative, sixteen);
    Value clamped = builder.create<arith::SelectOp>(
        op->getLoc(), aboveSixteen, sixteen, nonNegative);
    state.predicates[op->getResult(0)] =
        createPredTail(builder, op->getLoc(), zero, clamped);
    return success();
  }

  LogicalResult lowerConstant(arith::ConstantOp op) {
    Attribute value = op.getValue();
    Type resultType = op.getType();
    Location loc = op.getLoc();

    if (resultType.isIndex()) {
      auto integer = llvm::dyn_cast<IntegerAttr>(value);
      if (!integer)
        return op.emitOpError("index constant must be integer");
      state.values[op.getResult()] =
          createI32Constant(builder, loc, integer.getInt());
      return success();
    }
    if (resultType.isSignlessInteger(32)) {
      state.values[op.getResult()] = builder.create<arith::ConstantOp>(
          loc, builder.getI32Type(), llvm::cast<TypedAttr>(value));
      return success();
    }
    if (resultType.isF32()) {
      state.values[op.getResult()] =
          builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                            llvm::cast<TypedAttr>(value));
      return success();
    }

    auto dense = llvm::dyn_cast<DenseElementsAttr>(value);
    if (!dense)
      return emitPhase5Diagnostic(op.getOperation(), "non-dense vector constant");

    if (isVector16I1(resultType)) {
      bool sawTrue = false;
      bool sawFalse = false;
      for (APInt bit : dense.getValues<APInt>()) {
        sawTrue |= bit.getBoolValue();
        sawFalse |= !bit.getBoolValue();
      }
      if (sawTrue && sawFalse)
        return emitStagedDiagnostic(op.getOperation(), "sparse boolean vector constant mask");
      state.predicates[op.getResult()] = sawTrue ? createPredFull(builder, loc)
                                                 : createPredEmpty(builder, loc);
      return success();
    }

    if (isVector16Index(resultType)) {
      SmallVector<APInt, 16> values;
      for (APInt element : dense.getValues<APInt>())
        values.push_back(element.sextOrTrunc(32));
      auto converted = DenseIntElementsAttr::get(
          llvm::cast<ShapedType>(getVector16I32(ctx)), values);
      state.values[op.getResult()] =
          createFragmentConst(builder, loc, getVector16I32(ctx), converted);
      return success();
    }

    if (isVector16I32(resultType) || isVector16F32(resultType)) {
      state.values[op.getResult()] =
          createFragmentConst(builder, loc, resultType, dense);
      return success();
    }

    return emitStagedDiagnostic(op.getOperation(),
                                "constant result type outside Phase 5 subset");
  }

  LogicalResult lowerBinaryArith(Operation *op) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "binary arith shape");

    Value lhs = lookupValue(op, op->getOperand(0));
    Value rhs = lookupValue(op, op->getOperand(1));
    if (!lhs || !rhs)
      return failure();

    Type sourceType = op->getResult(0).getType();
    Type resultType = lowerValueType(sourceType, ctx);
    StringRef name = op->getName().getStringRef();

    if (resultType.isSignlessInteger(32) || resultType.isF32()) {
      if (name == "arith.addi")
        state.values[op->getResult(0)] = builder.create<arith::AddIOp>(op->getLoc(), lhs, rhs);
      else if (name == "arith.subi")
        state.values[op->getResult(0)] = builder.create<arith::SubIOp>(op->getLoc(), lhs, rhs);
      else if (name == "arith.muli")
        state.values[op->getResult(0)] = builder.create<arith::MulIOp>(op->getLoc(), lhs, rhs);
      else if (name == "arith.shli")
        state.values[op->getResult(0)] = builder.create<arith::ShLIOp>(op->getLoc(), lhs, rhs);
      else if (name == "arith.addf")
        state.values[op->getResult(0)] = builder.create<arith::AddFOp>(op->getLoc(), lhs, rhs);
      else if (name == "arith.subf")
        state.values[op->getResult(0)] = builder.create<arith::SubFOp>(op->getLoc(), lhs, rhs);
      else if (name == "arith.mulf")
        state.values[op->getResult(0)] = builder.create<arith::MulFOp>(op->getLoc(), lhs, rhs);
      else
        return emitPhase5Diagnostic(op, Twine("unsupported scalar arith op '") + name + "'");
      return success();
    }

    if (isVector16I32(sourceType) || isVector16Index(sourceType)) {
      if (name == "arith.addi")
        state.values[op->getResult(0)] = createAddPipe(
            builder, op->getLoc(), {lhs, rhs}, mlir::vc4kernel::AddALUOpcode::add, resultType);
      else if (name == "arith.subi")
        state.values[op->getResult(0)] = createAddPipe(
            builder, op->getLoc(), {lhs, rhs}, mlir::vc4kernel::AddALUOpcode::sub, resultType);
      else if (name == "arith.shli")
        state.values[op->getResult(0)] = createAddPipe(
            builder, op->getLoc(), {lhs, rhs}, mlir::vc4kernel::AddALUOpcode::shl, resultType);
      else if (name == "arith.muli") {
        if (!hasStringAttr(func.getOperation(), kI32MulPolicyAttr, "mul24_safe"))
          return emitPhase5Diagnostic(
              op, "vector i32 muli requires vc4value.i32_mul_policy = \"mul24_safe\" in Phase 5");
        state.values[op->getResult(0)] = createMulPipe(
            builder, op->getLoc(), {lhs, rhs}, mlir::vc4kernel::MulALUOpcode::mul24, resultType);
      } else {
        return emitPhase5Diagnostic(op, Twine("unsupported vector integer arith op '") + name + "'");
      }
      return success();
    }

    if (isVector16F32(sourceType)) {
      if (name == "arith.addf")
        state.values[op->getResult(0)] = createAddPipe(
            builder, op->getLoc(), {lhs, rhs}, mlir::vc4kernel::AddALUOpcode::fadd, resultType);
      else if (name == "arith.subf")
        state.values[op->getResult(0)] = createAddPipe(
            builder, op->getLoc(), {lhs, rhs}, mlir::vc4kernel::AddALUOpcode::fsub, resultType);
      else if (name == "arith.mulf")
        state.values[op->getResult(0)] = createMulPipe(
            builder, op->getLoc(), {lhs, rhs}, mlir::vc4kernel::MulALUOpcode::fmul, resultType);
      else
        return emitPhase5Diagnostic(op, Twine("unsupported vector f32 arith op '") + name + "'");
      return success();
    }

    return emitStagedDiagnostic(op, "arith result type outside Phase 5 subset");
  }

  static std::optional<mlir::vc4kernel::CmpPredicate>
  mapCmpIPredicate(arith::CmpIPredicate predicate) {
    switch (predicate) {
    case arith::CmpIPredicate::eq:
      return mlir::vc4kernel::CmpPredicate::eq;
    case arith::CmpIPredicate::ne:
      return mlir::vc4kernel::CmpPredicate::ne;
    case arith::CmpIPredicate::ult:
      return mlir::vc4kernel::CmpPredicate::ult;
    case arith::CmpIPredicate::ule:
      return mlir::vc4kernel::CmpPredicate::ule;
    case arith::CmpIPredicate::ugt:
      return mlir::vc4kernel::CmpPredicate::ugt;
    case arith::CmpIPredicate::uge:
      return mlir::vc4kernel::CmpPredicate::uge;
    case arith::CmpIPredicate::slt:
      return mlir::vc4kernel::CmpPredicate::slt;
    case arith::CmpIPredicate::sle:
      return mlir::vc4kernel::CmpPredicate::sle;
    case arith::CmpIPredicate::sgt:
      return mlir::vc4kernel::CmpPredicate::sgt;
    case arith::CmpIPredicate::sge:
      return mlir::vc4kernel::CmpPredicate::sge;
    }
    return std::nullopt;
  }

  LogicalResult lowerCmpI(arith::CmpIOp op) {
    if (!isVector16I32(op.getLhs().getType()) && !isVector16Index(op.getLhs().getType())) {
      if (isScalarI32OrIndex(op.getLhs().getType())) {
        Value lhs = lookupValue(op, op.getLhs());
        Value rhs = lookupValue(op, op.getRhs());
        if (!lhs || !rhs)
          return failure();
        state.values[op.getResult()] = builder.create<arith::CmpIOp>(
            op.getLoc(), op.getPredicate(), lhs, rhs);
        return success();
      }
      return emitStagedDiagnostic(op.getOperation(), "cmpi type outside Phase 5 subset");
    }
    std::optional<mlir::vc4kernel::CmpPredicate> predicate =
        mapCmpIPredicate(op.getPredicate());
    if (!predicate)
      return emitPhase5Diagnostic(op.getOperation(), "unsupported cmpi predicate");
    Value lhs = lookupValue(op, op.getLhs());
    Value rhs = lookupValue(op, op.getRhs());
    if (!lhs || !rhs)
      return failure();
    Value pred = createOpWithResult(
        builder, op.getLoc(), kFragmentCmpOpName, {lhs, rhs},
        {builder.getNamedAttr("predicate", mlir::vc4kernel::CmpPredicateAttr::get(ctx, *predicate))},
        getPred16(ctx));
    state.predicates[op.getResult()] = pred;
    return success();
  }

  static std::optional<mlir::vc4kernel::CmpPredicate>
  mapCmpFPredicate(arith::CmpFPredicate predicate) {
    switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      return mlir::vc4kernel::CmpPredicate::oeq;
    case arith::CmpFPredicate::ONE:
      return mlir::vc4kernel::CmpPredicate::one;
    case arith::CmpFPredicate::OLT:
      return mlir::vc4kernel::CmpPredicate::olt;
    case arith::CmpFPredicate::OLE:
      return mlir::vc4kernel::CmpPredicate::ole;
    case arith::CmpFPredicate::OGT:
      return mlir::vc4kernel::CmpPredicate::ogt;
    case arith::CmpFPredicate::OGE:
      return mlir::vc4kernel::CmpPredicate::oge;
    default:
      return std::nullopt;
    }
  }

  LogicalResult lowerCmpF(arith::CmpFOp op) {
    if (!isVector16F32(op.getLhs().getType()))
      return emitStagedDiagnostic(op.getOperation(), "cmpf type outside Phase 5 vector<16xf32> subset");
    if (!hasStringAttr(func.getOperation(), kFPDomainAttr, "finite"))
      return op.emitOpError()
             << "f32 comparisons require vc4value.fp_domain = \"finite\"; "
             << "not Phase 5 lowerable without explicit finite policy. "
             << "READY_FOR_TRITON remains NO";
    std::optional<mlir::vc4kernel::CmpPredicate> predicate =
        mapCmpFPredicate(op.getPredicate());
    if (!predicate)
      return emitPhase5Diagnostic(op.getOperation(), "unordered or NaN-sensitive f32 comparison");
    Value lhs = lookupValue(op, op.getLhs());
    Value rhs = lookupValue(op, op.getRhs());
    if (!lhs || !rhs)
      return failure();
    Value pred = createOpWithResult(
        builder, op.getLoc(), kFragmentCmpOpName, {lhs, rhs},
        {builder.getNamedAttr("predicate", mlir::vc4kernel::CmpPredicateAttr::get(ctx, *predicate)),
         builder.getNamedAttr("fp_policy", mlir::vc4kernel::FPCmpPolicyAttr::get(
                                               ctx, mlir::vc4kernel::FPCmpPolicy::finite_only))},
        getPred16(ctx));
    state.predicates[op.getResult()] = pred;
    return success();
  }

  LogicalResult lowerSelect(arith::SelectOp op) {
    Type resultType = op.getResult().getType();
    if (!isVector16Data(resultType)) {
      Value cond = lookupValue(op, op.getCondition());
      Value trueValue = lookupValue(op, op.getTrueValue());
      Value falseValue = lookupValue(op, op.getFalseValue());
      if (!cond || !trueValue || !falseValue)
        return failure();
      state.values[op.getResult()] = builder.create<arith::SelectOp>(
          op.getLoc(), cond, trueValue, falseValue);
      return success();
    }
    Value pred = lookupPredicate(op, op.getCondition());
    Value trueValue = lookupValue(op, op.getTrueValue());
    Value falseValue = lookupValue(op, op.getFalseValue());
    if (!pred || !trueValue || !falseValue)
      return failure();
    state.values[op.getResult()] =
        createFragmentSelect(builder, op.getLoc(), pred, trueValue, falseValue,
                             resultType);
    return success();
  }

  LogicalResult lowerIndexCast(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "arith.index_cast shape");
    Type input = op->getOperand(0).getType();
    Type result = op->getResult(0).getType();
    if (!((input.isIndex() && result.isSignlessInteger(32)) ||
          (input.isSignlessInteger(32) && result.isIndex())))
      return emitStagedDiagnostic(op, "index_cast outside index/i32 boundary");
    Value mapped = lookupValue(op, op->getOperand(0));
    if (!mapped)
      return failure();
    state.values[op->getResult(0)] = mapped;
    return success();
  }

  FailureOr<Value> lowerMaskOperand(Operation *op, std::optional<Value> mask) {
    if (!mask)
      return createPredFull(builder, op->getLoc());
    Value pred = lookupPredicate(op, *mask);
    if (!pred)
      return failure();
    return pred;
  }

  FailureOr<Value> createByteOffsets(Operation *op, Value elemIndex,
                                     int64_t elemBytes,
                                     std::optional<Value> maskPred) {
    Value index = lookupValue(op, elemIndex);
    if (!index)
      return failure();
    if (!index.getType().isSignlessInteger(32))
      return op->emitOpError("transfer index must lower to scalar i32");

    Value byteBase = index;
    if (elemBytes != 1) {
      int64_t shift = elemBytes == 2 ? 1 : elemBytes == 4 ? 2 : -1;
      if (shift < 0)
        return op->emitOpError("unsupported element byte width");
      byteBase = builder.create<arith::ShLIOp>(op->getLoc(), index,
                                               createI32Constant(builder, op->getLoc(), shift));
    }
    Value baseVec = createSplat(builder, op->getLoc(), byteBase, getVector16I32(ctx));
    Value laneBytes = createLaneByteOffsets(builder, op->getLoc(), elemBytes);
    Value offsets = createAddPipe(builder, op->getLoc(), {baseVec, laneBytes},
                                  mlir::vc4kernel::AddALUOpcode::add,
                                  getVector16I32(ctx));
    if (maskPred) {
      Value poison = createPoisonByteOffsets(builder, op->getLoc(), elemBytes);
      offsets = createFragmentSelect(builder, op->getLoc(), *maskPred, offsets,
                                     poison, getVector16I32(ctx));
    }
    return offsets;
  }

  LogicalResult lowerTransferRead(Operation *op) {
    if (op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "transfer_read result count");
    Type resultType = op->getResult(0).getType();
    if (!isVector16I32(resultType) && !isVector16F32(resultType))
      return emitStagedDiagnostic(op, "transfer_read result type outside vector<16xi32/f32>");
    if (op->getNumOperands() != 3 && op->getNumOperands() != 4)
      return emitStagedDiagnostic(op, "transfer_read rank or mask form");
    if (!isRank1IdentityTransferMap(op))
      return emitStagedDiagnostic(op, "transfer_read permutation map beyond rank-1 identity");

    Value memref = op->getOperand(0);
    Value index = op->getOperand(1);
    Value padding = op->getOperand(2);
    std::optional<Value> mask;
    if (op->getNumOperands() == 4)
      mask = op->getOperand(3);

    if (!isRank1IdentityGlobalMemref(memref.getType()))
      return emitStagedDiagnostic(op, "transfer_read memref outside rank-1 contiguous i32/f32 #vc4value.global");
    if (!isZeroConstant(padding))
      return emitPhase5Diagnostic(op, "transfer_read padding must be zero for inactive_load<zero>");

    Value base = lookupValue(op, memref);
    FailureOr<Value> pred = lowerMaskOperand(op, mask);
    if (!base || failed(pred))
      return failure();
    int64_t elemBytes = 4;
    FailureOr<Value> byteOffsets =
        createByteOffsets(op, index, elemBytes,
                          mask ? std::optional<Value>(*pred) : std::nullopt);
    if (failed(byteOffsets))
      return failure();
    Value safeOffset = createI32Constant(builder, op->getLoc(), 0);
    Value result = createOpWithResult(
        builder, op->getLoc(), kTMULoadFragmentOpName,
        {base, *byteOffsets, *pred, safeOffset},
        {builder.getNamedAttr("memory_path", mlir::vc4kernel::MemoryPathAttr::get(
                                                ctx, mlir::vc4kernel::MemoryPath::tmu_global_read)),
         builder.getNamedAttr("coherency", mlir::vc4kernel::CoherencyAttr::get(
                                             ctx, mlir::vc4kernel::Coherency::readonly_tmu)),
         builder.getNamedAttr("inactive_load", mlir::vc4kernel::InactiveLoadAttr::get(
                                                 ctx, mlir::vc4kernel::InactiveLoad::zero))},
        resultType);
    state.values[op->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerTransferWrite(Operation *op) {
    if (op->getNumOperands() != 3 && op->getNumOperands() != 4)
      return emitStagedDiagnostic(op, "transfer_write rank or mask form");
    Value vector = op->getOperand(0);
    Value memref = op->getOperand(1);
    Value index = op->getOperand(2);
    std::optional<Value> mask;
    if (op->getNumOperands() == 4)
      mask = op->getOperand(3);

    if (!isVector16I32(vector.getType()) && !isVector16F32(vector.getType()))
      return emitStagedDiagnostic(op, "transfer_write value outside vector<16xi32/f32>");
    if (!isRank1IdentityGlobalMemref(memref.getType()))
      return emitStagedDiagnostic(op, "transfer_write memref outside rank-1 contiguous i32/f32 #vc4value.global");
    if (!isRank1IdentityTransferWriteMap(op))
      return emitStagedDiagnostic(op, "transfer_write permutation map beyond rank-1 identity");
    if (mask && !hasName((*mask).getDefiningOp(), "vector.create_mask"))
      return emitStagedDiagnostic(op, "sparse or unknown transfer_write mask");

    Value value = lookupValue(op, vector);
    Value base = lookupValue(op, memref);
    FailureOr<Value> pred = lowerMaskOperand(op, mask);
    if (!value || !base || failed(pred))
      return failure();
    int64_t elemBytes = 4;
    // Do not poison store offsets.  Inactive preservation is the VDW policy;
    // sparse/unknown masks must be rejected before reaching this path.
    FailureOr<Value> byteOffsets = createByteOffsets(op, index, elemBytes, std::nullopt);
    if (failed(byteOffsets))
      return failure();
    createOp(builder, op->getLoc(), kVDWStoreFragmentOpName,
             {base, *byteOffsets, value, *pred},
             {builder.getNamedAttr("memory_path", mlir::vc4kernel::MemoryPathAttr::get(
                                                     ctx, mlir::vc4kernel::MemoryPath::vdw_global_store)),
              builder.getNamedAttr("coherency", mlir::vc4kernel::CoherencyAttr::get(
                                                  ctx, mlir::vc4kernel::Coherency::dma_ordered)),
              builder.getNamedAttr("inactive_store", mlir::vc4kernel::InactiveStoreAttr::get(
                                                   ctx, mlir::vc4kernel::InactiveStore::preserve))});
    return success();
  }

  ModuleOp module;
  func::FuncOp func;
  MLIRContext *ctx;
  OpBuilder builder;
  LoweringState state;
};

struct ConvertVC4ValueToVC4KernelPass
    : public PassWrapper<ConvertVC4ValueToVC4KernelPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVC4ValueToVC4KernelPass)

  StringRef getArgument() const final {
    return "convert-vc4-value-to-vc4kernel";
  }

  StringRef getDescription() const final {
    return "Lower the executable VC4 value-surface subset to VC4Kernel";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                    func::FuncDialect, memref::MemRefDialect,
                    vector::VectorDialect, mlir::vc4value::VC4ValueDialect,
                    mlir::vc4kernel::VC4KernelDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp, 4> kernels;
    SmallVector<Operation *, 4> illegalTopLevelOps;

    for (Operation &op : module.getBody()->getOperations()) {
      if (auto func = llvm::dyn_cast<func::FuncOp>(op)) {
        if (func->hasAttr(kKernelAttr))
          kernels.push_back(func);
        else
          illegalTopLevelOps.push_back(&op);
        continue;
      }
      // Allow symbols that future phases may remove only if they are absent from
      // Phase 5 tests.  Anything else would survive beside vc4kernel.kernel and
      // break the locked VC4Kernel verifier boundary.
      illegalTopLevelOps.push_back(&op);
    }

    if (kernels.empty()) {
      module.emitError("expected at least one func.func marked vc4value.kernel; "
                       "not Phase 5 lowerable. READY_FOR_TRITON remains NO");
      signalPassFailure();
      return;
    }
    if (!illegalTopLevelOps.empty()) {
      illegalTopLevelOps.front()->emitOpError()
          << "non-kernel top-level operation is not Phase 5 lowerable; "
          << "staged value-surface feature. READY_FOR_TRITON remains NO";
      signalPassFailure();
      return;
    }

    for (func::FuncOp func : kernels) {
      KernelLowerer lowerer(module, func);
      if (failed(lowerer.lower())) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vc4::createConvertVC4ValueToVC4KernelPass() {
  return std::make_unique<ConvertVC4ValueToVC4KernelPass>();
}

void mlir::vc4::registerConvertVC4ValueToVC4KernelPass() {
  // The file-scope PassRegistration below installs
  // --convert-vc4-value-to-vc4kernel when this translation unit is linked into
  // vc4-opt.  This function mirrors the existing conversion registration API.
}

static PassRegistration<ConvertVC4ValueToVC4KernelPass>
    registerConvertVC4ValueToVC4KernelPass;
