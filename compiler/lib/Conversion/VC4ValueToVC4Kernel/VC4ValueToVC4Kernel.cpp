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
// with standard cf.br/cf.cond_br multi-block control flow, and Phase 9 extends
// logical launch identity to grid ranks 1/2/3 while preserving a
// crisp boundary between:
//
//   * the broad value-surface contract, which is allowed to contain staged
//     fixed vectors, subword storage, reductions, contracts, scf/cf, and future
//     TTIR-importable patterns; and
//   * the executable subset, which is vector<16> i32/f32 elementwise code over
//     rank-1 contiguous/scalar-computed #vc4value.global memrefs plus Phase 11
//     rank-2 row-slice memory skeletons, Phase 8 V1 cf control flow, and
//     multi-axis program_id/num_programs identity.
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
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "llvm/ADT/DenseSet.h"
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
constexpr llvm::StringLiteral kReductionPolicyAttr("vc4value.reduction_policy");
constexpr llvm::StringLiteral kI32MulPolicyAttr("vc4value.i32_mul_policy");
constexpr llvm::StringLiteral kF16StoragePolicyAttr("vc4value.f16_storage_policy");
constexpr llvm::StringLiteral kShapeArgsAttr("vc4value.shape_args");
constexpr llvm::StringLiteral kStrideArgsAttr("vc4value.stride_args");
constexpr llvm::StringLiteral kMathPolicyAttr("vc4value.math_policy");

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
constexpr llvm::StringLiteral kFragmentReduceOpName("vc4kernel.fragment_reduce");
constexpr llvm::StringLiteral kFragmentSFUOpName("vc4kernel.fragment_sfu");
constexpr llvm::StringLiteral kFragmentUnpackOpName("vc4kernel.fragment_unpack");
constexpr llvm::StringLiteral kFragmentPackOpName("vc4kernel.fragment_pack");
constexpr llvm::StringLiteral kTMULoadFragmentOpName("vc4kernel.tmu_load_fragment");
constexpr llvm::StringLiteral kVDWStoreFragmentOpName("vc4kernel.vdw_store_fragment");
constexpr llvm::StringLiteral kVPMAllocOpName("vc4kernel.vpm_alloc");
constexpr llvm::StringLiteral kVPMReadFragmentOpName("vc4kernel.vpm_read_fragment");
constexpr llvm::StringLiteral kVPMWriteFragmentOpName("vc4kernel.vpm_write_fragment");
constexpr llvm::StringLiteral kVDRLoadRectToVPMOpName("vc4kernel.vdr_load_rect_to_vpm");
constexpr llvm::StringLiteral kVDWStoreRectFromVPMOpName("vc4kernel.vdw_store_rect_from_vpm");

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

static bool isVector16F16(Type type) {
  return isVectorOf(type, 16, [](Type elem) { return elem.isF16(); });
}

static bool isVector16I32OrIndex(Type type) {
  return isVector16I32(type) || isVector16Index(type);
}

static bool isVector16Data(Type type) {
  return isVector16I32(type) || isVector16F32(type);
}

static bool hasF16Type(Type type) {
  if (type.isF16())
    return true;
  if (auto vectorType = llvm::dyn_cast<VectorType>(type))
    return hasF16Type(vectorType.getElementType());
  return false;
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

static bool hasGlobalMemorySpace(MemRefType type) {
  return llvm::isa_and_nonnull<mlir::vc4value::GlobalMemorySpaceAttr>(
      type.getMemorySpace());
}

static bool isI32OrF32(Type type) {
  return type.isSignlessInteger(32) || type.isF32();
}

static bool isI32F32OrF16(Type type) {
  return isI32OrF32(type) || type.isF16();
}

static bool isBF16OrFP8StorageElement(Type type) {
  if (type.isBF16())
    return true;
  if (auto floatType = llvm::dyn_cast<FloatType>(type))
    return floatType.getWidth() == 8;
  return false;
}

static bool isQuantizedSubwordStorageElement(Type type) {
  return type.isSignlessInteger(8) || type.isSignlessInteger(16);
}

static bool isVector16Rank1(Type type) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  return vectorType && !vectorType.isScalable() && vectorType.getRank() == 1 &&
         vectorType.getDimSize(0) == 16;
}

static bool getStridesAndOffset(MemRefType type,
                                SmallVectorImpl<int64_t> &strides,
                                int64_t &offset) {
  MemRefLayoutAttrInterface layout = type.getLayout();
  if (!layout || layout.isIdentity()) {
    strides.clear();
    offset = 0;
    return true;
  }
  return succeeded(layout.getStridesAndOffset(type.getShape(), strides, offset));
}

static bool isRank2IdentityGlobalMemref(Type type) {
  auto memrefType = llvm::dyn_cast<MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 2)
    return false;
  if (!hasGlobalMemorySpace(memrefType) || !isI32OrF32(memrefType.getElementType()))
    return false;
  return memrefType.getLayout().isIdentity() &&
         ShapedType::isDynamic(memrefType.getDimSize(0)) &&
         ShapedType::isDynamic(memrefType.getDimSize(1));
}

static bool isRank2StridedOuterDynamicGlobalMemref(Type type) {
  auto memrefType = llvm::dyn_cast<MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 2)
    return false;
  if (!hasGlobalMemorySpace(memrefType) || !isI32OrF32(memrefType.getElementType()))
    return false;
  if (!ShapedType::isDynamic(memrefType.getDimSize(0)) ||
      !ShapedType::isDynamic(memrefType.getDimSize(1)))
    return false;
  SmallVector<int64_t, 2> strides;
  int64_t offset = 0;
  if (!getStridesAndOffset(memrefType, strides, offset))
    return false;
  return strides.size() == 2 && ShapedType::isDynamic(strides[0]) &&
         !ShapedType::isDynamic(strides[1]) && strides[1] == 1 && offset == 0;
}

static bool isPhase11LowerableGlobalMemref(Type type) {
  return isRank1IdentityGlobalMemref(type) || isRank2IdentityGlobalMemref(type) ||
         isRank2StridedOuterDynamicGlobalMemref(type);
}

static bool isPhase14StorageNumericGlobalMemref(Type type) {
  auto memrefType = llvm::dyn_cast<MemRefType>(type);
  if (!memrefType || !hasGlobalMemorySpace(memrefType) ||
      !isI32F32OrF16(memrefType.getElementType()))
    return false;

  if (memrefType.getRank() == 1)
    return memrefType.getLayout().isIdentity();

  if (memrefType.getRank() != 2 ||
      !ShapedType::isDynamic(memrefType.getDimSize(0)) ||
      !ShapedType::isDynamic(memrefType.getDimSize(1)))
    return false;

  if (memrefType.getLayout().isIdentity())
    return true;

  SmallVector<int64_t, 2> strides;
  int64_t offset = 0;
  if (!getStridesAndOffset(memrefType, strides, offset))
    return false;
  return strides.size() == 2 && ShapedType::isDynamic(strides[0]) &&
         !ShapedType::isDynamic(strides[1]) && strides[1] == 1 && offset == 0;
}

static bool isRank1IdentityTransferMap(Operation *op) {
  auto read = llvm::dyn_cast<vector::TransferReadOp>(op);
  if (!read)
    return false;
  AffineMap map = read.getPermutationMap();
  return map.getNumDims() == 1 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 1 && map.isMinorIdentity();
}

static bool isRank2InnermostRowSliceTransferMap(AffineMap map) {
  return map.getNumDims() == 2 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 1 && map.isMinorIdentity();
}

static bool isRank2InnermostRowSliceTransferMap(Operation *op) {
  if (auto read = llvm::dyn_cast<vector::TransferReadOp>(op))
    return isRank2InnermostRowSliceTransferMap(read.getPermutationMap());
  if (auto write = llvm::dyn_cast<vector::TransferWriteOp>(op))
    return isRank2InnermostRowSliceTransferMap(write.getPermutationMap());
  return false;
}

static bool isRank1IdentityTransferWriteMap(Operation *op) {
  auto write = llvm::dyn_cast<vector::TransferWriteOp>(op);
  if (!write)
    return false;
  AffineMap map = write.getPermutationMap();
  return map.getNumDims() == 1 && map.getNumSymbols() == 0 &&
         map.getNumResults() == 1 && map.isMinorIdentity();
}

static bool isArrayOfStringAttr(Attribute attr) {
  auto arrayAttr = llvm::dyn_cast_or_null<ArrayAttr>(attr);
  return arrayAttr && llvm::all_of(arrayAttr, [](Attribute element) {
           return llvm::isa<StringAttr>(element);
         });
}

static SmallVector<StringRef> getStringArrayValues(Attribute attr) {
  SmallVector<StringRef> values;
  auto arrayAttr = llvm::dyn_cast_or_null<ArrayAttr>(attr);
  if (!arrayAttr)
    return values;
  values.reserve(arrayAttr.size());
  for (Attribute element : arrayAttr)
    values.push_back(llvm::cast<StringAttr>(element).getValue());
  return values;
}

static std::optional<unsigned> getPublicMemRefArgIndex(func::FuncOp func,
                                                       Value value) {
  auto blockArg = llvm::dyn_cast<BlockArgument>(value);
  if (!blockArg || blockArg.getOwner() != &func.getBody().front())
    return std::nullopt;
  unsigned argIndex = blockArg.getArgNumber();
  if (argIndex >= func.getNumArguments() ||
      !llvm::isa<MemRefType>(func.getArgument(argIndex).getType()))
    return std::nullopt;
  return argIndex;
}

static std::optional<unsigned> getDynamicDimOrdinal(MemRefType type,
                                                    unsigned dim) {
  if (dim >= static_cast<unsigned>(type.getRank()) ||
      !ShapedType::isDynamic(type.getDimSize(dim)))
    return std::nullopt;
  unsigned ordinal = 0;
  for (unsigned i = 0; i < dim; ++i)
    if (ShapedType::isDynamic(type.getDimSize(i)))
      ++ordinal;
  return ordinal;
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

static StringRef getStorageElementTypeName(Type type) {
  if (type.isF16())
    return "u16";
  return getElementTypeName(type);
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

static Type getVector16F32(MLIRContext *ctx) {
  return VectorType::get({16}, Float32Type::get(ctx));
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

static Value createApproxSFU(OpBuilder &builder, Location loc, Value input,
                             mlir::vc4kernel::SFUKind kind,
                             mlir::vc4kernel::FPDomain domain) {
  MLIRContext *ctx = builder.getContext();
  return createOpWithResult(
      builder, loc, kFragmentSFUOpName, input,
      {builder.getNamedAttr("kind",
                            mlir::vc4kernel::SFUKindAttr::get(ctx, kind)),
       builder.getNamedAttr(
           "fp_policy", mlir::vc4kernel::FPMathPolicyAttr::get(
                            ctx, mlir::vc4kernel::FPMathPolicy::approx_sfu)),
       builder.getNamedAttr("domain",
                            mlir::vc4kernel::FPDomainAttr::get(ctx, domain))},
      getVector16F32(ctx));
}

static Value createFragmentSelect(OpBuilder &builder, Location loc, Value pred,
                                  Value trueValue, Value falseValue,
                                  Type resultType) {
  return createOpWithResult(builder, loc, kFragmentSelectOpName,
                            {pred, trueValue, falseValue}, {}, resultType);
}

static Value createF16StorageUnpack(OpBuilder &builder, Location loc,
                                    Value carrier) {
  MLIRContext *ctx = builder.getContext();
  return createOpWithResult(
      builder, loc, kFragmentUnpackOpName, carrier,
      {builder.getNamedAttr(
           "source", mlir::vc4kernel::SubwordTypeAttr::get(
                         ctx, mlir::vc4kernel::SubwordType::f16)),
       builder.getNamedAttr(
           "layout", mlir::vc4kernel::SubwordLayoutAttr::get(
                         ctx, mlir::vc4kernel::SubwordLayout::packed)),
       builder.getNamedAttr(
           "policy", mlir::vc4kernel::UnpackPolicyAttr::get(
                         ctx, mlir::vc4kernel::UnpackPolicy::to_f32))},
      getVector16F32(ctx));
}

static Value createF16StoragePack(OpBuilder &builder, Location loc,
                                  Value f32Fragment) {
  MLIRContext *ctx = builder.getContext();
  return createOpWithResult(
      builder, loc, kFragmentPackOpName, f32Fragment,
      {builder.getNamedAttr(
           "dest", mlir::vc4kernel::SubwordTypeAttr::get(
                       ctx, mlir::vc4kernel::SubwordType::f16)),
       builder.getNamedAttr(
           "layout", mlir::vc4kernel::SubwordLayoutAttr::get(
                         ctx, mlir::vc4kernel::SubwordLayout::packed)),
       builder.getNamedAttr(
           "policy", mlir::vc4kernel::PackPolicyAttr::get(
                         ctx, mlir::vc4kernel::PackPolicy::from_f32))},
      getVector16I32(ctx));
}

static SmallVector<NamedAttribute, 8>
getPackedW16VPMAttrs(OpBuilder &builder, StringRef memoryPath,
                     StringRef coherency, bool dmaAccess) {
  MLIRContext *ctx = builder.getContext();
  mlir::vc4kernel::MemoryPath path =
      memoryPath == "vdr"
          ? mlir::vc4kernel::MemoryPath::vdr_global_to_vpm
          : memoryPath == "vdw" ? mlir::vc4kernel::MemoryPath::vdw_global_store
                                 : mlir::vc4kernel::MemoryPath::vpm_qpu;
  mlir::vc4kernel::Coherency coherencyValue =
      coherency == "vpm" ? mlir::vc4kernel::Coherency::vpm_local
                          : mlir::vc4kernel::Coherency::dma_ordered;
  SmallVector<NamedAttribute, 8> attrs = {
      builder.getNamedAttr(
          "orientation", mlir::vc4kernel::VPMOrientationAttr::get(
                             ctx, mlir::vc4kernel::VPMOrientation::horizontal)),
      builder.getNamedAttr(
          "width", mlir::vc4kernel::VPMWidthAttr::get(
                       ctx, mlir::vc4kernel::VPMWidth::w16)),
      builder.getNamedAttr(
          "subword", mlir::vc4kernel::VPMSubwordAttr::get(
                         ctx, mlir::vc4kernel::VPMSubword::packed)),
      builder.getNamedAttr(
          "subword_selector", builder.getI32IntegerAttr(0)),
      builder.getNamedAttr(
          "memory_path", mlir::vc4kernel::MemoryPathAttr::get(ctx, path)),
      builder.getNamedAttr(
          "coherency", mlir::vc4kernel::CoherencyAttr::get(ctx, coherencyValue))};
  attrs.push_back(builder.getNamedAttr(
      dmaAccess ? "vpm_pitch" : "stride", builder.getI32IntegerAttr(1)));
  return attrs;
}

static NamedAttribute getOperandSegmentSizesAttr(OpBuilder &builder,
                                                 ArrayRef<int32_t> sizes) {
  return builder.getNamedAttr(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr(sizes));
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

static bool isVectorMulOp(Operation *op) {
  if (!op || op->getNumResults() != 1)
    return false;
  StringRef name = op->getName().getStringRef();
  return name == "arith.mulf" || name == "arith.muli";
}

static bool isDotCompositeProduct(Operation *op) {
  if (!isVectorMulOp(op) || !isVector16Rank1(op->getResult(0).getType()))
    return false;
  for (Operation *user : op->getResult(0).getUsers()) {
    auto reduction = llvm::dyn_cast<vector::ReductionOp>(user);
    if (reduction && reduction.getKind() == vector::CombiningKind::ADD &&
        reduction.getVector() == op->getResult(0))
      return true;
  }
  return false;
}

static bool isDotCompositeReductionValue(Value value) {
  auto reduction = value.getDefiningOp<vector::ReductionOp>();
  if (!reduction || reduction.getKind() != vector::CombiningKind::ADD)
    return false;
  return isDotCompositeProduct(reduction.getVector().getDefiningOp());
}

static bool isNonEntryBlockArgument(Value value) {
  auto arg = llvm::dyn_cast<BlockArgument>(value);
  return arg && arg.getOwner() &&
         !arg.getOwner()->isEntryBlock();
}

enum class MemoryMaskKind {
  Full,
  Empty,
  Tail,
  SparseOrUnknown,
  Unsupported
};

struct ClassifiedMemoryMask {
  MemoryMaskKind kind = MemoryMaskKind::Unsupported;
  Value start;
  Value count;
  Operation *source = nullptr;
  std::string reason;
  Value predicate;
};

struct MemoryAddressPlan {
  enum class Kind {
    Rank1ContiguousOrScalarComputed,
    Rank2RowSliceIdentity,
    Rank2RowSliceStridedOuterDynamic,
    Staged
  };

  Kind kind = Kind::Staged;
  Value basePointer;
  Value elementBaseIndex;
  int64_t elementBytes = 4;
  bool f16Storage = false;
  ClassifiedMemoryMask mask;
  Operation *source = nullptr;
  StringRef reason;
};

enum class TransferAccessKind { Read, Write };

struct ClassifiedMemoryTransfer {
  TransferAccessKind access = TransferAccessKind::Read;
  Value memref;
  Value vectorValue;
  Value padding;
  Type vectorType;
  Type storageCarrierType;
  MemoryAddressPlan addressPlan;
  std::string acceptedPath;
  std::string stagedReason;
};

struct LoweringState {
  explicit LoweringState(Operation *sourceKernel) : sourceKernel(sourceKernel) {}

  Operation *sourceKernel = nullptr;
  DenseMap<Block *, Block *> blocks;
  DenseMap<Value, Value> values;
  DenseMap<Value, Value> reductionFragments;
  DenseMap<Value, Value> predicates;
  DenseMap<Value, ClassifiedMemoryMask> memoryMasks;
  DenseSet<Value> i32DotProducts;
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
    auto gridRankAttr =
        llvm::dyn_cast_or_null<IntegerAttr>(func->getAttr(kGridRankAttr));
    if (!gridRankAttr)
      return func.emitOpError(
          "expected vc4value.grid_rank for Phase 9 value lowering");
    gridRank = gridRankAttr.getInt();
    if (gridRank < 1 || gridRank > 3)
      return func.emitOpError(
          "expected vc4value.grid_rank in [1, 3] for Phase 9 value lowering");
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
      if (!isPhase14StorageNumericGlobalMemref(type)) {
        InFlightDiagnostic diag =
            arg.getOwner()->getParentOp()->emitOpError();
        Type elementType = memrefType.getElementType();
        if (isBF16OrFP8StorageElement(elementType)) {
          diag << "bf16/fp8 storage is staged. READY_FOR_TRITON remains NO";
          return failure();
        }
        if (isQuantizedSubwordStorageElement(elementType)) {
          diag << "int8/int16 quantized storage is staged. "
               << "READY_FOR_TRITON remains NO";
          return failure();
        }
        bool supportedGlobalElement =
            hasGlobalMemorySpace(memrefType) &&
            isI32F32OrF16(elementType);
        if (!supportedGlobalElement)
          diag << "unsupported transfer element type; ";
        else
          diag << "ranked or strided memory beyond Phase 10 unless covered by "
               << "Phase 11 row-slice skeletons; ";
        diag << "expected rank-1 contiguous i32/f32/f16 #vc4value.global memref "
             << "or rank-2 row-slice i32/f32/f16 #vc4value.global memref "
             << "for Phase 11 argument " << arg.getArgNumber()
             << "; not Phase 5 lowerable unless accepted by the Phase 11 "
             << "row-slice and Phase 14 f16 storage contracts; staged value-surface feature. "
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
    if (type.isF32()) {
      DenseSet<Value> visiting;
      if (isReductionFragmentBlockArgument(arg, visiting))
        return VectorType::get({16}, builder.getF32Type());
    }
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
          "elem_type", builder.getStringAttr(
                           getStorageElementTypeName(memrefType.getElementType()))));
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
                                                   targetBlock->getArguments())) {
        DenseSet<Value> visiting;
        if (sourceArg.getType().isF32() &&
            isReductionFragmentBlockArgument(sourceArg, visiting))
          state.reductionFragments[sourceArg] = targetArg;
        else
          mapLoweredValue(sourceArg, targetArg);
      }
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

  Value lookupReductionFragment(Value value) {
    auto it = state.reductionFragments.find(value);
    if (it != state.reductionFragments.end())
      return it->second;
    return {};
  }

  FailureOr<Value> getF32FragmentOperand(Operation *op, Value value) {
    if (isVector16F32(value.getType())) {
      Value fragment = lookupValue(op, value);
      if (!fragment)
        return failure();
      return fragment;
    }
    if (value.getType().isF32()) {
      if (Value fragment = lookupReductionFragment(value))
        return fragment;
      Value scalar = lookupValue(op, value);
      if (!scalar)
        return failure();
      return createSplat(builder, op->getLoc(), scalar, getVector16F32(ctx));
    }
    return op->emitOpError("approximate SFU math requires scalar f32 or "
                           "vector<16xf32> operands");
  }

  void mapF32FragmentResult(Value source, Value fragment) {
    if (isVector16F32(source.getType()))
      state.values[source] = fragment;
    else
      state.reductionFragments[source] = fragment;
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
    if (Value fragment = lookupReductionFragment(value))
      return fragment;
    return lookupValue(op, value);
  }

  bool isReductionFragmentValue(Value value, DenseSet<Value> &visiting) {
    if (isDotCompositeReductionValue(value))
      return true;
    if (auto arg = llvm::dyn_cast<BlockArgument>(value))
      return isReductionFragmentBlockArgument(arg, visiting);

    Operation *def = value.getDefiningOp();
    if (!def || def->getName().getStringRef() != "arith.addf" ||
        def->getNumOperands() != 2 ||
        def->getNumResults() != 1 || !def->getResult(0).getType().isF32())
      return false;

    Value lhs = def->getOperand(0);
    Value rhs = def->getOperand(1);
    if (isNonEntryBlockArgument(lhs) || isNonEntryBlockArgument(rhs))
      return false;
    return isReductionFragmentValue(lhs, visiting) ||
           isReductionFragmentValue(rhs, visiting);
  }

  bool isReductionFragmentBlockArgument(BlockArgument arg,
                                        DenseSet<Value> &visiting) {
    if (!arg || !arg.getOwner() || arg.getOwner()->isEntryBlock())
      return false;
    Value asValue = arg;
    if (!visiting.insert(asValue).second)
      return false;

    bool sawIncoming = false;
    unsigned index = arg.getArgNumber();
    Block *block = arg.getOwner();
    for (Block *predecessor : block->getPredecessors()) {
      Operation *terminator = predecessor->getTerminator();
      if (auto branch = llvm::dyn_cast<cf::BranchOp>(terminator)) {
        if (branch.getDest() != block)
          continue;
        OperandRange operands = branch.getDestOperands();
        if (index >= operands.size() ||
            !isReductionFragmentValue(operands[index], visiting))
          return false;
        sawIncoming = true;
        continue;
      }
      if (auto condBranch = llvm::dyn_cast<cf::CondBranchOp>(terminator)) {
        bool matched = false;
        if (condBranch.getTrueDest() == block) {
          OperandRange operands = condBranch.getTrueDestOperands();
          if (index >= operands.size() ||
              !isReductionFragmentValue(operands[index], visiting))
            return false;
          matched = true;
        }
        if (condBranch.getFalseDest() == block) {
          OperandRange operands = condBranch.getFalseDestOperands();
          if (index >= operands.size() ||
              !isReductionFragmentValue(operands[index], visiting))
            return false;
          matched = true;
        }
        sawIncoming |= matched;
        continue;
      }
      return false;
    }
    return sawIncoming;
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
    if (auto switchOp = llvm::dyn_cast<cf::SwitchOp>(op))
      return lowerSwitch(switchOp);
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

  LogicalResult lowerSwitch(cf::SwitchOp switchOp) {
    Value flag = lookupValue(switchOp.getOperation(), switchOp.getFlag());
    if (!flag)
      return failure();
    if (!flag.getType().isSignlessInteger(32))
      return switchOp.emitOpError()
             << "flag must lower to scalar i32 for Phase 8R cf.switch "
                "control flow; READY_FOR_TRITON remains NO";

    Block *defaultTarget = lookupTargetBlock(switchOp.getOperation(),
                                             switchOp.getDefaultDestination());
    if (!defaultTarget)
      return failure();
    SmallVector<Value, 4> defaultOperands;
    if (failed(lowerSuccessorOperands(switchOp.getOperation(),
                                      switchOp.getDefaultOperands(),
                                      switchOp.getDefaultDestination(),
                                      defaultOperands)))
      return failure();

    DenseIntElementsAttr caseValuesAttr = switchOp.getCaseValuesAttr();
    if (!caseValuesAttr || caseValuesAttr.empty()) {
      builder.create<cf::BranchOp>(switchOp.getLoc(), defaultTarget,
                                   defaultOperands);
      return success();
    }

    SmallVector<APInt, 4> caseValues;
    for (APInt value : caseValuesAttr.getValues<APInt>())
      caseValues.push_back(value);

    SmallVector<Block *, 4> caseTargets;
    SmallVector<SmallVector<Value, 4>, 4> caseOperands;
    for (auto [index, destination] :
         llvm::enumerate(switchOp.getCaseDestinations())) {
      caseTargets.push_back(lookupTargetBlock(switchOp.getOperation(),
                                              destination));
      if (!caseTargets.back())
        return failure();
      SmallVector<Value, 4> operands;
      if (failed(lowerSuccessorOperands(switchOp.getOperation(),
                                        switchOp.getCaseOperands(index),
                                        destination, operands)))
        return failure();
      caseOperands.push_back(std::move(operands));
    }

    if (caseValues.size() != caseTargets.size())
      return switchOp.emitOpError()
             << "case value count does not match destination count; not Phase "
                "8R lowerable. READY_FOR_TRITON remains NO";

    Region *targetRegion = builder.getInsertionBlock()->getParent();
    for (auto [index, value] : llvm::enumerate(caseValues)) {
      Value constant =
          createI32Constant(builder, switchOp.getLoc(), value.getSExtValue());
      Value match = builder.create<arith::CmpIOp>(
          switchOp.getLoc(), arith::CmpIPredicate::eq, flag, constant);

      bool lastCase = index + 1 == caseValues.size();
      Block *falseTarget = defaultTarget;
      ValueRange falseOperands = defaultOperands;
      if (!lastCase) {
        Block *testBlock = builder.getInsertionBlock();
        falseTarget = builder.createBlock(targetRegion, targetRegion->end());
        falseOperands = ValueRange();
        builder.setInsertionPointToEnd(testBlock);
      }

      builder.create<cf::CondBranchOp>(switchOp.getLoc(), match,
                                       caseTargets[index], caseOperands[index],
                                       falseTarget, falseOperands);
      if (!lastCase)
        builder.setInsertionPointToEnd(falseTarget);
    }
    return success();
  }

  LogicalResult lowerOperation(Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "tt.dot")
      return op->emitOpError()
             << "tt.dot is staged. READY_FOR_TRITON remains NO";
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
    if (auto reduction = llvm::dyn_cast<vector::ReductionOp>(op))
      return lowerVectorReduction(reduction);
    if (llvm::isa<vector::ContractionOp>(op))
      return op->emitOpError()
             << "vector.contract is staged for Phase 15/contract. "
             << "READY_FOR_TRITON remains NO";
    if (name == "vector.multi_reduction")
      return emitStagedDiagnostic(op, "vector.multi_reduction is staged");
    if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op))
      return lowerConstant(constant);
    if (name == "arith.extf")
      return lowerExtF(op);
    if (name == "arith.truncf")
      return lowerTruncF(op);
    if (llvm::isa<math::ExpOp, math::LogOp, math::RsqrtOp>(op))
      return lowerApproxSFUMath(op);
    if (name == "arith.sitofp" || name == "arith.uitofp")
      return op->emitOpError()
             << "i32 to f32 numeric cast staged by lower-half gap. "
             << "READY_FOR_TRITON remains NO";
    if (name == "arith.fptosi" || name == "arith.fptoui")
      return op->emitOpError()
             << "fp-to-int numeric cast is staged. READY_FOR_TRITON remains NO";
    if (llvm::isa<arith::DivFOp>(op))
      return lowerApproxDivF(op);
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
    if (auto dim = llvm::dyn_cast<memref::DimOp>(op))
      return lowerMemRefDim(dim);
    if (auto store = llvm::dyn_cast<memref::StoreOp>(op))
      return lowerScalarMemrefStore(store);
    if (llvm::isa<memref::LoadOp>(op))
      return emitStagedDiagnostic(op, "scalar memref.load is staged");

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
    FailureOr<int64_t> axis = getLaunchAxis(op, "program_id");
    if (failed(axis))
      return failure();
    Value result = createOpWithResult(
        builder, op->getLoc(), kProgramIdOpName, {},
        {builder.getNamedAttr("axis", builder.getI32IntegerAttr(*axis))},
        builder.getI32Type());
    state.values[op->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerNumPrograms(Operation *op) {
    FailureOr<int64_t> axis = getLaunchAxis(op, "num_programs");
    if (failed(axis))
      return failure();
    Value result = createOpWithResult(
        builder, op->getLoc(), kNumProgramsOpName, {},
        {builder.getNamedAttr("axis", builder.getI32IntegerAttr(*axis))},
        builder.getI32Type());
    state.values[op->getResult(0)] = result;
    return success();
  }

  FailureOr<int64_t> getLaunchAxis(Operation *op, StringRef opName) {
    std::optional<int64_t> maybeAxis = getI32Attr(op, "axis");
    if (!maybeAxis) {
      op->emitOpError() << opName << " requires i32 axis attribute";
      return failure();
    }
    int64_t axis = *maybeAxis;
    if (axis < 0 || axis > 2) {
      op->emitOpError() << opName << " axis must be 0, 1, or 2";
      return failure();
    }
    if (axis >= gridRank) {
      op->emitOpError()
          << opName << " axis " << axis
          << " is outside vc4value.grid_rank " << gridRank
          << "; Phase 9 launch identity only lowers axes within the "
             "logical grid rank. READY_FOR_TRITON remains NO";
      return failure();
    }
    return axis;
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
    if (isVector16F32(resultType)) {
      if (Value fragment = lookupReductionFragment(op->getOperand(0))) {
        state.values[op->getResult(0)] = fragment;
        return success();
      }
    }
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

  Value createClampedMaskCount(Location loc, Value count, Value zero,
                               Value sixteen) {
    Value belowZero =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, count, zero);
    Value nonNegative =
        builder.create<arith::SelectOp>(loc, belowZero, zero, count);
    Value aboveSixteen = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, nonNegative, sixteen);
    return builder.create<arith::SelectOp>(loc, aboveSixteen, sixteen,
                                           nonNegative);
  }

  void rememberMemoryMask(Value source, const ClassifiedMemoryMask &mask) {
    state.memoryMasks[source] = mask;
    if (mask.predicate)
      state.predicates[source] = mask.predicate;
  }

  ClassifiedMemoryMask makeFullMask(Operation *source) {
    ClassifiedMemoryMask mask;
    mask.kind = MemoryMaskKind::Full;
    mask.start = createI32Constant(builder, source->getLoc(), 0);
    mask.count = createI32Constant(builder, source->getLoc(), 16);
    mask.source = source;
    mask.reason = "full transfer mask";
    mask.predicate = createPredFull(builder, source->getLoc());
    return mask;
  }

  ClassifiedMemoryMask makeEmptyMask(Operation *source) {
    ClassifiedMemoryMask mask;
    mask.kind = MemoryMaskKind::Empty;
    mask.start = createI32Constant(builder, source->getLoc(), 0);
    mask.count = createI32Constant(builder, source->getLoc(), 0);
    mask.source = source;
    mask.reason = "empty transfer mask";
    mask.predicate = createPredEmpty(builder, source->getLoc());
    return mask;
  }

  ClassifiedMemoryMask makeTailMask(Operation *source, Value start,
                                    Value clampedCount) {
    ClassifiedMemoryMask mask;
    mask.kind = MemoryMaskKind::Tail;
    mask.start = start;
    mask.count = clampedCount;
    mask.source = source;
    mask.reason = "vector.create_mask tail transfer mask";
    mask.predicate = createPredTail(builder, source->getLoc(), mask.start,
                                    clampedCount);
    return mask;
  }

  FailureOr<ClassifiedMemoryMask> classifyCreateMask(Operation *op) {
    Value count = lookupValue(op, op->getOperand(0));
    if (!count)
      return failure();

    std::optional<int64_t> constantCount = getIntegerConstant(op->getOperand(0));
    if (constantCount && *constantCount <= 0)
      return makeEmptyMask(op);
    if (constantCount && *constantCount >= 16)
      return makeFullMask(op);

    Value zero = createI32Constant(builder, op->getLoc(), 0);
    Value sixteen = createI32Constant(builder, op->getLoc(), 16);
    Value clamped = createClampedMaskCount(op->getLoc(), count, zero, sixteen);
    return makeTailMask(op, zero, clamped);
  }

  LogicalResult lowerCreateMask(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1 ||
        !isVector16I1(op->getResult(0).getType()))
      return emitStagedDiagnostic(op, "vector.create_mask shape other than vector<16xi1>");
    FailureOr<ClassifiedMemoryMask> mask = classifyCreateMask(op);
    if (failed(mask))
      return failure();
    rememberMemoryMask(op->getResult(0), *mask);
    return success();
  }

  LogicalResult lowerConstant(arith::ConstantOp op) {
    Attribute value = op.getValue();
    Type resultType = op.getType();
    Location loc = op.getLoc();

    if (isVector16Rank1(resultType) && !isVector16I32(resultType) &&
        !isVector16F32(resultType)) {
      for (Operation *user : op.getResult().getUsers()) {
        if (isDotCompositeProduct(user))
          return op.emitOpError("unsupported GEMV element type");
      }
    }

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
    if (resultType.isF16()) {
      if (!isZeroConstant(op.getResult()))
        return emitStagedDiagnostic(op.getOperation(),
                                    "nonzero scalar f16 constant");
      state.values[op.getResult()] = createI32Constant(builder, loc, 0);
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
      ClassifiedMemoryMask mask =
          sawTrue ? makeFullMask(op.getOperation()) : makeEmptyMask(op.getOperation());
      rememberMemoryMask(op.getResult(), mask);
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

  LogicalResult lowerExtF(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "arith.extf shape");
    if (!isVector16F16(op->getOperand(0).getType()) ||
        !isVector16F32(op->getResult(0).getType()))
      return emitStagedDiagnostic(
          op, "unsupported numeric conversion; only f16 storage extf to f32 "
              "compute is accepted in Phase 14");
    Value carrier = lookupValue(op, op->getOperand(0));
    if (!carrier)
      return failure();
    if (!isVector16I32(carrier.getType()))
      return op->emitOpError()
             << "f16 storage extf requires raw i32 storage carrier. "
             << "READY_FOR_TRITON remains NO";
    state.values[op->getResult(0)] =
        createF16StorageUnpack(builder, op->getLoc(), carrier);
    return success();
  }

  LogicalResult lowerTruncF(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "arith.truncf shape");
    if (!isVector16F32(op->getOperand(0).getType()) ||
        !isVector16F16(op->getResult(0).getType()))
      return emitStagedDiagnostic(
          op, "unsupported numeric conversion; only f32 compute truncf to "
              "f16 storage is accepted in Phase 14");
    if (!hasFiniteF16StoragePolicy(op))
      return op->emitOpError()
             << "f16 storage store requires explicit finite storage policy. "
             << "READY_FOR_TRITON remains NO";
    Value f32 = lookupValue(op, op->getOperand(0));
    if (!f32)
      return failure();
    if (!isVector16F32(f32.getType()))
      return op->emitOpError()
             << "f16 storage truncf requires f32 compute fragment. "
             << "READY_FOR_TRITON remains NO";
    state.values[op->getResult(0)] =
        createF16StoragePack(builder, op->getLoc(), f32);
    return success();
  }

  bool hasApproxMathPolicy(Operation *op) {
    return hasStringAttr(op, kMathPolicyAttr, "approx_sfu") ||
           hasStringAttr(func.getOperation(), kMathPolicyAttr, "approx_sfu");
  }

  bool hasFiniteMathDomain(Operation *op) {
    return hasStringAttr(op, kFPDomainAttr, "finite") ||
           hasStringAttr(func.getOperation(), kFPDomainAttr, "finite");
  }

  bool hasFiniteNonzeroMathDomain(Operation *op) {
    return hasStringAttr(op, kFPDomainAttr, "finite_nonzero") ||
           hasStringAttr(func.getOperation(), kFPDomainAttr, "finite_nonzero");
  }

  bool hasFinitePositiveMathDomain(Operation *op) {
    return hasStringAttr(op, kFPDomainAttr, "finite_positive") ||
           hasStringAttr(func.getOperation(), kFPDomainAttr, "finite_positive");
  }

  bool hasNaNInfMathDomain(Operation *op) {
    return hasStringAttr(op, kFPDomainAttr, "nan_inf") ||
           hasStringAttr(func.getOperation(), kFPDomainAttr, "nan_inf");
  }

  LogicalResult requireApproxSFUPolicy(Operation *op,
                                       mlir::vc4kernel::SFUKind kind) {
    if (hasNaNInfMathDomain(op))
      return op->emitOpError()
             << "NaN/Inf exact math semantics are staged. "
             << "READY_FOR_TRITON remains NO";
    if (!hasApproxMathPolicy(op))
      return op->emitOpError()
             << "exact/default math requires explicit approximate-SFU policy. "
             << "READY_FOR_TRITON remains NO";

    switch (kind) {
    case mlir::vc4kernel::SFUKind::exp:
      if (hasFiniteMathDomain(op) || hasFiniteNonzeroMathDomain(op) ||
          hasFinitePositiveMathDomain(op))
        return success();
      break;
    case mlir::vc4kernel::SFUKind::recip:
      if (hasFiniteNonzeroMathDomain(op) || hasFiniteMathDomain(op))
        return success();
      return op->emitOpError()
             << "generic division without approximate reciprocal policy is "
                "staged. READY_FOR_TRITON remains NO";
    case mlir::vc4kernel::SFUKind::rsqrt:
      if (hasFinitePositiveMathDomain(op))
        return success();
      return op->emitOpError()
             << "math.rsqrt SFU mode is staged by lower-half gap or missing "
                "finite_positive domain. READY_FOR_TRITON remains NO";
    case mlir::vc4kernel::SFUKind::log:
      if (hasFinitePositiveMathDomain(op))
        return success();
      return op->emitOpError()
             << "math.log SFU mode is staged by lower-half gap or missing "
                "finite_positive domain. READY_FOR_TRITON remains NO";
    }

    return op->emitOpError()
           << "exact/default math requires explicit approximate-SFU policy. "
           << "READY_FOR_TRITON remains NO";
  }

  LogicalResult lowerApproxSFUMath(Operation *op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "approximate SFU math arity");

    mlir::vc4kernel::SFUKind kind;
    mlir::vc4kernel::FPDomain domain;
    if (llvm::isa<math::ExpOp>(op)) {
      kind = mlir::vc4kernel::SFUKind::exp;
      domain = mlir::vc4kernel::FPDomain::finite;
    } else if (llvm::isa<math::LogOp>(op)) {
      kind = mlir::vc4kernel::SFUKind::log;
      domain = mlir::vc4kernel::FPDomain::finite_positive;
    } else if (llvm::isa<math::RsqrtOp>(op)) {
      kind = mlir::vc4kernel::SFUKind::rsqrt;
      domain = mlir::vc4kernel::FPDomain::finite_positive;
    } else {
      return emitStagedDiagnostic(op, "math dialect operation");
    }

    Type resultType = op->getResult(0).getType();
    if (!resultType.isF32() && !isVector16F32(resultType))
      return emitStagedDiagnostic(op, "approximate SFU math result type");
    if (failed(requireApproxSFUPolicy(op, kind)))
      return failure();

    FailureOr<Value> input = getF32FragmentOperand(op, op->getOperand(0));
    if (failed(input))
      return failure();
    Value result = createApproxSFU(builder, op->getLoc(), *input, kind, domain);
    mapF32FragmentResult(op->getResult(0), result);
    return success();
  }

  LogicalResult lowerApproxDivF(Operation *op) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "arith.divf shape");
    Type resultType = op->getResult(0).getType();
    if (!resultType.isF32() && !isVector16F32(resultType))
      return emitStagedDiagnostic(op, "arith.divf result type");
    if (!hasApproxMathPolicy(op))
      return op->emitOpError()
             << "generic division without approximate reciprocal policy is "
                "staged. READY_FOR_TRITON remains NO";
    if (failed(requireApproxSFUPolicy(op, mlir::vc4kernel::SFUKind::recip)))
      return failure();

    FailureOr<Value> lhs = getF32FragmentOperand(op, op->getOperand(0));
    FailureOr<Value> rhs = getF32FragmentOperand(op, op->getOperand(1));
    if (failed(lhs) || failed(rhs))
      return failure();
    Value recip = createApproxSFU(builder, op->getLoc(), *rhs,
                                  mlir::vc4kernel::SFUKind::recip,
                                  mlir::vc4kernel::FPDomain::finite_nonzero);
    Value result = createMulPipe(builder, op->getLoc(), {*lhs, recip},
                                 mlir::vc4kernel::MulALUOpcode::fmul,
                                 getVector16F32(ctx));
    mapF32FragmentResult(op->getResult(0), result);
    return success();
  }

  LogicalResult lowerBinaryArith(Operation *op) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "binary arith shape");

    Type sourceType = op->getResult(0).getType();
    Type resultType = lowerValueType(sourceType, ctx);
    StringRef name = op->getName().getStringRef();

    if (hasF16Type(sourceType))
      return op->emitOpError()
             << "native f16 arithmetic is staged. READY_FOR_TRITON remains NO";

    if ((name == "arith.mulf" || name == "arith.muli") &&
        isDotCompositeProduct(op) && !isVector16I32(sourceType) &&
        !isVector16F32(sourceType))
      return op->emitOpError("unsupported GEMV element type");

    if (name == "arith.addf" && resultType.isF32() &&
        ((isDotCompositeReductionValue(op->getOperand(0)) &&
          isNonEntryBlockArgument(op->getOperand(1))) ||
         (isDotCompositeReductionValue(op->getOperand(1)) &&
          isNonEntryBlockArgument(op->getOperand(0)))))
      return op->emitOpError("multi-block K accumulation is staged");

    if (name == "arith.addf" && resultType.isF32()) {
      Value reductionOperand;
      Value scalarOperand;
      if (isDotCompositeReductionValue(op->getOperand(0))) {
        reductionOperand = op->getOperand(0);
        scalarOperand = op->getOperand(1);
      } else if (isDotCompositeReductionValue(op->getOperand(1))) {
        reductionOperand = op->getOperand(1);
        scalarOperand = op->getOperand(0);
      }
      if (reductionOperand) {
        Value reductionFragment = lookupReductionFragment(reductionOperand);
        if (!reductionFragment)
          return failure();
        Value scalar = lookupValue(op, scalarOperand);
        if (!scalar)
          return failure();
        Type fragmentType = VectorType::get({16}, resultType);
        Value scalarFragment =
            createSplat(builder, op->getLoc(), scalar, fragmentType);
        state.reductionFragments[op->getResult(0)] = createAddPipe(
            builder, op->getLoc(), {reductionFragment, scalarFragment},
            mlir::vc4kernel::AddALUOpcode::fadd, fragmentType);
        return success();
      }
    }

    Value lhs = lookupValue(op, op->getOperand(0));
    Value rhs = lookupValue(op, op->getOperand(1));
    if (!lhs || !rhs)
      return failure();

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
        if (isDotCompositeProduct(op))
          state.i32DotProducts.insert(op->getResult(0));
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

  bool hasFiniteReductionPolicy(Operation *op) {
    return (hasStringAttr(op, kFPDomainAttr, "finite") ||
            hasStringAttr(func.getOperation(), kFPDomainAttr, "finite")) &&
           (hasStringAttr(op, kReductionPolicyAttr, "finite_tree") ||
            hasStringAttr(func.getOperation(), kReductionPolicyAttr,
                          "finite_tree"));
  }

  static std::optional<mlir::vc4kernel::ReduceKind>
  mapVectorReductionKind(vector::CombiningKind kind, Type elementType) {
    switch (kind) {
    case vector::CombiningKind::ADD:
      return mlir::vc4kernel::ReduceKind::add;
    case vector::CombiningKind::MAXNUMF:
    case vector::CombiningKind::MAXIMUMF:
      if (elementType.isF32())
        return mlir::vc4kernel::ReduceKind::fmax;
      return std::nullopt;
    default:
      return std::nullopt;
    }
  }

  LogicalResult lowerVectorReduction(vector::ReductionOp reduction) {
    VectorType sourceType = reduction.getSourceVectorType();
    Type elementType = sourceType.getElementType();
    Type resultType = reduction.getDest().getType();

    std::optional<mlir::vc4kernel::ReduceKind> reduceKind =
        mapVectorReductionKind(reduction.getKind(), elementType);
    if (!reduceKind)
      return reduction.emitOpError("unsupported reduction variant is staged");
    if (sourceType.getRank() != 1 || sourceType.getDimSize(0) != 16)
      return reduction.emitOpError("rank>1 reduction is staged");
    if (reduction.getAcc())
      return reduction.emitOpError("accumulator vector.reduction is staged");

    if (!elementType.isSignlessInteger(32) && !elementType.isF32())
      return isDotCompositeProduct(reduction.getVector().getDefiningOp())
                 ? reduction.emitOpError("unsupported GEMV element type")
                 : reduction.emitOpError("unsupported reduction element type");
    if (resultType != elementType)
      return isDotCompositeProduct(reduction.getVector().getDefiningOp())
                 ? reduction.emitOpError("unsupported GEMV element type")
                 : reduction.emitOpError("unsupported reduction element type");
    if (elementType.isF32() && !hasFiniteReductionPolicy(reduction))
      return isDotCompositeProduct(reduction.getVector().getDefiningOp())
                 ? reduction.emitOpError(
                       "exact f32 dot requires unsupported exact reduction policy")
                 : reduction.emitOpError(
                       "f32 vector.reduction requires explicit finite-tree policy");
    if (elementType.isSignlessInteger(32) &&
        state.i32DotProducts.contains(reduction.getVector()))
      return reduction.emitOpError("i32 dot is staged by current i32 multiply policy");

    Value input = lookupValue(reduction, reduction.getVector());
    if (!input)
      return failure();
    Value pred = createPredFull(builder, reduction.getLoc());

    SmallVector<NamedAttribute, 2> attrs;
    attrs.push_back(builder.getNamedAttr(
        "kind", mlir::vc4kernel::ReduceKindAttr::get(
                    ctx, *reduceKind)));
    if (elementType.isF32()) {
      attrs.push_back(builder.getNamedAttr(
          "fp_policy", mlir::vc4kernel::FPReducePolicyAttr::get(
                           ctx, mlir::vc4kernel::FPReducePolicy::finite_tree)));
    }

    Value fragment =
        createOpWithResult(builder, reduction.getLoc(), kFragmentReduceOpName,
                           {input, pred}, attrs, input.getType());
    state.reductionFragments[reduction.getResult()] = fragment;
    return success();
  }

  LogicalResult lowerMemRefDim(memref::DimOp dimOp) {
    Value source = dimOp.getSource();
    auto memrefType = llvm::dyn_cast<MemRefType>(source.getType());
    if (!memrefType || !hasGlobalMemorySpace(memrefType)) {
      return dimOp.emitOpError()
             << "memref.dim hidden descriptor ABI is rejected; source must be "
             << "a public #vc4value.global memref argument. READY_FOR_TRITON "
             << "remains NO";
    }
    std::optional<unsigned> argIndex = getPublicMemRefArgIndex(func, source);
    if (!argIndex) {
      return dimOp.emitOpError()
             << "memref.dim hidden descriptor ABI is rejected; source must be "
             << "a public #vc4value.global memref argument. READY_FOR_TRITON "
             << "remains NO";
    }

    std::optional<int64_t> maybeDim = getIntegerConstant(dimOp.getIndex());
    if (!maybeDim) {
      return dimOp.emitOpError()
             << "memref.dim hidden descriptor ABI is rejected; dimension index "
             << "must be constant for metadata lowering. READY_FOR_TRITON "
             << "remains NO";
    }
    int64_t dim = *maybeDim;
    if (dim < 0 || dim >= memrefType.getRank()) {
      return dimOp.emitOpError()
             << "memref.dim hidden descriptor ABI is rejected; dimension index "
             << "is out of range. READY_FOR_TRITON remains NO";
    }

    std::optional<unsigned> ordinal =
        getDynamicDimOrdinal(memrefType, static_cast<unsigned>(dim));
    if (!ordinal) {
      state.values[dimOp.getResult()] =
          createI32Constant(builder, dimOp.getLoc(), memrefType.getDimSize(dim));
      return success();
    }

    FailureOr<Value> extent = lookupShapeArgForDim(
        dimOp.getOperation(), source, static_cast<unsigned>(dim),
        "memref.dim requires explicit vc4value.shape_args metadata");
    if (failed(extent))
      return failure();
    state.values[dimOp.getResult()] = *extent;
    return success();
  }

  ClassifiedMemoryMask classifyTransferMask(Operation *op,
                                            std::optional<Value> maskValue) {
    if (!maskValue)
      return makeFullMask(op);

    auto it = state.memoryMasks.find(*maskValue);
    if (it != state.memoryMasks.end())
      return it->second;

    ClassifiedMemoryMask mask;
    mask.kind = MemoryMaskKind::SparseOrUnknown;
    mask.source = (*maskValue).getDefiningOp();
    mask.reason = "sparse or unknown transfer mask";
    return mask;
  }

  FailureOr<Value> lookupScalarArgByName(Operation *op, StringRef argName) {
    Block &entry = func.getBody().front();
    for (BlockArgument arg : entry.getArguments()) {
      unsigned index = arg.getArgNumber();
      StringAttr name = func.getArgAttrOfType<StringAttr>(index, kArgNameAttr);
      if (!name || name.getValue() != argName)
        continue;
    if (!isScalarI32OrIndex(arg.getType()))
      return op->emitOpError()
             << "metadata scalar argument '" << argName
             << "' must be index or i32 for Phase 11 memory lowering. "
             << "READY_FOR_TRITON remains NO";
      Value lowered = lookupValue(op, arg);
      if (!lowered)
        return failure();
      return lowered;
    }
    return op->emitOpError()
           << "metadata scalar argument '" << argName
           << "' was not found for Phase 11 memory lowering. "
           << "READY_FOR_TRITON remains NO";
  }

  FailureOr<Value> lookupShapeArgForDim(Operation *op, Value memref,
                                        unsigned dim, StringRef diagnostic) {
    auto memrefType = llvm::cast<MemRefType>(memref.getType());
    std::optional<unsigned> argIndex = getPublicMemRefArgIndex(func, memref);
    if (!argIndex)
      return op->emitOpError()
             << "hidden memref descriptor ABI is rejected; Phase 11 memory "
             << "lowering requires public #vc4value.global memref arguments. "
             << "READY_FOR_TRITON remains NO";
    std::optional<unsigned> ordinal = getDynamicDimOrdinal(memrefType, dim);
    if (!ordinal) {
      int64_t staticDim = memrefType.getDimSize(dim);
      return createI32Constant(builder, op->getLoc(), staticDim);
    }
    Attribute attr = func.getArgAttr(*argIndex, kShapeArgsAttr);
    if (!isArrayOfStringAttr(attr))
      return op->emitOpError()
             << diagnostic << ". READY_FOR_TRITON remains NO";
    SmallVector<StringRef> names = getStringArrayValues(attr);
    if (*ordinal >= names.size() || names[*ordinal].empty())
      return op->emitOpError()
             << diagnostic << ". READY_FOR_TRITON remains NO";
    return lookupScalarArgByName(op, names[*ordinal]);
  }

  FailureOr<Value> lookupOuterStrideArg(Operation *op, Value memref) {
    std::optional<unsigned> argIndex = getPublicMemRefArgIndex(func, memref);
    if (!argIndex)
      return op->emitOpError()
             << "hidden memref descriptor ABI is rejected; Phase 11 memory "
             << "lowering requires public #vc4value.global memref arguments. "
             << "READY_FOR_TRITON remains NO";
    Attribute attr = func.getArgAttr(*argIndex, kStrideArgsAttr);
    if (!isArrayOfStringAttr(attr))
      return op->emitOpError()
             << "rank-2 strided row-slice requires explicit "
             << "vc4value.stride_args metadata. READY_FOR_TRITON remains NO";
    SmallVector<StringRef> names = getStringArrayValues(attr);
    if (names.size() != 1 || names.front().empty())
      return op->emitOpError()
             << "rank-2 strided row-slice requires exactly one outer "
             << "vc4value.stride_args metadata entry. READY_FOR_TRITON "
             << "remains NO";
    return lookupScalarArgByName(op, names.front());
  }

  LogicalResult diagnoseTransferMemref(Operation *op, Value memref,
                                       StringRef accessName) {
    auto memrefType = llvm::dyn_cast<MemRefType>(memref.getType());
    if (!memrefType) {
      return emitStagedDiagnostic(
          op, Twine("ranked or strided memory beyond Phase 10; ") +
                  accessName +
                  " memref outside rank-1 contiguous i32/f32 #vc4value.global");
    }

    Type elementType = memrefType.getElementType();
    if (isBF16OrFP8StorageElement(elementType))
      return emitStagedDiagnostic(op, "bf16/fp8 storage is staged. "
                                      "READY_FOR_TRITON remains NO");
    if (isQuantizedSubwordStorageElement(elementType))
      return emitStagedDiagnostic(op, "int8/int16 quantized storage is staged. "
                                      "READY_FOR_TRITON remains NO");

    if (!isPhase14StorageNumericGlobalMemref(memref.getType())) {
      return emitStagedDiagnostic(
          op, Twine("memory form outside Phase 11 row-slice skeletons; ") +
                  accessName +
                  " memref outside rank-1 contiguous or rank-2 row-slice "
                  "i32/f32/f16 #vc4value.global");
    }

    if (!isI32F32OrF16(elementType)) {
      return emitStagedDiagnostic(
          op, Twine("unsupported transfer element type; ") + accessName +
                  " memref outside rank-1 contiguous or rank-2 row-slice "
                  "i32/f32/f16 #vc4value.global");
    }
    return success();
  }

  LogicalResult diagnoseVectorType(Operation *op, Type vectorType,
                                   StringRef accessName) {
    if (isVector16I32(vectorType) || isVector16F32(vectorType) ||
        isVector16F16(vectorType))
      return success();
    if (auto typedVector = llvm::dyn_cast<VectorType>(vectorType)) {
      Type elementType = typedVector.getElementType();
      if (isBF16OrFP8StorageElement(elementType))
        return emitStagedDiagnostic(op, "bf16/fp8 storage is staged. "
                                        "READY_FOR_TRITON remains NO");
      if (isQuantizedSubwordStorageElement(elementType))
        return emitStagedDiagnostic(
            op, "int8/int16 quantized storage is staged. "
                "READY_FOR_TRITON remains NO");
    }
    if (accessName == "transfer_read")
      return emitStagedDiagnostic(
          op, "unsupported transfer element type; transfer_read result type "
              "outside vector<16xi32/f32/f16>");
    return emitStagedDiagnostic(
        op, "unsupported transfer element type; transfer_write value outside "
            "vector<16xi32/f32/f16>");
  }

  LogicalResult checkTransferElementMatch(Operation *op, Value memref,
                                          Type vectorType) {
    auto memrefType = llvm::dyn_cast<MemRefType>(memref.getType());
    auto typedVector = llvm::dyn_cast<VectorType>(vectorType);
    if (!memrefType || !typedVector)
      return success();
    if (memrefType.getElementType() != typedVector.getElementType()) {
      return emitStagedDiagnostic(
          op, "unsupported transfer element type; memref element type and "
              "vector element type must match");
    }
    return success();
  }

  Type getTransferCarrierType(Type vectorType) {
    if (isVector16F16(vectorType))
      return getVector16I32(ctx);
    return lowerValueType(vectorType, ctx);
  }

  bool hasFiniteF16StoragePolicy(Operation *op) {
    return hasStringAttr(op, kF16StoragePolicyAttr, "finite") ||
           hasStringAttr(func.getOperation(), kF16StoragePolicyAttr, "finite");
  }

  FailureOr<MemoryAddressPlan>
  buildMemoryAddressPlan(Operation *op, Value memref, ValueRange indices,
                         const ClassifiedMemoryMask &mask) {
    MemoryAddressPlan plan;
    plan.source = op;
    plan.mask = mask;
    plan.reason = "staged memory form";

    auto memrefType = llvm::dyn_cast<MemRefType>(memref.getType());
    if (!memrefType)
      return emitStagedDiagnostic(op, "transfer memref must be ranked");

    Value basePointer = lookupValue(op, memref);
    if (!basePointer)
      return failure();
    plan.basePointer = basePointer;

    if (indices.size() != static_cast<size_t>(memrefType.getRank()))
      return emitStagedDiagnostic(op, "transfer rank/index count mismatch");
    for (Value index : indices) {
      if (!isScalarI32OrIndex(index.getType()))
        return emitStagedDiagnostic(op, "transfer base index must be scalar");
    }

    if (memrefType.getRank() == 1) {
      bool identityMap = llvm::isa<vector::TransferReadOp>(op)
                             ? isRank1IdentityTransferMap(op)
                             : isRank1IdentityTransferWriteMap(op);
      if (!identityMap) {
        StringRef accessName = llvm::isa<vector::TransferReadOp>(op)
                                   ? "transfer_read"
                                   : "transfer_write";
        return emitStagedDiagnostic(
            op, Twine(accessName) +
                    " permutation map beyond rank-1 identity");
      }
      Value loweredIndex = lookupValue(op, indices.front());
      if (!loweredIndex)
        return failure();
      plan.kind = MemoryAddressPlan::Kind::Rank1ContiguousOrScalarComputed;
      plan.elementBaseIndex = loweredIndex;
      plan.elementBytes = memrefType.getElementType().isF16() ? 2 : 4;
      plan.f16Storage = memrefType.getElementType().isF16();
      plan.reason = "rank-1 scalar-computed contiguous lane transfer";
      return plan;
    }

    if (memrefType.getRank() != 2)
      return emitStagedDiagnostic(op, "memref rank greater than 2");

    if (!isRank2InnermostRowSliceTransferMap(op)) {
      return emitStagedDiagnostic(
          op,
          "rank-2 row-slice transfer map must project the innermost dimension "
          "to vector lanes; column slices, transposes, and gather-like maps are "
          "staged");
    }

    Value row = lookupValue(op, indices[0]);
    Value col = lookupValue(op, indices[1]);
    if (!row || !col)
      return failure();

    Value stride;
    if (memrefType.getLayout().isIdentity()) {
      FailureOr<Value> cols = lookupShapeArgForDim(
          op, memref, 1,
          "rank-2 row-slice requires explicit vc4value.shape_args metadata");
      if (failed(cols))
        return failure();
      stride = *cols;
      plan.kind = MemoryAddressPlan::Kind::Rank2RowSliceIdentity;
      plan.reason = "rank-2 identity row-major row-slice transfer";
    } else {
      FailureOr<Value> outerStride = lookupOuterStrideArg(op, memref);
      if (failed(outerStride))
        return failure();
      stride = *outerStride;
      plan.kind = MemoryAddressPlan::Kind::Rank2RowSliceStridedOuterDynamic;
      plan.reason = "rank-2 dynamic outer stride row-slice transfer";
    }

    Value rowBase = builder.create<arith::MulIOp>(op->getLoc(), row, stride);
    plan.elementBaseIndex =
        builder.create<arith::AddIOp>(op->getLoc(), rowBase, col);
    plan.elementBytes = memrefType.getElementType().isF16() ? 2 : 4;
    plan.f16Storage = memrefType.getElementType().isF16();
    return plan;
  }

  FailureOr<ClassifiedMemoryTransfer>
  classifyTransferRead(Operation *op) {
    ClassifiedMemoryTransfer transfer;
    transfer.access = TransferAccessKind::Read;

    auto read = llvm::dyn_cast<vector::TransferReadOp>(op);
    if (!read)
      return emitPhase5Diagnostic(op, "transfer_read op kind");
    if (op->getNumResults() != 1)
      return emitPhase5Diagnostic(op, "transfer_read result count");
    transfer.vectorType = op->getResult(0).getType();
    if (failed(diagnoseVectorType(op, transfer.vectorType, "transfer_read")))
      return failure();
    transfer.storageCarrierType = getTransferCarrierType(transfer.vectorType);

    Value memref = read.getBase();
    if (failed(diagnoseTransferMemref(op, memref, "transfer_read")))
      return failure();
    if (failed(checkTransferElementMatch(op, memref, transfer.vectorType)))
      return failure();

    transfer.memref = memref;
    transfer.padding = read.getPadding();
    if (!isZeroConstant(transfer.padding)) {
      return emitPhase5Diagnostic(
          op, "transfer_read nonzero padding value; transfer_read padding must "
              "be zero for inactive_load<zero>");
    }

    std::optional<Value> maskValue;
    if (read.getMask())
      maskValue = read.getMask();
    ClassifiedMemoryMask mask = classifyTransferMask(op, maskValue);
    if (mask.kind == MemoryMaskKind::SparseOrUnknown)
      return emitStagedDiagnostic(op, "sparse or unknown transfer_read mask");
    if (mask.kind == MemoryMaskKind::Unsupported)
      return emitStagedDiagnostic(op, "unsupported transfer_read mask");
    FailureOr<MemoryAddressPlan> plan =
        buildMemoryAddressPlan(op, memref, read.getIndices(), mask);
    if (failed(plan))
      return failure();
    transfer.addressPlan = *plan;
    transfer.acceptedPath = "tmu safe-offset inactive-zero";
    return transfer;
  }

  FailureOr<ClassifiedMemoryTransfer>
  classifyTransferWrite(Operation *op) {
    ClassifiedMemoryTransfer transfer;
    transfer.access = TransferAccessKind::Write;

    auto write = llvm::dyn_cast<vector::TransferWriteOp>(op);
    if (!write)
      return emitPhase5Diagnostic(op, "transfer_write op kind");
    transfer.vectorValue = write.getVector();
    transfer.vectorType = transfer.vectorValue.getType();
    if (failed(diagnoseVectorType(op, transfer.vectorType, "transfer_write")))
      return failure();
    transfer.storageCarrierType = getTransferCarrierType(transfer.vectorType);

    Value memref = write.getBase();
    if (failed(diagnoseTransferMemref(op, memref, "transfer_write")))
      return failure();
    if (failed(checkTransferElementMatch(op, memref, transfer.vectorType)))
      return failure();
    if (isVector16F16(transfer.vectorType) && !hasFiniteF16StoragePolicy(op))
      return op->emitOpError()
             << "f16 storage store requires explicit finite storage policy. "
             << "READY_FOR_TRITON remains NO";

    transfer.memref = memref;

    std::optional<Value> maskValue;
    if (write.getMask())
      maskValue = write.getMask();
    ClassifiedMemoryMask mask = classifyTransferMask(op, maskValue);
    if (mask.kind == MemoryMaskKind::SparseOrUnknown)
      return emitStagedDiagnostic(op, "sparse or unknown transfer_write mask");
    if (mask.kind == MemoryMaskKind::Unsupported)
      return emitStagedDiagnostic(op, "unsupported transfer_write mask");
    FailureOr<MemoryAddressPlan> plan =
        buildMemoryAddressPlan(op, memref, write.getIndices(), mask);
    if (failed(plan))
      return failure();
    transfer.addressPlan = *plan;
    transfer.acceptedPath = "vdw preserve inactive-store";
    return transfer;
  }

  FailureOr<Value> createByteOffsets(Operation *op, Value loweredElemIndex,
                                     int64_t elemBytes,
                                     std::optional<Value> maskPred) {
    if (!loweredElemIndex.getType().isSignlessInteger(32))
      return op->emitOpError("transfer index must lower to scalar i32");

    Value byteBase = loweredElemIndex;
    if (elemBytes != 1) {
      int64_t shift = elemBytes == 2 ? 1 : elemBytes == 4 ? 2 : -1;
      if (shift < 0)
        return op->emitOpError("unsupported element byte width");
      byteBase = builder.create<arith::ShLIOp>(op->getLoc(), loweredElemIndex,
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

  FailureOr<Value> createScalarByteOffset(Operation *op, Value loweredElemIndex,
                                          int64_t elemBytes) {
    if (!loweredElemIndex.getType().isSignlessInteger(32))
      return op->emitOpError("transfer index must lower to scalar i32");
    if (elemBytes == 1)
      return loweredElemIndex;
    int64_t shift = elemBytes == 2 ? 1 : elemBytes == 4 ? 2 : -1;
    if (shift < 0)
      return op->emitOpError("unsupported element byte width");
    return builder.create<arith::ShLIOp>(
        op->getLoc(), loweredElemIndex,
        createI32Constant(builder, op->getLoc(), shift))
        .getResult();
  }

  Value createPhase14F16VPMTile(Location loc) {
    return createOpWithResult(
        builder, loc, kVPMAllocOpName, {},
        {builder.getNamedAttr("rows", builder.getI32IntegerAttr(1)),
         builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4))},
        mlir::vc4kernel::VPMTileType::get(ctx));
  }

  LogicalResult lowerF16TransferRead(Operation *op,
                                     const ClassifiedMemoryTransfer &transfer) {
    const MemoryAddressPlan &plan = transfer.addressPlan;
    FailureOr<Value> byteOffset =
        createScalarByteOffset(op, plan.elementBaseIndex, /*elemBytes=*/2);
    if (failed(byteOffset))
      return failure();

    Location loc = op->getLoc();
    Value tile = createPhase14F16VPMTile(loc);
    Value row = createI32Constant(builder, loc, 0);
    Value one = createI32Constant(builder, loc, 1);
    Value pitch = createI32Constant(builder, loc, 32);

    SmallVector<NamedAttribute, 10> vdrAttrs =
        getPackedW16VPMAttrs(builder, "vdr", "dma", /*dmaAccess=*/true);
    vdrAttrs.push_back(builder.getNamedAttr("max_rows",
                                            builder.getI32IntegerAttr(1)));
    vdrAttrs.push_back(builder.getNamedAttr("max_cols",
                                            builder.getI32IntegerAttr(16)));
    vdrAttrs.push_back(builder.getNamedAttr("elem_bytes",
                                            builder.getI32IntegerAttr(2)));
    vdrAttrs.push_back(builder.getNamedAttr("dst_x",
                                            builder.getI32IntegerAttr(0)));
    vdrAttrs.push_back(
        getOperandSegmentSizesAttr(builder, {1, 1, 1, 1, 1, 1, 1, 0, 0}));
    createOp(builder, loc, kVDRLoadRectToVPMOpName,
             {plan.basePointer, *byteOffset, tile, row, one, plan.mask.count,
              pitch},
             vdrAttrs);

    SmallVector<NamedAttribute, 8> readAttrs =
        getPackedW16VPMAttrs(builder, "vpm", "vpm", /*dmaAccess=*/false);
    readAttrs.push_back(
        getOperandSegmentSizesAttr(builder, {1, 1, 1, 0, 0}));
    Value raw = createOpWithResult(builder, loc, kVPMReadFragmentOpName,
                                   {tile, row, plan.mask.predicate}, readAttrs,
                                   getVector16I32(ctx));
    state.values[op->getResult(0)] = raw;
    return success();
  }

  LogicalResult lowerF16TransferWrite(Operation *op,
                                      const ClassifiedMemoryTransfer &transfer,
                                      Value packedValue) {
    const MemoryAddressPlan &plan = transfer.addressPlan;
    FailureOr<Value> byteOffset =
        createScalarByteOffset(op, plan.elementBaseIndex, /*elemBytes=*/2);
    if (failed(byteOffset))
      return failure();

    Location loc = op->getLoc();
    Value tile = createPhase14F16VPMTile(loc);
    Value row = createI32Constant(builder, loc, 0);
    Value one = createI32Constant(builder, loc, 1);
    Value pitch = createI32Constant(builder, loc, 32);

    SmallVector<NamedAttribute, 8> writeAttrs =
        getPackedW16VPMAttrs(builder, "vpm", "vpm", /*dmaAccess=*/false);
    writeAttrs.push_back(
        getOperandSegmentSizesAttr(builder, {1, 1, 1, 1, 0, 0}));
    createOp(builder, loc, kVPMWriteFragmentOpName,
             {tile, row, packedValue, plan.mask.predicate}, writeAttrs);

    SmallVector<NamedAttribute, 11> vdwAttrs =
        getPackedW16VPMAttrs(builder, "vdw", "dma", /*dmaAccess=*/true);
    vdwAttrs.push_back(builder.getNamedAttr("max_rows",
                                            builder.getI32IntegerAttr(1)));
    vdwAttrs.push_back(builder.getNamedAttr("max_cols",
                                            builder.getI32IntegerAttr(16)));
    vdwAttrs.push_back(builder.getNamedAttr("elem_bytes",
                                            builder.getI32IntegerAttr(2)));
    vdwAttrs.push_back(builder.getNamedAttr("src_x",
                                            builder.getI32IntegerAttr(0)));
    vdwAttrs.push_back(builder.getNamedAttr(
        "inactive_store", mlir::vc4kernel::InactiveStoreAttr::get(
                              ctx, mlir::vc4kernel::InactiveStore::preserve)));
    vdwAttrs.push_back(
        getOperandSegmentSizesAttr(builder, {1, 1, 1, 1, 1, 1, 1, 0, 0}));
    createOp(builder, loc, kVDWStoreRectFromVPMOpName,
             {tile, row, plan.basePointer, *byteOffset, one, plan.mask.count,
              pitch},
             vdwAttrs);
    return success();
  }

  LogicalResult lowerTransferRead(Operation *op) {
    FailureOr<ClassifiedMemoryTransfer> legality = classifyTransferRead(op);
    if (failed(legality))
      return failure();

    if (legality->addressPlan.f16Storage)
      return lowerF16TransferRead(op, *legality);

    const MemoryAddressPlan &plan = legality->addressPlan;
    std::optional<Value> offsetMask;
    if (plan.mask.kind != MemoryMaskKind::Full)
      offsetMask = plan.mask.predicate;
    FailureOr<Value> byteOffsets =
        createByteOffsets(op, plan.elementBaseIndex, plan.elementBytes,
                          offsetMask);
    if (failed(byteOffsets))
      return failure();
    Value safeOffset = createI32Constant(builder, op->getLoc(), 0);
    Value result = createOpWithResult(
        builder, op->getLoc(), kTMULoadFragmentOpName,
        {plan.basePointer, *byteOffsets, plan.mask.predicate, safeOffset},
        {builder.getNamedAttr("memory_path", mlir::vc4kernel::MemoryPathAttr::get(
                                                ctx, mlir::vc4kernel::MemoryPath::tmu_global_read)),
         builder.getNamedAttr("coherency", mlir::vc4kernel::CoherencyAttr::get(
                                             ctx, mlir::vc4kernel::Coherency::readonly_tmu)),
         builder.getNamedAttr("inactive_load", mlir::vc4kernel::InactiveLoadAttr::get(
                                                 ctx, mlir::vc4kernel::InactiveLoad::zero))},
        legality->storageCarrierType);
    state.values[op->getResult(0)] = result;
    return success();
  }

  LogicalResult lowerTransferWrite(Operation *op) {
    FailureOr<ClassifiedMemoryTransfer> legality = classifyTransferWrite(op);
    if (failed(legality))
      return failure();

    Value value = lookupValue(op, legality->vectorValue);
    if (!value)
      return failure();
    if (value.getType() != legality->storageCarrierType)
      return op->emitOpError()
             << "transfer_write storage carrier type mismatch. "
             << "READY_FOR_TRITON remains NO";

    if (legality->addressPlan.f16Storage)
      return lowerF16TransferWrite(op, *legality, value);

    const MemoryAddressPlan &plan = legality->addressPlan;
    // Do not poison store offsets.  Inactive preservation is the VDW policy;
    // sparse/unknown masks must be rejected before reaching this path.
    FailureOr<Value> byteOffsets =
        createByteOffsets(op, plan.elementBaseIndex, plan.elementBytes,
                          std::nullopt);
    if (failed(byteOffsets))
      return failure();
    createOp(builder, op->getLoc(), kVDWStoreFragmentOpName,
             {plan.basePointer, *byteOffsets, value, plan.mask.predicate},
             {builder.getNamedAttr("memory_path", mlir::vc4kernel::MemoryPathAttr::get(
                                                     ctx, mlir::vc4kernel::MemoryPath::vdw_global_store)),
              builder.getNamedAttr("coherency", mlir::vc4kernel::CoherencyAttr::get(
                                                  ctx, mlir::vc4kernel::Coherency::dma_ordered)),
              builder.getNamedAttr("inactive_store", mlir::vc4kernel::InactiveStoreAttr::get(
                                                   ctx, mlir::vc4kernel::InactiveStore::preserve))});
    return success();
  }

  LogicalResult diagnoseScalarStoreMemref(memref::StoreOp store,
                                          MemRefType memrefType) {
    if (!hasGlobalMemorySpace(memrefType))
      return emitStagedDiagnostic(store.getOperation(),
                                  "non-global scalar store memory space");
    if (memrefType.getRank() != 1)
      return emitStagedDiagnostic(store.getOperation(),
                                  "rank>1 scalar store is staged");
    if (!memrefType.getLayout().isIdentity())
      return emitStagedDiagnostic(store.getOperation(),
                                  "non-identity scalar store layout is staged");
    if (!isI32OrF32(memrefType.getElementType()))
      return store.emitOpError("unsupported scalar store element type");
    if (store.getIndices().size() != 1)
      return emitStagedDiagnostic(store.getOperation(),
                                  "scalar store rank/index count mismatch");
    Value index = store.getIndices().front();
    if (!isScalarI32OrIndex(index.getType()))
      return emitStagedDiagnostic(store.getOperation(),
                                  "scalar store index must be index or i32");

    std::optional<unsigned> argIndex =
        getPublicMemRefArgIndex(func, store.getMemRef());
    if (!argIndex)
      return store.emitOpError()
             << "scalar reduction-output store requires a public "
                "#vc4value.global memref argument. READY_FOR_TRITON remains NO";
    StringAttr direction =
        func.getArgAttrOfType<StringAttr>(*argIndex, kDirectionAttr);
    if (direction && direction.getValue() != "out" &&
        direction.getValue() != "inout") {
      return store.emitOpError()
             << "scalar reduction-output store requires output or inout "
                "direction metadata. READY_FOR_TRITON remains NO";
    }
    return success();
  }

  FailureOr<Value> getScalarStoreFragment(memref::StoreOp store,
                                          Type elementType) {
    Value source = store.getValueToStore();
    Value fragment = lookupReductionFragment(source);
    if (fragment)
      return fragment;

    if (!source.getType().isSignlessInteger(32) && !source.getType().isF32())
      return store.emitOpError("unsupported scalar store element type");
    if (source.getType() != elementType)
      return store.emitOpError("unsupported scalar store element type");
    Value scalar = lookupValue(store, source);
    if (!scalar)
      return failure();
    Type fragmentType =
        VectorType::get({16}, lowerValueType(elementType, ctx));
    return createSplat(builder, store.getLoc(), scalar, fragmentType);
  }

  LogicalResult lowerScalarMemrefStore(memref::StoreOp store) {
    auto memrefType = llvm::dyn_cast<MemRefType>(store.getMemRef().getType());
    if (!memrefType)
      return emitStagedDiagnostic(store.getOperation(),
                                  "scalar store memref must be ranked");
    if (failed(diagnoseScalarStoreMemref(store, memrefType)))
      return failure();
    Type elementType = memrefType.getElementType();
    if (store.getValueToStore().getType() != elementType)
      return store.emitOpError("unsupported scalar store element type");

    FailureOr<Value> storeFragment = getScalarStoreFragment(store, elementType);
    if (failed(storeFragment))
      return failure();

    Value basePointer = lookupValue(store, store.getMemRef());
    Value loweredIndex = lookupValue(store, store.getIndices().front());
    if (!basePointer || !loweredIndex)
      return failure();
    FailureOr<Value> byteOffsets =
        createByteOffsets(store, loweredIndex, /*elemBytes=*/4, std::nullopt);
    if (failed(byteOffsets))
      return failure();
    Value zero = createI32Constant(builder, store.getLoc(), 0);
    Value one = createI32Constant(builder, store.getLoc(), 1);
    Value oneLane = createPredTail(builder, store.getLoc(), zero, one);
    createOp(builder, store.getLoc(), kVDWStoreFragmentOpName,
             {basePointer, *byteOffsets, *storeFragment, oneLane},
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
  int64_t gridRank = 1;
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
                    math::MathDialect,
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
