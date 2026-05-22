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

#include "mlir/Dialect/Arith/IR/Arith.h"
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
constexpr llvm::StringLiteral kVC4TileProgramIdOpName("vc4tile.program_id");
constexpr llvm::StringLiteral kVC4TileLaneIdOpName("vc4tile.lane_id");
constexpr llvm::StringLiteral kVC4TileLaneRangeOpName("vc4tile.lane_range");
constexpr llvm::StringLiteral kVC4TileThreadIdOpName("vc4tile.thread_id");
constexpr llvm::StringLiteral kVC4TileMaskAllOpName("vc4tile.mask_all");
constexpr llvm::StringLiteral kVC4TileTailMaskOpName("vc4tile.tail_mask");

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
constexpr llvm::StringLiteral kVectorSplatOpName("vector.broadcast");

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

static DictionaryAttr buildLaunchABI(Operation *kernel, OpBuilder &builder) {
  if (DictionaryAttr carried = getCarriedLaunchABI(kernel))
    return carried;

  MLIRContext *ctx = builder.getContext();
  llvm::StringRef publicName = getPublicName(kernel);
  std::string codeSymbol = makeCIdentifier(publicName) + "_shader";
  StringAttr symName = getSymbolNameAttr(kernel);

  SmallVector<Attribute> args;
  SmallVector<Attribute> builtins;
  const bool needsLogicalRequest =
      kernelContains(kernel, kVC4TileProgramIdOpName) ||
      kernelContains(kernel, kVC4TileThreadIdOpName);

  if (needsLogicalRequest) {
    args.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("name", builder.getStringAttr("logical_request")),
        builder.getNamedAttr("kind", builder.getStringAttr("scalar")),
        builder.getNamedAttr("direction", builder.getStringAttr("by_value")),
        builder.getNamedAttr("type", builder.getStringAttr("u32")),
        builder.getNamedAttr("uniform_index", builder.getI32IntegerAttr(0)),
    }));
  } else {
    // The existing scheduled-VC4 verifier requires launch ABI uniform slots to
    // be described by dense arg/builtin indices.  Return-only kernels do not
    // consume launch uniforms, so use a harmless runtime launch-geometry
    // builtin rather than inventing a user-visible argument or weakening the
    // lower-half verifier contract.
    Attribute numQPUsKind = mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::num_qpus);
    builtins.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("name", builder.getStringAttr("num_qpus")),
        builder.getNamedAttr("kind", numQPUsKind),
        builder.getNamedAttr("materialization",
                             builder.getStringAttr("uniform_suffix")),
        builder.getNamedAttr("uniform_index", builder.getI32IntegerAttr(0)),
    }));
  }

  return builder.getDictionaryAttr({
      builder.getNamedAttr("public_name", builder.getStringAttr(publicName)),
      builder.getNamedAttr("symbol_name",
                           symName ? symName : builder.getStringAttr(publicName)),
      builder.getNamedAttr("code_symbol", builder.getStringAttr(codeSymbol)),
      builder.getNamedAttr("uniform_words_per_qpu",
                           builder.getI32IntegerAttr(1)),
      builder.getNamedAttr("args", builder.getArrayAttr(args)),
      builder.getNamedAttr("builtins", builder.getArrayAttr(builtins)),
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

static LogicalResult verifySupportedKernelShape(Operation *kernel) {
  if (kernel->getNumRegions() != 1 || kernel->getRegion(0).empty())
    return kernel->emitOpError("requires one non-empty body region");
  if (!llvm::hasSingleElement(kernel->getRegion(0))) {
    return kernel->emitOpError(
        "supports only a single-block body in this M4 lowering slice");
  }

  bool sawReturn = false;
  for (Operation &nested : kernel->getRegion(0).front()) {
    if (hasName(&nested, kVC4TileReturnOpName))
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

  addAttrIfPresent(kernel, state, "arg_attrs");
  addAttrIfPresent(kernel, state, "res_attrs");

  state.addRegion();
  moduleBuilder.setInsertionPointToEnd(&ssavc4Module->getRegion(0).front());
  Operation *func = moduleBuilder.create(state);
  func->getRegion(0).push_back(new Block());
  return func;
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

static LogicalResult lowerProgramId(Operation *op, OpBuilder &builder,
                                    llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  Type resultType = op->getResult(0).getType();
  if (!resultType.isSignlessInteger(32))
    return op->emitOpError("currently lowers only i32 program_id results");
  valueMap[op->getResult(0)] = createUniformRead(builder, op->getLoc(),
                                                 resultType, /*index=*/0);
  return success();
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
                                   llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  Type resultType = op->getResult(0).getType();
  if (!isVector16I32(resultType))
    return op->emitOpError("currently lowers only vector<16xi32> thread_id results");
  Type i32Type = builder.getI32Type();
  Value logicalRequest = createUniformRead(builder, op->getLoc(), i32Type,
                                           /*index=*/0);
  Value shiftFour = createLoadImmI32(builder, op->getLoc(), i32Type, 4);
  SmallVector<Value, 2> shiftOperands{logicalRequest, shiftFour};
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
  if (op->getNumResults() != 1)
    return op->emitOpError("expected one result");
  Value one = createLoadImmI32(builder, op->getLoc(), getVector16I32Type(builder),
                               1);
  Value zero = createLoadImmI32(builder, op->getLoc(), getVector16I32Type(builder),
                                0);
  SmallVector<Value, 2> operands{one, zero};
  valueMap[op->getResult(0)] =
      createMakeFlags(builder, op->getLoc(), operands,
                      mlir::ssavc4::FlagKind::compare);
  return success();
}

static LogicalResult lowerTailMask(Operation *op, OpBuilder &builder,
                                   llvm::DenseMap<Value, Value> &valueMap) {
  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return op->emitOpError("expected base, limit, and one result");

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

static LogicalResult lowerBodyOp(Operation *op, OpBuilder &builder,
                                 llvm::DenseMap<Value, Value> &valueMap) {
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
    return lowerProgramId(op, builder, valueMap);
  if (hasName(op, kVC4TileLaneRangeOpName))
    return lowerLaneRange(op, builder, valueMap);
  if (hasName(op, kVC4TileLaneIdOpName))
    return lowerLaneId(op, builder, valueMap);
  if (hasName(op, kVC4TileThreadIdOpName))
    return lowerThreadId(op, builder, valueMap);
  if (hasName(op, kVC4TileMaskAllOpName))
    return lowerMaskAll(op, builder, valueMap);
  if (hasName(op, kVC4TileTailMaskOpName))
    return lowerTailMask(op, builder, valueMap);

  return op->emitOpError(
      "is not supported yet by --convert-vc4tile-to-ssavc4 in this M4 slice");
}

static LogicalResult lowerKernelBody(Operation *kernel, Operation *func) {
  OpBuilder bodyBuilder(func->getContext());
  bodyBuilder.setInsertionPointToEnd(&func->getRegion(0).front());

  llvm::DenseMap<Value, Value> valueMap;
  bool sawReturn = false;

  for (Operation &nested : kernel->getRegion(0).front()) {
    if (hasName(&nested, kVC4TileReturnOpName)) {
      sawReturn = true;
      OperationState threadEndState(nested.getLoc(), kSSAVC4ThreadEndOpName);
      bodyBuilder.create(threadEndState);
      continue;
    }
    if (failed(lowerBodyOp(&nested, bodyBuilder, valueMap)))
      return failure();
  }

  if (!sawReturn)
    return kernel->emitOpError("requires a vc4tile.return terminator");
  return success();
}

struct ConvertVC4TileToSSAVC4Pass
    : public PassWrapper<ConvertVC4TileToSSAVC4Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVC4TileToSSAVC4Pass)

  StringRef getArgument() const override { return "convert-vc4tile-to-ssavc4"; }
  StringRef getDescription() const override {
    return "Lower supported VC4Tile kernels, IDs, masks, and metadata to SSAVC4";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::vector::VectorDialect,
                    mlir::vc4::VC4Dialect, mlir::ssavc4::SSAVC4Dialect,
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

std::unique_ptr<Pass> mlir::vc4::createConvertVC4TileToSSAVC4Pass() {
  return std::make_unique<ConvertVC4TileToSSAVC4Pass>();
}

void mlir::vc4::registerConvertVC4TileToSSAVC4Pass() {
  // This translation unit is linked into vc4-opt for M4, and the file-scope
  // PassRegistration below installs --convert-vc4tile-to-ssavc4.
}

static PassRegistration<ConvertVC4TileToSSAVC4Pass>
    registerVC4TileToSSAVC4Pass;
