//===- VC4Ops.cpp - VC4 dialect operations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"
#include "vc4/Dialect/VC4/IR/VC4SideEffects.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

static StringAttr getTypeAttrName(MLIRContext *context) {
  return StringAttr::get(context, "function_type");
}

static StringAttr getArgAttrsName(MLIRContext *context) {
  return StringAttr::get(context, "arg_attrs");
}

static StringAttr getResAttrsName(MLIRContext *context) {
  return StringAttr::get(context, "res_attrs");
}

static bool isI32OrVector16I32(Type type) {
  if (type.isSignlessInteger(32))
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable())
    return false;
  return vectorType.getDimSize(0) == 16 &&
         vectorType.getElementType().isSignlessInteger(32);
}

static bool isScalar32BitVC4ValueType(Type type) {
  return type.isSignlessInteger(32) || type.isF32();
}

// Structured arithmetic/value-shape SSA uses only 32-bit VC4 carrier words:
// scalar i32/f32 values or 16-lane vectors of those element types.
// pack/unpack stay in this carrier-word domain instead of materializing true
// i8/i16 storage types in the IR.
static bool isVector16Of32BitVC4ValueType(Type type) {
  auto vectorType = dyn_cast<VectorType>(type);
  if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable())
    return false;
  if (vectorType.getDimSize(0) != 16)
    return false;
  return isScalar32BitVC4ValueType(vectorType.getElementType());
}

static bool isVC4StructuredValueType(Type type) {
  return isScalar32BitVC4ValueType(type) ||
         isVector16Of32BitVC4ValueType(type);
}

static bool hasSameVC4Shape(Type lhs, Type rhs) {
  if (lhs.isSignlessInteger(32) || lhs.isF32())
    return rhs.isSignlessInteger(32) || rhs.isF32();

  auto lhsVector = dyn_cast<VectorType>(lhs);
  auto rhsVector = dyn_cast<VectorType>(rhs);
  if (!lhsVector || !rhsVector)
    return false;
  return lhsVector.getRank() == 1 && rhsVector.getRank() == 1 &&
         !lhsVector.isScalable() && !rhsVector.isScalable() &&
         lhsVector.getDimSize(0) == 16 && rhsVector.getDimSize(0) == 16;
}

static bool isStringOneOf(StringRef value, ArrayRef<StringRef> allowed) {
  for (StringRef candidate : allowed) {
    if (value == candidate)
      return true;
  }
  return false;
}

static bool isNonEmptyStringAttr(Attribute attr) {
  auto stringAttr = dyn_cast_or_null<StringAttr>(attr);
  return stringAttr && !stringAttr.getValue().empty();
}

static std::optional<int64_t> getSignlessI32AttrValue(DictionaryAttr dict,
                                                      StringRef name) {
  auto integerAttr = dyn_cast_or_null<IntegerAttr>(dict.get(name));
  if (!integerAttr || !integerAttr.getType().isSignlessInteger(32))
    return std::nullopt;
  return integerAttr.getInt();
}

static LogicalResult emitLaunchAbiError(mlir::vc4::FuncOp op,
                                        Twine message) {
  return op.emitOpError() << "\"vc4.launch_abi\" " << message;
}

static LogicalResult verifyLaunchAbiUniformIndex(mlir::vc4::FuncOp op,
                                                 DictionaryAttr dict,
                                                 StringRef entryKind,
                                                 SmallVectorImpl<int64_t>
                                                     &indices) {
  std::optional<int64_t> uniformIndex =
      getSignlessI32AttrValue(dict, "uniform_index");
  if (!uniformIndex) {
    return emitLaunchAbiError(op, Twine(entryKind) +
                                      Twine(" entry requires signless i32 "
                                            "'uniform_index'"));
  }
  if (*uniformIndex < 0) {
    return emitLaunchAbiError(
        op, Twine(entryKind) +
                Twine(" entry requires non-negative 'uniform_index'"));
  }
  indices.push_back(*uniformIndex);
  return success();
}

static LogicalResult verifyLaunchAbiArg(mlir::vc4::FuncOp op,
                                        DictionaryAttr arg,
                                        SmallVectorImpl<int64_t> &indices) {
  if (!isNonEmptyStringAttr(arg.get("name")))
    return emitLaunchAbiError(op,
                              "argument entry requires a non-empty string 'name'");

  auto kindAttr = dyn_cast_or_null<StringAttr>(arg.get("kind"));
  if (!kindAttr ||
      !isStringOneOf(kindAttr.getValue(), {"scalar", "buffer"})) {
    return emitLaunchAbiError(
        op, "argument entry requires kind = \"scalar\" or \"buffer\"");
  }

  auto directionAttr = dyn_cast_or_null<StringAttr>(arg.get("direction"));
  if (!directionAttr)
    return emitLaunchAbiError(op, "argument entry requires string 'direction'");

  if (kindAttr.getValue() == "scalar") {
    if (directionAttr.getValue() != "by_value") {
      return emitLaunchAbiError(
          op, "scalar argument entry requires direction = \"by_value\"");
    }
    auto typeAttr = dyn_cast_or_null<StringAttr>(arg.get("type"));
    if (!typeAttr ||
        !isStringOneOf(typeAttr.getValue(), {"i32", "u32", "f32", "index"})) {
      return emitLaunchAbiError(
          op, "scalar argument entry requires type = \"i32\", \"u32\", "
              "\"f32\", or \"index\"");
    }
    if (arg.get("elem_type")) {
      return emitLaunchAbiError(
          op, "scalar argument entry must not specify 'elem_type'");
    }
  } else {
    if (!isStringOneOf(directionAttr.getValue(), {"in", "out", "inout"})) {
      return emitLaunchAbiError(
          op, "buffer argument entry requires direction = \"in\", \"out\", "
              "or \"inout\"");
    }
    auto elemTypeAttr = dyn_cast_or_null<StringAttr>(arg.get("elem_type"));
    if (!elemTypeAttr ||
        !isStringOneOf(elemTypeAttr.getValue(),
                       {"i8", "u8", "i16", "u16", "i32", "u32", "f32"})) {
      return emitLaunchAbiError(
          op, "buffer argument entry requires elem_type = \"i8\", \"u8\", "
              "\"i16\", \"u16\", \"i32\", \"u32\", or \"f32\"");
    }
    if (arg.get("type"))
      return emitLaunchAbiError(op,
                                "buffer argument entry must not specify 'type'");
  }

  return verifyLaunchAbiUniformIndex(op, arg, "argument", indices);
}

static LogicalResult verifyLaunchAbiBuiltin(mlir::vc4::FuncOp op,
                                            DictionaryAttr builtin,
                                            SmallVectorImpl<int64_t> &indices) {
  if (!isNonEmptyStringAttr(builtin.get("name")))
    return emitLaunchAbiError(op,
                              "builtin entry requires a non-empty string 'name'");

  auto kindAttr =
      dyn_cast_or_null<mlir::vc4::BuiltinKindAttr>(builtin.get("kind"));
  if (!kindAttr)
    return emitLaunchAbiError(op, "builtin entry requires VC4 BuiltinKindAttr "
                                  "'kind'");

  auto materializationAttr =
      dyn_cast_or_null<StringAttr>(builtin.get("materialization"));
  if (!materializationAttr ||
      !isStringOneOf(materializationAttr.getValue(),
                     {"uniform_suffix", "register"})) {
    return emitLaunchAbiError(
        op, "builtin entry requires materialization = \"uniform_suffix\" or "
            "\"register\"");
  }

  mlir::vc4::BuiltinKind kind = kindAttr.getValue();
  StringRef materialization = materializationAttr.getValue();
  if (kind == mlir::vc4::BuiltinKind::elem_num) {
    return emitLaunchAbiError(
        op, "builtin kind #vc4.builtin_kind<elem_num> must not appear");
  }
  if (kind == mlir::vc4::BuiltinKind::num_qpus &&
      materialization != "uniform_suffix") {
    return emitLaunchAbiError(
        op, "builtin kind #vc4.builtin_kind<num_qpus> must use "
            "materialization = \"uniform_suffix\"");
  }

  if (materialization == "uniform_suffix")
    return verifyLaunchAbiUniformIndex(op, builtin, "builtin", indices);

  if (builtin.get("uniform_index")) {
    return emitLaunchAbiError(
        op, "register-materialized builtin entry must not specify "
            "'uniform_index'");
  }
  return success();
}

static LogicalResult verifyLaunchAbiUniformLayout(
    mlir::vc4::FuncOp op, int64_t uniformWordsPerQPU,
    ArrayRef<int64_t> indices) {
  if (static_cast<int64_t>(indices.size()) != uniformWordsPerQPU) {
    return emitLaunchAbiError(
        op, "uniform indices must be unique and dense in [0, "
            "uniform_words_per_qpu)");
  }

  llvm::SmallSet<int64_t, 8> seen;
  for (int64_t index : indices) {
    if (index < 0 || index >= uniformWordsPerQPU ||
        !seen.insert(index).second) {
      return emitLaunchAbiError(
          op, "uniform indices must be unique and dense in [0, "
              "uniform_words_per_qpu)");
    }
  }

  for (int64_t index = 0; index < uniformWordsPerQPU; ++index) {
    if (!seen.count(index)) {
      return emitLaunchAbiError(
          op, "uniform indices must be unique and dense in [0, "
              "uniform_words_per_qpu)");
    }
  }
  return success();
}

static LogicalResult verifyLaunchAbi(mlir::vc4::FuncOp op) {
  Attribute rawAttr = op->getAttr("vc4.launch_abi");
  if (!rawAttr)
    return success();

  auto launchAbi = dyn_cast<DictionaryAttr>(rawAttr);
  if (!launchAbi)
    return emitLaunchAbiError(op, "requires a dictionary attribute");

  if (!op.getKernelAttr()) {
    return emitLaunchAbiError(op,
                              "may appear only on vc4.func with 'kernel'");
  }
  if (!op.getDomain() || *op.getDomain() != mlir::vc4::ExecutionDomain::qpu) {
    return emitLaunchAbiError(
        op, "requires domain = #vc4.execution_domain<qpu>");
  }

  if (!isNonEmptyStringAttr(launchAbi.get("public_name"))) {
    return emitLaunchAbiError(
        op, "requires a non-empty string 'public_name'");
  }

  auto tailPolicyAttr =
      dyn_cast_or_null<StringAttr>(launchAbi.get("tail_policy"));
  if (!tailPolicyAttr ||
      !isStringOneOf(tailPolicyAttr.getValue(),
                     {"exact_multiple", "tail_safe"})) {
    return emitLaunchAbiError(
        op, "requires tail_policy = \"exact_multiple\" or \"tail_safe\"");
  }

  std::optional<int64_t> uniformWordsPerQPU =
      getSignlessI32AttrValue(launchAbi, "uniform_words_per_qpu");
  if (!uniformWordsPerQPU || *uniformWordsPerQPU <= 0) {
    return emitLaunchAbiError(
        op, "requires a positive signless i32 'uniform_words_per_qpu'");
  }

  auto argsAttr = dyn_cast_or_null<ArrayAttr>(launchAbi.get("args"));
  if (!argsAttr)
    return emitLaunchAbiError(op, "requires array 'args'");

  auto builtinsAttr = dyn_cast_or_null<ArrayAttr>(launchAbi.get("builtins"));
  if (!builtinsAttr)
    return emitLaunchAbiError(op, "requires array 'builtins'");

  SmallVector<int64_t> uniformIndices;
  for (Attribute argAttr : argsAttr) {
    auto arg = dyn_cast<DictionaryAttr>(argAttr);
    if (!arg)
      return emitLaunchAbiError(op, "argument entry must be a dictionary");
    if (failed(verifyLaunchAbiArg(op, arg, uniformIndices)))
      return failure();
  }

  for (Attribute builtinAttr : builtinsAttr) {
    auto builtin = dyn_cast<DictionaryAttr>(builtinAttr);
    if (!builtin)
      return emitLaunchAbiError(op, "builtin entry must be a dictionary");
    if (failed(verifyLaunchAbiBuiltin(op, builtin, uniformIndices)))
      return failure();
  }

  return verifyLaunchAbiUniformLayout(op, *uniformWordsPerQPU, uniformIndices);
}

static bool isVC4IntValueType(Type type) {
  if (type.isSignlessInteger(32))
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 && !vectorType.isScalable() &&
         vectorType.getDimSize(0) == 16 &&
         vectorType.getElementType().isSignlessInteger(32);
}

static bool isVC4FloatValueType(Type type) {
  if (type.isF32())
    return true;
  auto vectorType = dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 && !vectorType.isScalable() &&
         vectorType.getDimSize(0) == 16 && vectorType.getElementType().isF32();
}

static bool isScalarSignlessIntegerOrIndex(Type type) {
  return type.isSignlessIntOrIndex() && !isa<VectorType>(type);
}

static LogicalResult verifyPositiveI32Attr(Operation *op, StringRef attrName,
                                           IntegerAttr attr) {
  if (!attr)
    return success();
  if (!attr.getType().isSignlessInteger(32))
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be signless i32";
  if (attr.getInt() <= 0) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be greater than zero";
  }
  return success();
}

static LogicalResult verifyNonNegativeI32Attr(Operation *op, StringRef attrName,
                                              IntegerAttr attr) {
  if (!attr)
    return success();
  if (!attr.getType().isSignlessInteger(32))
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be signless i32";
  if (attr.getInt() < 0) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be non-negative";
  }
  return success();
}

static LogicalResult verifyStructuredFormOp(Operation *op) {
  auto func = op->getParentOfType<mlir::vc4::FuncOp>();
  if (!func)
    return op->emitOpError("must be nested in a vc4.func");
  if (!func.getForm() ||
      *func.getForm() != mlir::vc4::FunctionForm::structured)
    return op->emitOpError("is only legal in functions with form = structured");
  return success();
}

static LogicalResult verifyScheduledFormOp(Operation *op) {
  auto func = op->getParentOfType<mlir::vc4::FuncOp>();
  if (!func)
    return op->emitOpError("must be nested in a vc4.func");
  if (!func.getForm() ||
      *func.getForm() != mlir::vc4::FunctionForm::scheduled)
    return op->emitOpError("is only legal in functions with form = scheduled");
  return success();
}

static bool isVC4QPUOp(Operation &op) {
  return op.getName().getStringRef().starts_with("vc4.qpu.");
}

static bool isVC4StructuredOp(Operation &op) {
  StringRef name = op.getName().getStringRef();
  return name.starts_with("vc4.") && !name.starts_with("vc4.qpu.");
}

static bool isVC4HostDomainOnlyOp(Operation &op) {
  StringRef name = op.getName().getStringRef();
  return name == "vc4.enqueue_qpu" || name == "vc4.reserve_qpu" ||
         name == "vc4.v3d.query" || name == "vc4.v3d.configure";
}

static bool isVC4SharedStructuredDomainOp(Operation &op) {
  StringRef name = op.getName().getStringRef();
  return name == "vc4.async.wait" || name == "vc4.cf.branch" ||
         name == "vc4.return";
}

static bool isVC4QPUDomainOnlyOp(Operation &op) {
  StringRef name = op.getName().getStringRef();
  return name == "vc4.builtin" || name == "vc4.mov" || name == "vc4.read" ||
         name == "vc4.load_imm" || name == "vc4.pack" ||
         name == "vc4.unpack" || name == "vc4.rotate" ||
         name == "vc4.mutex" || name == "vc4.semaphore" ||
         name == "vc4.host_interrupt" || name == "vc4.thread_switch" ||
         name == "vc4.program_end" || name.starts_with("vc4.uniform.") ||
         name.starts_with("vc4.alu.") || name.starts_with("vc4.tmu.") ||
         name.starts_with("vc4.sfu.") || name.starts_with("vc4.vpm.") ||
         name.starts_with("vc4.dma.") || name.starts_with("vc4.qpu.");
}

static LogicalResult verifyAllOperandsAndResultAreVC4Values(Operation *op) {
  for (Type operandType : op->getOperandTypes()) {
    if (!isVC4StructuredValueType(operandType)) {
      return op->emitOpError(
          "operands must be i32, f32, vector<16xi32>, or vector<16xf32>");
    }
  }
  for (Type resultType : op->getResultTypes()) {
    if (!isVC4StructuredValueType(resultType)) {
      return op->emitOpError(
          "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
    }
  }
  return success();
}

static LogicalResult verifySameTypeOperandsAndResult(Operation *op) {
  Type resultType = op->getResult(0).getType();
  for (Type operandType : op->getOperandTypes()) {
    if (operandType != resultType) {
      return op->emitOpError(
                 "requires operand and result types to match exactly")
             << " (got operand type " << operandType << " and result type "
             << resultType << ")";
    }
  }
  return success();
}

static LogicalResult verifyBinaryALUTypes(Operation *op, bool requireFloat) {
  if (op->getNumOperands() != 2)
    return op->emitOpError("expects exactly 2 operands for this opcode");
  if (failed(verifyAllOperandsAndResultAreVC4Values(op)))
    return failure();

  Type lhsType = op->getOperand(0).getType();
  Type rhsType = op->getOperand(1).getType();
  Type resultType = op->getResult(0).getType();
  if (lhsType != rhsType || lhsType != resultType) {
    return op->emitOpError("requires both operands and the result to have the "
                           "same type");
  }

  if (requireFloat) {
    if (!isVC4FloatValueType(resultType))
      return op->emitOpError("requires f32 or vector<16xf32> types");
  } else {
    if (!isVC4IntValueType(resultType))
      return op->emitOpError("requires i32 or vector<16xi32> types");
  }
  return success();
}

static bool haveCompatibleVC4Shapes(TypeRange types) {
  if (types.empty())
    return true;
  Type firstType = types.front();
  for (Type type : types.drop_front()) {
    if (!hasSameVC4Shape(firstType, type))
      return false;
  }
  return true;
}

static std::optional<mlir::vc4::TMUMode> inferTMUModeFromDescriptor(Value value) {
  if (!value || !isa<mlir::vc4::TMUDescType>(value.getType()))
    return std::nullopt;
  if (auto descriptor = value.getDefiningOp<mlir::vc4::TMUDescriptorOp>())
    return descriptor.getMode();
  return std::nullopt;
}

static std::optional<mlir::vc4::VPMDescKind>
inferVPMDescKindFromDescriptor(Value value) {
  if (!value || !isa<mlir::vc4::VPMDescType>(value.getType()))
    return std::nullopt;
  if (auto descriptor = value.getDefiningOp<mlir::vc4::VPMDescriptorOp>())
    return descriptor.getKind();
  return std::nullopt;
}

static std::optional<mlir::vc4::DMADescKind>
inferDMADescKindFromDescriptor(Value value) {
  if (!value || !isa<mlir::vc4::DMADescType>(value.getType()))
    return std::nullopt;
  if (auto descriptor = value.getDefiningOp<mlir::vc4::DMADescriptorOp>())
    return descriptor.getKind();
  return std::nullopt;
}

using MemoryEffectList = SmallVectorImpl<MemoryEffects::EffectInstance>;

template <typename ResourceT, typename EffectT>
static void addEffect(MemoryEffectList &effects) {
  effects.emplace_back(EffectT::get(), ResourceT::get());
}

template <typename ResourceT>
static void addReadEffect(MemoryEffectList &effects) {
  addEffect<ResourceT, MemoryEffects::Read>(effects);
}

template <typename ResourceT>
static void addWriteEffect(MemoryEffectList &effects) {
  addEffect<ResourceT, MemoryEffects::Write>(effects);
}

template <typename ResourceT>
static void addReadWriteEffects(MemoryEffectList &effects) {
  addReadEffect<ResourceT>(effects);
  addWriteEffect<ResourceT>(effects);
}

static void addTMURequestEffects(mlir::vc4::TMUUnit unit,
                                 MemoryEffectList &effects) {
  switch (unit) {
  case mlir::vc4::TMUUnit::tmu0:
    addWriteEffect<mlir::vc4::effects::TMUReq0>(effects);
    break;
  case mlir::vc4::TMUUnit::tmu1:
    addWriteEffect<mlir::vc4::effects::TMUReq1>(effects);
    break;
  }
  addReadEffect<mlir::vc4::effects::MainMemory>(effects);
}

static void addTMUReadEffects(mlir::vc4::TMUUnit unit, MemoryEffectList &effects) {
  switch (unit) {
  case mlir::vc4::TMUUnit::tmu0:
    addReadEffect<mlir::vc4::effects::TMURcv0>(effects);
    break;
  case mlir::vc4::TMUUnit::tmu1:
    addReadEffect<mlir::vc4::effects::TMURcv1>(effects);
    break;
  }
}

static void addDMAQueueEffects(std::optional<mlir::vc4::DMADescKind> kind,
                               MemoryEffectList &effects, bool isWrite) {
  if (!kind || *kind == mlir::vc4::DMADescKind::load) {
    if (isWrite)
      addWriteEffect<mlir::vc4::effects::VDR>(effects);
    else
      addReadEffect<mlir::vc4::effects::VDR>(effects);
  }
  if (!kind || *kind == mlir::vc4::DMADescKind::store) {
    if (isWrite)
      addWriteEffect<mlir::vc4::effects::VDW>(effects);
    else
      addReadEffect<mlir::vc4::effects::VDW>(effects);
  }
}

static std::optional<mlir::vc4::DMADescKind> inferDMADescKindFromToken(Value token) {
  if (!token || !isa<mlir::vc4::AsyncTokenType>(token.getType()))
    return std::nullopt;
  if (auto start = token.getDefiningOp<mlir::vc4::DMAStartOp>())
    return inferDMADescKindFromDescriptor(start.getDescriptor());
  return std::nullopt;
}

static LogicalResult verifyQPUWriteAddressAttr(Operation *op, StringRef attrName,
                                               IntegerAttr attr) {
  if (!attr || !attr.getType().isSignlessInteger(32)) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be signless i32";
  }
  int64_t value = attr.getInt();
  if (value < 0 || value > 63) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be in range [0, 63]";
  }
  return success();
}

static LogicalResult verifyQPUBundleReadAddressAttr(Operation *op,
                                                    StringRef attrName,
                                                    IntegerAttr attr) {
  if (!attr || !attr.getType().isSignlessInteger(32)) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be signless i32";
  }
  int64_t value = attr.getInt();
  if (value < 0 || value > 63) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be in range [0, 63]";
  }
  return success();
}

static LogicalResult verifyQPUBranchReadAddressAttr(Operation *op,
                                                    StringRef attrName,
                                                    IntegerAttr attr) {
  if (!attr || !attr.getType().isSignlessInteger(32)) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be signless i32";
  }
  int64_t value = attr.getInt();
  if (value < 0 || value > 31) {
    return op->emitOpError() << "'" << attrName
                             << "' attribute must be in range [0, 31]";
  }
  return success();
}

static bool isQPUUpperWriteAddress(int64_t value) { return value >= 32; }

// Small-immediate selectors use the encoded hardware space directly:
// 0..47 are immediate integer/float literal selectors,
// 48 is the rotate-by-r5 selector,
// 49..63 are immediate rotate selectors.
static bool isQPUSmallImmLiteralSelector(int64_t value) {
  return value >= 0 && value <= 47;
}

static bool isQPUSmallImmRotateByR5Selector(int64_t value) {
  return value == 48;
}

static bool isQPUSmallImmImmediateRotateSelector(int64_t value) {
  return value >= 49 && value <= 63;
}

static bool isQPUSmallImmVectorRotateSelector(int64_t value) {
  return isQPUSmallImmRotateByR5Selector(value) ||
         isQPUSmallImmImmediateRotateSelector(value);
}

static bool isQPUValidSmallImmSelector(int64_t value) {
  return isQPUSmallImmLiteralSelector(value) ||
         isQPUSmallImmRotateByR5Selector(value) ||
         isQPUSmallImmImmediateRotateSelector(value);
}

static bool isQPUAccumulatorMuxR0ToR3(mlir::vc4::QPUMux mux) {
  return mux == mlir::vc4::QPUMux::r0 || mux == mlir::vc4::QPUMux::r1 ||
         mux == mlir::vc4::QPUMux::r2 || mux == mlir::vc4::QPUMux::r3;
}

// Semaphore instructions must not target closely-coupled peripheral write
// addresses that can stall the instruction stream.
static bool isQPUStallCapablePeripheralWriteAddress(int64_t value) {
  return (value >= 43 && value <= 47) || (value >= 52 && value <= 55) ||
         (value >= 56 && value <= 63);
}

static LogicalResult verifyQPUSemaWriteAddressAttr(Operation *op,
                                                   StringRef attrName,
                                                   IntegerAttr attr) {
  if (failed(verifyQPUWriteAddressAttr(op, attrName, attr)))
    return failure();

  int64_t value = attr.getInt();
  if (isQPUStallCapablePeripheralWriteAddress(value)) {
    return op->emitOpError()
           << "'" << attrName
           << "' must not target stall-capable peripheral write addresses "
              "(TLB 43..47, SFU 52..55, TMU 56..63) for vc4.qpu.sema";
  }
  return success();
}

static bool isQPUAddPipeActive(mlir::vc4::AddOpcode opcode) {
  return opcode != mlir::vc4::AddOpcode::nop;
}

static bool isQPUMulPipeActive(mlir::vc4::MulOpcode opcode) {
  return opcode != mlir::vc4::MulOpcode::nop;
}

static LogicalResult verifyQPUBundleWriteConflict(
    Operation *op, mlir::vc4::AddOpcode addOpcode,
    mlir::vc4::MulOpcode mulOpcode, IntegerAttr waddrAddAttr,
    IntegerAttr waddrMulAttr) {
  if (!isQPUAddPipeActive(addOpcode) || !isQPUMulPipeActive(mulOpcode))
    return success();

  int64_t waddrAdd = waddrAddAttr.getInt();
  int64_t waddrMul = waddrMulAttr.getInt();
  if (waddrAdd != waddrMul || !isQPUUpperWriteAddress(waddrAdd))
    return success();

  return op->emitOpError(
      "active ADD and MUL pipelines must not target the same accumulator/I/O "
      "write address");
}

static LogicalResult verifyQPUPackAttr(Operation *op, bool pm, Attribute packAttr) {
  if (!packAttr)
    return success();

  if (!pm) {
    if (!isa<mlir::vc4::RegfileAPackModeAttr>(packAttr)) {
      return op->emitOpError(
          "pm = false requires 'pack' to use #vc4.regfile_a_pack_mode");
    }
    return success();
  }

  if (!isa<mlir::vc4::MulPackModeAttr>(packAttr)) {
    return op->emitOpError(
        "pm = true requires 'pack' to use #vc4.mul_pack_mode");
  }
  return success();
}

static LogicalResult verifyQPUUnpackAttr(Operation *op, bool pm,
                                         Attribute unpackAttr) {
  if (!unpackAttr)
    return success();

  if (!pm) {
    if (!isa<mlir::vc4::RegfileAUnpackModeAttr>(unpackAttr)) {
      return op->emitOpError(
          "pm = false requires 'unpack' to use #vc4.regfile_a_unpack_mode");
    }
    return success();
  }

  if (!isa<mlir::vc4::R4UnpackModeAttr>(unpackAttr)) {
    return op->emitOpError(
        "pm = true requires 'unpack' to use #vc4.r4_unpack_mode");
  }
  return success();
}

static LogicalResult verifyQPULoadImmPayload(Operation *op,
                                             mlir::vc4::LoadImmMode mode,
                                             Attribute valueAttr) {
  switch (mode) {
  case mlir::vc4::LoadImmMode::splat32: {
    auto intAttr = dyn_cast<IntegerAttr>(valueAttr);
    if (!intAttr || !intAttr.getType().isSignlessInteger(32)) {
      return op->emitOpError(
          "splat32 mode requires a signless i32 'value' attribute");
    }
    return success();
  }
  case mlir::vc4::LoadImmMode::per_elem_i2:
  case mlir::vc4::LoadImmMode::per_elem_u2: {
    auto valuesAttr = dyn_cast<DenseI32ArrayAttr>(valueAttr);
    if (!valuesAttr) {
      return op->emitOpError(
          "per-element mode requires a dense i32 array 'value' attribute");
    }
    if (valuesAttr.asArrayRef().size() != 16)
      return op->emitOpError("per-element mode requires exactly 16 lane values");
    int32_t minValue =
        mode == mlir::vc4::LoadImmMode::per_elem_i2 ? -2 : 0;
    int32_t maxValue =
        mode == mlir::vc4::LoadImmMode::per_elem_i2 ? 1 : 3;
    for (int32_t laneValue : valuesAttr.asArrayRef()) {
      if (laneValue < minValue || laneValue > maxValue) {
        return op->emitOpError() << "lane values for mode "
                                 << mlir::vc4::stringifyLoadImmMode(mode)
                                 << " must be in range [" << minValue << ", "
                                 << maxValue << "]";
      }
    }
    return success();
  }
  }

  llvm_unreachable("unhandled vc4.qpu.ldi mode");
}

} // namespace

mlir::vc4::ModuleOp mlir::vc4::ModuleOp::create(Location loc, StringRef name) {
  OpBuilder builder(loc->getContext());
  OperationState state(loc, getOperationName());
  state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  Region *bodyRegion = state.addRegion();
  bodyRegion->push_back(new Block);
  return cast<mlir::vc4::ModuleOp>(Operation::create(state));
}

mlir::vc4::FuncOp mlir::vc4::FuncOp::create(
    Location location, StringRef name, FunctionType type,
    mlir::vc4::ThreadingMode threading, mlir::vc4::FunctionForm form,
    mlir::vc4::ExecutionDomain domain,
    ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  state.addAttribute(::mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(builder.getContext()), TypeAttr::get(type));
  state.addAttribute("threading",
                     mlir::vc4::ThreadingModeAttr::get(builder.getContext(),
                                                       threading));
  state.addAttribute("form",
                     mlir::vc4::FunctionFormAttr::get(builder.getContext(),
                                                      form));
  state.addAttribute("domain",
                     mlir::vc4::ExecutionDomainAttr::get(builder.getContext(),
                                                         domain));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
  return cast<mlir::vc4::FuncOp>(Operation::create(state));
}

ParseResult mlir::vc4::FuncOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> inputs, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag, std::string &) -> Type {
    return FunctionType::get(builder.getContext(), inputs, results);
  };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getTypeAttrName(parser.getContext()), buildFuncType,
      getArgAttrsName(parser.getContext()), getResAttrsName(parser.getContext()));
}

void mlir::vc4::FuncOp::print(OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, "function_type",
      getArgAttrsName(getContext()), getResAttrsName(getContext()));
}

ParseResult mlir::vc4::BuiltinOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  StringRef kindKeyword;
  SMLoc kindLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&kindKeyword))
    return failure();

  std::optional<mlir::vc4::BuiltinKind> kind =
      mlir::vc4::symbolizeBuiltinKind(kindKeyword);
  if (!kind)
    return parser.emitError(kindLoc)
           << "expected one of [elem_num, qpu_num, num_qpus] for vc4 builtin kind";

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Type resultType;
  if (parser.parseColonType(resultType))
    return failure();

  result.addTypes(resultType);
  result.addAttribute("kind",
                      mlir::vc4::BuiltinKindAttr::get(parser.getContext(), *kind));
  return success();
}

void mlir::vc4::BuiltinOp::print(OpAsmPrinter &printer) {
  printer << ' ' << mlir::vc4::stringifyBuiltinKind(getKind());
  printer.printOptionalAttrDict((*this)->getAttrs(), {"kind"});
  printer << " : " << getResult().getType();
}

LogicalResult mlir::vc4::ModuleOp::verify() {
  for (Operation &op : getBodyRegion().front()) {
    if (!isa<mlir::vc4::FuncOp>(op))
      return emitOpError("expects only vc4.func operations in the module body");
  }
  return success();
}

LogicalResult mlir::vc4::FuncOp::verify() {
  if (!getThreadingAttr())
    return emitOpError("requires a 'threading' attribute");
  if (!getFormAttr())
    return emitOpError("requires a 'form' attribute");
  if (!getDomainAttr())
    return emitOpError("requires a 'domain' attribute");

  mlir::vc4::FunctionForm form = *getForm();
  mlir::vc4::ExecutionDomain domain = *getDomain();
  if (getKernelAttr() && domain != mlir::vc4::ExecutionDomain::qpu) {
    return emitOpError(
        "the 'kernel' attribute is only legal with domain = #vc4.execution_domain<qpu>");
  }
  if (failed(verifyLaunchAbi(*this)))
    return failure();
  if (isExternal())
    return success();

  if (form == mlir::vc4::FunctionForm::structured) {
    for (Block &block : getBody()) {
      if (block.empty() || !block.back().mightHaveTrait<OpTrait::IsTerminator>()) {
        return emitOpError(
            "structured functions require every block to end in a terminator");
      }
    }
  }

  bool sawError = false;
  getBody().walk([&](Operation *op) {
    if (sawError)
      return WalkResult::interrupt();

    if (form == mlir::vc4::FunctionForm::structured) {
      if (domain == mlir::vc4::ExecutionDomain::qpu &&
          isVC4HostDomainOnlyOp(*op)) {
        op->emitOpError("is only legal in functions with domain = host");
        sawError = true;
        return WalkResult::interrupt();
      }
      if (domain == mlir::vc4::ExecutionDomain::host &&
          !isVC4HostDomainOnlyOp(*op) && !isVC4SharedStructuredDomainOp(*op) &&
          (isVC4QPUDomainOnlyOp(*op) || isVC4StructuredOp(*op) ||
           isVC4QPUOp(*op))) {
        op->emitOpError("is only legal in functions with domain = qpu");
        sawError = true;
        return WalkResult::interrupt();
      }
      if (isVC4QPUOp(*op)) {
        op->emitOpError("is only legal in functions with form = scheduled");
        sawError = true;
        return WalkResult::interrupt();
      }
      if (!isVC4StructuredOp(*op)) {
        op->emitOpError("is not a legal operation in functions with form = structured");
        sawError = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (domain == mlir::vc4::ExecutionDomain::host && isVC4QPUOp(*op)) {
      op->emitOpError("is only legal in functions with domain = qpu");
      sawError = true;
      return WalkResult::interrupt();
    }

    if (isVC4QPUOp(*op))
      return WalkResult::advance();

    if (isVC4StructuredOp(*op)) {
      op->emitOpError("is only legal in functions with form = structured");
      sawError = true;
      return WalkResult::interrupt();
    }

    op->emitOpError("is not a legal operation in functions with form = scheduled");
    sawError = true;
    return WalkResult::interrupt();
  });

  return failure(sawError);
}

LogicalResult mlir::vc4::ReturnOp::verify() {
  auto func = (*this)->getParentOfType<mlir::vc4::FuncOp>();
  if (!func)
    return emitOpError("must be nested in a vc4.func");
  if (!func.getForm() ||
      *func.getForm() != mlir::vc4::FunctionForm::structured)
    return emitOpError("is only legal in functions with form = structured");

  FunctionType functionType = func.getFunctionType();
  if (getNumOperands() != functionType.getNumResults())
    return emitOpError() << "expected " << functionType.getNumResults()
                         << " operands to match the enclosing function signature";

  for (auto [index, operandType, resultType] :
       llvm::zip_equal(llvm::seq<unsigned>(0, getNumOperands()),
                       getOperandTypes(), functionType.getResults())) {
    if (operandType != resultType) {
      return emitOpError() << "type of return operand #" << index << " ("
                           << operandType
                           << ") must match the enclosing function result type ("
                           << resultType << ")";
    }
  }

  return success();
}

LogicalResult mlir::vc4::BuiltinOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  if (!isI32OrVector16I32(getResult().getType()))
    return emitOpError("result type must be i32 or vector<16xi32>");
  return success();
}

void mlir::vc4::UniformReadOp::getEffects(MemoryEffectList &effects) {
  addReadEffect<mlir::vc4::effects::UniformStream>(effects);
}

void mlir::vc4::UniformSeekOp::getEffects(MemoryEffectList &effects) {
  addWriteEffect<mlir::vc4::effects::UniformStream>(effects);
}

LogicalResult mlir::vc4::UniformReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::UniformSeekOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  Type offsetType = getOffset().getType();
  if (isa<VectorType>(offsetType))
    return emitOpError("operand must be a scalar signless integer or index");
  if (!offsetType.isSignlessIntOrIndex())
    return emitOpError("operand must be a signless integer or index");
  return success();
}

LogicalResult mlir::vc4::MovOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::ReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();
  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::ALUAddOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  switch (getOp()) {
  case mlir::vc4::AddOpcode::nop:
    return emitOpError("structured vc4.alu.add does not support the nop opcode");
  case mlir::vc4::AddOpcode::fadd:
  case mlir::vc4::AddOpcode::fsub:
  case mlir::vc4::AddOpcode::fmin:
  case mlir::vc4::AddOpcode::fmax:
  case mlir::vc4::AddOpcode::fminabs:
  case mlir::vc4::AddOpcode::fmaxabs:
    return verifyBinaryALUTypes(getOperation(), /*requireFloat=*/true);
  case mlir::vc4::AddOpcode::add:
  case mlir::vc4::AddOpcode::sub:
  case mlir::vc4::AddOpcode::shr:
  case mlir::vc4::AddOpcode::asr:
  case mlir::vc4::AddOpcode::ror:
  case mlir::vc4::AddOpcode::shl:
  case mlir::vc4::AddOpcode::min:
  case mlir::vc4::AddOpcode::max:
  case mlir::vc4::AddOpcode::bit_and:
  case mlir::vc4::AddOpcode::bit_or:
  case mlir::vc4::AddOpcode::bit_xor:
  case mlir::vc4::AddOpcode::v8adds:
  case mlir::vc4::AddOpcode::v8subs:
    return verifyBinaryALUTypes(getOperation(), /*requireFloat=*/false);
  case mlir::vc4::AddOpcode::bit_not:
  case mlir::vc4::AddOpcode::clz:
    if (getNumOperands() != 1)
      return emitOpError("expects exactly 1 operand for this opcode");
    if (failed(verifyAllOperandsAndResultAreVC4Values(getOperation())))
      return failure();
    if (failed(verifySameTypeOperandsAndResult(getOperation())))
      return failure();
    if (!isVC4IntValueType(getResult().getType()))
      return emitOpError("requires i32 or vector<16xi32> types");
    return success();
  case mlir::vc4::AddOpcode::ftoi:
    if (getNumOperands() != 1)
      return emitOpError("expects exactly 1 operand for this opcode");
    if (failed(verifyAllOperandsAndResultAreVC4Values(getOperation())))
      return failure();
    if (!isVC4FloatValueType(getOperand(0).getType()))
      return emitOpError("requires an f32 or vector<16xf32> operand");
    if (!isVC4IntValueType(getResult().getType()))
      return emitOpError("requires an i32 or vector<16xi32> result");
    if (!hasSameVC4Shape(getOperand(0).getType(), getResult().getType()))
      return emitOpError("operand and result must have compatible scalar or "
                         "16-lane vector shapes");
    return success();
  case mlir::vc4::AddOpcode::itof:
    if (getNumOperands() != 1)
      return emitOpError("expects exactly 1 operand for this opcode");
    if (failed(verifyAllOperandsAndResultAreVC4Values(getOperation())))
      return failure();
    if (!isVC4IntValueType(getOperand(0).getType()))
      return emitOpError("requires an i32 or vector<16xi32> operand");
    if (!isVC4FloatValueType(getResult().getType()))
      return emitOpError("requires an f32 or vector<16xf32> result");
    if (!hasSameVC4Shape(getOperand(0).getType(), getResult().getType()))
      return emitOpError("operand and result must have compatible scalar or "
                         "16-lane vector shapes");
    return success();
  }

  llvm_unreachable("unhandled vc4.alu.add opcode");
}

LogicalResult mlir::vc4::ALUMulOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  switch (getOp()) {
  case mlir::vc4::MulOpcode::nop:
    return emitOpError("structured vc4.alu.mul does not support the nop opcode");
  case mlir::vc4::MulOpcode::fmul:
    return verifyBinaryALUTypes(getOperation(), /*requireFloat=*/true);
  case mlir::vc4::MulOpcode::mul24:
  case mlir::vc4::MulOpcode::v8muld:
  case mlir::vc4::MulOpcode::v8min:
  case mlir::vc4::MulOpcode::v8max:
  case mlir::vc4::MulOpcode::v8adds:
  case mlir::vc4::MulOpcode::v8subs:
    return verifyBinaryALUTypes(getOperation(), /*requireFloat=*/false);
  }

  llvm_unreachable("unhandled vc4.alu.mul opcode");
}

LogicalResult mlir::vc4::LoadImmOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  // Structured vc4.load_imm remains on the SSA side of the dialect even when
  // it corresponds closely to a later vc4.qpu.ldi sink form.
  Attribute valueAttr = getValueAttr();
  if (!valueAttr)
    return emitOpError("requires a 'value' attribute");

  Type resultType = getResult().getType();
  switch (getMode()) {
  case mlir::vc4::LoadImmMode::splat32: {
    auto intAttr = dyn_cast<IntegerAttr>(valueAttr);
    if (!intAttr || !intAttr.getType().isSignlessInteger(32))
      return emitOpError("splat32 mode requires a signless i32 'value' attribute");
    if (!isVC4StructuredValueType(resultType)) {
      return emitOpError("splat32 mode result type must be i32, f32, "
                         "vector<16xi32>, or vector<16xf32>");
    }
    return success();
  }
  case mlir::vc4::LoadImmMode::per_elem_i2:
  case mlir::vc4::LoadImmMode::per_elem_u2: {
    auto valuesAttr = dyn_cast<DenseI32ArrayAttr>(valueAttr);
    if (!valuesAttr)
      return emitOpError("per-element mode requires a dense i32 array 'value' attribute");
    if (valuesAttr.asArrayRef().size() != 16)
      return emitOpError("per-element mode requires exactly 16 lane values");
    int32_t minValue =
        getMode() == mlir::vc4::LoadImmMode::per_elem_i2 ? -2 : 0;
    int32_t maxValue =
        getMode() == mlir::vc4::LoadImmMode::per_elem_i2 ? 1 : 3;
    for (int32_t laneValue : valuesAttr.asArrayRef()) {
      if (laneValue < minValue || laneValue > maxValue) {
        return emitOpError() << "lane values for mode "
                             << mlir::vc4::stringifyLoadImmMode(getMode())
                             << " must be in range [" << minValue << ", "
                             << maxValue << "]";
      }
    }
    auto vectorType = dyn_cast<VectorType>(resultType);
    if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable() ||
        vectorType.getDimSize(0) != 16 ||
        !vectorType.getElementType().isSignlessInteger(32)) {
      return emitOpError(
          "per-element mode result type must be vector<16xi32>");
    }
    return success();
  }
  }

  llvm_unreachable("unhandled vc4.load_imm mode");
}

LogicalResult mlir::vc4::PackOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  bool hasRegfileAMode = static_cast<bool>(getRegfileAModeAttr());
  bool hasMulMode = static_cast<bool>(getMulModeAttr());
  if (hasRegfileAMode == hasMulMode) {
    return emitOpError(
        "requires exactly one of 'regfile_a_mode' or 'mul_mode'");
  }

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (!hasSameVC4Shape(inputType, resultType)) {
    return emitOpError("input and result must have compatible scalar or "
                       "16-lane vector shapes");
  }
  if (!isVC4IntValueType(resultType))
    return emitOpError(
        "result type must be a 32-bit carrier word: i32 or vector<16xi32>");

  if (hasRegfileAMode) {
    switch (*getRegfileAMode()) {
    case mlir::vc4::RegfileAPackMode::none:
      return emitOpError("regfile_a_mode must not be <none>");
    case mlir::vc4::RegfileAPackMode::to_16a:
    case mlir::vc4::RegfileAPackMode::to_16b:
      if (!isVC4StructuredValueType(inputType)) {
        return emitOpError(
            "input type must be i32, f32, vector<16xi32>, or vector<16xf32>");
      }
      return success();
    case mlir::vc4::RegfileAPackMode::sat32:
    case mlir::vc4::RegfileAPackMode::to_8888:
    case mlir::vc4::RegfileAPackMode::to_8a:
    case mlir::vc4::RegfileAPackMode::to_8b:
    case mlir::vc4::RegfileAPackMode::to_8c:
    case mlir::vc4::RegfileAPackMode::to_8d:
    case mlir::vc4::RegfileAPackMode::sat16a:
    case mlir::vc4::RegfileAPackMode::sat16b:
    case mlir::vc4::RegfileAPackMode::sat8888:
    case mlir::vc4::RegfileAPackMode::sat8a:
    case mlir::vc4::RegfileAPackMode::sat8b:
    case mlir::vc4::RegfileAPackMode::sat8c:
    case mlir::vc4::RegfileAPackMode::sat8d:
      if (!isVC4IntValueType(inputType))
        return emitOpError("regfile_a_mode requires i32 or vector<16xi32> input");
      return success();
    }
    llvm_unreachable("unhandled vc4.pack regfile_a_mode");
  }

  switch (*getMulMode()) {
  case mlir::vc4::MulPackMode::none:
    return emitOpError("mul_mode must not be <none>");
  case mlir::vc4::MulPackMode::to_8888:
  case mlir::vc4::MulPackMode::to_8a:
  case mlir::vc4::MulPackMode::to_8b:
  case mlir::vc4::MulPackMode::to_8c:
  case mlir::vc4::MulPackMode::to_8d:
    if (!isVC4FloatValueType(inputType))
      return emitOpError("mul_mode requires f32 or vector<16xf32> input");
    return success();
  }

  llvm_unreachable("unhandled vc4.pack mul_mode");
}

LogicalResult mlir::vc4::UnpackOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  bool hasRegfileAMode = static_cast<bool>(getRegfileAModeAttr());
  bool hasR4Mode = static_cast<bool>(getR4ModeAttr());
  if (hasRegfileAMode == hasR4Mode) {
    return emitOpError(
        "requires exactly one of 'regfile_a_mode' or 'r4_mode'");
  }

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (!hasSameVC4Shape(inputType, resultType)) {
    return emitOpError("input and result must have compatible scalar or "
                       "16-lane vector shapes");
  }
  if (!isVC4IntValueType(inputType))
    return emitOpError(
        "input type must be a 32-bit carrier word: i32 or vector<16xi32>");

  if (hasRegfileAMode) {
    switch (*getRegfileAMode()) {
    case mlir::vc4::RegfileAUnpackMode::none:
      return emitOpError("regfile_a_mode must not be <none>");
    case mlir::vc4::RegfileAUnpackMode::f16a_or_i16a:
    case mlir::vc4::RegfileAUnpackMode::f16b_or_i16b:
    case mlir::vc4::RegfileAUnpackMode::color8a:
    case mlir::vc4::RegfileAUnpackMode::color8b:
    case mlir::vc4::RegfileAUnpackMode::color8c:
    case mlir::vc4::RegfileAUnpackMode::color8d:
      if (!isVC4StructuredValueType(resultType)) {
        return emitOpError(
            "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
      }
      return success();
    case mlir::vc4::RegfileAUnpackMode::replicate_8d:
      if (!isVC4IntValueType(resultType))
        return emitOpError("replicate_8d requires i32 or vector<16xi32> result");
      return success();
    }
    llvm_unreachable("unhandled vc4.unpack regfile_a_mode");
  }

  switch (*getR4Mode()) {
  case mlir::vc4::R4UnpackMode::none:
    return emitOpError("r4_mode must not be <none>");
  case mlir::vc4::R4UnpackMode::f16a:
  case mlir::vc4::R4UnpackMode::f16b:
  case mlir::vc4::R4UnpackMode::color8a:
  case mlir::vc4::R4UnpackMode::color8b:
  case mlir::vc4::R4UnpackMode::color8c:
  case mlir::vc4::R4UnpackMode::color8d:
    if (!isVC4FloatValueType(resultType))
      return emitOpError("r4_mode requires f32 or vector<16xf32> result");
    return success();
  case mlir::vc4::R4UnpackMode::replicate_8d:
    if (!isVC4IntValueType(resultType))
      return emitOpError("replicate_8d requires i32 or vector<16xi32> result");
    return success();
  }

  llvm_unreachable("unhandled vc4.unpack r4_mode");
}

LogicalResult mlir::vc4::RotateOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (inputType != resultType)
    return emitOpError("input and result types must match exactly");
  if (!isVector16Of32BitVC4ValueType(inputType))
    return emitOpError(
        "structured vc4.rotate requires vector<16xi32> or vector<16xf32> "
        "input and result types");

  bool hasAmount = static_cast<bool>(getAmount());
  bool hasImmediate = static_cast<bool>(getImmediateAttr());
  if (hasAmount == hasImmediate) {
    return emitOpError(
        "requires exactly one of an amount operand or an immediate attribute");
  }

  if (hasAmount) {
    if (!isScalarSignlessIntegerOrIndex(getAmount().getType()))
      return emitOpError("amount operand must be a scalar signless integer or index");
    return success();
  }

  auto immediateAttr = getImmediateAttr();
  if (!immediateAttr.getType().isSignlessInteger(32))
    return emitOpError("immediate attribute must be signless i32");
  int64_t value = immediateAttr.getInt();
  if (value < 0 || value > 15)
    return emitOpError("immediate rotate amount must be in range [0, 15]");
  return success();
}

LogicalResult mlir::vc4::TMUDescriptorOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (failed(verifyPositiveI32Attr(getOperation(), "mip_levels",
                                   getMipLevelsAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "width", getWidthAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "height", getHeightAttr())))
    return failure();
  if (failed(verifyPositiveI32Attr(getOperation(), "cube_map_stride",
                                   getCubeMapStrideAttr())))
    return failure();

  if (static_cast<bool>(getWidthAttr()) != static_cast<bool>(getHeightAttr())) {
    return emitOpError(
        "requires 'width' and 'height' to be provided together");
  }

  if (getChildImageFieldsAttr() && getChildImageFieldsAttr().empty()) {
    return emitOpError("'child_image_fields' attribute must not be empty");
  }
  if (getBiasFlagsAttr() && getBiasFlagsAttr().empty()) {
    return emitOpError("'bias_flags' attribute must not be empty");
  }

  bool hasTextureOnlyFields = getBaseAttr() || getTextureTypeAttr() ||
                              getMipLevelsAttr() || getWidthAttr() ||
                              getHeightAttr() || getMagFilterAttr() ||
                              getMinFilterAttr() || getWrapSAttr() ||
                              getWrapTAttr() || getFlipYAttr() ||
                              getCubeMapStrideAttr() ||
                              getChildImageFieldsAttr() || getBiasFlagsAttr();

  switch (getMode()) {
  case mlir::vc4::TMUMode::direct:
    if (hasTextureOnlyFields) {
      return emitOpError(
          "direct mode must not carry texture setup attributes");
    }
    return success();
  case mlir::vc4::TMUMode::texture2d:
    if (getCubeMapStrideAttr()) {
      return emitOpError(
          "'cube_map_stride' is only legal for mode = cubemap");
    }
    return success();
  case mlir::vc4::TMUMode::cubemap:
    return success();
  }

  llvm_unreachable("unhandled vc4.tmu.descriptor mode");
}

void mlir::vc4::TMURequestOp::getEffects(MemoryEffectList &effects) {
  addTMURequestEffects(getUnit(), effects);
}

LogicalResult mlir::vc4::TMURequestOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  ValueRange operands = getOperands();
  if (operands.empty())
    return emitOpError("requires at least one operand");

  unsigned descriptorCount = llvm::count_if(operands, [](Value operand) {
    return isa<mlir::vc4::TMUDescType>(operand.getType());
  });
  if (descriptorCount > 1)
    return emitOpError("accepts at most one !vc4.tmu.desc operand");

  bool hasDescriptor =
      isa<mlir::vc4::TMUDescType>(operands.back().getType());
  if (descriptorCount == 1 && !hasDescriptor) {
    return emitOpError("descriptor operand must be the last operand");
  }

  ValueRange valueOperands =
      hasDescriptor ? operands.drop_back() : operands;
  if (valueOperands.empty()) {
    return emitOpError(
        "requires at least one address or coordinate operand");
  }

  auto verifyTextureOperands = [&](unsigned minCount,
                                   unsigned maxCount) -> LogicalResult {
    if (valueOperands.size() < minCount || valueOperands.size() > maxCount) {
      return emitOpError() << "expects between " << minCount << " and "
                           << maxCount
                           << " coordinate operands for the selected TMU mode";
    }
    TypeRange valueOperandTypes = valueOperands.getTypes();
    for (Type type : valueOperandTypes) {
      if (!isVC4StructuredValueType(type)) {
        return emitOpError("coordinate operands must be i32, f32, "
                           "vector<16xi32>, or vector<16xf32>");
      }
    }
    if (!haveCompatibleVC4Shapes(valueOperandTypes)) {
      return emitOpError("coordinate operands must have compatible scalar or "
                         "16-lane vector shapes");
    }
    return success();
  };

  if (!hasDescriptor) {
    if (valueOperands.size() != 1) {
      return emitOpError(
          "requests without a descriptor are only legal in direct mode and "
          "require exactly one address operand");
    }
    if (!isI32OrVector16I32(valueOperands.front().getType())) {
      return emitOpError(
          "direct-mode address operand must be i32 or vector<16xi32>");
    }
    return success();
  }

  std::optional<mlir::vc4::TMUMode> mode =
      inferTMUModeFromDescriptor(operands.back());
  if (!mode)
    return verifyTextureOperands(/*minCount=*/1, /*maxCount=*/4);

  switch (*mode) {
  case mlir::vc4::TMUMode::direct:
    if (valueOperands.size() != 1) {
      return emitOpError(
          "direct-mode descriptors require exactly one address operand");
    }
    if (!isI32OrVector16I32(valueOperands.front().getType())) {
      return emitOpError(
          "direct-mode address operand must be i32 or vector<16xi32>");
    }
    return success();
  case mlir::vc4::TMUMode::texture2d:
    return verifyTextureOperands(/*minCount=*/1, /*maxCount=*/3);
  case mlir::vc4::TMUMode::cubemap:
    return verifyTextureOperands(/*minCount=*/3, /*maxCount=*/4);
  }

  llvm_unreachable("unhandled vc4.tmu.request mode");
}

void mlir::vc4::TMUReadOp::getEffects(MemoryEffectList &effects) {
  addTMUReadEffects(getUnit(), effects);
}

LogicalResult mlir::vc4::TMUReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (Value token = getToken()) {
    if (Operation *definingOp = token.getDefiningOp()) {
      auto request = dyn_cast<mlir::vc4::TMURequestOp>(definingOp);
      if (!request) {
        return emitOpError(
            "token operand must come from vc4.tmu.request or be a block argument");
      }
      if (request.getUnit() != getUnit()) {
        return emitOpError("token unit must match the selected read unit");
      }
    }
  }

  switch (getPart()) {
  case mlir::vc4::TMUReadPart::raw32:
    if (!isVC4StructuredValueType(getResult().getType())) {
      return emitOpError("part = raw32 requires i32, f32, vector<16xi32>, "
                         "or vector<16xf32> result type");
    }
    return success();
  case mlir::vc4::TMUReadPart::rgba8888:
  case mlir::vc4::TMUReadPart::rg1616:
  case mlir::vc4::TMUReadPart::ba1616:
    if (!isI32OrVector16I32(getResult().getType())) {
      return emitOpError("packed TMU read parts require i32 or vector<16xi32> "
                         "result type");
    }
    return success();
  }

  llvm_unreachable("unhandled vc4.tmu.read part");
}

void mlir::vc4::TMUNoSwapOp::getEffects(MemoryEffectList &effects) {
  addWriteEffect<mlir::vc4::effects::V3DSystem>(effects);
}

LogicalResult mlir::vc4::TMUNoSwapOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  bool hasValue = static_cast<bool>(getValue());
  bool hasDisableAttr = static_cast<bool>(getDisableAttr());
  if (hasValue == hasDisableAttr) {
    return emitOpError(
        "requires exactly one of a value operand or a 'disable' attribute");
  }

  if (hasValue && isa<VectorType>(getValue().getType())) {
    return emitOpError("value operand must be a scalar signless integer");
  }

  return success();
}

void mlir::vc4::SFUIssueOp::getEffects(MemoryEffectList &effects) {
  addWriteEffect<mlir::vc4::effects::SFU>(effects);
}

LogicalResult mlir::vc4::SFUIssueOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isVC4StructuredValueType(getInput().getType())) {
    return emitOpError(
        "input type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

void mlir::vc4::SFUReadOp::getEffects(MemoryEffectList &effects) {
  addReadEffect<mlir::vc4::effects::SFU>(effects);
}

LogicalResult mlir::vc4::SFUReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }
  return success();
}

LogicalResult mlir::vc4::VPMDescriptorOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (failed(verifyNonNegativeI32Attr(getOperation(), "addr", getAddrAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "num_vectors",
                                getNumVectorsAttr())))
    return failure();

  switch (getKind()) {
  case mlir::vc4::VPMDescKind::read:
    if (!getNumVectorsAttr())
      return emitOpError("kind = read requires a 'num_vectors' attribute");
    return success();
  case mlir::vc4::VPMDescKind::write:
    if (getNumVectorsAttr())
      return emitOpError("kind = write must not carry 'num_vectors'");
    return success();
  }

  llvm_unreachable("unhandled vc4.vpm.desc kind");
}

void mlir::vc4::VPMReadOp::getEffects(MemoryEffectList &effects) {
  addReadEffect<mlir::vc4::effects::VPMReadFIFO>(effects);
}

LogicalResult mlir::vc4::VPMReadOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isVC4StructuredValueType(getResult().getType())) {
    return emitOpError(
        "result type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }

  std::optional<mlir::vc4::VPMDescKind> kind =
      inferVPMDescKindFromDescriptor(getDescriptor());
  if (kind && *kind != mlir::vc4::VPMDescKind::read)
    return emitOpError("descriptor kind must be <read>");

  return success();
}

void mlir::vc4::VPMWriteOp::getEffects(MemoryEffectList &effects) {
  addWriteEffect<mlir::vc4::effects::VPMWriteFIFO>(effects);
}

LogicalResult mlir::vc4::VPMWriteOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isVC4StructuredValueType(getValue().getType())) {
    return emitOpError(
        "value type must be i32, f32, vector<16xi32>, or vector<16xf32>");
  }

  std::optional<mlir::vc4::VPMDescKind> kind =
      inferVPMDescKindFromDescriptor(getDescriptor());
  if (kind && *kind != mlir::vc4::VPMDescKind::write)
    return emitOpError("descriptor kind must be <write>");

  return success();
}

LogicalResult mlir::vc4::DMADescriptorOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (failed(verifyNonNegativeI32Attr(getOperation(), "start_offset",
                                      getStartOffsetAttr())))
    return failure();
  if (failed(verifyNonNegativeI32Attr(getOperation(), "mpitch",
                                      getMpitchAttr())))
    return failure();
  if (failed(verifyNonNegativeI32Attr(getOperation(), "vpitch",
                                      getVpitchAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "nrows", getNrowsAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "rowlen", getRowlenAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "units", getUnitsAttr())))
    return failure();
  if (failed(
          verifyPositiveI32Attr(getOperation(), "depth", getDepthAttr())))
    return failure();
  if (failed(verifyNonNegativeI32Attr(getOperation(), "vpm_base",
                                      getVpmBaseAttr())))
    return failure();
  if (failed(
          verifyNonNegativeI32Attr(getOperation(), "stride", getStrideAttr())))
    return failure();
  if (failed(verifyNonNegativeI32Attr(getOperation(), "extended_stride",
                                      getExtendedStrideAttr())))
    return failure();

  if (auto startOffset = getStartOffsetAttr()) {
    if (startOffset.getInt() > 3) {
      return emitOpError(
          "'start_offset' attribute must be in range [0, 3]");
    }
  }

  if (static_cast<bool>(getMpitchAttr()) != static_cast<bool>(getVpitchAttr())) {
    return emitOpError(
        "requires 'mpitch' and 'vpitch' to be provided together");
  }
  if (static_cast<bool>(getNrowsAttr()) != static_cast<bool>(getRowlenAttr())) {
    return emitOpError(
        "requires 'nrows' and 'rowlen' to be provided together");
  }
  if (getExtendedStrideAttr() && !getStrideAttr()) {
    return emitOpError(
        "'extended_stride' requires the base 'stride' attribute");
  }
  if (getUnitsAttr() && getDepthAttr()) {
    return emitOpError(
        "must not specify both 'units' and 'depth' in one descriptor");
  }

  switch (getKind()) {
  case mlir::vc4::DMADescKind::load:
  case mlir::vc4::DMADescKind::store:
    return success();
  }

  llvm_unreachable("unhandled vc4.dma.desc kind");
}

void mlir::vc4::DMAStartOp::getEffects(MemoryEffectList &effects) {
  std::optional<mlir::vc4::DMADescKind> kind =
      inferDMADescKindFromDescriptor(getDescriptor());
  addDMAQueueEffects(kind, effects, /*isWrite=*/true);
  if (!kind || *kind == mlir::vc4::DMADescKind::load)
    addReadEffect<mlir::vc4::effects::MainMemory>(effects);
  if (!kind || *kind == mlir::vc4::DMADescKind::store)
    addWriteEffect<mlir::vc4::effects::MainMemory>(effects);
}

LogicalResult mlir::vc4::DMAStartOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (!isScalarSignlessIntegerOrIndex(getBase().getType())) {
    return emitOpError("base address must be a scalar signless integer or index");
  }

  std::optional<mlir::vc4::DMADescKind> kind =
      inferDMADescKindFromDescriptor(getDescriptor());
  if (kind && *kind != mlir::vc4::DMADescKind::load &&
      *kind != mlir::vc4::DMADescKind::store) {
    return emitOpError("descriptor kind must be <load> or <store>");
  }
  return success();
}

void mlir::vc4::DMAStatusOp::getEffects(MemoryEffectList &effects) {
  addDMAQueueEffects(getKind(), effects, /*isWrite=*/false);
}

LogicalResult mlir::vc4::DMAStatusOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  Type resultType = getResult().getType();
  if (!resultType.isSignlessIntOrIndex() || isa<VectorType>(resultType)) {
    return emitOpError("result type must be a scalar signless integer or index");
  }
  return success();
}

void mlir::vc4::DMAWaitOp::getEffects(MemoryEffectList &effects) {
  std::optional<mlir::vc4::DMADescKind> kind =
      getKindAttr()
          ? std::optional<mlir::vc4::DMADescKind>(getKind())
          : inferDMADescKindFromToken(getToken());
  addDMAQueueEffects(kind, effects, /*isWrite=*/false);
}

LogicalResult mlir::vc4::DMAWaitOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  bool hasToken = static_cast<bool>(getToken());
  bool hasKind = static_cast<bool>(getKindAttr());
  if (hasToken == hasKind) {
    return emitOpError(
        "requires exactly one of a token operand or a 'kind' attribute");
  }

  if (Value token = getToken()) {
    if (Operation *definingOp = token.getDefiningOp()) {
      auto start = dyn_cast<mlir::vc4::DMAStartOp>(definingOp);
      if (!start) {
        return emitOpError(
            "token operand must come from vc4.dma.start or be a block argument");
      }
    }
  }

  return success();
}

void mlir::vc4::MutexOp::getEffects(MemoryEffectList &effects) {
  addReadWriteEffects<mlir::vc4::effects::Mutex>(effects);
}

LogicalResult mlir::vc4::MutexOp::verify() {
  return verifyStructuredFormOp(getOperation());
}

void mlir::vc4::SemaphoreOp::getEffects(MemoryEffectList &effects) {
  addReadWriteEffects<mlir::vc4::effects::Semaphore>(effects);
}

LogicalResult mlir::vc4::SemaphoreOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  int64_t id = getIdAttr().getInt();
  if (id < 0 || id > 15)
    return emitOpError("semaphore 'id' attribute must be in range [0, 15]");
  return success();
}

void mlir::vc4::HostInterruptOp::getEffects(MemoryEffectList &effects) {
  addWriteEffect<mlir::vc4::effects::HostIRQ>(effects);
}

LogicalResult mlir::vc4::HostInterruptOp::verify() {
  return verifyStructuredFormOp(getOperation());
}

void mlir::vc4::ThreadSwitchOp::getEffects(MemoryEffectList &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult mlir::vc4::ThreadSwitchOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  auto func = (*this)->getParentOfType<mlir::vc4::FuncOp>();
  if (!func || !func.getThreading() ||
      *func.getThreading() != mlir::vc4::ThreadingMode::threadable) {
    return emitOpError(
        "is only legal in functions with threading = threadable");
  }
  return success();
}

void mlir::vc4::ProgramEndOp::getEffects(MemoryEffectList &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult mlir::vc4::ProgramEndOp::verify() {
  return verifyStructuredFormOp(getOperation());
}

void mlir::vc4::AsyncWaitOp::getEffects(MemoryEffectList &effects) {
  effects.emplace_back(MemoryEffects::Write::get());
}

LogicalResult mlir::vc4::AsyncWaitOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (getTokens().empty())
    return emitOpError("requires at least one async token operand");
  return success();
}

LogicalResult mlir::vc4::CFBranchOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  unsigned numSuccessors = getNumSuccessors();
  if (getCond() == mlir::vc4::BranchCond::always) {
    if (numSuccessors != 1)
      return emitOpError("cond = always requires exactly one successor");
    return success();
  }

  if (numSuccessors != 2)
    return emitOpError("conditional branch requires exactly two successors");
  return success();
}

mlir::SuccessorOperands mlir::vc4::CFBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "successor index out of range");
  return mlir::SuccessorOperands(
      mlir::MutableOperandRange(getOperation(), /*start=*/0, /*length=*/0));
}

LogicalResult mlir::vc4::EnqueueQPUOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  if (getNumOperands() != 0 && getNumOperands() != 2) {
    return emitOpError(
        "supports either no operands or exactly two operands for uniforms base and length");
  }
  if (getNumOperands() == 2) {
    if (!isScalarSignlessIntegerOrIndex(getOperand(0).getType()) ||
        !isScalarSignlessIntegerOrIndex(getOperand(1).getType())) {
      return emitOpError(
          "uniforms base and length operands must be scalar signless integers or index");
    }
  }

  Operation *symbol = SymbolTable::lookupNearestSymbolFrom(getOperation(), getEntryAttr());
  auto func = dyn_cast_or_null<mlir::vc4::FuncOp>(symbol);
  if (!func)
    return emitOpError("referenced 'entry' must resolve to a vc4.func symbol");
  if (!func.getKernelAttr())
    return emitOpError("referenced function must be marked with the 'kernel' attribute");
  return success();
}

void mlir::vc4::EnqueueQPUOp::getEffects(MemoryEffectList &effects) {
  addWriteEffect<mlir::vc4::effects::QPUScheduler>(effects);
}

LogicalResult mlir::vc4::ReserveQPUOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  int64_t mask = getMaskAttr().getInt();
  if (mask < 0 || mask > 0xFFF)
    return emitOpError("mask attribute must fit the 12-QPU target range [0, 4095]");
  return success();
}

void mlir::vc4::ReserveQPUOp::getEffects(MemoryEffectList &effects) {
  addWriteEffect<mlir::vc4::effects::QPUScheduler>(effects);
}

LogicalResult mlir::vc4::V3DQueryOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  auto verifyScalarIntLike = [&](Type type, StringRef what) -> LogicalResult {
    if (!isScalarSignlessIntegerOrIndex(type)) {
      return emitOpError() << what
                           << " must be a scalar signless integer or index";
    }
    return success();
  };

  if (getNumResults() != 1)
    return emitOpError("currently requires exactly one result");
  if (failed(verifyScalarIntLike(getResult(0).getType(), "result type")))
    return failure();

  switch (getKind()) {
  case mlir::vc4::V3DQueryKind::ident:
  case mlir::vc4::V3DQueryKind::queue_status:
  case mlir::vc4::V3DQueryKind::interrupt_status:
  case mlir::vc4::V3DQueryKind::error_status:
    if (getNumOperands() != 0)
      return emitOpError("selected query kind does not accept selector operands");
    return success();
  case mlir::vc4::V3DQueryKind::perf_counter:
  case mlir::vc4::V3DQueryKind::scratch:
    if (getNumOperands() != 1)
      return emitOpError("selected query kind requires exactly one selector operand");
    return verifyScalarIntLike(getOperand(0).getType(), "selector operand");
  }

  llvm_unreachable("unhandled vc4.v3d.query kind");
}

void mlir::vc4::V3DQueryOp::getEffects(MemoryEffectList &effects) {
  addReadEffect<mlir::vc4::effects::V3DSystem>(effects);
}

LogicalResult mlir::vc4::V3DConfigureOp::verify() {
  if (failed(verifyStructuredFormOp(getOperation())))
    return failure();

  auto verifyScalarOperands = [&](unsigned expected) -> LogicalResult {
    if (getNumOperands() != expected) {
      return emitOpError() << "selected configure kind requires exactly "
                           << expected << " payload operand"
                           << (expected == 1 ? "" : "s");
    }
    for (Type type : getOperandTypes()) {
      if (!isScalarSignlessIntegerOrIndex(type)) {
        return emitOpError(
            "payload operands must be scalar signless integers or index");
      }
    }
    return success();
  };

  switch (getKind()) {
  case mlir::vc4::V3DConfigureKind::cache_control:
  case mlir::vc4::V3DConfigureKind::interrupt_enable:
  case mlir::vc4::V3DConfigureKind::interrupt_disable:
  case mlir::vc4::V3DConfigureKind::perf_enable:
  case mlir::vc4::V3DConfigureKind::vpm_reservation:
  case mlir::vc4::V3DConfigureKind::vpm_allocator:
    return verifyScalarOperands(/*expected=*/1);
  case mlir::vc4::V3DConfigureKind::perf_map:
  case mlir::vc4::V3DConfigureKind::scratch:
    return verifyScalarOperands(/*expected=*/2);
  case mlir::vc4::V3DConfigureKind::perf_clear:
    return verifyScalarOperands(/*expected=*/0);
  }

  llvm_unreachable("unhandled vc4.v3d.configure kind");
}

void mlir::vc4::V3DConfigureOp::getEffects(MemoryEffectList &effects) {
  addWriteEffect<mlir::vc4::effects::V3DSystem>(effects);
}

LogicalResult mlir::vc4::QPULDIOp::verify() {
  if (failed(verifyScheduledFormOp(getOperation())))
    return failure();

  if (failed(verifyQPULoadImmPayload(getOperation(), getMode(), getValueAttr())))
    return failure();
  if (failed(verifyQPUPackAttr(getOperation(), getPm(), getPackAttr())))
    return failure();
  if (failed(
          verifyQPUWriteAddressAttr(getOperation(), "waddr_add", getWaddrAddAttr())))
    return failure();
  if (failed(
          verifyQPUWriteAddressAttr(getOperation(), "waddr_mul", getWaddrMulAttr())))
    return failure();
  return success();
}

void mlir::vc4::QPUSemaOp::getEffects(MemoryEffectList &effects) {
  addReadWriteEffects<mlir::vc4::effects::Semaphore>(effects);
}

LogicalResult mlir::vc4::QPUSemaOp::verify() {
  if (failed(verifyScheduledFormOp(getOperation())))
    return failure();

  int64_t id = getIdAttr().getInt();
  if (id < 0 || id > 15)
    return emitOpError("'id' attribute must be in range [0, 15]");
  if (failed(verifyQPUPackAttr(getOperation(), getPm(), getPackAttr())))
    return failure();
  if (failed(
          verifyQPUSemaWriteAddressAttr(getOperation(), "waddr_add",
                                        getWaddrAddAttr())))
    return failure();
  if (failed(
          verifyQPUSemaWriteAddressAttr(getOperation(), "waddr_mul",
                                        getWaddrMulAttr())))
    return failure();
  return success();
}

LogicalResult mlir::vc4::QPUBundleOp::verify() {
  if (failed(verifyScheduledFormOp(getOperation())))
    return failure();

  bool hasRaddrB = static_cast<bool>(getRaddrBAttr());
  bool hasSmallImm = static_cast<bool>(getSmallImmAttr());
  if (hasRaddrB == hasSmallImm) {
    return emitOpError(
        "requires exactly one of 'raddr_b' or 'small_imm'");
  }

  if (failed(verifyQPUUnpackAttr(getOperation(), getPm(), getUnpackAttr())))
    return failure();
  if (failed(verifyQPUPackAttr(getOperation(), getPm(), getPackAttr())))
    return failure();
  if (failed(
          verifyQPUWriteAddressAttr(getOperation(), "waddr_add", getWaddrAddAttr())))
    return failure();
  if (failed(
          verifyQPUWriteAddressAttr(getOperation(), "waddr_mul", getWaddrMulAttr())))
    return failure();
  if (failed(verifyQPUBundleWriteConflict(getOperation(), getOpAdd(), getOpMul(),
                                          getWaddrAddAttr(), getWaddrMulAttr())))
    return failure();
  if (failed(
          verifyQPUBundleReadAddressAttr(getOperation(), "raddr_a",
                                         getRaddrAAttr())))
    return failure();
  if (hasRaddrB &&
      failed(verifyQPUBundleReadAddressAttr(getOperation(), "raddr_b",
                                            getRaddrBAttr()))) {
    return failure();
  }

  if (hasSmallImm) {
    int64_t smallImm = getSmallImmAttr().getInt();
    if (!isQPUValidSmallImmSelector(smallImm)) {
      return emitOpError(
          "'small_imm' attribute must be an encoded selector in range [0, 63]");
    }
    if (isQPUSmallImmVectorRotateSelector(smallImm) &&
        (!isQPUAccumulatorMuxR0ToR3(getMulA()) ||
         !isQPUAccumulatorMuxR0ToR3(getMulB()))) {
      return emitOpError(
          "vector-rotate small_imm selectors 48..63 require both MUL inputs "
          "to come from accumulators r0..r3");
    }
  }

  switch (getSig()) {
  case mlir::vc4::QPUSignal::small_imm:
    if (!hasSmallImm) {
      return emitOpError(
          "sig = #vc4.qpu_signal<small_imm> requires a 'small_imm' attribute");
    }
    break;
  case mlir::vc4::QPUSignal::load_imm:
    return emitOpError(
        "sig = #vc4.qpu_signal<load_imm> is represented by vc4.qpu.ldi");
  case mlir::vc4::QPUSignal::branch:
    return emitOpError(
        "sig = #vc4.qpu_signal<branch> is represented by vc4.qpu.branch");
  default:
    if (hasSmallImm) {
      return emitOpError(
          "'small_imm' attribute requires sig = #vc4.qpu_signal<small_imm>");
    }
    break;
  }

  return success();
}

LogicalResult mlir::vc4::QPUBranchOp::verify() {
  if (failed(verifyScheduledFormOp(getOperation())))
    return failure();

  if (failed(
          verifyQPUBranchReadAddressAttr(getOperation(), "raddr_a",
                                         getRaddrAAttr())))
    return failure();
  if (failed(
          verifyQPUWriteAddressAttr(getOperation(), "waddr_add", getWaddrAddAttr())))
    return failure();
  if (failed(
          verifyQPUWriteAddressAttr(getOperation(), "waddr_mul", getWaddrMulAttr())))
    return failure();

  if (!getDelaySlots().hasOneBlock())
    return emitOpError("delay-slot region must contain exactly one block");

  Block &delaySlotBlock = getDelaySlots().front();
  if (delaySlotBlock.getNumArguments() != 0)
    return emitOpError("delay-slot region block must not take arguments");

  unsigned delaySlotCount = 0;
  for (Operation &op : delaySlotBlock) {
    ++delaySlotCount;
    if (!isVC4QPUOp(op)) {
      return op.emitOpError(
          "is not a legal delay-slot operation; expected a vc4.qpu.* op");
    }
  }

  if (delaySlotCount != 3) {
    return emitOpError() << "delay-slot region must contain exactly 3 "
                         << "scheduled QPU ops";
  }

  return success();
}

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4/IR/VC4Ops.cpp.inc"
