//===- VC4Ops.cpp - VC4 dialect operations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4/IR/VC4Ops.h"
#include "vc4/Dialect/VC4/IR/VC4SideEffects.h"
#include "vc4/Support/VC4ResourceMetadata.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/SmallSet.h"
#include <algorithm>
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

static bool isStringOneOf(StringRef value, ArrayRef<StringRef> allowed) {
  for (StringRef candidate : allowed) {
    if (value == candidate)
      return true;
  }
  return false;
}

static bool isRuntimeLaunchABIName(StringRef value) {
  return isStringOneOf(value,
                       {"logical_request",
                        "total_requests",
                        "logical_block_id",
                        "logical_warp_id",
                        "warps_per_block",
                        "vpm_base_row",
                        "vpm_rows",
                        "semaphore_base",
                        "barrier_arrive_sem",
                        "barrier_go_sem",
                        "barrier_depart_sem",
                        "barrier_reset_sem",
                        "resident_request_id",
                        "spill_frame_base",
                        "spill_frame_bytes",
                        "spill_frame_stride_bytes",
                        "spill_vpm_row"});
}

static bool isLaneIdentityLaunchABIName(StringRef value) {
  return isStringOneOf(value, {"ELEMENT_NUMBER", "element_number", "elem_num",
                              "lane_id", "lane_range"});
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
  auto nameAttr = cast<StringAttr>(arg.get("name"));
  if (isRuntimeLaunchABIName(nameAttr.getValue())) {
    return emitLaunchAbiError(
        op, Twine("argument entry '") + nameAttr.getValue() +
                Twine("' is runtime metadata and must be a builtin"));
  }
  if (isLaneIdentityLaunchABIName(nameAttr.getValue())) {
    return emitLaunchAbiError(
        op, Twine("argument entry '") + nameAttr.getValue() +
                Twine("' is lane identity and must not be launch ABI metadata"));
  }

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
  auto nameAttr = cast<StringAttr>(builtin.get("name"));
  if (isLaneIdentityLaunchABIName(nameAttr.getValue())) {
    return emitLaunchAbiError(
        op, Twine("builtin entry '") + nameAttr.getValue() +
                Twine("' is lane identity and must not be launch ABI metadata"));
  }

  auto kindAttr =
      dyn_cast_or_null<mlir::vc4::BuiltinKindAttr>(builtin.get("kind"));
  if (!kindAttr)
    return emitLaunchAbiError(op,
                              "builtin entry requires VC4 BuiltinKindAttr");

  auto materializationAttr =
      dyn_cast_or_null<StringAttr>(builtin.get("materialization"));
  if (!materializationAttr ||
      materializationAttr.getValue() != "uniform_suffix") {
    return emitLaunchAbiError(
        op, "builtin entry requires materialization = \"uniform_suffix\"");
  }

  if (isLaneIdentityLaunchABIName(
          stringifyBuiltinKind(kindAttr.getValue()))) {
    return emitLaunchAbiError(
        op, "lane identity must not appear as a launch ABI builtin");
  }

  return verifyLaunchAbiUniformIndex(op, builtin, "builtin", indices);
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
  if (!uniformWordsPerQPU || *uniformWordsPerQPU < 0) {
    return emitLaunchAbiError(
        op, "requires a non-negative signless i32 'uniform_words_per_qpu'");
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

static LogicalResult emitResourceError(mlir::vc4::FuncOp op, Twine message) {
  return op.emitOpError() << "\"vc4.resource\" " << message;
}

static LogicalResult verifyResourceMetadata(mlir::vc4::FuncOp op) {
  Attribute rawAttr = op->getAttr("vc4.resource");
  if (!rawAttr) return success();
  auto resource = dyn_cast<DictionaryAttr>(rawAttr);
  if (!resource) return emitResourceError(op, "requires a dictionary attribute");
  if (!op.getKernelAttr()) return emitResourceError(op, "may appear only on vc4.func with 'kernel'");
  if (!op.getDomain() || *op.getDomain() != mlir::vc4::ExecutionDomain::qpu)
    return emitResourceError(op, "requires domain = #vc4.execution_domain<qpu>");
  static constexpr llvm::StringLiteral requiredFields[] = {
      "schedule_mode",
      "warps_per_block",
      "user_vpm_rows_per_block",
      "compiler_vpm_staging_rows_per_warp",
      "compiler_vpm_staging_rows_per_block",
      "total_vpm_rows_per_block",
      "uses_tmu",
      "uses_vpm",
      "uses_vpm_qpu_read",
      "uses_vpm_qpu_write",
      "uses_vdr",
      "uses_vdw",
      "uses_barrier",
      "semaphore_count_per_block",
      "requires_vpm_base_row_builtin",
      "requires_semaphore_base_builtin",
  };
  for (llvm::StringLiteral field : requiredFields) {
    if (!resource.get(field))
      return emitResourceError(op, Twine("requires semantic field '") +
                                      field + Twine("'"));
  }
  mlir::vc4::SemanticResourceInfo info;
  if (failed(mlir::vc4::parseSemanticResourceMetadata(
          op.getOperation(), resource, info, /*allowAbsent=*/false)))
    return failure();
  if (mlir::vc4::getMaxResidentBlocksForSemanticResource(info) <= 0)
    return emitResourceError(
        op, "resource request leaves zero resident_blocks; check "
            "warps_per_block, total_vpm_rows_per_block, and "
            "semaphore_count_per_block");
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

static LogicalResult verifyOptionalQPUReadAddressAttr(Operation *op,
                                                      StringRef attrName,
                                                      IntegerAttr attr) {
  if (!attr)
    return success();
  return verifyQPUBundleReadAddressAttr(op, attrName, attr);
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

static LogicalResult verifyOptionalLiteralAddress(Operation *op,
                                                  StringRef attrName,
                                                  IntegerAttr attr,
                                                  int64_t expected) {
  if (!attr)
    return success();
  if (failed(verifyQPUWriteAddressAttr(op, attrName, attr)))
    return failure();
  if (attr.getInt() != expected) {
    return op->emitOpError() << "'" << attrName << "' must be "
                             << expected << " when present";
  }
  return success();
}

static LogicalResult verifyQPUOperandSelectorAttrs(
    Operation *op, IntegerAttr raddrAAttr, IntegerAttr raddrBAttr,
    IntegerAttr smallImmAttr, mlir::vc4::QPUMux mulA,
    mlir::vc4::QPUMux mulB) {
  bool hasRaddrB = static_cast<bool>(raddrBAttr);
  bool hasSmallImm = static_cast<bool>(smallImmAttr);
  if (hasRaddrB == hasSmallImm) {
    return op->emitOpError(
        "requires exactly one of 'raddr_b' or 'small_imm'");
  }

  if (failed(verifyQPUBundleReadAddressAttr(op, "raddr_a", raddrAAttr)))
    return failure();
  if (hasRaddrB &&
      failed(verifyQPUBundleReadAddressAttr(op, "raddr_b", raddrBAttr)))
    return failure();

  if (!hasSmallImm)
    return success();

  int64_t smallImm = smallImmAttr.getInt();
  if (!isQPUValidSmallImmSelector(smallImm)) {
    return op->emitOpError(
        "'small_imm' attribute must be an encoded selector in range [0, 63]");
  }
  if (isQPUSmallImmVectorRotateSelector(smallImm) &&
      (!isQPUAccumulatorMuxR0ToR3(mulA) ||
       !isQPUAccumulatorMuxR0ToR3(mulB))) {
    return op->emitOpError(
        "vector-rotate small_imm selectors 48..63 require both MUL inputs "
        "to come from accumulators r0..r3");
  }
  return success();
}

static LogicalResult verifyQPUVPMVCDWritePseudoOp(
    Operation *op, int64_t expectedWaddrAdd, mlir::vc4::Cond condAdd,
    mlir::vc4::Cond condMul, mlir::vc4::AddOpcode opAdd,
    mlir::vc4::MulOpcode opMul, IntegerAttr raddrAAttr,
    IntegerAttr raddrBAttr, IntegerAttr smallImmAttr, mlir::vc4::QPUMux mulA,
    mlir::vc4::QPUMux mulB) {
  if (failed(verifyScheduledFormOp(op)))
    return failure();

  if (condAdd == mlir::vc4::Cond::never || opAdd == mlir::vc4::AddOpcode::nop) {
    return op->emitOpError(
        "requires an active ADD-side write to the implied VPM/VCD/VDW "
        "control address");
  }
  if (condMul != mlir::vc4::Cond::never &&
      opMul != mlir::vc4::MulOpcode::nop) {
    return op->emitOpError(
        "must not carry an active MUL-side ordinary register write");
  }

  if (failed(verifyOptionalLiteralAddress(
          op, "waddr_add", op->getAttrOfType<IntegerAttr>("waddr_add"),
          expectedWaddrAdd)))
    return failure();
  if (failed(verifyOptionalLiteralAddress(
          op, "waddr_mul", op->getAttrOfType<IntegerAttr>("waddr_mul"), 32)))
    return failure();

  return verifyQPUOperandSelectorAttrs(op, raddrAAttr, raddrBAttr,
                                       smallImmAttr, mulA, mulB);
}

static LogicalResult verifyNoRawVPMVCDControlWrite(Operation *op,
                                                   StringRef opName,
                                                   mlir::vc4::Cond cond,
                                                   IntegerAttr addressAttr) {
  if (cond == mlir::vc4::Cond::never || !addressAttr)
    return success();

  int64_t address = addressAttr.getInt();
  if (address == 49) {
    return op->emitOpError()
           << "raw " << opName
           << " writes to VPM/VCD/VDW setup address 49 require "
              "vc4.qpu.vpmvcd_setup or vc4.qpu.vpmvcd_setup_ldi";
  }
  if (address == 50) {
    return op->emitOpError()
           << "raw " << opName
           << " writes to VPM/VCD/VDW address register 50 require "
              "vc4.qpu.vpmvcd_addr";
  }
  return success();
}

static LogicalResult verifyNoRawVPMVCDWaitRead(Operation *op,
                                               IntegerAttr addressAttr) {
  if (!addressAttr || addressAttr.getInt() != 50)
    return success();
  return op->emitOpError()
         << "raw vc4.qpu.bundle reads from VPM/VCD/VDW wait address 50 "
            "require vc4.qpu.vpmvcd_wait";
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
  if (failed(verifyResourceMetadata(*this)))
    return failure();
  if (isExternal())
    return success();

  if (form != mlir::vc4::FunctionForm::scheduled) {
    return emitOpError(
        "requires form = #vc4.function_form<scheduled>");
  }
  if (domain != mlir::vc4::ExecutionDomain::qpu) {
    return emitOpError(
        "scheduled VC4 functions require domain = #vc4.execution_domain<qpu>");
  }

  bool sawError = false;
  getBody().walk([&](Operation *op) {
    if (sawError)
      return WalkResult::interrupt();

    if (isVC4QPUOp(*op))
      return WalkResult::advance();

    op->emitOpError(
        "is not legal in scheduled VC4 functions; expected a vc4.qpu.* op");
    sawError = true;
    return WalkResult::interrupt();
  });

  return failure(sawError);
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
  if (failed(verifyNoRawVPMVCDControlWrite(
          getOperation(), "vc4.qpu.ldi", getCondAdd(), getWaddrAddAttr())))
    return failure();
  if (failed(verifyNoRawVPMVCDControlWrite(
          getOperation(), "vc4.qpu.ldi", getCondMul(), getWaddrMulAttr())))
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
  if (failed(verifyNoRawVPMVCDControlWrite(
          getOperation(), "vc4.qpu.bundle", getCondAdd(), getWaddrAddAttr())))
    return failure();
  if (failed(verifyNoRawVPMVCDControlWrite(
          getOperation(), "vc4.qpu.bundle", getCondMul(), getWaddrMulAttr())))
    return failure();
  if (failed(
          verifyQPUBundleReadAddressAttr(getOperation(), "raddr_a",
                                         getRaddrAAttr())))
    return failure();
  if (failed(verifyNoRawVPMVCDWaitRead(getOperation(), getRaddrAAttr())))
    return failure();
  if (hasRaddrB &&
      failed(verifyQPUBundleReadAddressAttr(getOperation(), "raddr_b",
                                            getRaddrBAttr()))) {
    return failure();
  }
  if (hasRaddrB &&
      failed(verifyNoRawVPMVCDWaitRead(getOperation(), getRaddrBAttr())))
    return failure();

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

void mlir::vc4::QPUVPMVCDSetupOp::getEffects(MemoryEffectList &effects) {
  if (getSide() == mlir::vc4::VPMVCDSide::read)
    addWriteEffect<mlir::vc4::effects::VDR>(effects);
  else
    addWriteEffect<mlir::vc4::effects::VDW>(effects);
}

LogicalResult mlir::vc4::QPUVPMVCDSetupOp::verify() {
  return verifyQPUVPMVCDWritePseudoOp(
      getOperation(), 49, getCondAdd(), getCondMul(), getOpAdd(), getOpMul(),
      getRaddrAAttr(), getRaddrBAttr(), getSmallImmAttr(), getMulA(),
      getMulB());
}

void mlir::vc4::QPUVPMVCDSetupLDIOp::getEffects(MemoryEffectList &effects) {
  if (getSide() == mlir::vc4::VPMVCDSide::read)
    addWriteEffect<mlir::vc4::effects::VDR>(effects);
  else
    addWriteEffect<mlir::vc4::effects::VDW>(effects);
}

LogicalResult mlir::vc4::QPUVPMVCDSetupLDIOp::verify() {
  if (failed(verifyScheduledFormOp(getOperation())))
    return failure();

  if (getMode() != mlir::vc4::LoadImmMode::splat32)
    return emitOpError("requires mode #vc4.load_imm_mode<splat32>");
  if (!getValueAttr().getType().isSignlessInteger(32))
    return emitOpError("requires a signless i32 'value' attribute");
  if (getPm())
    return emitOpError("requires pm = false");
  if (getCondAdd() == mlir::vc4::Cond::never)
    return emitOpError(
        "requires an active ADD-side write to VPM/VCD/VDW setup address 49");
  if (getCondMul() != mlir::vc4::Cond::never)
    return emitOpError("must not carry an active MUL-side write");
  if (failed(verifyOptionalLiteralAddress(getOperation(), "waddr_add",
                                          getWaddrAddAttr(), 49)))
    return failure();
  if (failed(verifyOptionalLiteralAddress(getOperation(), "waddr_mul",
                                          getWaddrMulAttr(), 32)))
    return failure();
  return success();
}

void mlir::vc4::QPUVPMVCDAddrOp::getEffects(MemoryEffectList &effects) {
  if (getSide() == mlir::vc4::VPMVCDSide::read)
    addWriteEffect<mlir::vc4::effects::VDR>(effects);
  else
    addWriteEffect<mlir::vc4::effects::VDW>(effects);
}

LogicalResult mlir::vc4::QPUVPMVCDAddrOp::verify() {
  return verifyQPUVPMVCDWritePseudoOp(
      getOperation(), 50, getCondAdd(), getCondMul(), getOpAdd(), getOpMul(),
      getRaddrAAttr(), getRaddrBAttr(), getSmallImmAttr(), getMulA(),
      getMulB());
}

void mlir::vc4::QPUVPMVCDWaitOp::getEffects(MemoryEffectList &effects) {
  if (getSide() == mlir::vc4::VPMVCDSide::read)
    addReadEffect<mlir::vc4::effects::VDR>(effects);
  else
    addReadEffect<mlir::vc4::effects::VDW>(effects);
}

LogicalResult mlir::vc4::QPUVPMVCDWaitOp::verify() {
  if (failed(verifyScheduledFormOp(getOperation())))
    return failure();

  if (failed(verifyOptionalQPUReadAddressAttr(getOperation(), "raddr_a",
                                              getRaddrAAttr())))
    return failure();
  if (getRaddrAAttr() && getRaddrAAttr().getInt() != 50)
    return emitOpError("'raddr_a' must be 50 when present");

  if (getWaddrAddAttr())
    return emitOpError("must be read-only and must not carry 'waddr_add'");
  if (getWaddrMulAttr())
    return emitOpError("must be read-only and must not carry 'waddr_mul'");
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
