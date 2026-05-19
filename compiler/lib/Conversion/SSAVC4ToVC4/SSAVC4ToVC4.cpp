//===- SSAVC4ToVC4.cpp - SSAVC4 to scheduled VC4 lowering ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.h"
#include "vc4/Dialect/VC4/IR/VC4Ops.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cstdint>
#include <optional>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kSSAVC4ModuleOpName("ssavc4.module");
constexpr llvm::StringLiteral kSSAVC4FuncOpName("ssavc4.func");
constexpr llvm::StringLiteral kSSAVC4ThreadEndOpName("ssavc4.thread_end");
constexpr llvm::StringLiteral kSSAVC4LoadImmOpName("ssavc4.load_imm");
constexpr llvm::StringLiteral kSSAVC4ElementNumberOpName("ssavc4.element_number");
constexpr llvm::StringLiteral kSSAVC4UniformReadOpName("ssavc4.uniform.read");
constexpr llvm::StringLiteral kSSAVC4SplatOpName("ssavc4.splat");
constexpr llvm::StringLiteral kSSAVC4MovOpName("ssavc4.mov");
constexpr llvm::StringLiteral kSSAVC4ALUAddOpName("ssavc4.alu.add");
constexpr llvm::StringLiteral kSSAVC4ALUMulOpName("ssavc4.alu.mul");
constexpr llvm::StringLiteral kSSAVC4MakeFlagsOpName("ssavc4.make_flags");
constexpr llvm::StringLiteral kSSAVC4BranchOpName("ssavc4.br");
constexpr llvm::StringLiteral kSSAVC4CondBranchOpName("ssavc4.cond_br");
constexpr llvm::StringLiteral kSSAVC4VDWStoreOpName("ssavc4.vdw.store");
constexpr llvm::StringLiteral kSSAVC4PackOpName("ssavc4.pack");
constexpr llvm::StringLiteral kSSAVC4UnpackOpName("ssavc4.unpack");
constexpr llvm::StringLiteral kSSAVC4RotateOpName("ssavc4.rotate");
constexpr llvm::StringLiteral kSSAVC4TMURequestOpName("ssavc4.tmu.request");
constexpr llvm::StringLiteral kSSAVC4TMUReadOpName("ssavc4.tmu.read");
constexpr llvm::StringLiteral kSSAVC4SemaAcquireOpName("ssavc4.sema.acquire");
constexpr llvm::StringLiteral kSSAVC4SemaReleaseOpName("ssavc4.sema.release");
constexpr llvm::StringLiteral kSSAVC4BarrierOpName("ssavc4.barrier");
constexpr llvm::StringLiteral kSSAVC4VPMWriteOpName("ssavc4.vpm.write");
constexpr llvm::StringLiteral kSSAVC4VPMReadOpName("ssavc4.vpm.read");

static bool hasName(Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static StringAttr getSymbolNameAttr(Operation *op) {
  return llvm::dyn_cast_or_null<StringAttr>(
      op->getAttr(SymbolTable::getSymbolAttrName()));
}

static FunctionType getFunctionType(Operation *op, MLIRContext *ctx) {
  if (auto typeAttr = llvm::dyn_cast_or_null<TypeAttr>(op->getAttr("function_type"))) {
    if (auto fnType = llvm::dyn_cast<FunctionType>(typeAttr.getValue()))
      return fnType;
  }
  return FunctionType::get(ctx, {}, {});
}

static void addAttrIfPresent(Operation *source, OperationState &state,
                             llvm::StringRef name) {
  if (Attribute attr = source->getAttr(name))
    state.addAttribute(name, attr);
}

static void addCommonBundleAttrs(OpBuilder &builder, OperationState &state,
                                 mlir::vc4::QPUSignal signal,
                                 mlir::vc4::Cond condAdd,
                                 mlir::vc4::Cond condMul,
                                 int64_t waddrAdd, int64_t waddrMul,
                                 mlir::vc4::AddOpcode addOpcode,
                                 mlir::vc4::MulOpcode mulOpcode,
                                 int64_t raddrA, int64_t raddrB,
                                 mlir::vc4::QPUMux addA,
                                 mlir::vc4::QPUMux addB,
                                 mlir::vc4::QPUMux mulA,
                                 mlir::vc4::QPUMux mulB) {
  MLIRContext *ctx = builder.getContext();
  state.addAttribute("sig", mlir::vc4::QPUSignalAttr::get(ctx, signal));
  state.addAttribute("pm", builder.getBoolAttr(false));
  state.addAttribute("cond_add", mlir::vc4::CondAttr::get(ctx, condAdd));
  state.addAttribute("cond_mul", mlir::vc4::CondAttr::get(ctx, condMul));
  state.addAttribute("waddr_add", builder.getI32IntegerAttr(waddrAdd));
  state.addAttribute("waddr_mul", builder.getI32IntegerAttr(waddrMul));
  state.addAttribute("op_add", mlir::vc4::AddOpcodeAttr::get(ctx, addOpcode));
  state.addAttribute("op_mul", mlir::vc4::MulOpcodeAttr::get(ctx, mulOpcode));
  state.addAttribute("raddr_a", builder.getI32IntegerAttr(raddrA));
  state.addAttribute("raddr_b", builder.getI32IntegerAttr(raddrB));
  state.addAttribute("add_a", mlir::vc4::QPUMuxAttr::get(ctx, addA));
  state.addAttribute("add_b", mlir::vc4::QPUMuxAttr::get(ctx, addB));
  state.addAttribute("mul_a", mlir::vc4::QPUMuxAttr::get(ctx, mulA));
  state.addAttribute("mul_b", mlir::vc4::QPUMuxAttr::get(ctx, mulB));
}

static Operation *createNopBundle(OpBuilder &builder, Location loc) {
  OperationState state(loc, "vc4.qpu.bundle");
  addCommonBundleAttrs(builder, state, mlir::vc4::QPUSignal::none,
                       mlir::vc4::Cond::never, mlir::vc4::Cond::never,
                       /*waddrAdd=*/32, /*waddrMul=*/33,
                       mlir::vc4::AddOpcode::nop, mlir::vc4::MulOpcode::nop,
                       /*raddrA=*/39, /*raddrB=*/39, mlir::vc4::QPUMux::a,
                       mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                       mlir::vc4::QPUMux::r1);
  return builder.create(state);
}

static Operation *createThreadEndBundle(OpBuilder &builder, Location loc) {
  OperationState state(loc, "vc4.qpu.bundle");
  addCommonBundleAttrs(builder, state, mlir::vc4::QPUSignal::thrend,
                       mlir::vc4::Cond::never, mlir::vc4::Cond::never,
                       /*waddrAdd=*/32, /*waddrMul=*/33,
                       mlir::vc4::AddOpcode::nop, mlir::vc4::MulOpcode::nop,
                       /*raddrA=*/0, /*raddrB=*/1, mlir::vc4::QPUMux::a,
                       mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                       mlir::vc4::QPUMux::r1);
  return builder.create(state);
}

static Operation *createScheduledBranch(OpBuilder &builder, Location loc,
                                        mlir::vc4::BranchCond cond,
                                        int64_t immediate,
                                        int64_t raddrA = 0,
                                        int64_t waddrAdd = 31,
                                        int64_t waddrMul = 30) {
  MLIRContext *ctx = builder.getContext();
  OperationState state(loc, "vc4.qpu.branch");
  state.addAttribute("cond", mlir::vc4::BranchCondAttr::get(ctx, cond));
  state.addAttribute("relative", builder.getBoolAttr(true));
  state.addAttribute("use_reg", builder.getBoolAttr(false));
  state.addAttribute("raddr_a", builder.getI32IntegerAttr(raddrA));
  state.addAttribute("immediate", builder.getI32IntegerAttr(immediate));
  state.addAttribute("waddr_add", builder.getI32IntegerAttr(waddrAdd));
  state.addAttribute("waddr_mul", builder.getI32IntegerAttr(waddrMul));
  state.addRegion();
  Operation *branch = builder.create(state);
  branch->getRegion(0).push_back(new Block());
  OpBuilder delayBuilder(ctx);
  delayBuilder.setInsertionPointToEnd(&branch->getRegion(0).front());
  createNopBundle(delayBuilder, loc);
  createNopBundle(delayBuilder, loc);
  createNopBundle(delayBuilder, loc);
  return branch;
}


static Operation *createScheduledBundle(
    OpBuilder &builder, Location loc, mlir::vc4::QPUSignal signal,
    mlir::vc4::Cond condAdd, mlir::vc4::Cond condMul, int64_t waddrAdd,
    int64_t waddrMul, mlir::vc4::AddOpcode addOpcode,
    mlir::vc4::MulOpcode mulOpcode, int64_t raddrA, int64_t raddrB,
    mlir::vc4::QPUMux addA, mlir::vc4::QPUMux addB, mlir::vc4::QPUMux mulA,
    mlir::vc4::QPUMux mulB, std::optional<int64_t> smallImm = std::nullopt,
    bool setFlags = false, bool writeSwap = false) {
  MLIRContext *ctx = builder.getContext();
  OperationState state(loc, "vc4.qpu.bundle");
  state.addAttribute("sig", mlir::vc4::QPUSignalAttr::get(ctx, signal));
  state.addAttribute("pm", builder.getBoolAttr(false));
  state.addAttribute("cond_add", mlir::vc4::CondAttr::get(ctx, condAdd));
  state.addAttribute("cond_mul", mlir::vc4::CondAttr::get(ctx, condMul));
  state.addAttribute("waddr_add", builder.getI32IntegerAttr(waddrAdd));
  state.addAttribute("waddr_mul", builder.getI32IntegerAttr(waddrMul));
  state.addAttribute("op_add", mlir::vc4::AddOpcodeAttr::get(ctx, addOpcode));
  state.addAttribute("op_mul", mlir::vc4::MulOpcodeAttr::get(ctx, mulOpcode));
  state.addAttribute("raddr_a", builder.getI32IntegerAttr(raddrA));
  if (smallImm)
    state.addAttribute("small_imm", builder.getI32IntegerAttr(*smallImm));
  else
    state.addAttribute("raddr_b", builder.getI32IntegerAttr(raddrB));
  state.addAttribute("add_a", mlir::vc4::QPUMuxAttr::get(ctx, addA));
  state.addAttribute("add_b", mlir::vc4::QPUMuxAttr::get(ctx, addB));
  state.addAttribute("mul_a", mlir::vc4::QPUMuxAttr::get(ctx, mulA));
  state.addAttribute("mul_b", mlir::vc4::QPUMuxAttr::get(ctx, mulB));
  if (setFlags)
    state.addAttribute("set_flags", builder.getUnitAttr());
  if (writeSwap)
    state.addAttribute("write_swap", builder.getUnitAttr());
  return builder.create(state);
}

static Operation *createVPMVCDWritePseudoOp(
    OpBuilder &builder, Location loc, llvm::StringRef opName,
    mlir::vc4::VPMVCDSide side, mlir::vc4::Cond condAdd,
    mlir::vc4::Cond condMul, mlir::vc4::AddOpcode addOpcode,
    mlir::vc4::MulOpcode mulOpcode, int64_t raddrA, int64_t raddrB,
    mlir::vc4::QPUMux addA, mlir::vc4::QPUMux addB,
    mlir::vc4::QPUMux mulA, mlir::vc4::QPUMux mulB,
    std::optional<int64_t> smallImm = std::nullopt) {
  MLIRContext *ctx = builder.getContext();
  OperationState state(loc, opName);
  state.addAttribute("side", mlir::vc4::VPMVCDSideAttr::get(ctx, side));
  state.addAttribute("pm", builder.getBoolAttr(false));
  state.addAttribute("cond_add", mlir::vc4::CondAttr::get(ctx, condAdd));
  state.addAttribute("cond_mul", mlir::vc4::CondAttr::get(ctx, condMul));
  state.addAttribute("op_add", mlir::vc4::AddOpcodeAttr::get(ctx, addOpcode));
  state.addAttribute("op_mul", mlir::vc4::MulOpcodeAttr::get(ctx, mulOpcode));
  state.addAttribute("raddr_a", builder.getI32IntegerAttr(raddrA));
  if (smallImm)
    state.addAttribute("small_imm", builder.getI32IntegerAttr(*smallImm));
  else
    state.addAttribute("raddr_b", builder.getI32IntegerAttr(raddrB));
  state.addAttribute("add_a", mlir::vc4::QPUMuxAttr::get(ctx, addA));
  state.addAttribute("add_b", mlir::vc4::QPUMuxAttr::get(ctx, addB));
  state.addAttribute("mul_a", mlir::vc4::QPUMuxAttr::get(ctx, mulA));
  state.addAttribute("mul_b", mlir::vc4::QPUMuxAttr::get(ctx, mulB));
  return builder.create(state);
}

static Operation *createVPMVCDSetup(OpBuilder &builder, Location loc,
                                    mlir::vc4::VPMVCDSide side,
                                    mlir::vc4::Cond condAdd,
                                    mlir::vc4::Cond condMul,
                                    mlir::vc4::AddOpcode addOpcode,
                                    mlir::vc4::MulOpcode mulOpcode,
                                    int64_t raddrA, int64_t raddrB,
                                    mlir::vc4::QPUMux addA,
                                    mlir::vc4::QPUMux addB,
                                    mlir::vc4::QPUMux mulA,
                                    mlir::vc4::QPUMux mulB,
                                    std::optional<int64_t> smallImm =
                                        std::nullopt) {
  return createVPMVCDWritePseudoOp(
      builder, loc, "vc4.qpu.vpmvcd_setup", side, condAdd, condMul,
      addOpcode, mulOpcode, raddrA, raddrB, addA, addB, mulA, mulB, smallImm);
}

static Operation *createVPMVCDAddr(OpBuilder &builder, Location loc,
                                   mlir::vc4::VPMVCDSide side,
                                   mlir::vc4::Cond condAdd,
                                   mlir::vc4::Cond condMul,
                                   mlir::vc4::AddOpcode addOpcode,
                                   mlir::vc4::MulOpcode mulOpcode,
                                   int64_t raddrA, int64_t raddrB,
                                   mlir::vc4::QPUMux addA,
                                   mlir::vc4::QPUMux addB,
                                   mlir::vc4::QPUMux mulA,
                                   mlir::vc4::QPUMux mulB,
                                   std::optional<int64_t> smallImm =
                                       std::nullopt) {
  return createVPMVCDWritePseudoOp(
      builder, loc, "vc4.qpu.vpmvcd_addr", side, condAdd, condMul, addOpcode,
      mulOpcode, raddrA, raddrB, addA, addB, mulA, mulB, smallImm);
}

static Operation *createVPMVCDWait(OpBuilder &builder, Location loc,
                                   mlir::vc4::VPMVCDSide side) {
  MLIRContext *ctx = builder.getContext();
  OperationState state(loc, "vc4.qpu.vpmvcd_wait");
  state.addAttribute("side", mlir::vc4::VPMVCDSideAttr::get(ctx, side));
  return builder.create(state);
}

static Operation *createSplat32LDIWithMul(OpBuilder &builder, Location loc,
                                          int64_t value, int64_t waddrAdd,
                                          int64_t waddrMul) {
  MLIRContext *ctx = builder.getContext();
  OperationState state(loc, "vc4.qpu.ldi");
  state.addAttribute("mode", mlir::vc4::LoadImmModeAttr::get(ctx, mlir::vc4::LoadImmMode::splat32));
  state.addAttribute("value", builder.getI32IntegerAttr(value));
  state.addAttribute("pm", builder.getBoolAttr(false));
  state.addAttribute("cond_add", mlir::vc4::CondAttr::get(ctx, mlir::vc4::Cond::always));
  state.addAttribute("cond_mul",
                     mlir::vc4::CondAttr::get(
                         ctx, waddrMul == 32 ? mlir::vc4::Cond::never
                                             : mlir::vc4::Cond::always));
  state.addAttribute("waddr_add", builder.getI32IntegerAttr(waddrAdd));
  state.addAttribute("waddr_mul", builder.getI32IntegerAttr(waddrMul));
  return builder.create(state);
}

static Operation *createSplat32LDI(OpBuilder &builder, Location loc,
                                   int64_t value, int64_t waddrAdd) {
  return createSplat32LDIWithMul(builder, loc, value, waddrAdd, /*waddrMul=*/32);
}

static Operation *createNopLDISlot(OpBuilder &builder, Location loc) {
  MLIRContext *ctx = builder.getContext();
  OperationState state(loc, "vc4.qpu.ldi");
  state.addAttribute("mode", mlir::vc4::LoadImmModeAttr::get(
                                 ctx, mlir::vc4::LoadImmMode::splat32));
  state.addAttribute("value", builder.getI32IntegerAttr(0));
  state.addAttribute("pm", builder.getBoolAttr(false));
  state.addAttribute("cond_add",
                     mlir::vc4::CondAttr::get(ctx, mlir::vc4::Cond::never));
  state.addAttribute("cond_mul",
                     mlir::vc4::CondAttr::get(ctx, mlir::vc4::Cond::never));
  state.addAttribute("waddr_add", builder.getI32IntegerAttr(32));
  state.addAttribute("waddr_mul", builder.getI32IntegerAttr(33));
  return builder.create(state);
}

static Operation *emitMutexRelease(OpBuilder &builder, Location loc) {
  return createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                               mlir::vc4::Cond::always,
                               mlir::vc4::Cond::never,
                               /*waddrAdd=*/51, /*waddrMul=*/32,
                               mlir::vc4::AddOpcode::add,
                               mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                               /*raddrB=*/0, mlir::vc4::QPUMux::b,
                               mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                               mlir::vc4::QPUMux::r1,
                               /*smallImm=*/0);
}

static int64_t getI32IntegerAttrOr(Operation *op, llvm::StringRef name,
                                   int64_t fallback) {
  auto attr = llvm::dyn_cast_or_null<IntegerAttr>(op->getAttr(name));
  if (!attr)
    return fallback;
  return attr.getInt();
}

static bool isVector16I32Type(Type type) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 && vectorType.getDimSize(0) == 16 &&
         vectorType.getElementType().isSignlessInteger(32);
}

static bool isVector16F32Type(Type type) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  return vectorType && vectorType.getRank() == 1 && vectorType.getDimSize(0) == 16 &&
         vectorType.getElementType().isF32();
}

static bool isScalarI32OrVector16I32(Type type) {
  return type.isSignlessInteger(32) || isVector16I32Type(type);
}

static bool isMirroredLoadImm(Value value) {
  return hasName(value.getDefiningOp(), kSSAVC4LoadImmOpName);
}

// Lowering-private representation seams.  Slice 3 uses deliberately small
// implementations, but keeps the explicit phase boundaries needed by later
// instruction selection/templates, virtual values, liveness, no-spill
// allocation, conservative scheduling, hazard insertion, and branch layout.
struct VirtualValue {
  Value value;
  unsigned ordinal = 0;
};

struct InstructionTemplate {
  enum class Kind {
    LoadImm,
    ElementNumber,
    UniformRead,
    Splat,
    Mov,
    ALUAdd,
    ALUMul,
    MakeFlags,
    Branch,
    CondBranch,
    Pack,
    Unpack,
    Rotate,
    TMURequest,
    TMURead,
    SemaAcquire,
    SemaRelease,
    Barrier,
    VPMWrite,
    VPMRead,
    VDWStore,
    ThreadEnd
  } kind;
  Operation *source = nullptr;
  Block *sourceBlock = nullptr;
  SmallVector<Value, 2> operands;
  std::optional<Value> result;
};

struct LivenessSummary {
  unsigned virtualValueCount = 0;
  DenseMap<Value, unsigned> lastUseIndex;
  DenseMap<Value, SmallVector<unsigned, 4>> useIndices;
};

enum class LocationKind { Register, SpillSlot };

struct SpillSlot {
  unsigned index = 0;
  uint32_t offsetBytes = 0;
  uint32_t sizeBytes = 64;
};

struct ValueLocation {
  LocationKind kind = LocationKind::Register;
  int64_t physicalReg = -1;
  unsigned spillSlotIndex = 0;
};

struct SpillPlan {
  DenseMap<Value, ValueLocation> locations;
  DenseMap<Value, SpillSlot> spillSlots;
  uint32_t spillFrameBytes = 0;
  uint32_t spillSlotCount = 0;
};

struct SpillAction {
  enum class Kind { Store, Reload };
  Kind kind = Kind::Store;
  Value value;
  SpillSlot slot;
  int64_t physicalReg = -1;
};

struct PerTemplateAllocation {
  DenseMap<Value, int64_t> operandRegisters;
  std::optional<int64_t> resultRegister;
  SmallVector<SpillAction, 4> preActions;
};

static bool isAllocatableSSAValue(Value value,
                                  ArrayRef<VirtualValue> virtualValues) {
  for (const VirtualValue &virtualValue : virtualValues)
    if (virtualValue.value == value)
      return true;
  return false;
}

static LivenessSummary computeLiveness(
    ArrayRef<InstructionTemplate> templates, ArrayRef<VirtualValue> virtualValues) {
  LivenessSummary summary;
  summary.virtualValueCount = virtualValues.size();
  for (unsigned index = 0; index < templates.size(); ++index) {
    for (Value operand : templates[index].operands) {
      if (isAllocatableSSAValue(operand, virtualValues)) {
        summary.lastUseIndex[operand] = index;
        summary.useIndices[operand].push_back(index);
      }
    }
  }
  return summary;
}

static DictionaryAttr getResourceMetadata(Operation *func);
static std::optional<llvm::StringRef> getResourceString(DictionaryAttr resource,
                                                        llvm::StringRef name);

class SpillAwareAllocator {
public:
  LogicalResult allocate(Operation *diagnosticAnchor,
                         ArrayRef<InstructionTemplate> templates,
                         ArrayRef<VirtualValue> virtualValues,
                         const LivenessSummary &liveness) {
    if (succeeded(allocateWithoutSpills(templates, virtualValues, liveness)))
      return success();

    if (failed(verifyS3SpillingSupported(diagnosticAnchor, templates)))
      return failure();

    return allocateWithSpills(diagnosticAnchor, templates, virtualValues,
                              liveness);
  }

  std::optional<int64_t> lookup(Value value) const {
    auto it = registers.find(value);
    if (it == registers.end())
      return std::nullopt;
    return it->second;
  }

  std::optional<int64_t> lookup(const InstructionTemplate &templ,
                                Value value) const {
    if (!hasSpills())
      return lookup(value);
    if (templ.source) {
      auto planIt = perTemplate.find(templ.source);
      if (planIt != perTemplate.end()) {
        auto operandIt = planIt->second.operandRegisters.find(value);
        if (operandIt != planIt->second.operandRegisters.end())
          return operandIt->second;
        if (templ.result && *templ.result == value &&
            planIt->second.resultRegister)
          return *planIt->second.resultRegister;
      }
    }
    return std::nullopt;
  }

  ArrayRef<SpillAction> getPreActions(const InstructionTemplate &templ) const {
    static const SmallVector<SpillAction, 0> empty;
    if (!templ.source)
      return empty;
    auto it = perTemplate.find(templ.source);
    if (it == perTemplate.end())
      return empty;
    return it->second.preActions;
  }

  bool hasSpills() const { return spillPlan.spillSlotCount != 0; }

  uint32_t getSpillSlotCount() const { return spillPlan.spillSlotCount; }
  uint32_t getSpillFrameBytes() const { return spillPlan.spillFrameBytes; }
  static constexpr int64_t spillBaseReg() { return kSpillBaseReg; }
  static constexpr int64_t spillOffsetReg() { return kSpillOffsetReg; }
  static constexpr int64_t spillAddrReg() { return kSpillAddrReg; }
  static constexpr int64_t spillLaneReg() { return kSpillLaneReg; }
  static constexpr int64_t spillRowReg() { return kSpillRowReg; }

private:
  static constexpr int64_t kForbiddenThreadEndHazardReg = 14;
  static constexpr int64_t kSpillBaseReg = 28;
  static constexpr int64_t kSpillOffsetReg = 29;
  static constexpr int64_t kSpillAddrReg = 30;
  static constexpr int64_t kSpillLaneReg = 31;
  static constexpr int64_t kSpillRowReg = 27;

  LogicalResult allocateWithoutSpills(ArrayRef<InstructionTemplate> templates,
                                      ArrayRef<VirtualValue> virtualValues,
                                      const LivenessSummary &liveness) {
    static constexpr int64_t kForbiddenThreadEndHazardReg = 14;
    int64_t nextRegister = 0;
    SmallVector<int64_t, 8> freeRegisters;
    auto isReservedOrForbidden = [&](int64_t reg) {
      return reg == kForbiddenThreadEndHazardReg;
    };
    auto allocateRegister = [&]() -> std::optional<int64_t> {
      if (!freeRegisters.empty()) {
        int64_t reg = freeRegisters.pop_back_val();
        return reg;
      }
      while (isReservedOrForbidden(nextRegister))
        ++nextRegister;
      if (nextRegister > 31)
        return std::nullopt;
      return nextRegister++;
    };
    DenseMap<Value, bool> releasedValues;
    auto releaseIfLastUse = [&](Value value, unsigned index) {
      if (releasedValues.find(value) != releasedValues.end())
        return;
      auto regIt = registers.find(value);
      if (regIt == registers.end())
        return;
      auto lastUseIt = liveness.lastUseIndex.find(value);
      if (lastUseIt == liveness.lastUseIndex.end() || lastUseIt->second != index)
        return;
      freeRegisters.push_back(regIt->second);
      releasedValues[value] = true;
    };

    DenseMap<Value, bool> needsAllocation;
    for (const VirtualValue &virtualValue : virtualValues)
      needsAllocation[virtualValue.value] = true;

    for (unsigned index = 0; index < templates.size(); ++index) {
      const InstructionTemplate &templ = templates[index];
      if (templ.result && needsAllocation.find(*templ.result) != needsAllocation.end()) {
        std::optional<int64_t> reg = allocateRegister();
        if (!reg) {
          registers.clear();
          return failure();
        }
        registers.try_emplace(*templ.result, *reg);
      }
      for (Value operand : templ.operands)
        releaseIfLastUse(operand, index);
    }
    return success();
  }

  LogicalResult verifyS3SpillingSupported(
      Operation *diagnosticAnchor, ArrayRef<InstructionTemplate> templates) const {
    DictionaryAttr resource = getResourceMetadata(diagnosticAnchor);
    std::optional<llvm::StringRef> scheduleMode =
        getResourceString(resource, "schedule_mode");
    if (scheduleMode && *scheduleMode == "cooperative_block") {
      return diagnosticAnchor->emitError()
             << "S3 spilling supports only independent-vector data values; "
                "cooperative/control path requires later spill support";
    }

    bool sawNonUniform = false;
    DenseMap<Block *, unsigned> blockOrder;
    for (const InstructionTemplate &templ : templates) {
      if (templ.sourceBlock && !blockOrder.count(templ.sourceBlock))
        blockOrder[templ.sourceBlock] = blockOrder.size();
    }

    for (const InstructionTemplate &templ : templates) {
      switch (templ.kind) {
      case InstructionTemplate::Kind::Branch:
      case InstructionTemplate::Kind::CondBranch:
        if (!templ.source || !templ.sourceBlock ||
            templ.source->getNumSuccessors() == 0)
          return diagnosticAnchor->emitError()
                 << "internal lowering error: malformed branch template";
        for (Block *successor : templ.source->getSuccessors()) {
          auto sourceIt = blockOrder.find(templ.sourceBlock);
          auto successorIt = blockOrder.find(successor);
          if (sourceIt == blockOrder.end() || successorIt == blockOrder.end())
            return templ.source->emitOpError()
                   << "S3 spilling requires branch successors to remain in "
                      "the current function layout";
          if (successorIt->second <= sourceIt->second)
            return templ.source->emitOpError()
                   << "S3 spilling does not support loop/backedge branch "
                      "layouts requiring path-sensitive liveness";
        }
        sawNonUniform = true;
        break;
      case InstructionTemplate::Kind::Barrier:
      case InstructionTemplate::Kind::SemaAcquire:
      case InstructionTemplate::Kind::SemaRelease:
      case InstructionTemplate::Kind::VPMWrite:
      case InstructionTemplate::Kind::VPMRead:
        return templ.source->emitOpError()
               << "S3 spilling supports only independent-vector data values; "
                  "value in cooperative/control path requires later spill "
                  "support";
      case InstructionTemplate::Kind::UniformRead:
        if (sawNonUniform) {
          return templ.source->emitOpError()
                 << "S3 spilling requires all ssavc4.uniform.read operations "
                    "before spillable data operations";
        }
        break;
      default:
        sawNonUniform = true;
        break;
      }
    }
    return success();
  }

  bool isReservedInSpillMode(int64_t reg) const {
    return reg == kForbiddenThreadEndHazardReg || reg == kSpillBaseReg ||
           reg == kSpillOffsetReg || reg == kSpillAddrReg ||
           reg == kSpillLaneReg || reg == kSpillRowReg;
  }

  bool isValueSpillable(Value value) const {
    Type type = value.getType();
    return isVector16I32Type(type) || isVector16F32Type(type);
  }

  std::optional<unsigned> nextUseAfter(Value value, unsigned index,
                                       const LivenessSummary &liveness) const {
    auto it = liveness.useIndices.find(value);
    if (it == liveness.useIndices.end())
      return std::nullopt;
    for (unsigned useIndex : it->second)
      if (useIndex > index)
        return useIndex;
    return std::nullopt;
  }

  SpillSlot getOrCreateSpillSlot(Value value) {
    auto it = spillPlan.spillSlots.find(value);
    if (it != spillPlan.spillSlots.end())
      return it->second;
    SpillSlot slot;
    slot.index = spillPlan.spillSlotCount++;
    slot.offsetBytes = slot.index * slot.sizeBytes;
    spillPlan.spillFrameBytes = spillPlan.spillSlotCount * slot.sizeBytes;
    spillPlan.spillSlots[value] = slot;
    return slot;
  }

  LogicalResult allocateWithSpills(Operation *diagnosticAnchor,
                                   ArrayRef<InstructionTemplate> templates,
                                   ArrayRef<VirtualValue> virtualValues,
                                   const LivenessSummary &liveness) {
    SmallVector<int64_t, 32> freeRegisters;
    for (int64_t reg = 31; reg >= 0; --reg)
      if (!isReservedInSpillMode(reg))
        freeRegisters.push_back(reg);

    DenseMap<Value, int64_t> activeRegisters;
    DenseMap<int64_t, Value> registerValues;
    DenseMap<Value, bool> slotValid;
    DenseMap<Value, bool> needsAllocation;
    for (const VirtualValue &virtualValue : virtualValues) {
      needsAllocation[virtualValue.value] = true;
    }

    auto popFreeRegister = [&]() -> std::optional<int64_t> {
      if (freeRegisters.empty())
        return std::nullopt;
      return freeRegisters.pop_back_val();
    };

    auto protectContains = [](ArrayRef<Value> protectedValues, Value value) {
      for (Value protectedValue : protectedValues)
        if (protectedValue == value)
          return true;
      return false;
    };

    auto acquireRegister =
        [&](unsigned index, ArrayRef<Value> protectedValues,
            SmallVectorImpl<SpillAction> &actions) -> FailureOr<int64_t> {
      if (std::optional<int64_t> reg = popFreeRegister())
        return *reg;

      Value victim;
      int64_t victimReg = -1;
      std::optional<unsigned> farthestNextUse;
      bool foundVictim = false;
      for (auto &entry : activeRegisters) {
        Value candidate = entry.first;
        if (protectContains(protectedValues, candidate))
          continue;
        if (!isValueSpillable(candidate))
          continue;
        std::optional<unsigned> nextUse =
            nextUseAfter(candidate, index, liveness);
        if (!foundVictim || !nextUse ||
            (farthestNextUse && *nextUse > *farthestNextUse)) {
          victim = candidate;
          victimReg = entry.second;
          farthestNextUse = nextUse;
          foundVictim = true;
          if (!nextUse)
            break;
        }
      }
      if (!foundVictim)
        return failure();

      SpillSlot slot = getOrCreateSpillSlot(victim);
      if (!slotValid[victim]) {
        actions.push_back(
            SpillAction{SpillAction::Kind::Store, victim, slot, victimReg});
        slotValid[victim] = true;
      }
      spillPlan.locations[victim] =
          ValueLocation{LocationKind::SpillSlot, -1, slot.index};
      activeRegisters.erase(victim);
      registerValues.erase(victimReg);
      return victimReg;
    };

    auto ensureFutureLiveValuesHaveBranchSlots =
        [&](unsigned index, SmallVectorImpl<SpillAction> &actions) {
          for (const VirtualValue &virtualValue : virtualValues) {
            Value value = virtualValue.value;
            auto activeIt = activeRegisters.find(value);
            if (activeIt == activeRegisters.end())
              continue;
            if (!isValueSpillable(value))
              continue;
            if (!nextUseAfter(value, index, liveness))
              continue;
            SpillSlot slot = getOrCreateSpillSlot(value);
            if (slotValid[value])
              continue;
            actions.push_back(
                SpillAction{SpillAction::Kind::Store, value, slot,
                            activeIt->second});
            slotValid[value] = true;
          }
        };

    for (unsigned index = 0; index < templates.size(); ++index) {
      const InstructionTemplate &templ = templates[index];
      PerTemplateAllocation &allocation = perTemplate[templ.source];

      SmallVector<Value, 2> protectedOperands;
      for (Value operand : templ.operands)
        if (needsAllocation.count(operand))
          protectedOperands.push_back(operand);

      for (Value operand : templ.operands) {
        if (!needsAllocation.count(operand))
          continue;
        auto activeIt = activeRegisters.find(operand);
        if (activeIt == activeRegisters.end()) {
          SpillSlot slot = getOrCreateSpillSlot(operand);
          FailureOr<int64_t> reg =
              acquireRegister(index, protectedOperands, allocation.preActions);
          if (failed(reg)) {
            return templ.source->emitOpError()
                   << "S3 spilling supports only independent-vector data "
                      "values; no scratch register was available to reload an "
                      "operand";
          }
          activeRegisters[operand] = *reg;
          registerValues[*reg] = operand;
          allocation.preActions.push_back(
              SpillAction{SpillAction::Kind::Reload, operand, slot, *reg});
          activeIt = activeRegisters.find(operand);
        }
        allocation.operandRegisters[operand] = activeIt->second;
      }

      if (templ.kind == InstructionTemplate::Kind::Branch ||
          templ.kind == InstructionTemplate::Kind::CondBranch)
        ensureFutureLiveValuesHaveBranchSlots(index, allocation.preActions);

      if (templ.result && needsAllocation.count(*templ.result)) {
        FailureOr<int64_t> reg =
            acquireRegister(index, protectedOperands, allocation.preActions);
        if (failed(reg)) {
          return templ.source->emitOpError()
                 << "S3 spilling supports only independent-vector data values; "
                    "no register was available for a spillable result";
        }
        allocation.resultRegister = *reg;
        registers[*templ.result] = *reg;
        activeRegisters[*templ.result] = *reg;
        registerValues[*reg] = *templ.result;
        spillPlan.locations[*templ.result] =
            ValueLocation{LocationKind::Register, *reg, 0};
        slotValid[*templ.result] = false;
      }

      for (Value operand : templ.operands) {
        if (!needsAllocation.count(operand))
          continue;
        auto lastUseIt = liveness.lastUseIndex.find(operand);
        if (lastUseIt == liveness.lastUseIndex.end() ||
            lastUseIt->second != index)
          continue;
        auto activeIt = activeRegisters.find(operand);
        if (activeIt == activeRegisters.end())
          continue;
        freeRegisters.push_back(activeIt->second);
        registerValues.erase(activeIt->second);
        activeRegisters.erase(activeIt);
      }

      if (templ.result && needsAllocation.count(*templ.result) &&
          liveness.lastUseIndex.find(*templ.result) ==
              liveness.lastUseIndex.end()) {
        int64_t reg = *allocation.resultRegister;
        freeRegisters.push_back(reg);
        registerValues.erase(reg);
        activeRegisters.erase(*templ.result);
      }
    }

    if (spillPlan.spillSlotCount == 0) {
      return diagnosticAnchor->emitError()
             << "ssavc4-to-vc4 allocator exhausted available QPU registers "
                "but did not produce a spill plan";
    }
    return success();
  }

  DenseMap<Value, int64_t> registers;
  DenseMap<Operation *, PerTemplateAllocation> perTemplate;
  SpillPlan spillPlan;
};

static bool isPhysicalRegFileWriteAddress(int64_t waddr) {
  return waddr >= 0 && waddr < 32;
}

static bool emitsPhysicalRegFileResultWrite(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator) {
  if (!templ.result)
    return false;
  if (templ.result->use_empty())
    return false;
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  return resultReg && isPhysicalRegFileWriteAddress(*resultReg);
}

static unsigned getRegfileResultSpacerSlotCount(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator) {
  return emitsPhysicalRegFileResultWrite(templ, allocator) ? 1 : 0;
}

static unsigned getSpillSlotBaseSlotCount(const SpillSlot &slot) {
  return slot.offsetBytes == 0 ? 2 : 4;
}

static unsigned getRawVDWStoreSlotCount(bool useMutex,
                                        bool hasDynamicActiveLanes,
                                        bool hasDynamicVPMRow,
                                        int64_t vpmRow) {
  (void)hasDynamicActiveLanes;
  unsigned count = 17;
  if (useMutex)
    count += 2;
  if (hasDynamicVPMRow && vpmRow != 0)
    count += 4;
  return count;
}

static unsigned getSpillActionSlotCount(const SpillAction &action) {
  if (action.kind == SpillAction::Kind::Store)
    return getSpillSlotBaseSlotCount(action.slot) +
           getRawVDWStoreSlotCount(/*useMutex=*/true,
                                   /*hasDynamicActiveLanes=*/false,
                                   /*hasDynamicVPMRow=*/true,
                                   /*vpmRow=*/0);
  return getSpillSlotBaseSlotCount(action.slot) + 13;
}

static void emitRegfileResultSpacer(OpBuilder &builder, Location loc,
                                    const InstructionTemplate &templ,
                                    const SpillAwareAllocator &allocator) {
  if (emitsPhysicalRegFileResultWrite(templ, allocator))
    createNopLDISlot(builder, loc);
}

struct ScheduledTemplate {
  InstructionTemplate templ;
};

class ConservativeScheduler {
public:
  SmallVector<ScheduledTemplate, 8>
  schedule(ArrayRef<InstructionTemplate> templates, const LivenessSummary &) {
    SmallVector<ScheduledTemplate, 8> scheduled;
    for (const InstructionTemplate &templ : templates)
      scheduled.push_back(ScheduledTemplate{templ});
    return scheduled;
  }
};

struct LayoutSummary {
  DenseMap<Block *, unsigned> blockStartSlots;
  DenseMap<Operation *, unsigned> opStartSlots;
  DenseMap<Operation *, int64_t> branchImmediates;
};

static unsigned getFlattenedSlotCount(const InstructionTemplate &templ,
                                      const SpillAwareAllocator &allocator) {
  unsigned spillActionSlots = 0;
  for (const SpillAction &action : allocator.getPreActions(templ))
    spillActionSlots += getSpillActionSlotCount(action);

  unsigned resultSpacer = getRegfileResultSpacerSlotCount(templ, allocator);
  switch (templ.kind) {
  case InstructionTemplate::Kind::Branch:
  case InstructionTemplate::Kind::CondBranch:
    return spillActionSlots + 4;
  case InstructionTemplate::Kind::ThreadEnd:
    return spillActionSlots + 3;
  case InstructionTemplate::Kind::TMURequest:
    return spillActionSlots + 3;
  case InstructionTemplate::Kind::TMURead:
    return spillActionSlots + 2 + resultSpacer;
  case InstructionTemplate::Kind::Barrier:
    if (templ.operands.size() == 2)
      return spillActionSlots + 65;
    return spillActionSlots + 8;
  case InstructionTemplate::Kind::VPMWrite:
    if (auto serialize =
            llvm::dyn_cast_or_null<StringAttr>(templ.source->getAttr("serialize")))
      return spillActionSlots + (serialize.getValue() == "mutex" ? 6 : 3);
    return spillActionSlots + 3;
  case InstructionTemplate::Kind::VPMRead:
    if (auto serialize =
            llvm::dyn_cast_or_null<StringAttr>(templ.source->getAttr("serialize")))
      return spillActionSlots + (serialize.getValue() == "mutex" ? 8 : 6) +
             resultSpacer;
    return spillActionSlots + 6 + resultSpacer;
  case InstructionTemplate::Kind::VDWStore:
    if (auto serialize =
            llvm::dyn_cast_or_null<StringAttr>(templ.source->getAttr("serialize")))
      return spillActionSlots + (serialize.getValue() == "mutex" ? 19 : 17);
    return spillActionSlots + 17;
  case InstructionTemplate::Kind::Rotate:
    return spillActionSlots + 3 + resultSpacer;
  case InstructionTemplate::Kind::LoadImm:
  case InstructionTemplate::Kind::ElementNumber:
  case InstructionTemplate::Kind::UniformRead:
  case InstructionTemplate::Kind::Pack:
  case InstructionTemplate::Kind::Unpack:
    return spillActionSlots + 1 + resultSpacer;
  case InstructionTemplate::Kind::SemaAcquire:
  case InstructionTemplate::Kind::SemaRelease:
    return spillActionSlots + 1;
  case InstructionTemplate::Kind::Splat:
  case InstructionTemplate::Kind::Mov:
    return spillActionSlots + 1 + resultSpacer;
  case InstructionTemplate::Kind::ALUAdd:
    return spillActionSlots +
           (templ.operands.size() == 2 && !isMirroredLoadImm(templ.operands[1])
                ? 1
                : 0) +
           1 + resultSpacer;
  case InstructionTemplate::Kind::ALUMul:
    return spillActionSlots +
           (templ.operands.size() == 2 && !isMirroredLoadImm(templ.operands[1])
                ? 1
                : 0) +
           2 + (2 * resultSpacer);
  case InstructionTemplate::Kind::MakeFlags:
    return spillActionSlots +
           (templ.operands.size() == 2 && !isMirroredLoadImm(templ.operands[1])
               ? 2
               : 1);
  }
  return spillActionSlots + 1;
}

class BranchLayoutPlanner {
public:
  LogicalResult compute(Operation *diagnosticAnchor,
                        ArrayRef<ScheduledTemplate> scheduled,
                        const SpillAwareAllocator &allocator,
                        LayoutSummary &layout) {
    unsigned slot = 0;
    for (const ScheduledTemplate &scheduledTemplate : scheduled) {
      const InstructionTemplate &templ = scheduledTemplate.templ;
      if (templ.sourceBlock)
        layout.blockStartSlots.try_emplace(templ.sourceBlock, slot);
      unsigned preActionSlots = 0;
      for (const SpillAction &action : allocator.getPreActions(templ))
        preActionSlots += getSpillActionSlotCount(action);
      if (templ.source)
        layout.opStartSlots.try_emplace(templ.source, slot + preActionSlots);
      slot += getFlattenedSlotCount(templ, allocator);
    }

    for (const ScheduledTemplate &scheduledTemplate : scheduled) {
      const InstructionTemplate &templ = scheduledTemplate.templ;
      if (templ.kind != InstructionTemplate::Kind::Branch &&
          templ.kind != InstructionTemplate::Kind::CondBranch)
        continue;

      Operation *branchOp = templ.source;
      if (!branchOp || branchOp->getNumSuccessors() == 0)
        return diagnosticAnchor->emitError()
               << "internal lowering error: branch template has no successor";

      Block *target = branchOp->getSuccessor(0);
      auto sourceIt = layout.opStartSlots.find(branchOp);
      auto targetIt = layout.blockStartSlots.find(target);
      if (sourceIt == layout.opStartSlots.end() ||
          targetIt == layout.blockStartSlots.end())
        return branchOp->emitError()
               << "could not compute final scheduled branch layout for successor";

      int64_t sourceSlot = static_cast<int64_t>(sourceIt->second);
      int64_t targetSlot = static_cast<int64_t>(targetIt->second);
      layout.branchImmediates[branchOp] = (targetSlot - sourceSlot) * 8;
    }
    return success();
  }
};

static bool hasStringAttr(Operation *op, llvm::StringRef name,
                          llvm::StringRef expected) {
  auto attr = llvm::dyn_cast_or_null<StringAttr>(op->getAttr(name));
  return attr && attr.getValue() == expected;
}

static DictionaryAttr getResourceMetadata(Operation *func) {
  return llvm::dyn_cast_or_null<DictionaryAttr>(func->getAttr("vc4.resource"));
}

static std::optional<int64_t> getResourceI32(DictionaryAttr resource,
                                            llvm::StringRef name) {
  if (!resource)
    return std::nullopt;
  auto attr = llvm::dyn_cast_or_null<IntegerAttr>(resource.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static std::optional<bool> getResourceBool(DictionaryAttr resource,
                                           llvm::StringRef name) {
  if (!resource)
    return std::nullopt;
  if (auto attr = llvm::dyn_cast_or_null<BoolAttr>(resource.get(name)))
    return attr.getValue();
  return std::nullopt;
}

static std::optional<llvm::StringRef> getResourceString(DictionaryAttr resource,
                                                        llvm::StringRef name) {
  if (!resource)
    return std::nullopt;
  auto attr = llvm::dyn_cast_or_null<StringAttr>(resource.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

static std::optional<int64_t> getConstantI32FromLoadImm(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, kSSAVC4LoadImmOpName))
    return std::nullopt;
  auto attr = llvm::dyn_cast_or_null<IntegerAttr>(def->getAttr("value"));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static std::optional<int64_t> getSmallImmLiteralSelector(Value value) {
  std::optional<int64_t> constant = getConstantI32FromLoadImm(value);
  if (!constant || *constant < 0 || *constant > 15)
    return std::nullopt;
  return constant;
}

static LogicalResult verifyCooperativeBarrierResources(Operation *func) {
  if (func->getNumRegions() == 0 || func->getRegion(0).empty())
    return success();

  bool sawBarrier = false;
  int64_t requiredSemaphores = 0;
  for (Block &block : func->getRegion(0)) {
    for (Operation &op : block) {
      if (hasName(&op, kSSAVC4BarrierOpName)) {
        sawBarrier = true;
        int64_t arrive = getI32IntegerAttrOr(&op, "arrive_offset", -1);
        int64_t go = getI32IntegerAttrOr(&op, "go_offset", -1);
        int64_t depart = getI32IntegerAttrOr(&op, "depart_offset", -1);
        int64_t reset = getI32IntegerAttrOr(&op, "reset_offset", -1);
        if (arrive < 0 || go < 0 || depart < 0 || reset < 0)
          return op.emitOpError()
                 << "requires non-negative arrive/go/depart/reset semaphore offsets";
        requiredSemaphores = std::max(requiredSemaphores, arrive + 1);
        requiredSemaphores = std::max(requiredSemaphores, go + 1);
        requiredSemaphores = std::max(requiredSemaphores, depart + 1);
        requiredSemaphores = std::max(requiredSemaphores, reset + 1);
      }

      if (hasName(&op, kSSAVC4SemaAcquireOpName) ||
          hasName(&op, kSSAVC4SemaReleaseOpName)) {
        if (op.getNumOperands() != 1)
          return op.emitOpError("requires exactly one semaphore id operand");
        if (!op.getOperand(0).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 semaphore id operand");
        if (std::optional<int64_t> sem = getConstantI32FromLoadImm(op.getOperand(0))) {
          if (*sem < 0 || *sem > 15)
            return op.emitOpError()
                   << "requires compile-time semaphore id in hardware range [0, 15]";
          requiredSemaphores = std::max(requiredSemaphores, *sem + 1);
        }
      }
    }
  }

  DictionaryAttr resource = getResourceMetadata(func);
  if (sawBarrier) {
    if (!resource)
      return func->emitOpError()
             << "uses ssavc4.barrier but lacks vc4.resource metadata";

    std::optional<llvm::StringRef> scheduleMode =
        getResourceString(resource, "schedule_mode");
    if (!scheduleMode || *scheduleMode != "cooperative_block")
      return func->emitOpError()
             << "uses ssavc4.barrier and requires vc4.resource schedule_mode = \"cooperative_block\"";

    std::optional<bool> usesBarrier = getResourceBool(resource, "uses_barrier");
    if (!usesBarrier || !*usesBarrier)
      return func->emitOpError()
             << "uses ssavc4.barrier but vc4.resource uses_barrier is not true";

    std::optional<bool> fullResidency =
        getResourceBool(resource, "require_full_block_residency");
    if (!fullResidency || !*fullResidency)
      return func->emitOpError()
             << "uses ssavc4.barrier but vc4.resource require_full_block_residency is not true";

    std::optional<int64_t> warpsPerBlockMax =
        getResourceI32(resource, "warps_per_block_max");
    if (!warpsPerBlockMax || *warpsPerBlockMax <= 0 || *warpsPerBlockMax > 12)
      return func->emitOpError()
             << "uses ssavc4.barrier but vc4.resource warps_per_block_max is not in [1, 12]";

    requiredSemaphores = std::max<int64_t>(requiredSemaphores, 4);
  }

  if (resource && requiredSemaphores > 0) {
    std::optional<int64_t> semaphoresPerBlock =
        getResourceI32(resource, "semaphores_per_block");
    if (!semaphoresPerBlock || *semaphoresPerBlock < requiredSemaphores)
      return func->emitOpError()
             << "vc4.resource semaphores_per_block is too small for SSAVC4 semaphore/barrier use";
  }

  return success();
}


static LogicalResult verifyCooperativeVPMResource(Operation *func,
                                                  Operation *vpmOp) {
  DictionaryAttr resource = getResourceMetadata(func);
  if (!resource)
    return vpmOp->emitOpError()
           << "requires vc4.resource metadata with schedule_mode = cooperative_block";

  std::optional<llvm::StringRef> scheduleMode =
      getResourceString(resource, "schedule_mode");
  if (!scheduleMode || *scheduleMode != "cooperative_block")
    return vpmOp->emitOpError()
           << "requires vc4.resource schedule_mode = cooperative_block";

  std::optional<bool> usesSharedVPM = getResourceBool(resource, "uses_shared_vpm");
  if (!usesSharedVPM || !*usesSharedVPM)
    return vpmOp->emitOpError()
           << "requires vc4.resource uses_shared_vpm = true";

  std::optional<bool> fullResidency =
      getResourceBool(resource, "require_full_block_residency");
  if (!fullResidency || !*fullResidency)
    return vpmOp->emitOpError()
           << "requires vc4.resource require_full_block_residency = true";

  std::optional<int64_t> sharedVPMBytes = getResourceI32(resource, "shared_vpm_bytes");
  if (!sharedVPMBytes || *sharedVPMBytes <= 0)
    return vpmOp->emitOpError()
           << "requires positive vc4.resource shared_vpm_bytes";

  std::optional<int64_t> warpsPerBlockMax =
      getResourceI32(resource, "warps_per_block_max");
  if (!warpsPerBlockMax || *warpsPerBlockMax <= 0 || *warpsPerBlockMax > 12)
    return vpmOp->emitOpError()
           << "requires vc4.resource warps_per_block_max in [1, 12]";
  return success();
}

static LogicalResult verifyVPMSubset(Operation *op, Type valueType) {
  int64_t elemBytes = getI32IntegerAttrOr(op, "elem_bytes", -1);
  int64_t lanes = getI32IntegerAttrOr(op, "lanes", -1);
  if (elemBytes != 4)
    return op->emitOpError("supports only 32-bit VPM elements in M3 lowering");
  if (lanes != 16)
    return op->emitOpError("supports only full 16-lane VPM vectors in M3 lowering");
  if (!isVector16I32Type(valueType) && !isVector16F32Type(valueType))
    return op->emitOpError(
        "supports only vector<16xi32> or vector<16xf32> VPM values in M3 lowering");
  if (auto orientation = llvm::dyn_cast_or_null<StringAttr>(op->getAttr("orientation"))) {
    if (orientation.getValue() != "horizontal" && orientation.getValue() != "vertical")
      return op->emitOpError("supports only horizontal or vertical VPM orientation in M3 v1");
  }
  return success();
}

static LogicalResult verifyRestrictedFlagUses(Operation *func) {
  if (func->getNumRegions() == 0 || func->getRegion(0).empty())
    return success();

  for (Block &block : func->getRegion(0)) {
    for (Operation &op : block) {
      if (hasName(&op, kSSAVC4MakeFlagsOpName)) {
        if (op.getNumResults() != 1)
          return op.emitOpError("requires exactly one flag result");
        Value flags = op.getResult(0);
        if (!flags.hasOneUse())
          return op.emitOpError()
                 << "result must be consumed exactly once by ssavc4.cond_br in M3 v1";
        Operation *user = *flags.getUsers().begin();
        if (!hasName(user, kSSAVC4CondBranchOpName))
          return op.emitOpError()
                 << "result must be consumed by ssavc4.cond_br in M3 v1";
      }

      if (hasName(&op, kSSAVC4CondBranchOpName)) {
        if (op.getNumOperands() != 1)
          return op.emitOpError("requires exactly one flags operand");
        Operation *definingOp = op.getOperand(0).getDefiningOp();
        if (!hasName(definingOp, kSSAVC4MakeFlagsOpName))
          return op.emitOpError()
                 << "requires flags produced directly by ssavc4.make_flags in M3 v1";
      }
    }
  }
  return success();
}

static LogicalResult selectInstructionTemplates(
    Operation *func, SmallVectorImpl<InstructionTemplate> &templates,
    SmallVectorImpl<VirtualValue> &virtualValues) {
  if (func->getNumRegions() == 0 || func->getRegion(0).empty())
    return success();

  if (failed(verifyRestrictedFlagUses(func)))
    return failure();
  if (failed(verifyCooperativeBarrierResources(func)))
    return failure();

  SmallVector<Block *, 8> blocks;
  for (Block &block : func->getRegion(0))
    blocks.push_back(&block);

  DenseMap<Block *, Block *> nextBlock;
  for (size_t i = 0; i + 1 < blocks.size(); ++i)
    nextBlock[blocks[i]] = blocks[i + 1];

  DenseMap<int64_t, Value> uniformReadByIndex;
  unsigned nextVirtualOrdinal = 0;
  int64_t nextUniformOrdinal = 0;
  for (Block *block : blocks) {
    for (Operation &op : *block) {
      if (hasName(&op, kSSAVC4ThreadEndOpName)) {
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::ThreadEnd;
        templ.source = &op;
        templ.sourceBlock = block;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4BranchOpName)) {
        if (op.getNumSuccessors() != 1)
          return op.emitOpError("requires exactly one successor");
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::Branch;
        templ.source = &op;
        templ.sourceBlock = block;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4CondBranchOpName)) {
        if (op.getNumSuccessors() != 2)
          return op.emitOpError("requires true and false successors");
        auto nextIt = nextBlock.find(block);
        if (nextIt == nextBlock.end() || op.getSuccessor(1) != nextIt->second)
          return op.emitOpError()
                 << "requires the false successor to be the next linear block in M3 v1";
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::CondBranch;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4LoadImmOpName)) {
        if (op.getNumResults() != 1)
          return op.emitOpError("requires exactly one result for M3 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::LoadImm;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.result = result;
        templates.push_back(std::move(templ));
        uniformReadByIndex[nextUniformOrdinal++] = result;
        if (auto uniformRead = llvm::dyn_cast<mlir::ssavc4::UniformReadOp>(op))
          uniformReadByIndex[uniformRead.getIndex()] = result;
        continue;
      }

      if (hasName(&op, kSSAVC4ElementNumberOpName)) {
        if (op.getNumOperands() != 0 || op.getNumResults() != 1)
          return op.emitOpError("requires exactly one result for M3 lowering");
        if (!isVector16I32Type(op.getResult(0).getType()))
          return op.emitOpError("supports only vector<16xi32> element numbers in M3 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::ElementNumber;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.result = result;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4UniformReadOpName)) {
        if (op.getNumOperands() != 0 || op.getNumResults() != 1)
          return op.emitOpError("requires exactly one result for M3 lowering");
        Type resultType = op.getResult(0).getType();
        if (!resultType.isSignlessInteger(32) && !isVector16I32Type(resultType) &&
            !isVector16F32Type(resultType))
          return op.emitOpError("supports only i32, vector<16xi32>, or vector<16xf32> uniform reads in M3 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::UniformRead;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.result = result;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4SplatOpName)) {
        if (op.getNumOperands() != 1 || op.getNumResults() != 1)
          return op.emitOpError("requires one input operand and one result for M3 lowering");
        if (!op.getOperand(0).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 scalar input for M3 lowering");
        if (!isVector16I32Type(op.getResult(0).getType()) &&
            !isVector16F32Type(op.getResult(0).getType()))
          return op.emitOpError("supports only vector<16xi32> or vector<16xf32> splat results in M3 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::Splat;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4MovOpName)) {
        if (op.getNumOperands() != 1 || op.getNumResults() != 1)
          return op.emitOpError("requires one input operand and one result for M3 lowering");
        if (op.getOperand(0).getType() != op.getResult(0).getType())
          return op.emitOpError("requires identical input and result types for M3 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::Mov;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4ALUAddOpName) || hasName(&op, kSSAVC4ALUMulOpName)) {
        if (op.getNumResults() != 1)
          return op.emitOpError("requires exactly one result for M3 lowering");
        if (op.getNumOperands() == 0 || op.getNumOperands() > 2)
          return op.emitOpError("supports only unary/binary ALU operations in M3 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = hasName(&op, kSSAVC4ALUAddOpName)
                         ? InstructionTemplate::Kind::ALUAdd
                         : InstructionTemplate::Kind::ALUMul;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4MakeFlagsOpName)) {
        if (op.getNumResults() != 1)
          return op.emitOpError("requires exactly one flag result");
        if (op.getNumOperands() == 0 || op.getNumOperands() > 2)
          return op.emitOpError("supports only unary/binary flag compares in M3 v1");
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::MakeFlags;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = op.getResult(0);
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4PackOpName) || hasName(&op, kSSAVC4UnpackOpName) ||
          hasName(&op, kSSAVC4RotateOpName)) {
        if (op.getNumOperands() != 1 || op.getNumResults() != 1)
          return op.emitOpError("requires one input operand and one result for M3 lowering");
        if (op.getOperand(0).getType() != op.getResult(0).getType())
          return op.emitOpError("requires identical input and result carrier types for M3 lowering");
        if (!isVector16I32Type(op.getResult(0).getType()) &&
            !isVector16F32Type(op.getResult(0).getType()))
          return op.emitOpError("supports only vector<16xi32> or vector<16xf32> values in M3 lowering");
        if (hasName(&op, kSSAVC4RotateOpName)) {
          int64_t amount = getI32IntegerAttrOr(&op, "amount", -1);
          if (amount < 0 || amount > 15)
            return op.emitOpError("requires immediate rotate amount in [0, 15]");
        }
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = hasName(&op, kSSAVC4RotateOpName)
                         ? InstructionTemplate::Kind::Rotate
                         : (hasName(&op, kSSAVC4PackOpName)
                                ? InstructionTemplate::Kind::Pack
                                : InstructionTemplate::Kind::Unpack);
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4TMURequestOpName)) {
        if (op.getNumOperands() != 1 || op.getNumResults() != 1)
          return op.emitOpError("requires one address operand and one async token result");
        if (!isScalarI32OrVector16I32(op.getOperand(0).getType()))
          return op.emitOpError("supports only i32 or vector<16xi32> direct addresses in M3 v1");
        if (!hasStringAttr(&op, "unit", "tmu0"))
          return op.emitOpError("supports only unit = \"tmu0\" in M3 v1");
        if (!hasStringAttr(&op, "mode", "direct"))
          return op.emitOpError("supports only mode = \"direct\" in M3 v1");
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::TMURequest;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = op.getResult(0);
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4TMUReadOpName)) {
        if (op.getNumOperands() != 1 || op.getNumResults() != 1)
          return op.emitOpError("requires one async token operand and one data result");
        if (!hasStringAttr(&op, "unit", "tmu0"))
          return op.emitOpError("supports only unit = \"tmu0\" in M3 v1");
        if (!hasStringAttr(&op, "part", "raw32"))
          return op.emitOpError("supports only part = \"raw32\" in M3 v1");
        Operation *request = op.getOperand(0).getDefiningOp();
        if (!hasName(request, kSSAVC4TMURequestOpName))
          return op.emitOpError("requires a token produced directly by ssavc4.tmu.request in M3 v1");
        if (!hasStringAttr(request, "unit", "tmu0"))
          return op.emitOpError("requires a token from a matching tmu0 request");
        Type resultType = op.getResult(0).getType();
        if (!isVector16I32Type(resultType) && !isVector16F32Type(resultType))
          return op.emitOpError("supports only vector<16xi32> or vector<16xf32> raw32 results in M3 v1");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::TMURead;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4SemaAcquireOpName) ||
          hasName(&op, kSSAVC4SemaReleaseOpName)) {
        if (op.getNumOperands() != 1)
          return op.emitOpError("requires exactly one semaphore id operand");
        if (!op.getOperand(0).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 semaphore id operand");
        if (!getConstantI32FromLoadImm(op.getOperand(0)))
          return op.emitOpError()
                 << "requires semaphore id produced by ssavc4.load_imm with an integer value in M3 v1";
        InstructionTemplate templ;
        templ.kind = hasName(&op, kSSAVC4SemaAcquireOpName)
                         ? InstructionTemplate::Kind::SemaAcquire
                         : InstructionTemplate::Kind::SemaRelease;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4BarrierOpName)) {
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::Barrier;
        templ.source = &op;
        templ.sourceBlock = block;
        if (op.getNumOperands() != 0 && op.getNumOperands() != 2)
          return op.emitOpError()
                 << "requires either no operands or logical_warp_id and "
                    "warps_per_block operands";
        templ.operands.append(op.operand_begin(), op.operand_end());
        if (templ.operands.empty()) {
          auto logicalIndex =
              llvm::dyn_cast_or_null<IntegerAttr>(op.getAttr("logical_warp_id_uniform"));
          auto warpsIndex =
              llvm::dyn_cast_or_null<IntegerAttr>(op.getAttr("warps_per_block_uniform"));
          if (logicalIndex || warpsIndex) {
          if (!logicalIndex || !warpsIndex)
            return op.emitOpError()
                   << "requires both logical_warp_id_uniform and "
                      "warps_per_block_uniform when either is present";
          auto logicalIt = uniformReadByIndex.find(logicalIndex.getInt());
          auto warpsIt = uniformReadByIndex.find(warpsIndex.getInt());
          if (logicalIt == uniformReadByIndex.end() ||
              warpsIt == uniformReadByIndex.end())
            return op.emitOpError()
                   << "requires barrier uniform-index attrs to reference prior "
                      "ssavc4.uniform.read results";
          templ.operands.push_back(logicalIt->second);
          templ.operands.push_back(warpsIt->second);
          }
        }
        templates.push_back(std::move(templ));
        continue;
      }


      if (hasName(&op, kSSAVC4VPMWriteOpName)) {
        if (failed(verifyCooperativeVPMResource(func, &op)))
          return failure();
        if (op.getNumOperands() != 2)
          return op.emitOpError("requires row and vector value operands");
        if (!op.getOperand(0).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 VPM row operand for M3 lowering");
        if (failed(verifyVPMSubset(&op, op.getOperand(1).getType())))
          return failure();
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::VPMWrite;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4VPMReadOpName)) {
        if (failed(verifyCooperativeVPMResource(func, &op)))
          return failure();
        if (op.getNumOperands() != 1 || op.getNumResults() != 1)
          return op.emitOpError("requires one row operand and one vector result");
        if (!op.getOperand(0).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 VPM row operand for M3 lowering");
        if (failed(verifyVPMSubset(&op, op.getResult(0).getType())))
          return failure();
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::VPMRead;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        templates.push_back(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4VDWStoreOpName)) {
        if (op.getNumOperands() < 2 || op.getNumOperands() > 4)
          return op.emitOpError("requires address, vector value, and optional active-lane and VPM row operands");
        if (!op.getOperand(0).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 global address operand for M3 lowering");
        if (!isVector16I32Type(op.getOperand(1).getType()) &&
            !isVector16F32Type(op.getOperand(1).getType()))
          return op.emitOpError("requires a vector<16xi32> or vector<16xf32> value operand for M3 lowering");
        int64_t elemBytes = getI32IntegerAttrOr(&op, "elem_bytes", -1);
        int64_t activeLanes = getI32IntegerAttrOr(&op, "active_lanes", -1);
        if (elemBytes != 4)
          return op.emitOpError("supports only 32-bit elements in M3 lowering");
        if (op.getNumOperands() == 3) {
          if (!op.getOperand(2).getType().isSignlessInteger(32))
            return op.emitOpError("requires an i32 dynamic active-lane operand");
        } else if (op.getNumOperands() == 4) {
          if (!op.getOperand(2).getType().isSignlessInteger(32))
            return op.emitOpError("requires an i32 dynamic active-lane operand");
          if (!op.getOperand(3).getType().isSignlessInteger(32))
            return op.emitOpError("requires an i32 dynamic VPM row operand");
        } else if (activeLanes != 16) {
          return op.emitOpError("supports only full 16-lane static VDW stores in M3 lowering");
        }
        if (auto serialize = llvm::dyn_cast_or_null<StringAttr>(op.getAttr("serialize"))) {
          if (serialize.getValue() != "mutex" && serialize.getValue() != "none")
            return op.emitOpError("supports only serialize = \"mutex\" or \"none\" in M3 lowering");
        }
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::VDWStore;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.operands.append(op.operand_begin(), op.operand_end());
        templates.push_back(std::move(templ));
        continue;
      }

      return op.emitOpError()
             << "is not supported by the M3 SSAVC4 lowering; supported operations are "
                "ssavc4.load_imm, ssavc4.element_number, ssavc4.uniform.read, "
                "ssavc4.splat, ssavc4.mov, "
                "ssavc4.alu.add, ssavc4.alu.mul, "
                "ssavc4.make_flags, ssavc4.br, ssavc4.cond_br, "
                "ssavc4.pack, ssavc4.unpack, ssavc4.rotate, "
                "ssavc4.tmu.request, ssavc4.tmu.read, "
                "ssavc4.sema.acquire, ssavc4.sema.release, "
                "ssavc4.barrier, ssavc4.vpm.write, ssavc4.vpm.read, "
                "ssavc4.vdw.store, and ssavc4.thread_end";
    }
  }

  return success();
}

static LogicalResult emitLoadImm(OpBuilder &builder,
                                 const InstructionTemplate &templ,
                                 const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  std::optional<int64_t> destination = allocator.lookup(templ, *templ.result);
  if (!destination)
    return source->emitError("internal lowering error: result register was not allocated");

  OperationState state(source->getLoc(), "vc4.qpu.ldi");
  MLIRContext *ctx = builder.getContext();
  Attribute mode = source->getAttr("mode");
  if (!mode)
    mode = mlir::vc4::LoadImmModeAttr::get(ctx, mlir::vc4::LoadImmMode::splat32);
  Attribute value = source->getAttr("value");
  if (!value)
    value = builder.getI32IntegerAttr(0);

  state.addAttribute("mode", mode);
  if (Attribute values = source->getAttr("values"))
    state.addAttribute("values", values);
  else
    state.addAttribute("value", value);
  state.addAttribute("pm", builder.getBoolAttr(false));
  state.addAttribute("cond_add", mlir::vc4::CondAttr::get(ctx, mlir::vc4::Cond::always));
  state.addAttribute("cond_mul", mlir::vc4::CondAttr::get(ctx, mlir::vc4::Cond::always));
  state.addAttribute("waddr_add", builder.getI32IntegerAttr(*destination));
  state.addAttribute("waddr_mul", builder.getI32IntegerAttr(*destination));
  builder.create(state);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  return success();
}

static LogicalResult emitElementNumber(OpBuilder &builder,
                                       const InstructionTemplate &templ,
                                       const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  if (!resultReg)
    return source->emitError("internal lowering error: result register was not allocated");

  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/38,
                        /*raddrB=*/38, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  return success();
}

static LogicalResult emitUniformRead(OpBuilder &builder,
                                     const InstructionTemplate &templ,
                                     const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  if (!resultReg)
    return source->emitError("internal lowering error: result register was not allocated");

  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/32,
                        /*raddrB=*/0, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1,
                        /*smallImm=*/0);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  return success();
}

static LogicalResult emitMov(OpBuilder &builder,
                             const InstructionTemplate &templ,
                             const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  std::optional<int64_t> inputReg = allocator.lookup(templ, templ.operands.front());
  if (!resultReg || !inputReg)
    return source->emitOpError()
           << "uses a value that is not defined by a lowerable SSAVC4 op in M3";

  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, *inputReg, *inputReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  return success();
}

static LogicalResult emitALU(OpBuilder &builder, const InstructionTemplate &templ,
                             const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  if (!resultReg)
    return source->emitError("internal lowering error: result register was not allocated");

  SmallVector<int64_t, 2> operandRegs;
  for (Value operand : templ.operands) {
    std::optional<int64_t> reg = allocator.lookup(templ, operand);
    if (!reg)
      return source->emitOpError()
             << "uses a value that is not defined by a lowerable SSAVC4 op in "
                "the current M3 slice 3 skeleton";
    operandRegs.push_back(*reg);
  }
  mlir::vc4::AddOpcode addOpcode = mlir::vc4::AddOpcode::nop;
  mlir::vc4::MulOpcode mulOpcode = mlir::vc4::MulOpcode::nop;
  int64_t waddrAdd = 32;
  int64_t waddrMul = 33;
  mlir::vc4::Cond condAdd = mlir::vc4::Cond::never;
  mlir::vc4::Cond condMul = mlir::vc4::Cond::never;

  if (templ.kind == InstructionTemplate::Kind::ALUAdd) {
    if (auto attr = llvm::dyn_cast_or_null<mlir::vc4::AddOpcodeAttr>(source->getAttr("opcode")))
      addOpcode = attr.getValue();
    else
      addOpcode = mlir::vc4::AddOpcode::bit_or;
    condAdd = mlir::vc4::Cond::always;
    waddrAdd = *resultReg;
  } else {
    if (auto attr = llvm::dyn_cast_or_null<mlir::vc4::MulOpcodeAttr>(source->getAttr("opcode")))
      mulOpcode = attr.getValue();
    else
      mulOpcode = mlir::vc4::MulOpcode::mul24;
    condMul = mlir::vc4::Cond::always;
    waddrMul = *resultReg;
  }

  std::optional<int64_t> smallImm;
  if (templ.kind == InstructionTemplate::Kind::ALUAdd &&
      templ.operands.size() == 2)
    smallImm = getSmallImmLiteralSelector(templ.operands[1]);

  bool useMirroredSecond = !allocator.hasSpills() && !smallImm &&
                           templ.operands.size() == 2 &&
                           isMirroredLoadImm(templ.operands[1]);
  int64_t raddrA = operandRegs.empty() ? 0 : operandRegs.front();
  int64_t raddrB = operandRegs.size() < 2 ? raddrA
                  : useMirroredSecond   ? operandRegs[1]
                                        : 0;
  mlir::vc4::QPUMux addB = operandRegs.size() < 2 ? mlir::vc4::QPUMux::a
                         : (useMirroredSecond || smallImm)
                             ? mlir::vc4::QPUMux::b
                             : mlir::vc4::QPUMux::r1;
  mlir::vc4::QPUMux mulB = operandRegs.size() < 2 ? mlir::vc4::QPUMux::a
                         : useMirroredSecond     ? mlir::vc4::QPUMux::b
                                                  : mlir::vc4::QPUMux::r1;

  if (operandRegs.size() == 2 && !useMirroredSecond && !smallImm) {
    createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, operandRegs[1],
                          operandRegs[1], mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
  }

  mlir::vc4::QPUSignal signal =
      smallImm ? mlir::vc4::QPUSignal::small_imm : mlir::vc4::QPUSignal::none;
  createScheduledBundle(builder, source->getLoc(), signal, condAdd, condMul,
                        waddrAdd, waddrMul, addOpcode, mulOpcode, raddrA,
                        raddrB, mlir::vc4::QPUMux::a, addB,
                        mlir::vc4::QPUMux::a, mulB, smallImm);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  if (templ.kind == InstructionTemplate::Kind::ALUMul) {
    createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          *resultReg, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                          *resultReg, mlir::vc4::QPUMux::b,
                          mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
    emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  }
  return success();
}


static LogicalResult emitPackOrUnpack(OpBuilder &builder,
                                      const InstructionTemplate &templ,
                                      const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  std::optional<int64_t> inputReg = allocator.lookup(templ, templ.operands.front());
  if (!resultReg || !inputReg)
    return source->emitOpError()
           << "uses a value that is not defined by a lowerable SSAVC4 op in M3";

  OperationState state(source->getLoc(), "vc4.qpu.bundle");
  addCommonBundleAttrs(builder, state, mlir::vc4::QPUSignal::none,
                       mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                       *resultReg, /*waddrMul=*/32,
                       mlir::vc4::AddOpcode::bit_or,
                       mlir::vc4::MulOpcode::nop, *inputReg, *inputReg,
                       mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                       mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  if (Attribute mode = source->getAttr("mode")) {
    if (templ.kind == InstructionTemplate::Kind::Pack)
      state.addAttribute("pack", mode);
    else
      state.addAttribute("unpack", mode);
  }
  builder.create(state);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  return success();
}

static LogicalResult emitRotate(OpBuilder &builder,
                                const InstructionTemplate &templ,
                                const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  std::optional<int64_t> inputReg = allocator.lookup(templ, templ.operands.front());
  if (!resultReg || !inputReg)
    return source->emitOpError()
           << "uses a value that is not defined by a lowerable SSAVC4 op in M3";
  int64_t amount = getI32IntegerAttrOr(source, "amount", -1);
  if (amount < 0 || amount > 15)
    return source->emitOpError("requires immediate rotate amount in [0, 15]");

  // VC4's small-immediate vector rotate path rotates an accumulator source.
  // Copy the SSA input to r2, then emit the rotate selector 48 + amount.
  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, *inputReg, *inputReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createNopBundle(builder, source->getLoc());
  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1,
                        /*smallImm=*/48 + amount);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  return success();
}

static LogicalResult emitTMURequest(OpBuilder &builder,
                                    const InstructionTemplate &templ,
                                    const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (templ.operands.empty())
    return source->emitError("internal lowering error: TMU request has no address operand");
  std::optional<int64_t> addressReg = allocator.lookup(templ, templ.operands.front());
  if (!addressReg)
    return source->emitOpError()
           << "uses a direct TMU address value that is not defined by a lowerable SSAVC4 op";

  // Direct TMU0 requests are issued by writing the per-lane address to t0s.
  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/56, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, *addressReg, /*raddrB=*/0,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);

  // M3 v1 is deliberately non-overlapped: leave a conservative spacer before
  // the matching receive signal.
  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::never, mlir::vc4::Cond::never,
                        /*waddrAdd=*/39, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::nop, mlir::vc4::MulOpcode::nop,
                        /*raddrA=*/0, /*raddrB=*/1, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r3);
  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::never, mlir::vc4::Cond::never,
                        /*waddrAdd=*/39, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::nop, mlir::vc4::MulOpcode::nop,
                        /*raddrA=*/0, /*raddrB=*/1, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r3);
  return success();
}

static LogicalResult emitTMURead(OpBuilder &builder,
                                 const InstructionTemplate &templ,
                                 const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  if (!resultReg)
    return source->emitError("internal lowering error: TMU result register was not allocated");

  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::ldtmu0,
                        mlir::vc4::Cond::never, mlir::vc4::Cond::never,
                        /*waddrAdd=*/39, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::nop, mlir::vc4::MulOpcode::nop,
                        /*raddrA=*/0, /*raddrB=*/1, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r3);

  // The TMU receive value is only valid in r4 for the following instruction.
  // Copy it immediately into the allocated SSA value register before any other
  // instruction can clobber the r4 lifetime.
  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/1, mlir::vc4::QPUMux::r4,
                        mlir::vc4::QPUMux::r4, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  return success();
}

static Operation *createScheduledSema(OpBuilder &builder, Location loc,
                                      mlir::vc4::SemaphoreMode mode,
                                      int64_t id) {
  MLIRContext *ctx = builder.getContext();
  OperationState state(loc, "vc4.qpu.sema");
  state.addAttribute("mode", mlir::vc4::SemaphoreModeAttr::get(ctx, mode));
  state.addAttribute("id", builder.getI32IntegerAttr(id));
  state.addAttribute("pm", builder.getBoolAttr(false));
  state.addAttribute("cond_add", mlir::vc4::CondAttr::get(ctx, mlir::vc4::Cond::never));
  state.addAttribute("cond_mul", mlir::vc4::CondAttr::get(ctx, mlir::vc4::Cond::never));
  state.addAttribute("waddr_add", builder.getI32IntegerAttr(32));
  state.addAttribute("waddr_mul", builder.getI32IntegerAttr(33));
  return builder.create(state);
}

static LogicalResult emitSema(OpBuilder &builder,
                              const InstructionTemplate &templ) {
  Operation *source = templ.source;
  if (templ.operands.size() != 1)
    return source->emitError("internal lowering error: semaphore template has no id operand");
  std::optional<int64_t> id = getConstantI32FromLoadImm(templ.operands.front());
  if (!id)
    return source->emitOpError()
           << "requires compile-time semaphore id in M3 v1";
  if (*id < 0 || *id > 15)
    return source->emitOpError()
           << "requires compile-time semaphore id in hardware range [0, 15]";
  mlir::vc4::SemaphoreMode mode =
      templ.kind == InstructionTemplate::Kind::SemaAcquire
          ? mlir::vc4::SemaphoreMode::acquire
          : mlir::vc4::SemaphoreMode::release;
  createScheduledSema(builder, source->getLoc(), mode, *id);
  return success();
}

static LogicalResult emitBarrier(OpBuilder &builder,
                                 const InstructionTemplate &templ,
                                 const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  int64_t arrive = getI32IntegerAttrOr(source, "arrive_offset", -1);
  int64_t go = getI32IntegerAttrOr(source, "go_offset", -1);
  int64_t depart = getI32IntegerAttrOr(source, "depart_offset", -1);
  int64_t reset = getI32IntegerAttrOr(source, "reset_offset", -1);
  if (arrive < 0 || go < 0 || depart < 0 || reset < 0)
    return source->emitOpError()
           << "requires non-negative arrive/go/depart/reset semaphore offsets";

  Location loc = source->getLoc();
  if (templ.operands.empty()) {
    // Degenerate barrier form retained for explicit semaphore smoke tests.
    // Cooperative shared-memory kernels should provide logical warp and block
    // size operands through the barrier uniform-index attrs below.
    createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::release, arrive);
    createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::acquire, arrive);
    createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::release, go);
    createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::acquire, go);
    createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::release, depart);
    createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::acquire, depart);
    createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::release, reset);
    createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::acquire, reset);
    return success();
  }

  if (templ.operands.size() != 2)
    return source->emitError("internal lowering error: barrier has wrong operand count");
  std::optional<int64_t> logicalWarpReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> warpsPerBlockReg = allocator.lookup(templ, templ.operands[1]);
  if (!logicalWarpReg || !warpsPerBlockReg)
    return source->emitOpError()
           << "uses barrier metadata values that are not defined by lowerable SSAVC4 ops";

  auto subSetFlagsSmallImm = [&](mlir::vc4::QPUMux lhs, int64_t raddrA,
                                 int64_t smallImm) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/31, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::sub,
                          mlir::vc4::MulOpcode::nop, raddrA, /*raddrB=*/0,
                          lhs, mlir::vc4::QPUMux::b,
                          mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                          smallImm, /*setFlags=*/true);
  };
  auto moveRegToR3 = [&](int64_t reg) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/35, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, reg, reg,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  };
  auto decrementR3 = [&]() {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/35, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::sub,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                          /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                          mlir::vc4::QPUMux::b,
                          mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                          /*smallImm=*/1);
  };
  auto branch = [&](mlir::vc4::BranchCond cond, int64_t slotDelta) {
    createScheduledBranch(builder, loc, cond, slotDelta * 8,
                          /*raddrA=*/0, /*waddrAdd=*/39, /*waddrMul=*/39);
  };
  auto testR3Zero = [&]() {
    subSetFlagsSmallImm(mlir::vc4::QPUMux::r3, /*raddrA=*/0,
                        /*smallImm=*/0);
  };

  // Full cooperative barrier. One leader drains arrival tokens, releases the
  // block, waits for departure tokens, then resets the final semaphore. Other
  // warps signal arrival, wait for release, signal departure, and wait reset.
  subSetFlagsSmallImm(mlir::vc4::QPUMux::a, *logicalWarpReg, /*smallImm=*/0);
  branch(mlir::vc4::BranchCond::all_z_set, /*slotDelta=*/12);
  createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::release, arrive);
  createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::acquire, go);
  createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::release, depart);
  createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::acquire, reset);
  branch(mlir::vc4::BranchCond::always, /*slotDelta=*/56);

  moveRegToR3(*warpsPerBlockReg);
  decrementR3();

  testR3Zero();
  branch(mlir::vc4::BranchCond::all_z_set, /*slotDelta=*/10);
  createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::acquire, arrive);
  decrementR3();
  branch(mlir::vc4::BranchCond::always, /*slotDelta=*/-7);

  moveRegToR3(*warpsPerBlockReg);
  decrementR3();

  testR3Zero();
  branch(mlir::vc4::BranchCond::all_z_set, /*slotDelta=*/10);
  createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::release, go);
  decrementR3();
  branch(mlir::vc4::BranchCond::always, /*slotDelta=*/-7);

  moveRegToR3(*warpsPerBlockReg);
  decrementR3();

  testR3Zero();
  branch(mlir::vc4::BranchCond::all_z_set, /*slotDelta=*/10);
  createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::acquire, depart);
  decrementR3();
  branch(mlir::vc4::BranchCond::always, /*slotDelta=*/-7);

  moveRegToR3(*warpsPerBlockReg);
  decrementR3();

  testR3Zero();
  branch(mlir::vc4::BranchCond::all_z_set, /*slotDelta=*/10);
  createScheduledSema(builder, loc, mlir::vc4::SemaphoreMode::release, reset);
  decrementR3();
  branch(mlir::vc4::BranchCond::always, /*slotDelta=*/-7);
  return success();
}

static LogicalResult emitMakeFlags(OpBuilder &builder,
                                   const InstructionTemplate &templ,
                                   const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  SmallVector<int64_t, 2> operandRegs;
  for (Value operand : templ.operands) {
    std::optional<int64_t> reg = allocator.lookup(templ, operand);
    if (!reg)
      return source->emitOpError()
             << "uses a value that is not defined by a lowerable SSAVC4 op in M3 v1";
    operandRegs.push_back(*reg);
  }
  bool useMirroredSecond =
      templ.operands.size() == 2 && isMirroredLoadImm(templ.operands[1]);
  int64_t raddrA = operandRegs.empty() ? 0 : operandRegs.front();
  int64_t raddrB = operandRegs.size() < 2 ? raddrA
                  : useMirroredSecond   ? operandRegs[1]
                                        : 0;
  mlir::vc4::QPUMux addB = operandRegs.size() < 2 ? mlir::vc4::QPUMux::a
                         : useMirroredSecond     ? mlir::vc4::QPUMux::b
                                                  : mlir::vc4::QPUMux::r1;

  if (operandRegs.size() == 2 && !useMirroredSecond) {
    createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, operandRegs[1],
                          operandRegs[1], mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
  }

  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/31, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::sub,
                        mlir::vc4::MulOpcode::nop, raddrA, raddrB,
                        mlir::vc4::QPUMux::a, addB,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        std::nullopt, /*setFlags=*/true);
  return success();
}

static LogicalResult emitScheduledBranch(OpBuilder &builder,
                                         const InstructionTemplate &templ,
                                         const LayoutSummary &layout) {
  Operation *source = templ.source;
  auto immediateIt = layout.branchImmediates.find(source);
  if (immediateIt == layout.branchImmediates.end())
    return source->emitError("internal lowering error: missing scheduled branch immediate");

  mlir::vc4::BranchCond cond = mlir::vc4::BranchCond::always;
  if (templ.kind == InstructionTemplate::Kind::CondBranch) {
    auto condAttr = llvm::dyn_cast_or_null<mlir::vc4::BranchCondAttr>(source->getAttr("cond"));
    if (!condAttr)
      return source->emitOpError("requires a vc4.branch_cond condition attribute");
    cond = condAttr.getValue();
  }

  createScheduledBranch(builder, source->getLoc(), cond, immediateIt->second,
                        /*raddrA=*/0, /*waddrAdd=*/31, /*waddrMul=*/30);
  return success();
}


static LogicalResult emitVPMWrite(OpBuilder &builder,
                                  const InstructionTemplate &templ,
                                  const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (templ.operands.size() != 2)
    return source->emitError("internal lowering error: VPM write has wrong operand count");
  std::optional<int64_t> rowReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> valueReg = allocator.lookup(templ, templ.operands[1]);
  if (!rowReg || !valueReg)
    return source->emitOpError()
           << "uses a VPM row/value that is not defined by a lowerable SSAVC4 op";

  int64_t setupBase = 1055232;
  if (auto orientation =
          llvm::dyn_cast_or_null<StringAttr>(source->getAttr("orientation"))) {
    if (orientation.getValue() == "vertical")
      setupBase = 1053184;
  }

  Location loc = source->getLoc();
  bool useMutex = hasStringAttr(source, "serialize", "mutex");
  if (useMutex) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/31, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/51,
                          /*raddrB=*/51, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
  }
  createSplat32LDI(builder, loc, setupBase, 35);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                    *rowReg, /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                    mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/48, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, *valueReg, *valueReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  if (useMutex) {
    createVPMVCDWait(builder, loc, mlir::vc4::VPMVCDSide::write);
    emitMutexRelease(builder, loc);
  }
  return success();
}

static LogicalResult emitVPMRead(OpBuilder &builder,
                                 const InstructionTemplate &templ,
                                 const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (templ.operands.size() != 1 || !templ.result)
    return source->emitError("internal lowering error: VPM read template is malformed");
  std::optional<int64_t> rowReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  if (!rowReg || !resultReg)
    return source->emitOpError()
           << "uses a VPM row/result that is not defined by a lowerable SSAVC4 op";

  int64_t setupBase = 1055232;
  if (auto orientation =
          llvm::dyn_cast_or_null<StringAttr>(source->getAttr("orientation"))) {
    if (orientation.getValue() == "vertical")
      setupBase = 1053184;
  }

  Location loc = source->getLoc();
  bool useMutex = hasStringAttr(source, "serialize", "mutex");
  if (useMutex) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/31, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/51,
                          /*raddrB=*/51, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
  }
  createSplat32LDI(builder, loc, setupBase, 35);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::read,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                    *rowReg, /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                    mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/48,
                        /*raddrB=*/48, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  emitRegfileResultSpacer(builder, loc, templ, allocator);
  if (useMutex)
    emitMutexRelease(builder, loc);
  return success();
}

static void emitRawVDWStore(OpBuilder &builder, Location loc,
                            int64_t addressReg, int64_t valueReg,
                            std::optional<int64_t> dynamicActiveLanesReg,
                            std::optional<int64_t> dynamicVPMRowReg,
                            int64_t activeLanes, int64_t vpmRow,
                            bool useMutex) {
  if (useMutex) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/31, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/51,
                          /*raddrB=*/51, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
  }

  // Stage the SSA vector value into the requested VPM row.
  createSplat32LDI(builder, loc, 1055232, 35);
  if (dynamicVPMRowReg) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, *dynamicVPMRowReg,
                          *dynamicVPMRowReg, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
    if (vpmRow != 0) {
      createSplat32LDI(builder, loc, vpmRow, 34);
      createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                            mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                            /*waddrAdd=*/33, /*waddrMul=*/32,
                            mlir::vc4::AddOpcode::add,
                            mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                            /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                            mlir::vc4::QPUMux::r2,
                            mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
    }
  } else {
    createSplat32LDI(builder, loc, vpmRow, 33);
  }
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                    mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/48, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, valueReg, valueReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createVPMVCDWait(builder, loc, mlir::vc4::VPMVCDSide::write);

  // VDW store setup: depth in bits 16..23, VPM row in bits 7.., and the
  // explicit SSA byte address in vw_addr.
  createSplat32LDI(builder, loc, -1073741824, 35);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                    mlir::vc4::QPUMux::r3, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  if (dynamicActiveLanesReg) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, *dynamicActiveLanesReg,
                          *dynamicActiveLanesReg, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
  } else {
    createSplat32LDI(builder, loc, activeLanes, 33);
  }
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/8);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/8);
  createSplat32LDI(builder, loc, -2139078656, 35);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  if (dynamicVPMRowReg) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, *dynamicVPMRowReg,
                          *dynamicVPMRowReg, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
    if (vpmRow != 0) {
      createSplat32LDI(builder, loc, vpmRow, 36);
      createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                            mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                            /*waddrAdd=*/33, /*waddrMul=*/32,
                            mlir::vc4::AddOpcode::add,
                            mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                            /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                            mlir::vc4::QPUMux::r4,
                            mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
    }
  } else {
    createSplat32LDI(builder, loc, vpmRow, 33);
  }
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/7);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                    mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createVPMVCDAddr(builder, loc, mlir::vc4::VPMVCDSide::write,
                   mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                   mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                   addressReg, addressReg, mlir::vc4::QPUMux::a,
                   mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                   mlir::vc4::QPUMux::r1);
  createVPMVCDWait(builder, loc, mlir::vc4::VPMVCDSide::write);
  if (useMutex)
    emitMutexRelease(builder, loc);
}

static LogicalResult emitVDWStore(OpBuilder &builder,
                                  const InstructionTemplate &templ,
                                  const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (templ.operands.size() < 2 || templ.operands.size() > 4)
    return source->emitError("internal lowering error: VDW store has wrong operand count");
  std::optional<int64_t> addressReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> valueReg = allocator.lookup(templ, templ.operands[1]);
  if (!addressReg || !valueReg)
    return source->emitOpError()
           << "uses a VDW address/value that is not defined by a lowerable SSAVC4 op";
  std::optional<int64_t> dynamicActiveLanesReg;
  if (templ.operands.size() >= 3) {
    dynamicActiveLanesReg = allocator.lookup(templ, templ.operands[2]);
    if (!dynamicActiveLanesReg)
      return source->emitOpError()
             << "uses a dynamic VDW active-lane value that is not defined by a lowerable SSAVC4 op";
  }
  std::optional<int64_t> dynamicVPMRowReg;
  if (templ.operands.size() == 4) {
    dynamicVPMRowReg = allocator.lookup(templ, templ.operands[3]);
    if (!dynamicVPMRowReg)
      return source->emitOpError()
             << "uses a dynamic VPM row value that is not defined by a lowerable SSAVC4 op";
  }

  int64_t elemBytes = getI32IntegerAttrOr(source, "elem_bytes", -1);
  int64_t activeLanes = getI32IntegerAttrOr(source, "active_lanes", -1);
  int64_t vpmRow = getI32IntegerAttrOr(source, "vpm_row", 0);
  if (elemBytes != 4)
    return source->emitOpError("supports only 32-bit elements in M3 lowering");
  if (!dynamicActiveLanesReg && activeLanes != 16)
    return source->emitOpError("supports only full 16-lane static VDW stores in M3 lowering");
  if (vpmRow < 0)
    return source->emitOpError("requires a non-negative VPM row in M3 lowering");

  emitRawVDWStore(builder, source->getLoc(), *addressReg, *valueReg,
                  dynamicActiveLanesReg, dynamicVPMRowReg, activeLanes,
                  vpmRow, hasStringAttr(source, "serialize", "mutex"));
  return success();
}

static unsigned getSourceUniformWordsPerQPU(Operation *sourceFunc) {
  auto launchABI =
      llvm::dyn_cast_or_null<DictionaryAttr>(sourceFunc->getAttr("vc4.launch_abi"));
  if (!launchABI)
    return 0;
  auto attr = llvm::dyn_cast_or_null<IntegerAttr>(
      launchABI.get("uniform_words_per_qpu"));
  if (!attr || attr.getInt() < 0)
    return 0;
  return static_cast<unsigned>(attr.getInt());
}

static void emitUniformReadToReg(OpBuilder &builder, Location loc,
                                 int64_t destinationReg) {
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        destinationReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/32,
                        /*raddrB=*/0, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1,
                        /*smallImm=*/0);
  createNopLDISlot(builder, loc);
}

static void emitSpillSlotBase(OpBuilder &builder, Location loc,
                              const SpillSlot &slot) {
  if (slot.offsetBytes == 0) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          SpillAwareAllocator::spillAddrReg(), /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop,
                          SpillAwareAllocator::spillBaseReg(),
                          SpillAwareAllocator::spillBaseReg(),
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
    createNopLDISlot(builder, loc);
    return;
  }

  createSplat32LDIWithMul(builder, loc, slot.offsetBytes,
                          SpillAwareAllocator::spillOffsetReg(),
                          SpillAwareAllocator::spillOffsetReg());
  createNopLDISlot(builder, loc);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        SpillAwareAllocator::spillAddrReg(), /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                        SpillAwareAllocator::spillBaseReg(),
                        SpillAwareAllocator::spillOffsetReg(),
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createNopLDISlot(builder, loc);
}

static void emitSpillStoreAction(OpBuilder &builder, Location loc,
                                 const SpillAction &action) {
  emitSpillSlotBase(builder, loc, action.slot);
  emitRawVDWStore(builder, loc, SpillAwareAllocator::spillAddrReg(),
                  action.physicalReg, std::nullopt,
                  SpillAwareAllocator::spillRowReg(),
                  /*activeLanes=*/16, /*vpmRow=*/0, /*useMutex=*/true);
}

static void emitSpillReloadAction(OpBuilder &builder, Location loc,
                                  const SpillAction &action) {
  emitSpillSlotBase(builder, loc, action.slot);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        SpillAwareAllocator::spillLaneReg(), /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/38,
                        /*raddrB=*/38, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  createNopLDISlot(builder, loc);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        SpillAwareAllocator::spillLaneReg(), /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl, mlir::vc4::MulOpcode::nop,
                        SpillAwareAllocator::spillLaneReg(), /*raddrB=*/0,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/2);
  createNopLDISlot(builder, loc);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop,
                        SpillAwareAllocator::spillLaneReg(),
                        SpillAwareAllocator::spillLaneReg(),
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        SpillAwareAllocator::spillAddrReg(), /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                        SpillAwareAllocator::spillAddrReg(), /*raddrB=*/0,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createNopLDISlot(builder, loc);

  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/56, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop,
                        SpillAwareAllocator::spillAddrReg(), /*raddrB=*/0,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::never, mlir::vc4::Cond::never,
                        /*waddrAdd=*/39, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::nop, mlir::vc4::MulOpcode::nop,
                        /*raddrA=*/0, /*raddrB=*/1, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r3);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::never, mlir::vc4::Cond::never,
                        /*waddrAdd=*/39, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::nop, mlir::vc4::MulOpcode::nop,
                        /*raddrA=*/0, /*raddrB=*/1, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r3);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::ldtmu0,
                        mlir::vc4::Cond::never, mlir::vc4::Cond::never,
                        /*waddrAdd=*/39, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::nop, mlir::vc4::MulOpcode::nop,
                        /*raddrA=*/0, /*raddrB=*/1, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r3);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        action.physicalReg, /*waddrMul=*/39,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/1, mlir::vc4::QPUMux::r4,
                        mlir::vc4::QPUMux::r4, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  createNopLDISlot(builder, loc);
}

static void emitSpillAction(OpBuilder &builder, Location loc,
                            const SpillAction &action) {
  if (action.kind == SpillAction::Kind::Store)
    emitSpillStoreAction(builder, loc, action);
  else
    emitSpillReloadAction(builder, loc, action);
}

static LogicalResult emitScheduledFunctionBody(
    Operation *sourceFunc, OpBuilder &builder, ArrayRef<ScheduledTemplate> scheduled,
    const SpillAwareAllocator &allocator, const LayoutSummary &layout) {
  bool emittedThreadEnd = false;
  bool emittedSpillBaseUniform = !allocator.hasSpills();
  unsigned consumedPublicUniforms = 0;
  unsigned sourceUniformWords = getSourceUniformWordsPerQPU(sourceFunc);
  for (const ScheduledTemplate &scheduledTemplate : scheduled) {
    const InstructionTemplate &templ = scheduledTemplate.templ;
    if (templ.kind == InstructionTemplate::Kind::UniformRead)
      ++consumedPublicUniforms;
    if (!emittedSpillBaseUniform &&
        templ.kind != InstructionTemplate::Kind::UniformRead) {
      while (consumedPublicUniforms < sourceUniformWords) {
        emitUniformReadToReg(builder, templ.source->getLoc(),
                             SpillAwareAllocator::spillOffsetReg());
        ++consumedPublicUniforms;
      }
      emitUniformReadToReg(builder, templ.source->getLoc(),
                           SpillAwareAllocator::spillBaseReg());
      emitUniformReadToReg(builder, templ.source->getLoc(),
                           SpillAwareAllocator::spillRowReg());
      emittedSpillBaseUniform = true;
    }
    for (const SpillAction &action : allocator.getPreActions(templ))
      emitSpillAction(builder, templ.source->getLoc(), action);

    switch (templ.kind) {
    case InstructionTemplate::Kind::LoadImm:
      if (failed(emitLoadImm(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::ElementNumber:
      if (failed(emitElementNumber(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::UniformRead:
      if (failed(emitUniformRead(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::Splat:
    case InstructionTemplate::Kind::Mov:
      if (failed(emitMov(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::ALUAdd:
    case InstructionTemplate::Kind::ALUMul:
      if (failed(emitALU(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::Pack:
    case InstructionTemplate::Kind::Unpack:
      if (failed(emitPackOrUnpack(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::Rotate:
      if (failed(emitRotate(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::MakeFlags:
      if (failed(emitMakeFlags(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::TMURequest:
      if (failed(emitTMURequest(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::TMURead:
      if (failed(emitTMURead(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::SemaAcquire:
    case InstructionTemplate::Kind::SemaRelease:
      if (failed(emitSema(builder, templ)))
        return failure();
      break;
    case InstructionTemplate::Kind::Barrier:
      if (failed(emitBarrier(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::Branch:
    case InstructionTemplate::Kind::CondBranch:
      if (failed(emitScheduledBranch(builder, templ, layout)))
        return failure();
      break;
    case InstructionTemplate::Kind::VPMWrite:
      if (failed(emitVPMWrite(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::VPMRead:
      if (failed(emitVPMRead(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::VDWStore:
      if (failed(emitVDWStore(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::ThreadEnd:
      createThreadEndBundle(builder, templ.source->getLoc());
      createNopBundle(builder, templ.source->getLoc());
      createNopBundle(builder, templ.source->getLoc());
      emittedThreadEnd = true;
      break;
    }
  }

  if (!emittedThreadEnd) {
    Location loc = sourceFunc->getLoc();
    createThreadEndBundle(builder, loc);
    createNopBundle(builder, loc);
    createNopBundle(builder, loc);
  }
  return success();
}

static Operation *createVC4Module(Operation *sourceModule, OpBuilder &builder) {
  OperationState state(sourceModule->getLoc(), "vc4.module");
  StringAttr symName = getSymbolNameAttr(sourceModule);
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     symName ? symName : builder.getStringAttr("ssavc4_lowered"));
  state.addRegion();
  Operation *vc4Module = builder.create(state);
  vc4Module->getRegion(0).push_back(new Block());
  return vc4Module;
}

static Operation *createVC4FuncShell(Operation *sourceFunc, OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  OperationState state(sourceFunc->getLoc(), "vc4.func");

  StringAttr symName = getSymbolNameAttr(sourceFunc);
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     symName ? symName : builder.getStringAttr("ssavc4_kernel"));
  state.addAttribute("function_type", TypeAttr::get(getFunctionType(sourceFunc, ctx)));

  Attribute threading = sourceFunc->getAttr("threading");
  if (!threading)
    threading = mlir::vc4::ThreadingModeAttr::get(ctx, mlir::vc4::ThreadingMode::single);
  state.addAttribute("threading", threading);
  state.addAttribute("form", mlir::vc4::FunctionFormAttr::get(ctx, mlir::vc4::FunctionForm::scheduled));
  state.addAttribute("domain", mlir::vc4::ExecutionDomainAttr::get(ctx, mlir::vc4::ExecutionDomain::qpu));

  addAttrIfPresent(sourceFunc, state, "kernel");
  addAttrIfPresent(sourceFunc, state, "vc4.launch_abi");
  addAttrIfPresent(sourceFunc, state, "vc4.resource");
  addAttrIfPresent(sourceFunc, state, "arg_attrs");
  addAttrIfPresent(sourceFunc, state, "res_attrs");

  state.addRegion();
  Operation *vc4Func = builder.create(state);
  vc4Func->getRegion(0).push_back(new Block());
  return vc4Func;
}

static DictionaryAttr appendSpillFrameBaseBuiltin(OpBuilder &builder,
                                                  DictionaryAttr launchABI) {
  int64_t oldUniformWords = 0;
  if (auto attr = llvm::dyn_cast_or_null<IntegerAttr>(
          launchABI.get("uniform_words_per_qpu")))
    oldUniformWords = attr.getInt();

  SmallVector<Attribute, 8> builtins;
  if (auto existing =
          llvm::dyn_cast_or_null<ArrayAttr>(launchABI.get("builtins"))) {
    builtins.append(existing.begin(), existing.end());
  }
  builtins.push_back(builder.getDictionaryAttr({
      builder.getNamedAttr("name",
                           builder.getStringAttr("__vc4_spill_frame_base")),
      builder.getNamedAttr("kind", builder.getStringAttr("hidden_runtime")),
      builder.getNamedAttr("materialization",
                           builder.getStringAttr("uniform_suffix")),
      builder.getNamedAttr("uniform_index",
                           builder.getI32IntegerAttr(oldUniformWords)),
  }));
  builtins.push_back(builder.getDictionaryAttr({
      builder.getNamedAttr("name",
                           builder.getStringAttr("__vc4_resident_request_id")),
      builder.getNamedAttr("kind", builder.getStringAttr("hidden_runtime")),
      builder.getNamedAttr("materialization",
                           builder.getStringAttr("uniform_suffix")),
      builder.getNamedAttr("uniform_index",
                           builder.getI32IntegerAttr(oldUniformWords + 1)),
  }));

  SmallVector<NamedAttribute, 8> attrs;
  for (NamedAttribute attr : launchABI)
    attrs.push_back(attr);

  auto replaceAttr = [&](StringRef name, Attribute value) {
    StringAttr nameAttr = builder.getStringAttr(name);
    for (NamedAttribute &attr : attrs) {
      if (attr.getName() == nameAttr) {
        attr.setValue(value);
        return;
      }
    }
    attrs.push_back(builder.getNamedAttr(name, value));
  };

  replaceAttr("uniform_words_per_qpu",
              builder.getI32IntegerAttr(oldUniformWords + 2));
  replaceAttr("builtins", builder.getArrayAttr(builtins));
  return builder.getDictionaryAttr(attrs);
}

static void attachSpillFrameMetadata(Operation *vc4Func, OpBuilder &builder,
                                     const SpillAwareAllocator &allocator) {
  if (!allocator.hasSpills())
    return;

  vc4Func->setAttr("spill_frame_bytes",
                   builder.getI32IntegerAttr(allocator.getSpillFrameBytes()));
  vc4Func->setAttr("spill_frame_stride_bytes",
                   builder.getI32IntegerAttr(allocator.getSpillFrameBytes()));

  auto launchABI =
      llvm::dyn_cast_or_null<DictionaryAttr>(vc4Func->getAttr("vc4.launch_abi"));
  if (launchABI)
    vc4Func->setAttr("vc4.launch_abi",
                     appendSpillFrameBaseBuiltin(builder, launchABI));
}

static LogicalResult lowerFunction(Operation *sourceFunc, Operation *vc4Module,
                                   OpBuilder &topBuilder) {
  SmallVector<InstructionTemplate, 8> templates;
  SmallVector<VirtualValue, 8> virtualValues;
  if (failed(selectInstructionTemplates(sourceFunc, templates, virtualValues)))
    return failure();

  LivenessSummary liveness = computeLiveness(templates, virtualValues);

  SpillAwareAllocator allocator;
  if (failed(allocator.allocate(sourceFunc, templates, virtualValues, liveness)))
    return failure();

  ConservativeScheduler scheduler;
  BranchLayoutPlanner branchLayout;
  SmallVector<ScheduledTemplate, 8> scheduled =
      scheduler.schedule(templates, liveness);
  LayoutSummary layout;
  if (failed(branchLayout.compute(sourceFunc, scheduled, allocator, layout)))
    return failure();

  OpBuilder moduleBuilder = topBuilder;
  moduleBuilder.setInsertionPointToEnd(&vc4Module->getRegion(0).front());
  Operation *vc4Func = createVC4FuncShell(sourceFunc, moduleBuilder);
  attachSpillFrameMetadata(vc4Func, moduleBuilder, allocator);

  OpBuilder bodyBuilder(vc4Func->getContext());
  bodyBuilder.setInsertionPointToEnd(&vc4Func->getRegion(0).front());
  return emitScheduledFunctionBody(sourceFunc, bodyBuilder, scheduled, allocator, layout);
}

struct ConvertSSAVC4ToVC4Pass
    : public PassWrapper<ConvertSSAVC4ToVC4Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSSAVC4ToVC4Pass)

  StringRef getArgument() const override { return "convert-ssavc4-to-vc4"; }
  StringRef getDescription() const override {
    return "Lower SSAVC4 machine SSA kernels to scheduled VC4 QPU sink ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::vc4::VC4Dialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *, 4> ssavc4Modules;
    for (Operation &op : module.getBody()->getOperations()) {
      if (hasName(&op, kSSAVC4ModuleOpName))
        ssavc4Modules.push_back(&op);
    }

    if (ssavc4Modules.empty())
      return;

    OpBuilder builder(module.getContext());
    for (Operation *sourceModule : ssavc4Modules) {
      builder.setInsertionPoint(sourceModule);
      Operation *vc4Module = createVC4Module(sourceModule, builder);

      if (sourceModule->getNumRegions() != 1 || sourceModule->getRegion(0).empty()) {
        sourceModule->emitOpError("requires one body region for SSAVC4 lowering");
        signalPassFailure();
        return;
      }

      SmallVector<Operation *, 8> funcs;
      for (Operation &op : sourceModule->getRegion(0).front()) {
        if (hasName(&op, kSSAVC4FuncOpName))
          funcs.push_back(&op);
        else {
          op.emitOpError("is not legal in an SSAVC4 module for M3 slice 3 lowering; expected ssavc4.func");
          signalPassFailure();
          return;
        }
      }

      for (Operation *func : funcs) {
        if (failed(lowerFunction(func, vc4Module, builder))) {
          signalPassFailure();
          return;
        }
      }

      sourceModule->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vc4::createConvertSSAVC4ToVC4Pass() {
  return std::make_unique<ConvertSSAVC4ToVC4Pass>();
}

void mlir::vc4::registerConvertSSAVC4ToVC4Pass() {
  // This translation unit is linked into vc4-opt for M3 slice 3, and the
  // file-scope PassRegistration below installs --convert-ssavc4-to-vc4.
}

static PassRegistration<ConvertSSAVC4ToVC4Pass> registerSSAVC4ToVC4Pass;
