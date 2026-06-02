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
#include "vc4/Support/VC4ResourceMetadata.h"

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
#include "llvm/ADT/BitVector.h"
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
constexpr llvm::StringLiteral kSSAVC4CondSelectOpName("ssavc4.cond_select");
constexpr llvm::StringLiteral kSSAVC4BranchOpName("ssavc4.br");
constexpr llvm::StringLiteral kSSAVC4CondBranchOpName("ssavc4.cond_br");
constexpr llvm::StringLiteral kSSAVC4VDWStoreOpName("ssavc4.vdw.store");
constexpr llvm::StringLiteral kSSAVC4VDWStoreVPMOpName("ssavc4.vdw.store_vpm");
constexpr llvm::StringLiteral
    kSSAVC4VDWStoreRectDynamicOpName("ssavc4.vdw.store_rect.dynamic");
constexpr llvm::StringLiteral kSSAVC4VDRLoadOpName("ssavc4.vdr.load");
constexpr llvm::StringLiteral
    kSSAVC4VDRLoadRectDynamicOpName("ssavc4.vdr.load_rect.dynamic");
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
// instruction selection/templates, virtual values, liveness, allocation and
// spill planning, conservative scheduling, hazard insertion, and branch layout.
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
    CondSelect,
    EdgeCopy,
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
    VDRLoad,
    VDRLoadRectDynamic,
    VDWStore,
    VDWStoreVPM,
    VDWStoreRectDynamic,
    ThreadEnd
  } kind;
  Operation *source = nullptr;
  Block *sourceBlock = nullptr;
  unsigned ordinal = 0;
  unsigned layoutBlockId = 0;
  std::optional<unsigned> branchTargetBlockId;
  SmallVector<Value, 2> operands;
  std::optional<Value> result;
  Operation *flagSource = nullptr;
  unsigned flagOperandCount = 0;
  bool syntheticThreadEndBranch = false;
};

struct LivenessSummary {
  unsigned virtualValueCount = 0;
  DenseMap<Value, unsigned> lastUseIndex;
  DenseMap<Value, SmallVector<unsigned, 4>> useIndices;
  DenseMap<Value, bool> pinnedValues;
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

struct EdgeCopyEndpoint {
  LocationKind kind = LocationKind::Register;
  int64_t physicalReg = -1;
  SpillSlot slot;
};

struct EdgeCopyAllocation {
  EdgeCopyEndpoint source;
  EdgeCopyEndpoint destination;
};

struct PerTemplateAllocation {
  DenseMap<Value, int64_t> operandRegisters;
  std::optional<int64_t> resultRegister;
  std::optional<EdgeCopyAllocation> edgeCopy;
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
  DenseMap<Block *, unsigned> blockOrder;
  for (const InstructionTemplate &templ : templates) {
    if (templ.sourceBlock && !blockOrder.count(templ.sourceBlock))
      blockOrder[templ.sourceBlock] = blockOrder.size();
  }

  for (unsigned index = 0; index < templates.size(); ++index) {
    for (Value operand : templates[index].operands) {
      if (isAllocatableSSAValue(operand, virtualValues)) {
        summary.lastUseIndex[operand] = index;
        summary.useIndices[operand].push_back(index);
      }
    }
  }

  auto getDefiningBlock = [](Value value) -> Block * {
    if (auto argument = llvm::dyn_cast<BlockArgument>(value))
      return argument.getOwner();
    Operation *def = value.getDefiningOp();
    return def ? def->getBlock() : nullptr;
  };
  for (const VirtualValue &virtualValue : virtualValues) {
    if (llvm::isa<BlockArgument>(virtualValue.value))
      summary.pinnedValues[virtualValue.value] = true;
  }
  for (const InstructionTemplate &branchTempl : templates) {
    if ((branchTempl.kind != InstructionTemplate::Kind::Branch &&
         branchTempl.kind != InstructionTemplate::Kind::CondBranch) ||
        !branchTempl.sourceBlock || !branchTempl.branchTargetBlockId)
      continue;
    auto sourceIt = blockOrder.find(branchTempl.sourceBlock);
    if (sourceIt == blockOrder.end() ||
        *branchTempl.branchTargetBlockId > sourceIt->second)
      continue;

    unsigned loopHeader = *branchTempl.branchTargetBlockId;
    unsigned loopLatch = sourceIt->second;
    for (const InstructionTemplate &templ : templates) {
      if (!templ.sourceBlock)
        continue;
      auto blockIt = blockOrder.find(templ.sourceBlock);
      if (blockIt == blockOrder.end() || blockIt->second < loopHeader ||
          blockIt->second > loopLatch)
        continue;
      for (Value operand : templ.operands) {
        if (!isAllocatableSSAValue(operand, virtualValues))
          continue;
        Block *defBlock = getDefiningBlock(operand);
        auto defIt = defBlock ? blockOrder.find(defBlock) : blockOrder.end();
        if (!defBlock || defIt == blockOrder.end() ||
            defIt->second < loopHeader || defIt->second > loopLatch)
          summary.pinnedValues[operand] = true;
      }
    }
  }
  return summary;
}

static bool hasSuccessorOperandsForEdge(Operation *op, unsigned successorIndex) {
  if (auto branch = llvm::dyn_cast<mlir::ssavc4::BranchOp>(op))
    return successorIndex == 0 && !branch.getTargetOperands().empty();
  if (auto condBranch = llvm::dyn_cast<mlir::ssavc4::CondBranchOp>(op)) {
    if (successorIndex == 0)
      return !condBranch.getTrueDestOperands().empty();
    if (successorIndex == 1)
      return !condBranch.getFalseDestOperands().empty();
  }
  return false;
}

static DictionaryAttr getResourceMetadata(Operation *func);

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
    auto planIt = perTemplate.find(templ.ordinal);
    if (planIt != perTemplate.end()) {
      auto operandIt = planIt->second.operandRegisters.find(value);
      if (operandIt != planIt->second.operandRegisters.end())
        return operandIt->second;
      if (templ.result && *templ.result == value &&
          planIt->second.resultRegister)
        return *planIt->second.resultRegister;
    }
    return std::nullopt;
  }

  ArrayRef<SpillAction> getPreActions(const InstructionTemplate &templ) const {
    static const SmallVector<SpillAction, 0> empty;
    auto it = perTemplate.find(templ.ordinal);
    if (it == perTemplate.end())
      return empty;
    return it->second.preActions;
  }

  std::optional<EdgeCopyAllocation>
  getEdgeCopyAllocation(const InstructionTemplate &templ) const {
    auto it = perTemplate.find(templ.ordinal);
    if (it == perTemplate.end())
      return std::nullopt;
    return it->second.edgeCopy;
  }

  bool hasSpills() const { return spillPlan.spillSlotCount != 0; }

  uint32_t getSpillSlotCount() const { return spillPlan.spillSlotCount; }
  uint32_t getSpillFrameBytes() const { return spillPlan.spillFrameBytes; }
  static constexpr int64_t spillBaseReg() { return kSpillBaseReg; }
  static constexpr int64_t spillOffsetReg() { return kSpillOffsetReg; }
  static constexpr int64_t spillAddrReg() { return kSpillAddrReg; }
  static constexpr int64_t spillLaneReg() { return kSpillLaneReg; }
  static constexpr int64_t spillRowReg() { return kSpillRowReg; }
  static constexpr int64_t edgeCopyScratchReg() { return kEdgeCopyScratchReg; }

private:
  static constexpr int64_t kForbiddenThreadEndHazardReg = 14;
  static constexpr int64_t kSpillBaseReg = 28;
  static constexpr int64_t kSpillOffsetReg = 29;
  static constexpr int64_t kSpillAddrReg = 30;
  static constexpr int64_t kSpillLaneReg = 31;
  static constexpr int64_t kFixedScratchReg = 31;
  static constexpr int64_t kSpillRowReg = 27;
  static constexpr int64_t kEdgeCopyScratchReg = 26;

  LogicalResult allocateWithoutSpills(ArrayRef<InstructionTemplate> templates,
                                      ArrayRef<VirtualValue> virtualValues,
                                      const LivenessSummary &liveness) {
    static constexpr int64_t kForbiddenThreadEndHazardReg = 14;
    int64_t nextRegister = 0;
    SmallVector<int64_t, 8> freeRegisters;
    auto isReservedOrForbidden = [&](int64_t reg) {
      return reg == kForbiddenThreadEndHazardReg || reg == kFixedScratchReg;
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
      if (liveness.pinnedValues.count(value))
        return;
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
      if (templ.result &&
          needsAllocation.find(*templ.result) != needsAllocation.end() &&
          registers.find(*templ.result) == registers.end()) {
        std::optional<int64_t> reg = allocateRegister();
        if (!reg) {
          registers.clear();
          return failure();
        }
        registers.try_emplace(*templ.result, *reg);
        if (!liveness.pinnedValues.count(*templ.result) &&
            liveness.lastUseIndex.find(*templ.result) ==
                liveness.lastUseIndex.end()) {
          freeRegisters.push_back(*reg);
          releasedValues[*templ.result] = true;
        }
      }
      for (Value operand : templ.operands)
        releaseIfLastUse(operand, index);
    }
    return success();
  }

  LogicalResult verifyS3SpillingSupported(
      Operation *diagnosticAnchor, ArrayRef<InstructionTemplate> templates) const {
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
        if (templ.syntheticThreadEndBranch) {
          sawNonUniform = true;
          break;
        }
        if (!templ.source || !templ.sourceBlock ||
            templ.source->getNumSuccessors() == 0)
          return diagnosticAnchor->emitError()
                 << "internal lowering error: malformed branch template";
        for (auto [successorIndex, successor] :
             llvm::enumerate(templ.source->getSuccessors())) {
          auto sourceIt = blockOrder.find(templ.sourceBlock);
          auto successorIt = blockOrder.find(successor);
          if (sourceIt == blockOrder.end() || successorIt == blockOrder.end())
            return templ.source->emitOpError()
                   << "S3 spilling requires branch successors to remain in "
                      "the current function layout";
          if (successorIt->second <= sourceIt->second &&
              !hasSuccessorOperandsForEdge(templ.source, successorIndex))
            return templ.source->emitOpError()
                   << "S3 spilling does not support loop/backedge branch "
                      "layouts requiring path-sensitive liveness";
        }
        sawNonUniform = true;
        break;
      case InstructionTemplate::Kind::EdgeCopy:
        sawNonUniform = true;
        break;
      case InstructionTemplate::Kind::SemaAcquire:
      case InstructionTemplate::Kind::SemaRelease:
        return templ.source->emitOpError()
               << "S4 spilling does not support spilling semaphore/control "
                  "protocol values";
      case InstructionTemplate::Kind::Barrier:
      case InstructionTemplate::Kind::VPMWrite:
      case InstructionTemplate::Kind::VPMRead:
        sawNonUniform = true;
        break;
      case InstructionTemplate::Kind::UniformRead:
        if (sawNonUniform) {
          return templ.source->emitOpError()
                 << "S4 spilling requires all ssavc4.uniform.read operations "
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
           reg == kSpillLaneReg || reg == kSpillRowReg ||
           reg == kEdgeCopyScratchReg;
  }

  bool isValueSpillable(Value value) const {
    Type type = value.getType();
    return type.isSignlessInteger(32) || type.isF32() ||
           isVector16I32Type(type) || isVector16F32Type(type);
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
    SmallVector<int64_t, 8> reservedRegisterHomes;
    SmallVector<int64_t, 32> freeRegisters;
    auto resetFreeRegisters = [&]() {
      freeRegisters.clear();
      for (int64_t reg = 31; reg >= 0; --reg)
        if (!isReservedInSpillMode(reg) &&
            std::find(reservedRegisterHomes.begin(), reservedRegisterHomes.end(),
                      reg) == reservedRegisterHomes.end())
          freeRegisters.push_back(reg);
    };
    resetFreeRegisters();

    DenseMap<Value, int64_t> activeRegisters;
    DenseMap<int64_t, Value> registerValues;
    DenseMap<Value, bool> slotValid;
    DenseMap<Value, bool> needsAllocation;
    for (const VirtualValue &virtualValue : virtualValues) {
      needsAllocation[virtualValue.value] = true;
    }

    DenseMap<Block *, unsigned> naturalBlockIds;
    for (const InstructionTemplate &templ : templates)
      if (templ.sourceBlock && !naturalBlockIds.count(templ.sourceBlock))
        naturalBlockIds[templ.sourceBlock] = templ.layoutBlockId;

    DenseMap<Block *, SmallVector<Block *, 4>> predecessors;
    for (const InstructionTemplate &templ : templates) {
      if ((templ.kind != InstructionTemplate::Kind::Branch &&
           templ.kind != InstructionTemplate::Kind::CondBranch) ||
          !templ.sourceBlock || !templ.source || templ.syntheticThreadEndBranch)
        continue;
      for (Block *successor : templ.source->getSuccessors()) {
        SmallVectorImpl<Block *> &preds = predecessors[successor];
        bool alreadyRecorded = false;
        for (Block *predecessor : preds)
          if (predecessor == templ.sourceBlock)
            alreadyRecorded = true;
        if (!alreadyRecorded)
          preds.push_back(templ.sourceBlock);
      }
    }

    DenseMap<Block *, bool> loopRegisterHomeBlocks;
    for (const InstructionTemplate &templ : templates) {
      if ((templ.kind != InstructionTemplate::Kind::Branch &&
           templ.kind != InstructionTemplate::Kind::CondBranch) ||
          !templ.sourceBlock || !templ.branchTargetBlockId)
        continue;
      auto sourceIt = naturalBlockIds.find(templ.sourceBlock);
      if (sourceIt == naturalBlockIds.end() ||
          *templ.branchTargetBlockId > sourceIt->second)
        continue;
      for (auto &entry : naturalBlockIds)
        if (entry.second >= *templ.branchTargetBlockId &&
            entry.second <= sourceIt->second)
          loopRegisterHomeBlocks[entry.first] = true;
    }

    DenseMap<Value, int64_t> blockArgumentRegisterHomes;
    int64_t nextHomeReg = 0;
    auto reserveNextLoopCarriedHomeReg = [&]() -> std::optional<int64_t> {
      while (nextHomeReg < 32 && isReservedInSpillMode(nextHomeReg))
        ++nextHomeReg;
      if (nextHomeReg >= 32)
        return std::nullopt;
      return nextHomeReg++;
    };
    for (const InstructionTemplate &templ : templates) {
      if (templ.kind != InstructionTemplate::Kind::EdgeCopy || !templ.result)
        continue;
      auto blockArg = llvm::dyn_cast<BlockArgument>(*templ.result);
      if (!blockArg || !loopRegisterHomeBlocks[blockArg.getOwner()])
        continue;
      if (blockArgumentRegisterHomes.count(*templ.result))
        continue;
      std::optional<int64_t> reg = reserveNextLoopCarriedHomeReg();
      if (!reg)
        return diagnosticAnchor->emitError()
               << "SSAVC4 block-argument lowering could not reserve register "
                  "homes for loop-carried spilled block arguments";
      blockArgumentRegisterHomes[*templ.result] = *reg;
      reservedRegisterHomes.push_back(*reg);
    }
    resetFreeRegisters();

    DenseMap<Value, SpillSlot> blockArgumentHomeSlots;
    for (const InstructionTemplate &templ : templates) {
      if (templ.kind != InstructionTemplate::Kind::EdgeCopy || !templ.result)
        continue;
      if (!llvm::isa<BlockArgument>(*templ.result))
        continue;
      if (blockArgumentRegisterHomes.count(*templ.result))
        continue;
      blockArgumentHomeSlots.try_emplace(*templ.result,
                                         getOrCreateSpillSlot(*templ.result));
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

    auto forgetRegisterValue = [&](int64_t reg) {
      auto oldIt = registerValues.find(reg);
      if (oldIt == registerValues.end())
        return;
      activeRegisters.erase(oldIt->second);
      registerValues.erase(oldIt);
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

    auto getActiveOrSpilledEndpoint =
        [&](Value value) -> std::optional<EdgeCopyEndpoint> {
      auto activeIt = activeRegisters.find(value);
      if (activeIt != activeRegisters.end()) {
        EdgeCopyEndpoint endpoint;
        endpoint.kind = LocationKind::Register;
        endpoint.physicalReg = activeIt->second;
        return endpoint;
      }
      auto slotIt = spillPlan.spillSlots.find(value);
      if (slotIt != spillPlan.spillSlots.end() && slotValid[value]) {
        EdgeCopyEndpoint endpoint;
        endpoint.kind = LocationKind::SpillSlot;
        endpoint.slot = slotIt->second;
        return endpoint;
      }
      return std::nullopt;
    };

    auto releaseIfLastUse = [&](Value value, unsigned index) {
      if (!needsAllocation.count(value) || liveness.pinnedValues.count(value))
        return;
      auto lastUseIt = liveness.lastUseIndex.find(value);
      if (lastUseIt == liveness.lastUseIndex.end() ||
          lastUseIt->second != index)
        return;
      auto activeIt = activeRegisters.find(value);
      if (activeIt == activeRegisters.end())
        return;
      freeRegisters.push_back(activeIt->second);
      registerValues.erase(activeIt->second);
      activeRegisters.erase(activeIt);
    };

    auto allocateEdgeCopy =
        [&](unsigned index, const InstructionTemplate &templ,
            PerTemplateAllocation &allocation) -> LogicalResult {
      if (!templ.result || templ.operands.size() != 1)
        return templ.source->emitError(
            "internal lowering error: malformed edge copy");
      Value source = templ.operands.front();
      Value destination = *templ.result;

      std::optional<EdgeCopyEndpoint> sourceEndpoint =
          getActiveOrSpilledEndpoint(source);
      if (!sourceEndpoint)
        return templ.source->emitOpError()
               << "SSAVC4 block-argument edge copy source value is not "
                  "available in a register or spill slot";

      std::optional<EdgeCopyEndpoint> destinationEndpoint;
      auto registerHomeIt = blockArgumentRegisterHomes.find(destination);
      if (registerHomeIt != blockArgumentRegisterHomes.end()) {
        EdgeCopyEndpoint endpoint;
        endpoint.kind = LocationKind::Register;
        endpoint.physicalReg = registerHomeIt->second;
        destinationEndpoint = endpoint;
        forgetRegisterValue(endpoint.physicalReg);
        activeRegisters[destination] = endpoint.physicalReg;
        registerValues[endpoint.physicalReg] = destination;
        spillPlan.locations[destination] =
            ValueLocation{LocationKind::Register, endpoint.physicalReg, 0};
        slotValid[destination] = false;
      } else if (auto blockArgHomeIt =
                     blockArgumentHomeSlots.find(destination);
                 blockArgHomeIt != blockArgumentHomeSlots.end()) {
        EdgeCopyEndpoint endpoint;
        endpoint.kind = LocationKind::SpillSlot;
        endpoint.slot = blockArgHomeIt->second;
        destinationEndpoint = endpoint;
        auto activeDestIt = activeRegisters.find(destination);
        if (activeDestIt != activeRegisters.end()) {
          registerValues.erase(activeDestIt->second);
          activeRegisters.erase(activeDestIt);
        }
        spillPlan.locations[destination] =
            ValueLocation{LocationKind::SpillSlot, -1, endpoint.slot.index};
        slotValid[destination] = true;
      } else {
        destinationEndpoint = getActiveOrSpilledEndpoint(destination);
      }
      if (!destinationEndpoint) {
        auto existingDestSlot = spillPlan.spillSlots.find(destination);
        if (existingDestSlot != spillPlan.spillSlots.end()) {
          EdgeCopyEndpoint endpoint;
          endpoint.kind = LocationKind::SpillSlot;
          endpoint.slot = existingDestSlot->second;
          destinationEndpoint = endpoint;
          spillPlan.locations[destination] = ValueLocation{
              LocationKind::SpillSlot, -1, endpoint.slot.index};
          slotValid[destination] = true;
        }
      }
      if (!destinationEndpoint) {
        SmallVector<Value, 2> protectedValues;
        if (sourceEndpoint->kind == LocationKind::Register)
          protectedValues.push_back(source);
        FailureOr<int64_t> reg =
            acquireRegister(index, protectedValues, allocation.preActions);
        if (succeeded(reg)) {
          EdgeCopyEndpoint endpoint;
          endpoint.kind = LocationKind::Register;
          endpoint.physicalReg = *reg;
          destinationEndpoint = endpoint;
          forgetRegisterValue(*reg);
          activeRegisters[destination] = *reg;
          registerValues[*reg] = destination;
          spillPlan.locations[destination] =
              ValueLocation{LocationKind::Register, *reg, 0};
          slotValid[destination] = false;
        } else {
          if (!isValueSpillable(destination))
            return templ.source->emitOpError()
                   << "SSAVC4 block-argument edge copy needs a spill slot for "
                      "a non-spillable scalar destination";
          SpillSlot slot = getOrCreateSpillSlot(destination);
          EdgeCopyEndpoint endpoint;
          endpoint.kind = LocationKind::SpillSlot;
          endpoint.slot = slot;
          destinationEndpoint = endpoint;
          spillPlan.locations[destination] =
              ValueLocation{LocationKind::SpillSlot, -1, slot.index};
          slotValid[destination] = true;
        }
      } else if (destinationEndpoint->kind == LocationKind::Register) {
        forgetRegisterValue(destinationEndpoint->physicalReg);
        activeRegisters[destination] = destinationEndpoint->physicalReg;
        registerValues[destinationEndpoint->physicalReg] = destination;
        spillPlan.locations[destination] = ValueLocation{
            LocationKind::Register, destinationEndpoint->physicalReg, 0};
        slotValid[destination] = false;
      } else {
        spillPlan.locations[destination] = ValueLocation{
            LocationKind::SpillSlot, -1, destinationEndpoint->slot.index};
        slotValid[destination] = true;
      }

      allocation.edgeCopy = EdgeCopyAllocation{*sourceEndpoint,
                                               *destinationEndpoint};
      if (destinationEndpoint->kind == LocationKind::Register)
        allocation.resultRegister = destinationEndpoint->physicalReg;
      if (sourceEndpoint->kind == LocationKind::Register)
        allocation.operandRegisters[source] = sourceEndpoint->physicalReg;

      releaseIfLastUse(source, index);
      return success();
    };

    std::optional<unsigned> activeLayoutBlockId;
    Block *activeSourceBlock = nullptr;
    for (unsigned index = 0; index < templates.size(); ++index) {
      const InstructionTemplate &templ = templates[index];
      PerTemplateAllocation &allocation = perTemplate[templ.ordinal];

      if (!activeLayoutBlockId || *activeLayoutBlockId != templ.layoutBlockId) {
        bool keepActiveState = false;
        if (activeLayoutBlockId && templ.sourceBlock && activeSourceBlock) {
          auto predIt = predecessors.find(templ.sourceBlock);
          auto naturalIt = naturalBlockIds.find(activeSourceBlock);
          keepActiveState =
              predIt != predecessors.end() && predIt->second.size() == 1 &&
              predIt->second.front() == activeSourceBlock &&
              naturalIt != naturalBlockIds.end() &&
              *activeLayoutBlockId == naturalIt->second;
        }
        if (activeLayoutBlockId && !keepActiveState) {
          activeRegisters.clear();
          registerValues.clear();
          resetFreeRegisters();
        }
        activeLayoutBlockId = templ.layoutBlockId;
        activeSourceBlock = templ.sourceBlock;
        if (templ.sourceBlock) {
          for (BlockArgument arg : templ.sourceBlock->getArguments()) {
            auto homeIt = blockArgumentRegisterHomes.find(arg);
            if (homeIt == blockArgumentRegisterHomes.end())
              continue;
            forgetRegisterValue(homeIt->second);
            activeRegisters[arg] = homeIt->second;
            registerValues[homeIt->second] = arg;
            spillPlan.locations[arg] =
                ValueLocation{LocationKind::Register, homeIt->second, 0};
            slotValid[arg] = false;
          }
        }
      }

      if (templ.kind == InstructionTemplate::Kind::EdgeCopy) {
        if (failed(allocateEdgeCopy(index, templ, allocation)))
          return failure();
        continue;
      }

      SmallVector<Value, 2> protectedOperands;
      for (Value operand : templ.operands)
        if (needsAllocation.count(operand))
          protectedOperands.push_back(operand);

      for (Value operand : templ.operands) {
        if (!needsAllocation.count(operand))
          continue;
        auto activeIt = activeRegisters.find(operand);
        if (activeIt == activeRegisters.end()) {
          auto slotIt = spillPlan.spillSlots.find(operand);
          if (slotIt == spillPlan.spillSlots.end() || !slotValid[operand])
            return templ.source->emitOpError()
                   << "S4 spilling expected inactive operand to have a valid "
                      "spill slot at layout block entry";
          SpillSlot slot = slotIt->second;
          FailureOr<int64_t> reg =
              acquireRegister(index, protectedOperands, allocation.preActions);
          if (failed(reg)) {
            return templ.source->emitOpError()
                   << "S4 spilling supports only data vector values; no "
                      "scratch register was available to reload an operand";
          }
          forgetRegisterValue(*reg);
          activeRegisters[operand] = *reg;
          registerValues[*reg] = operand;
          allocation.preActions.push_back(
              SpillAction{SpillAction::Kind::Reload, operand, slot, *reg});
          activeIt = activeRegisters.find(operand);
        }
        allocation.operandRegisters[operand] = activeIt->second;
      }

      if (templ.kind == InstructionTemplate::Kind::Branch) {
        ensureFutureLiveValuesHaveBranchSlots(index, allocation.preActions);
      } else if (templ.kind == InstructionTemplate::Kind::CondBranch) {
        SmallVectorImpl<SpillAction> *branchSlotActions = &allocation.preActions;
        if (index > 0 &&
            templates[index - 1].kind == InstructionTemplate::Kind::MakeFlags &&
            templates[index - 1].sourceBlock == templ.sourceBlock)
          branchSlotActions =
              &perTemplate[templates[index - 1].ordinal].preActions;
        ensureFutureLiveValuesHaveBranchSlots(index, *branchSlotActions);
      }

      if (templ.result && needsAllocation.count(*templ.result)) {
        FailureOr<int64_t> reg =
            acquireRegister(index, protectedOperands, allocation.preActions);
        if (failed(reg)) {
          return templ.source->emitOpError()
                 << "S4 spilling supports only data vector values; no register "
                    "was available for a spillable result";
        }
        allocation.resultRegister = *reg;
        registers[*templ.result] = *reg;
        forgetRegisterValue(*reg);
        activeRegisters[*templ.result] = *reg;
        registerValues[*reg] = *templ.result;
        spillPlan.locations[*templ.result] =
            ValueLocation{LocationKind::Register, *reg, 0};
        slotValid[*templ.result] = false;
      }

      for (Value operand : templ.operands)
        releaseIfLastUse(operand, index);

      if (templ.result && needsAllocation.count(*templ.result) &&
          !liveness.pinnedValues.count(*templ.result) &&
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
  DenseMap<unsigned, PerTemplateAllocation> perTemplate;
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

static unsigned
getEdgeCopySlotCount(const InstructionTemplate &templ,
                     const SpillAwareAllocator &allocator) {
  std::optional<EdgeCopyAllocation> edgeCopy =
      allocator.getEdgeCopyAllocation(templ);
  if (!edgeCopy)
    return 1;

  bool sourceReg = edgeCopy->source.kind == LocationKind::Register;
  bool destReg = edgeCopy->destination.kind == LocationKind::Register;
  if (sourceReg && destReg)
    return 1;
  if (sourceReg && !destReg)
    return getSpillActionSlotCount(SpillAction{
        SpillAction::Kind::Store, templ.operands.front(),
        edgeCopy->destination.slot, edgeCopy->source.physicalReg});
  if (!sourceReg && destReg)
    return getSpillActionSlotCount(SpillAction{
        SpillAction::Kind::Reload, templ.operands.front(),
        edgeCopy->source.slot, edgeCopy->destination.physicalReg});
  if (edgeCopy->source.slot.index == edgeCopy->destination.slot.index)
    return 0;
  return getSpillActionSlotCount(SpillAction{
             SpillAction::Kind::Reload, templ.operands.front(),
             edgeCopy->source.slot,
             SpillAwareAllocator::edgeCopyScratchReg()}) +
         getSpillActionSlotCount(SpillAction{
             SpillAction::Kind::Store, *templ.result,
             edgeCopy->destination.slot,
             SpillAwareAllocator::edgeCopyScratchReg()});
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
  DenseMap<unsigned, unsigned> blockStartSlots;
  DenseMap<Operation *, unsigned> opStartSlots;
  DenseMap<unsigned, unsigned> templateStartSlots;
  DenseMap<unsigned, int64_t> branchImmediates;
};

static bool canUseMirroredSecondOperandForALU(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator,
    bool smallImm);
static bool hasStringAttr(Operation *op, llvm::StringRef name,
                          llvm::StringRef expected);
static bool aluNeedsSecondOperandAccumulatorMove(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator);
static bool canUseMirroredSecondOperandForMakeFlags(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator);
static bool makeFlagsNeedsSecondOperandAccumulatorMove(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator);
static unsigned getVDRLoadRectDynamicSlotCount(Operation *op);
static unsigned getVDWStoreRectDynamicSlotCount(Operation *op);

static unsigned getFlattenedSlotCount(const InstructionTemplate &templ,
                                      const SpillAwareAllocator &allocator) {
  unsigned spillActionSlots = 0;
  for (const SpillAction &action : allocator.getPreActions(templ))
    spillActionSlots += getSpillActionSlotCount(action);

  unsigned resultSpacer = getRegfileResultSpacerSlotCount(templ, allocator);
  switch (templ.kind) {
  case InstructionTemplate::Kind::EdgeCopy:
    return spillActionSlots + getEdgeCopySlotCount(templ, allocator);
  case InstructionTemplate::Kind::Branch:
    return spillActionSlots + 4;
  case InstructionTemplate::Kind::CondBranch:
    return spillActionSlots +
           (makeFlagsNeedsSecondOperandAccumulatorMove(templ, allocator) ? 2
                                                                         : 1) +
           4;
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
  case InstructionTemplate::Kind::VDRLoad:
    if (auto serialize =
            llvm::dyn_cast_or_null<StringAttr>(templ.source->getAttr("serialize")))
      return spillActionSlots + (serialize.getValue() == "mutex" ? 9 : 7);
    return spillActionSlots + 7;
  case InstructionTemplate::Kind::VDRLoadRectDynamic:
    return spillActionSlots + getVDRLoadRectDynamicSlotCount(templ.source);
  case InstructionTemplate::Kind::VDWStore:
    if (auto serialize =
            llvm::dyn_cast_or_null<StringAttr>(templ.source->getAttr("serialize")))
      return spillActionSlots + (serialize.getValue() == "mutex" ? 19 : 17);
    return spillActionSlots + 17;
  case InstructionTemplate::Kind::VDWStoreVPM:
  {
    if (auto serialize =
            llvm::dyn_cast_or_null<StringAttr>(templ.source->getAttr("serialize")))
      return spillActionSlots + (serialize.getValue() == "mutex"
                                     ? 19
                                     : 17);
    return spillActionSlots + 17;
  }
  case InstructionTemplate::Kind::VDWStoreRectDynamic:
    return spillActionSlots + getVDWStoreRectDynamicSlotCount(templ.source);
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
  case InstructionTemplate::Kind::CondSelect:
    return spillActionSlots +
           (makeFlagsNeedsSecondOperandAccumulatorMove(templ, allocator) ? 2
                                                                         : 1) +
           2 + resultSpacer;
  case InstructionTemplate::Kind::ALUAdd:
    return spillActionSlots +
           (aluNeedsSecondOperandAccumulatorMove(templ, allocator) ? 1 : 0) +
           1 + resultSpacer;
  case InstructionTemplate::Kind::ALUMul:
    return spillActionSlots +
           (aluNeedsSecondOperandAccumulatorMove(templ, allocator) ? 1 : 0) +
           2 + (2 * resultSpacer);
  case InstructionTemplate::Kind::MakeFlags:
    return spillActionSlots +
           (makeFlagsNeedsSecondOperandAccumulatorMove(templ, allocator) ? 2
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
      layout.blockStartSlots.try_emplace(templ.layoutBlockId, slot);
      unsigned preActionSlots = 0;
      for (const SpillAction &action : allocator.getPreActions(templ))
        preActionSlots += getSpillActionSlotCount(action);
      unsigned instructionPreludeSlots = 0;
      if (templ.kind == InstructionTemplate::Kind::CondBranch) {
        instructionPreludeSlots =
            makeFlagsNeedsSecondOperandAccumulatorMove(templ, allocator) ? 2
                                                                         : 1;
      }
      if (templ.source)
        layout.opStartSlots.try_emplace(
            templ.source, slot + preActionSlots + instructionPreludeSlots);
      layout.templateStartSlots.try_emplace(templ.ordinal,
                                            slot + preActionSlots +
                                                instructionPreludeSlots);
      slot += getFlattenedSlotCount(templ, allocator);
    }

    for (const ScheduledTemplate &scheduledTemplate : scheduled) {
      const InstructionTemplate &templ = scheduledTemplate.templ;
      if (templ.kind != InstructionTemplate::Kind::Branch &&
          templ.kind != InstructionTemplate::Kind::CondBranch)
        continue;

      Operation *branchOp = templ.source;
      if (!branchOp || !templ.branchTargetBlockId)
        return diagnosticAnchor->emitError()
               << "internal lowering error: branch template has no successor";

      auto sourceIt = layout.templateStartSlots.find(templ.ordinal);
      auto targetIt = layout.blockStartSlots.find(*templ.branchTargetBlockId);
      if (sourceIt == layout.templateStartSlots.end() ||
          targetIt == layout.blockStartSlots.end())
        return branchOp->emitError()
               << "could not compute final scheduled branch layout for successor";

      int64_t sourceSlot = static_cast<int64_t>(sourceIt->second);
      int64_t targetSlot = static_cast<int64_t>(targetIt->second);
      layout.branchImmediates[templ.ordinal] = (targetSlot - sourceSlot) * 8;
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

static std::optional<int64_t> getConstantI32FromLoadImm(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, kSSAVC4LoadImmOpName))
    return std::nullopt;
  auto attr = llvm::dyn_cast_or_null<IntegerAttr>(def->getAttr("value"));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static unsigned getRowOffsetScratchSlotCount(int64_t row) {
  return row == 0 ? 0 : 2;
}

static unsigned getDynamicVDRPitchRowSlotCount(bool useMutex, int64_t row) {
  unsigned rawVDRSlots = useMutex ? 9 : 7;
  unsigned dynamicAddressSlots = row == 0 ? 0 : 1;
  return getRowOffsetScratchSlotCount(row) + dynamicAddressSlots + rawVDRSlots;
}

static unsigned getDynamicVDWStoreRowsSlotCount(bool useMutex,
                                                bool dynamicPitch) {
  unsigned slots = useMutex ? 21 : 19;
  return dynamicPitch ? slots + 2 : slots;
}

static unsigned getVDRLoadRectDynamicSlotCount(Operation *op) {
  bool useMutex = hasStringAttr(op, "serialize", "mutex");
  unsigned rawVDRSlots = useMutex ? 9 : 7;
  std::optional<int64_t> activeRows =
      op->getNumOperands() == 5 ? getConstantI32FromLoadImm(op->getOperand(2))
                                : std::nullopt;
  std::optional<int64_t> activeCols =
      op->getNumOperands() == 5 ? getConstantI32FromLoadImm(op->getOperand(3))
                                : std::nullopt;
  std::optional<int64_t> pitch =
      op->getNumOperands() == 5 ? getConstantI32FromLoadImm(op->getOperand(4))
                                : std::nullopt;
  int64_t maxRows = getI32IntegerAttrOr(op, "max_rows", -1);
  int64_t maxCols = getI32IntegerAttrOr(op, "max_cols", -1);
  if (!activeRows && activeCols && *activeCols == maxCols && maxRows >= 1 &&
      maxRows <= 2) {
    unsigned zeroFillSlots = useMutex ? 7 : 4;
    unsigned guardSlots = 8;
    unsigned total = 0;
    for (int64_t row = 0; row < maxRows; ++row)
      total += getRowOffsetScratchSlotCount(row) + zeroFillSlots;
    if (!pitch) {
      for (int64_t row = 0; row < maxRows; ++row)
        total += guardSlots +
                 getDynamicVDRPitchRowSlotCount(useMutex, row);
      return total;
    }
    unsigned clampSlots = 4;
    unsigned dynamicRowsVDRSlots = useMutex ? 16 : 14;
    return total + clampSlots + guardSlots + dynamicRowsVDRSlots;
  }
  if (!pitch)
    return rawVDRSlots;
  if (!activeRows)
    return rawVDRSlots;
  if (activeCols && *activeRows == maxRows && *activeCols == maxCols)
    return rawVDRSlots;
  if (!activeCols && activeRows && *activeRows == maxRows && maxRows == 1) {
    unsigned zeroFillSlots = useMutex ? 7 : 4;
    unsigned clampSlots = 4;
    unsigned guardSlots = 8;
    unsigned dynamicVDRSlots = useMutex ? 18 : 16;
    return zeroFillSlots + clampSlots + guardSlots + dynamicVDRSlots;
  }
  if (maxRows == 1) {
    unsigned zeroFillSlots = useMutex ? 7 : 4;
    int64_t clampedRows = std::clamp(*activeRows, int64_t(0), maxRows);
    int64_t clampedCols = std::clamp(*activeCols, int64_t(0), maxCols);
    return zeroFillSlots + (clampedRows == 0 || clampedCols == 0 ? 0
                                                                 : rawVDRSlots);
  }
  return rawVDRSlots;
}

static unsigned getVDWStoreRectDynamicSlotCount(Operation *op) {
  bool useMutex = hasStringAttr(op, "serialize", "mutex");
  unsigned rawStaticSlots = useMutex ? 17 : 15;
  if (op->getNumOperands() != 5)
    return rawStaticSlots;
  std::optional<int64_t> activeRows =
      getConstantI32FromLoadImm(op->getOperand(2));
  std::optional<int64_t> activeCols =
      getConstantI32FromLoadImm(op->getOperand(3));
  int64_t maxRows = getI32IntegerAttrOr(op, "max_rows", -1);
  int64_t maxCols = getI32IntegerAttrOr(op, "max_cols", -1);
  if (!activeRows && activeCols &&
      std::clamp(*activeCols, int64_t(0), maxCols) == maxCols &&
      maxRows >= 1 && maxRows <= 2) {
    unsigned clampSlots = 4;
    unsigned guardSlots = 8;
    std::optional<int64_t> strideBytes =
        getConstantI32FromLoadImm(op->getOperand(4));
    unsigned dynamicRowsVDWSlots =
        getDynamicVDWStoreRowsSlotCount(useMutex, !strideBytes);
    return clampSlots + guardSlots + dynamicRowsVDWSlots;
  }
  if (activeCols)
    return rawStaticSlots;
  return rawStaticSlots + (maxCols == 16 ? 3 : 2);
}

static std::optional<int64_t> getSmallImmLiteralSelector(Value value) {
  std::optional<int64_t> constant = getConstantI32FromLoadImm(value);
  if (!constant || *constant < 0 || *constant > 15)
    return std::nullopt;
  return constant;
}

static bool canUseMirroredSecondOperandForALU(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator,
    bool smallImm) {
  return !allocator.hasSpills() && !smallImm && templ.operands.size() == 2 &&
         isMirroredLoadImm(templ.operands[1]);
}

static bool aluNeedsSecondOperandAccumulatorMove(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator) {
  if (templ.operands.size() != 2)
    return false;
  bool smallImm = templ.kind == InstructionTemplate::Kind::ALUAdd &&
                  getSmallImmLiteralSelector(templ.operands[1]).has_value();
  return !smallImm &&
         !canUseMirroredSecondOperandForALU(templ, allocator, smallImm);
}

static bool canUseMirroredSecondOperandForMakeFlags(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator) {
  unsigned flagOperandCount =
      templ.flagOperandCount ? templ.flagOperandCount : templ.operands.size();
  return !allocator.hasSpills() && flagOperandCount == 2 &&
         isMirroredLoadImm(templ.operands[1]);
}

static bool makeFlagsNeedsSecondOperandAccumulatorMove(
    const InstructionTemplate &templ, const SpillAwareAllocator &allocator) {
  unsigned flagOperandCount =
      templ.flagOperandCount ? templ.flagOperandCount : templ.operands.size();
  if (flagOperandCount != 2)
    return false;
  return !canUseMirroredSecondOperandForMakeFlags(templ, allocator);
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
  mlir::vc4::SemanticResourceInfo resourceInfo;
  if (resource &&
      failed(mlir::vc4::parseSemanticResourceMetadata(
          func, resource, resourceInfo, /*allowAbsent=*/false)))
    return failure();
  if (sawBarrier) {
    if (!resource)
      return func->emitOpError()
             << "uses ssavc4.barrier but lacks vc4.resource metadata";

    if (resourceInfo.scheduleMode != "cooperative_block")
      return func->emitOpError()
             << "uses ssavc4.barrier and requires vc4.resource schedule_mode = \"cooperative_block\"";

    if (!resourceInfo.usesBarrier)
      return func->emitOpError()
             << "uses ssavc4.barrier but vc4.resource uses_barrier is not true";

    requiredSemaphores = std::max<int64_t>(requiredSemaphores, 4);
  }

  if (resource && requiredSemaphores > 0) {
    if (resourceInfo.semaphoreCountPerBlock < requiredSemaphores)
      return func->emitOpError()
             << "vc4.resource semaphore_count_per_block is too small for SSAVC4 semaphore/barrier use";
  }

  return success();
}


static LogicalResult verifyVPMResource(Operation *func, Operation *vpmOp) {
  DictionaryAttr resource = getResourceMetadata(func);
  if (!resource)
    return vpmOp->emitOpError()
           << "requires semantic vc4.resource metadata with VPM usage";

  mlir::vc4::SemanticResourceInfo resourceInfo;
  if (failed(mlir::vc4::parseSemanticResourceMetadata(
          vpmOp, resource, resourceInfo, /*allowAbsent=*/false)))
    return failure();
  if (!resourceInfo.usesVPM || resourceInfo.totalVPMRowsPerBlock <= 0 ||
      !resourceInfo.requiresVPMBaseRowBuiltin)
    return vpmOp->emitOpError()
           << "requires semantic vc4.resource VPM rows and vpm_base_row";
  return success();
}

static LogicalResult verifyVPMSubset(Operation *op, Type valueType) {
  int64_t lanes = getI32IntegerAttrOr(op, "lanes", -1);
  if (lanes != 16)
    return op->emitOpError("supports only full 16-lane VPM vectors in M3 lowering");
  if (!isVector16I32Type(valueType) && !isVector16F32Type(valueType))
    return op->emitOpError(
        "supports only vector<16xi32> or vector<16xf32> VPM values in M3 lowering");
  auto orientation =
      llvm::dyn_cast_or_null<mlir::ssavc4::VPMOrientationAttr>(
          op->getAttr("orientation"));
  if (!orientation)
    return op->emitOpError("requires typed VPM orientation attr");
  auto width = llvm::dyn_cast_or_null<mlir::ssavc4::VPMElemWidthAttr>(
      op->getAttr("width"));
  if (!width)
    return op->emitOpError("requires typed VPM width attr");
  if (width.getValue() != mlir::ssavc4::VPMElemWidth::w32)
    return op->emitOpError(
        "supports only width = #ssavc4.vpm_elem_width<w32> in executable v1");
  auto subword = llvm::dyn_cast_or_null<mlir::ssavc4::VPMSubwordAttr>(
      op->getAttr("subword"));
  if (!subword)
    return op->emitOpError("requires typed VPM subword attr");
  if (subword.getValue() != mlir::ssavc4::VPMSubword::none)
    return op->emitOpError(
        "supports only subword = #ssavc4.vpm_subword<none> in executable v1");
  int64_t x = getI32IntegerAttrOr(op, "x", -1);
  int64_t stride = getI32IntegerAttrOr(op, "stride", -1);
  if (x < 0 || x > 15)
    return op->emitOpError("requires VPM x coordinate in range [0, 15]");
  if (orientation.getValue() == mlir::ssavc4::VPMOrientation::horizontal &&
      x != 0)
    return op->emitOpError(
        "horizontal 32-bit VPM QPU access requires x = 0 in executable v1");
  if (stride <= 0)
    return op->emitOpError("requires positive VPM stride");
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
                 << "result must be consumed exactly once by ssavc4.cond_br "
                    "or ssavc4.cond_select in M5";
        Operation *user = *flags.getUsers().begin();
        if (!hasName(user, kSSAVC4CondBranchOpName) &&
            !hasName(user, kSSAVC4CondSelectOpName))
          return op.emitOpError()
                 << "result must be consumed by ssavc4.cond_br or "
                    "ssavc4.cond_select in M5";
      }

      if (hasName(&op, kSSAVC4CondBranchOpName)) {
        auto condBranch = llvm::cast<mlir::ssavc4::CondBranchOp>(op);
        Operation *definingOp = condBranch.getFlags().getDefiningOp();
        if (!hasName(definingOp, kSSAVC4MakeFlagsOpName))
          return op.emitOpError()
                 << "requires flags produced directly by ssavc4.make_flags in M3 v1";
      }

      if (hasName(&op, kSSAVC4CondSelectOpName)) {
        auto condSelect = llvm::cast<mlir::ssavc4::CondSelectOp>(op);
        Operation *definingOp = condSelect.getFlags().getDefiningOp();
        if (!hasName(definingOp, kSSAVC4MakeFlagsOpName))
          return op.emitOpError()
                 << "requires flags produced directly by ssavc4.make_flags in M5";
      }
    }
  }
  return success();
}

static bool isP2BlockArgumentType(Type type) {
  return type.isSignlessInteger(32) || type.isF32() ||
         isVector16I32Type(type) || isVector16F32Type(type);
}

static LogicalResult verifyP2BlockArgumentType(Operation *op, Value value,
                                               StringRef role) {
  if (isP2BlockArgumentType(value.getType()))
    return success();
  return op->emitOpError()
         << "SSAVC4 block-argument lowering supports only i32, f32, "
            "vector<16xi32>, or vector<16xf32> "
         << role << " values";
}

static constexpr llvm::StringLiteral kP3UnsupportedLoopDiagnostic(
    "SSAVC4 block-argument lowering supports only natural loops with "
    "conservative loop-carried data values");

static constexpr llvm::StringLiteral kP3CyclicLoopCopyDiagnostic(
    "SSAVC4 block-argument lowering does not yet support cyclic parallel "
    "copies involving spilled values");

static DenseMap<Block *, BitVector>
computeDominance(ArrayRef<Block *> blocks,
                 const DenseMap<Block *, SmallVector<Block *, 4>> &predecessors) {
  DenseMap<Block *, BitVector> dominators;
  if (blocks.empty())
    return dominators;

  unsigned blockCount = blocks.size();
  BitVector allBlocks(blockCount, true);
  for (auto [index, block] : llvm::enumerate(blocks)) {
    dominators[block] = allBlocks;
    if (index == 0) {
      dominators[block].reset();
      dominators[block].set(index);
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto [index, block] : llvm::enumerate(blocks)) {
      if (index == 0)
        continue;

      BitVector next(blockCount, true);
      auto predIt = predecessors.find(block);
      if (predIt == predecessors.end() || predIt->second.empty()) {
        next.reset();
      } else {
        bool sawPred = false;
        for (Block *pred : predIt->second) {
          auto domIt = dominators.find(pred);
          if (domIt == dominators.end())
            continue;
          if (!sawPred) {
            next = domIt->second;
            sawPred = true;
            continue;
          }
          next &= domIt->second;
        }
        if (!sawPred)
          next.reset();
      }
      next.set(index);
      if (next != dominators[block]) {
        dominators[block] = std::move(next);
        changed = true;
      }
    }
  }
  return dominators;
}

static bool isBackedge(Block *sourceBlock, Block *targetBlock,
                       const DenseMap<Block *, unsigned> &blockOrder) {
  auto sourceIt = blockOrder.find(sourceBlock);
  auto targetIt = blockOrder.find(targetBlock);
  return sourceIt != blockOrder.end() && targetIt != blockOrder.end() &&
         targetIt->second <= sourceIt->second;
}

static LogicalResult verifySuccessorOperandLoopShape(
    Operation *op, Block *sourceBlock, Block *targetBlock,
    ValueRange successorOperands, const DenseMap<Block *, unsigned> &blockOrder,
    const DenseMap<Block *, BitVector> &dominators) {
  if (successorOperands.empty() || !isBackedge(sourceBlock, targetBlock, blockOrder))
    return success();

  auto sourceIt = blockOrder.find(sourceBlock);
  auto targetIt = blockOrder.find(targetBlock);
  auto domIt = dominators.find(sourceBlock);
  if (sourceIt == blockOrder.end() || targetIt == blockOrder.end() ||
      domIt == dominators.end())
    return op->emitOpError() << kP3UnsupportedLoopDiagnostic;
  if (targetIt->second >= domIt->second.size() ||
      !domIt->second.test(targetIt->second))
    return op->emitOpError() << kP3UnsupportedLoopDiagnostic;
  return success();
}

static LogicalResult verifyParallelEdgeCopyGroup(
    Operation *op, Block *sourceBlock, Block *targetBlock,
    ValueRange successorOperands, const DenseMap<Block *, unsigned> &blockOrder) {
  bool loopBackedge = !successorOperands.empty() &&
                      isBackedge(sourceBlock, targetBlock, blockOrder);
  for (auto [index, successorOperand] : llvm::enumerate(successorOperands)) {
    for (BlockArgument argument : targetBlock->getArguments()) {
      if (successorOperand != argument)
        continue;
      if (argument.getArgNumber() == index)
        break;
      if (loopBackedge)
        return op->emitOpError() << kP3CyclicLoopCopyDiagnostic;
      return op->emitOpError()
             << "SSAVC4 block-argument lowering requires a scratch register for "
                "overlapping parallel edge copies, but a scratch register is "
                "unavailable";
    }
  }
  return success();
}

static LogicalResult verifySuccessorOperandEdge(
    Operation *op, Block *sourceBlock, Block *targetBlock,
    ValueRange successorOperands, const DenseMap<Block *, unsigned> &blockOrder,
    const DenseMap<Block *, BitVector> &dominators) {
  if (successorOperands.size() != targetBlock->getNumArguments())
    return op->emitOpError()
           << "successor operand count does not match target block argument "
              "count";
  for (auto [index, successorOperand] : llvm::enumerate(successorOperands)) {
    BlockArgument argument = targetBlock->getArgument(index);
    if (successorOperand.getType() != argument.getType())
      return op->emitOpError()
             << "successor operand type does not match target block argument "
                "type";
    if (failed(verifyP2BlockArgumentType(op, successorOperand,
                                         "successor operand")) ||
        failed(verifyP2BlockArgumentType(op, argument, "block argument")))
      return failure();
  }
  if (failed(verifySuccessorOperandLoopShape(op, sourceBlock, targetBlock,
                                             successorOperands, blockOrder,
                                             dominators)))
    return failure();
  return verifyParallelEdgeCopyGroup(op, sourceBlock, targetBlock,
                                     successorOperands, blockOrder);
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

  DenseMap<Block *, unsigned> blockOrder;
  DenseMap<Block *, unsigned> blockIds;
  for (auto [index, block] : llvm::enumerate(blocks)) {
    blockOrder[block] = index;
    blockIds[block] = index;
  }

  DenseMap<Block *, Block *> nextBlock;
  for (size_t i = 0; i + 1 < blocks.size(); ++i)
    nextBlock[blocks[i]] = blocks[i + 1];

  DenseMap<Block *, SmallVector<Block *, 4>> predecessors;
  for (Block *block : blocks) {
    Operation *terminator = block->getTerminator();
    if (!terminator)
      continue;
    for (Block *successor : terminator->getSuccessors())
      predecessors[successor].push_back(block);
  }
  DenseMap<Block *, BitVector> dominators =
      computeDominance(blocks, predecessors);

  unsigned threadEndOpCount = 0;
  for (Block *block : blocks)
    for (Operation &op : *block)
      if (hasName(&op, kSSAVC4ThreadEndOpName))
        ++threadEndOpCount;
  bool needsThreadEndEpilogue = threadEndOpCount > 1;

  DenseMap<int64_t, Value> uniformReadByIndex;
  unsigned nextVirtualOrdinal = 0;
  for (Block *block : blocks) {
    for (BlockArgument argument : block->getArguments()) {
      if (!isP2BlockArgumentType(argument.getType()))
        return func->emitError()
              << "SSAVC4 block-argument lowering supports only i32, f32, "
                  "vector<16xi32>, or vector<16xf32> block arguments";
      virtualValues.push_back({argument, nextVirtualOrdinal++});
    }
  }

  unsigned nextLayoutBlockId = blocks.size();
  unsigned threadEndEpilogueBlockId = nextLayoutBlockId++;
  Operation *firstThreadEndOp = nullptr;
  bool sawThreadEnd = false;
  auto appendTemplate = [&](InstructionTemplate templ) {
    templ.ordinal = templates.size();
    templates.push_back(std::move(templ));
  };
  auto attachFlagProducer = [&](Operation *consumer, Value flags,
                                InstructionTemplate &templ) -> LogicalResult {
    Operation *definingOp = flags.getDefiningOp();
    if (!hasName(definingOp, kSSAVC4MakeFlagsOpName))
      return consumer->emitOpError()
             << "requires flags produced directly by ssavc4.make_flags";
    if (definingOp->getNumOperands() == 0 || definingOp->getNumOperands() > 2)
      return definingOp->emitOpError()
             << "supports only unary/binary flag compares in M3 v1";
    templ.flagSource = definingOp;
    templ.flagOperandCount = definingOp->getNumOperands();
    templ.operands.append(definingOp->operand_begin(),
                          definingOp->operand_end());
    return success();
  };
  auto appendEdgeCopies = [&](Operation *source, Block *sourceBlock,
                              unsigned layoutBlockId, Block *targetBlock,
                              ValueRange successorOperands) -> LogicalResult {
    if (failed(verifySuccessorOperandEdge(source, sourceBlock, targetBlock,
                                          successorOperands, blockOrder,
                                          dominators)))
      return failure();
    for (auto [index, successorOperand] : llvm::enumerate(successorOperands)) {
      InstructionTemplate templ;
      templ.kind = InstructionTemplate::Kind::EdgeCopy;
      templ.source = source;
      templ.sourceBlock = sourceBlock;
      templ.layoutBlockId = layoutBlockId;
      templ.operands.push_back(successorOperand);
      templ.result = targetBlock->getArgument(index);
      appendTemplate(std::move(templ));
    }
    return success();
  };

  int64_t nextUniformOrdinal = 0;
  for (Block *block : blocks) {
    for (Operation &op : *block) {
      if (hasName(&op, kSSAVC4ThreadEndOpName)) {
        if (!needsThreadEndEpilogue) {
          InstructionTemplate templ;
          templ.kind = InstructionTemplate::Kind::ThreadEnd;
          templ.source = &op;
          templ.sourceBlock = block;
          templ.layoutBlockId = blockIds[block];
          appendTemplate(std::move(templ));
          continue;
        }

        if (!firstThreadEndOp)
          firstThreadEndOp = &op;
        sawThreadEnd = true;

        // A QPU shader has one physical thread-end epilogue: a single thrend
        // signal followed by two delay slots at the end of the flattened
        // scheduled stream.  Multiple SSA CFG exits branch to that epilogue
        // instead of each emitting their own inline thrend.
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::Branch;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        templ.branchTargetBlockId = threadEndEpilogueBlockId;
        templ.syntheticThreadEndBranch = true;
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4BranchOpName)) {
        if (op.getNumSuccessors() != 1)
          return op.emitOpError("requires exactly one successor");
        auto branch = llvm::cast<mlir::ssavc4::BranchOp>(op);
        if (failed(appendEdgeCopies(&op, block, blockIds[block],
                                    branch.getTarget(),
                                    branch.getTargetOperands())))
          return failure();
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::Branch;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        templ.branchTargetBlockId = blockIds[op.getSuccessor(0)];
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4CondBranchOpName)) {
        if (op.getNumSuccessors() != 2)
          return op.emitOpError("requires true and false successors");
        auto condBranch = llvm::cast<mlir::ssavc4::CondBranchOp>(op);
        bool hasSuccessorOperands =
            !condBranch.getTrueDestOperands().empty() ||
            !condBranch.getFalseDestOperands().empty();
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::CondBranch;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        if (failed(attachFlagProducer(&op, condBranch.getFlags(), templ)))
          return failure();
        if (!hasSuccessorOperands) {
          auto nextIt = nextBlock.find(block);
          if (nextIt != nextBlock.end() &&
              op.getSuccessor(1) == nextIt->second) {
            templ.branchTargetBlockId = blockIds[condBranch.getTrueDest()];
            appendTemplate(std::move(templ));
            continue;
          }
          // VC4 conditional branches still fall through on the false path, but
          // valid SSAVC4 CFGs are not required to place the false successor as
          // the next source-layout block.  Reuse the successor-operand lowering
          // shape with empty edge-copy lists: a synthetic false fallthrough
          // block branches to the real false destination, and a synthetic true
          // block branches to the real true destination.
        }

        unsigned falseCopyBlockId = nextLayoutBlockId++;
        unsigned trueCopyBlockId = nextLayoutBlockId++;
        templ.branchTargetBlockId = trueCopyBlockId;
        appendTemplate(std::move(templ));

        if (failed(appendEdgeCopies(&op, block, falseCopyBlockId,
                                    condBranch.getFalseDest(),
                                    condBranch.getFalseDestOperands())))
          return failure();
        InstructionTemplate falseBranch;
        falseBranch.kind = InstructionTemplate::Kind::Branch;
        falseBranch.source = &op;
        falseBranch.sourceBlock = block;
        falseBranch.layoutBlockId = falseCopyBlockId;
        falseBranch.branchTargetBlockId = blockIds[condBranch.getFalseDest()];
        appendTemplate(std::move(falseBranch));

        if (failed(appendEdgeCopies(&op, block, trueCopyBlockId,
                                    condBranch.getTrueDest(),
                                    condBranch.getTrueDestOperands())))
          return failure();
        InstructionTemplate trueBranch;
        trueBranch.kind = InstructionTemplate::Kind::Branch;
        trueBranch.source = &op;
        trueBranch.sourceBlock = block;
        trueBranch.layoutBlockId = trueCopyBlockId;
        trueBranch.branchTargetBlockId = blockIds[condBranch.getTrueDest()];
        appendTemplate(std::move(trueBranch));
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
        templ.layoutBlockId = blockIds[block];
        templ.result = result;
        appendTemplate(std::move(templ));
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
        templ.layoutBlockId = blockIds[block];
        templ.result = result;
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4UniformReadOpName)) {
        if (op.getNumOperands() != 0 || op.getNumResults() != 1)
          return op.emitOpError("requires exactly one result for M3 lowering");
        Type resultType = op.getResult(0).getType();
        if (!resultType.isSignlessInteger(32) && !resultType.isF32() &&
            !isVector16I32Type(resultType) &&
            !isVector16F32Type(resultType))
          return op.emitOpError("supports only i32, f32, vector<16xi32>, or vector<16xf32> uniform reads in M3 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::UniformRead;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        templ.result = result;
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4SplatOpName)) {
        if (op.getNumOperands() != 1 || op.getNumResults() != 1)
          return op.emitOpError("requires one input operand and one result for M3 lowering");
        Type inputType = op.getOperand(0).getType();
        Type resultType = op.getResult(0).getType();
        bool inputI32 = inputType.isSignlessInteger(32);
        bool inputF32 = inputType.isF32();
        if (!inputI32 && !inputF32)
          return op.emitOpError("requires an i32 or f32 scalar input for M3 lowering");
        if (!isVector16I32Type(resultType) && !isVector16F32Type(resultType))
          return op.emitOpError("supports only vector<16xi32> or vector<16xf32> splat results in M3 lowering");
        if ((inputI32 && !isVector16I32Type(resultType)) ||
            (inputF32 && !isVector16F32Type(resultType)))
          return op.emitOpError("requires input and result to have the same SSAVC4 element domain in M3 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::Splat;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        appendTemplate(std::move(templ));
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4CondSelectOpName)) {
        if (op.getNumOperands() != 3 || op.getNumResults() != 1)
          return op.emitOpError(
              "requires flags, true_value, false_value, and one result for M5 lowering");
        if (op.getOperand(1).getType() != op.getOperand(2).getType() ||
            op.getOperand(1).getType() != op.getResult(0).getType())
          return op.emitOpError(
              "requires identical true_value, false_value, and result types for M5 lowering");
        if (!isVector16I32Type(op.getResult(0).getType()) &&
            !isVector16F32Type(op.getResult(0).getType()) &&
            !op.getResult(0).getType().isSignlessInteger(32) &&
            !op.getResult(0).getType().isF32())
          return op.emitOpError(
              "supports only i32, f32, vector<16xi32>, or vector<16xf32> values in M5 lowering");
        Value result = op.getResult(0);
        virtualValues.push_back({result, nextVirtualOrdinal++});
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::CondSelect;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        if (failed(attachFlagProducer(&op, op.getOperand(0), templ)))
          return failure();
        templ.operands.push_back(op.getOperand(1));
        templ.operands.push_back(op.getOperand(2));
        templ.result = result;
        appendTemplate(std::move(templ));
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4MakeFlagsOpName)) {
        if (op.getNumResults() != 1)
          return op.emitOpError("requires exactly one flag result");
        if (op.getNumOperands() == 0 || op.getNumOperands() > 2)
          return op.emitOpError("supports only unary/binary flag compares in M3 v1");
        // !ssavc4.flags is pseudo-SSA for transient physical condition
        // state.  It is rematerialized immediately before its single
        // conditional consumer rather than scheduled as an independent value.
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        appendTemplate(std::move(templ));
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = op.getResult(0);
        appendTemplate(std::move(templ));
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        appendTemplate(std::move(templ));
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4BarrierOpName)) {
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::Barrier;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        if (op.getNumOperands() != 0 && op.getNumOperands() != 2 &&
            op.getNumOperands() != 3)
          return op.emitOpError()
                 << "requires either no operands, logical_warp_id and "
                    "warps_per_block operands, or logical_warp_id, "
                    "warps_per_block, and semaphore_base operands";
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
        appendTemplate(std::move(templ));
        continue;
      }


      if (hasName(&op, kSSAVC4VPMWriteOpName)) {
        if (failed(verifyVPMResource(func, &op)))
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4VPMReadOpName)) {
        if (failed(verifyVPMResource(func, &op)))
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        templ.result = result;
        appendTemplate(std::move(templ));
        continue;
      }


      if (hasName(&op, kSSAVC4VDRLoadOpName)) {
        if (failed(verifyVPMResource(func, &op)))
          return failure();
        if (op.getNumOperands() != 2)
          return op.emitOpError("requires i32 global base address and VPM base row operands");
        if (!op.getOperand(0).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 global base address operand for M5 lowering");
        if (!op.getOperand(1).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 VPM base row operand for M5 lowering");
        int64_t rowLen = getI32IntegerAttrOr(&op, "row_len", -1);
        int64_t nrows = getI32IntegerAttrOr(&op, "nrows", -1);
        int64_t memoryPitchBytes = getI32IntegerAttrOr(&op, "memory_pitch_bytes", -1);
        int64_t vpmX = getI32IntegerAttrOr(&op, "vpm_x", -1);
        int64_t vpmPitch = getI32IntegerAttrOr(&op, "vpm_pitch", -1);
        auto width = llvm::dyn_cast_or_null<mlir::ssavc4::VPMElemWidthAttr>(
            op.getAttr("width"));
        auto subword = llvm::dyn_cast_or_null<mlir::ssavc4::VPMSubwordAttr>(
            op.getAttr("subword"));
        if (!width || !subword)
          return op.emitOpError("requires typed VDR width and subword attrs");
        if (width.getValue() != mlir::ssavc4::VPMElemWidth::w32)
          return op.emitOpError("supports only width = #ssavc4.vpm_elem_width<w32> in executable v1");
        if (subword.getValue() != mlir::ssavc4::VPMSubword::none)
          return op.emitOpError("supports only subword = #ssavc4.vpm_subword<none> in executable v1");
        if (rowLen <= 0 || rowLen > 16)
          return op.emitOpError("requires row_len in range [1, 16] for M5 lowering");
        if (nrows <= 0 || nrows > 16)
          return op.emitOpError("requires nrows in range [1, 16] for M5 lowering");
        if (memoryPitchBytes <= 0 || memoryPitchBytes % 4 != 0 ||
            memoryPitchBytes < rowLen * 4)
          return op.emitOpError("requires memory_pitch_bytes to cover whole 32-bit rows for M5 lowering");
        if (vpmX < 0 || vpmX > 15)
          return op.emitOpError("requires vpm_x in range [0, 15] for M5 lowering");
        if (vpmPitch <= 0 || vpmPitch > 16)
          return op.emitOpError("requires vpm_pitch in range [1, 16] for M5 lowering");
        if (auto serialize = llvm::dyn_cast_or_null<StringAttr>(op.getAttr("serialize"))) {
          if (serialize.getValue() != "mutex" && serialize.getValue() != "none")
            return op.emitOpError("supports only serialize = \"mutex\" or \"none\" in M5 VDR lowering");
        }
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::VDRLoad;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4VDRLoadRectDynamicOpName)) {
        if (failed(verifyVPMResource(func, &op)))
          return failure();
        if (op.getNumOperands() != 5)
          return op.emitOpError()
                 << "requires address, VPM base row, active rows, active "
                    "cols, and memory pitch operands";
        for (Value operand : op.getOperands()) {
          if (!operand.getType().isSignlessInteger(32))
            return op.emitOpError()
                   << "requires all dynamic rectangular VDR operands to be i32";
        }
        int64_t maxRows = getI32IntegerAttrOr(&op, "max_rows", -1);
        int64_t maxCols = getI32IntegerAttrOr(&op, "max_cols", -1);
        int64_t elemBytes = getI32IntegerAttrOr(&op, "elem_bytes", -1);
        int64_t dstX = getI32IntegerAttrOr(&op, "dst_x", -1);
        int64_t vpmPitch = getI32IntegerAttrOr(&op, "vpm_pitch", -1);
        auto orientation =
            llvm::dyn_cast_or_null<mlir::ssavc4::VPMOrientationAttr>(
                op.getAttr("orientation"));
        auto width = llvm::dyn_cast_or_null<mlir::ssavc4::VPMElemWidthAttr>(
            op.getAttr("width"));
        auto subword = llvm::dyn_cast_or_null<mlir::ssavc4::VPMSubwordAttr>(
            op.getAttr("subword"));
        auto zeroFill =
            llvm::dyn_cast_or_null<BoolAttr>(op.getAttr("zero_fill"));
        if (!orientation || !width || !subword || !zeroFill)
          return op.emitOpError()
                 << "requires typed orientation/width/subword and zero_fill attrs";
        if (!zeroFill.getValue())
          return op.emitOpError(
              "requires zero_fill = true for executable lowering");
        if (maxRows <= 0 || maxRows > 16 || maxCols <= 0 || maxCols > 16)
          return op.emitOpError("requires max_rows/max_cols in range [1, 16]");
        if (elemBytes != 4)
          return op.emitOpError("supports only elem_bytes = 4");
        if (width.getValue() != mlir::ssavc4::VPMElemWidth::w32)
          return op.emitOpError(
              "supports only width = #ssavc4.vpm_elem_width<w32>");
        if (subword.getValue() != mlir::ssavc4::VPMSubword::none)
          return op.emitOpError(
              "supports only subword = #ssavc4.vpm_subword<none>");
        if (dstX < 0 || dstX > 15)
          return op.emitOpError("requires dst_x in range [0, 15]");
        if (vpmPitch <= 0 || vpmPitch > 16)
          return op.emitOpError("requires vpm_pitch in range [1, 16]");
        if (auto serialize =
                llvm::dyn_cast_or_null<StringAttr>(op.getAttr("serialize"))) {
          if (serialize.getValue() != "mutex" && serialize.getValue() != "none")
            return op.emitOpError(
                "supports only serialize = \"mutex\" or \"none\"");
        }
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::VDRLoadRectDynamic;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        appendTemplate(std::move(templ));
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
        int64_t activeLanes = getI32IntegerAttrOr(&op, "active_lanes", -1);
        auto width = llvm::dyn_cast_or_null<mlir::ssavc4::VPMElemWidthAttr>(
            op.getAttr("width"));
        auto subword = llvm::dyn_cast_or_null<mlir::ssavc4::VPMSubwordAttr>(
            op.getAttr("subword"));
        if (!width || !subword)
          return op.emitOpError("requires typed VDW width and subword attrs");
        if (width.getValue() != mlir::ssavc4::VPMElemWidth::w32)
          return op.emitOpError("supports only width = #ssavc4.vpm_elem_width<w32> in executable v1");
        if (subword.getValue() != mlir::ssavc4::VPMSubword::none)
          return op.emitOpError("supports only subword = #ssavc4.vpm_subword<none> in executable v1");
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
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4VDWStoreVPMOpName)) {
        if (op.getNumOperands() < 3 || op.getNumOperands() > 4)
          return op.emitOpError(
              "requires address, VPM y, VPM x, and optional active-lane operand");
        if (!op.getOperand(0).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 global address operand for M3 lowering");
        if (!op.getOperand(1).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 VPM y-coordinate operand for M3 lowering");
        if (!op.getOperand(2).getType().isSignlessInteger(32))
          return op.emitOpError("requires an i32 VPM x-coordinate operand for M3 lowering");
        int64_t activeLanes = getI32IntegerAttrOr(&op, "active_lanes", -1);
        int64_t rowLen = getI32IntegerAttrOr(&op, "row_len", -1);
        int64_t nrows = getI32IntegerAttrOr(&op, "nrows", -1);
        int64_t memoryPitchBytes =
            getI32IntegerAttrOr(&op, "memory_pitch_bytes", -1);
        auto width = llvm::dyn_cast_or_null<mlir::ssavc4::VPMElemWidthAttr>(
            op.getAttr("width"));
        auto subword = llvm::dyn_cast_or_null<mlir::ssavc4::VPMSubwordAttr>(
            op.getAttr("subword"));
        if (!width || !subword)
          return op.emitOpError("requires typed VDW width and subword attrs");
        if (width.getValue() != mlir::ssavc4::VPMElemWidth::w32)
          return op.emitOpError("supports only width = #ssavc4.vpm_elem_width<w32> in executable v1");
        if (subword.getValue() != mlir::ssavc4::VPMSubword::none)
          return op.emitOpError("supports only subword = #ssavc4.vpm_subword<none> in executable v1");
        if (rowLen < 1 || rowLen > 16)
          return op.emitOpError("requires row_len in range [1, 16] for M3 lowering");
        if (nrows < 1 || nrows > 16)
          return op.emitOpError("requires nrows in range [1, 16] for M3 lowering");
        if (memoryPitchBytes < rowLen * 4 ||
            memoryPitchBytes % 4 != 0)
          return op.emitOpError("requires memory_pitch_bytes to cover whole 32-bit rows for M3 lowering");
        if (op.getNumOperands() == 4) {
          if (!op.getOperand(3).getType().isSignlessInteger(32))
            return op.emitOpError("requires an i32 dynamic active-lane operand");
        } else if (activeLanes != -1 && activeLanes != rowLen) {
          return op.emitOpError("requires static active_lanes to match row_len in M3 lowering");
        }
        if (auto serialize = llvm::dyn_cast_or_null<StringAttr>(op.getAttr("serialize"))) {
          if (serialize.getValue() != "mutex" && serialize.getValue() != "none")
            return op.emitOpError("supports only serialize = \"mutex\" or \"none\" in M3 lowering");
        }
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::VDWStoreVPM;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        appendTemplate(std::move(templ));
        continue;
      }

      if (hasName(&op, kSSAVC4VDWStoreRectDynamicOpName)) {
        if (failed(verifyVPMResource(func, &op)))
          return failure();
        if (op.getNumOperands() != 5)
          return op.emitOpError()
                 << "requires address, VPM source row, active rows, active "
                    "cols, and memory stride operands";
        for (Value operand : op.getOperands()) {
          if (!operand.getType().isSignlessInteger(32))
            return op.emitOpError()
                   << "requires all dynamic rectangular VDW operands to be i32";
        }
        int64_t maxRows = getI32IntegerAttrOr(&op, "max_rows", -1);
        int64_t maxCols = getI32IntegerAttrOr(&op, "max_cols", -1);
        int64_t elemBytes = getI32IntegerAttrOr(&op, "elem_bytes", -1);
        int64_t srcX = getI32IntegerAttrOr(&op, "src_x", -1);
        int64_t vpmPitch = getI32IntegerAttrOr(&op, "vpm_pitch", -1);
        auto orientation =
            llvm::dyn_cast_or_null<mlir::ssavc4::VPMOrientationAttr>(
                op.getAttr("orientation"));
        auto width = llvm::dyn_cast_or_null<mlir::ssavc4::VPMElemWidthAttr>(
            op.getAttr("width"));
        auto subword = llvm::dyn_cast_or_null<mlir::ssavc4::VPMSubwordAttr>(
            op.getAttr("subword"));
        auto preserve =
            llvm::dyn_cast_or_null<BoolAttr>(op.getAttr("preserve_inactive"));
        if (!orientation || !width || !subword || !preserve)
          return op.emitOpError()
                 << "requires typed orientation/width/subword and "
                    "preserve_inactive attrs";
        if (!preserve.getValue())
          return op.emitOpError(
              "requires preserve_inactive = true for executable lowering");
        if (maxRows <= 0 || maxRows > 16 || maxCols <= 0 || maxCols > 16)
          return op.emitOpError("requires max_rows/max_cols in range [1, 16]");
        if (elemBytes != 4)
          return op.emitOpError("supports only elem_bytes = 4");
        if (width.getValue() != mlir::ssavc4::VPMElemWidth::w32)
          return op.emitOpError(
              "supports only width = #ssavc4.vpm_elem_width<w32>");
        if (subword.getValue() != mlir::ssavc4::VPMSubword::none)
          return op.emitOpError(
              "supports only subword = #ssavc4.vpm_subword<none>");
        if (srcX < 0 || srcX > 15)
          return op.emitOpError("requires src_x in range [0, 15]");
        if (vpmPitch <= 0 || vpmPitch > 16)
          return op.emitOpError("requires vpm_pitch in range [1, 16]");
        if (auto serialize =
                llvm::dyn_cast_or_null<StringAttr>(op.getAttr("serialize"))) {
          if (serialize.getValue() != "mutex" && serialize.getValue() != "none")
            return op.emitOpError(
                "supports only serialize = \"mutex\" or \"none\"");
        }
        InstructionTemplate templ;
        templ.kind = InstructionTemplate::Kind::VDWStoreRectDynamic;
        templ.source = &op;
        templ.sourceBlock = block;
        templ.layoutBlockId = blockIds[block];
        templ.operands.append(op.operand_begin(), op.operand_end());
        appendTemplate(std::move(templ));
        continue;
      }

      return op.emitOpError()
             << "is not supported by the M3 SSAVC4 lowering; supported operations are "
                "ssavc4.load_imm, ssavc4.element_number, ssavc4.uniform.read, "
                "ssavc4.splat, ssavc4.mov, "
                "ssavc4.alu.add, ssavc4.alu.mul, "
                "ssavc4.make_flags, ssavc4.cond_select, ssavc4.br, "
                "ssavc4.cond_br, "
                "ssavc4.pack, ssavc4.unpack, ssavc4.rotate, "
                "ssavc4.tmu.request, ssavc4.tmu.read, "
                "ssavc4.sema.acquire, ssavc4.sema.release, "
                "ssavc4.barrier, ssavc4.vpm.write, ssavc4.vpm.read, "
                "ssavc4.vdr.load, ssavc4.vdr.load_rect.dynamic, "
                "ssavc4.vdw.store, ssavc4.vdw.store_rect.dynamic, and "
                "ssavc4.thread_end";
    }
  }

  if (sawThreadEnd) {
    InstructionTemplate templ;
    templ.kind = InstructionTemplate::Kind::ThreadEnd;
    templ.source = firstThreadEndOp;
    templ.layoutBlockId = threadEndEpilogueBlockId;
    appendTemplate(std::move(templ));
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
    state.addAttribute("value", values);
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

static LogicalResult emitMakeFlags(OpBuilder &builder,
                                   const InstructionTemplate &templ,
                                   const SpillAwareAllocator &allocator);

static LogicalResult emitCondSelect(OpBuilder &builder,
                                    const InstructionTemplate &templ,
                                    const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (!templ.result || templ.flagOperandCount == 0 ||
      templ.operands.size() != templ.flagOperandCount + 2)
    return source->emitError("internal lowering error: malformed cond_select");

  auto condAttr =
      llvm::dyn_cast_or_null<mlir::vc4::CondAttr>(source->getAttr("cond"));
  if (!condAttr)
    return source->emitOpError("requires a vc4.cond condition attribute");

  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  std::optional<int64_t> trueReg =
      allocator.lookup(templ, templ.operands[templ.flagOperandCount]);
  std::optional<int64_t> falseReg =
      allocator.lookup(templ, templ.operands[templ.flagOperandCount + 1]);
  if (!resultReg || !trueReg || !falseReg)
    return source->emitOpError()
           << "uses a value that is not defined by a lowerable SSAVC4 op in M5";

  if (failed(emitMakeFlags(builder, templ, allocator)))
    return failure();

  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, *falseReg, *falseReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        condAttr.getValue(), mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, *trueReg, *trueReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  emitRegfileResultSpacer(builder, source->getLoc(), templ, allocator);
  return success();
}

static void emitSpillStoreAction(OpBuilder &builder, Location loc,
                                 const SpillAction &action);
static void emitSpillReloadAction(OpBuilder &builder, Location loc,
                                  const SpillAction &action);

static LogicalResult emitEdgeCopy(OpBuilder &builder,
                                  const InstructionTemplate &templ,
                                  const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (!templ.result || templ.operands.size() != 1)
    return source->emitError("internal lowering error: malformed edge copy");

  std::optional<EdgeCopyAllocation> edgeCopy =
      allocator.getEdgeCopyAllocation(templ);
  if (edgeCopy) {
    bool sourceReg = edgeCopy->source.kind == LocationKind::Register;
    bool destReg = edgeCopy->destination.kind == LocationKind::Register;
    if (sourceReg && destReg) {
      createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                            mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                            edgeCopy->destination.physicalReg,
                            /*waddrMul=*/32, mlir::vc4::AddOpcode::bit_or,
                            mlir::vc4::MulOpcode::nop,
                            edgeCopy->source.physicalReg,
                            edgeCopy->source.physicalReg,
                            mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                            mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
      return success();
    }
    if (sourceReg && !destReg) {
      emitSpillStoreAction(
          builder, source->getLoc(),
          SpillAction{SpillAction::Kind::Store, *templ.result,
                      edgeCopy->destination.slot,
                      edgeCopy->source.physicalReg});
      return success();
    }
    if (!sourceReg && destReg) {
      emitSpillReloadAction(
          builder, source->getLoc(),
          SpillAction{SpillAction::Kind::Reload, templ.operands.front(),
                      edgeCopy->source.slot,
                      edgeCopy->destination.physicalReg});
      return success();
    }
    if (edgeCopy->source.slot.index == edgeCopy->destination.slot.index)
      return success();
    emitSpillReloadAction(
        builder, source->getLoc(),
        SpillAction{SpillAction::Kind::Reload, templ.operands.front(),
                    edgeCopy->source.slot,
                    SpillAwareAllocator::edgeCopyScratchReg()});
    emitSpillStoreAction(
        builder, source->getLoc(),
        SpillAction{SpillAction::Kind::Store, *templ.result,
                    edgeCopy->destination.slot,
                    SpillAwareAllocator::edgeCopyScratchReg()});
    return success();
  }

  std::optional<int64_t> resultReg = allocator.lookup(templ, *templ.result);
  std::optional<int64_t> inputReg = allocator.lookup(templ, templ.operands.front());
  if (!resultReg || !inputReg)
    return source->emitOpError()
           << "SSAVC4 block-argument lowering uses a value that is not available "
              "in a QPU register";

  createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        *resultReg, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, *inputReg, *inputReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
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

  bool useMirroredSecond = canUseMirroredSecondOperandForALU(
      templ, allocator, smallImm.has_value());
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

  if (aluNeedsSecondOperandAccumulatorMove(templ, allocator)) {
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

  if (templ.operands.size() != 2 && templ.operands.size() != 3)
    return source->emitError("internal lowering error: barrier has wrong operand count");
  std::optional<int64_t> logicalWarpReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> warpsPerBlockReg = allocator.lookup(templ, templ.operands[1]);
  std::optional<int64_t> semaphoreBaseReg;
  if (templ.operands.size() == 3)
    semaphoreBaseReg = allocator.lookup(templ, templ.operands[2]);
  if (!logicalWarpReg || !warpsPerBlockReg)
    return source->emitOpError()
           << "uses barrier metadata values that are not defined by lowerable SSAVC4 ops";
  if (templ.operands.size() == 3 && !semaphoreBaseReg)
    return source->emitOpError()
           << "uses a semaphore_base value that is not defined by lowerable SSAVC4 ops";

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
  Operation *source = templ.flagSource ? templ.flagSource : templ.source;
  unsigned flagOperandCount =
      templ.flagOperandCount ? templ.flagOperandCount : templ.operands.size();
  SmallVector<int64_t, 2> operandRegs;
  for (unsigned index = 0; index < flagOperandCount; ++index) {
    Value operand = templ.operands[index];
    std::optional<int64_t> reg = allocator.lookup(templ, operand);
    if (!reg)
      return source->emitOpError()
             << "uses a value that is not defined by a lowerable SSAVC4 op in M3 v1";
    operandRegs.push_back(*reg);
  }
  bool useMirroredSecond =
      canUseMirroredSecondOperandForMakeFlags(templ, allocator);
  int64_t raddrA = operandRegs.empty() ? 0 : operandRegs.front();
  int64_t raddrB = flagOperandCount < 2 ? raddrA
                  : useMirroredSecond   ? operandRegs[1]
                                        : 0;
  mlir::vc4::QPUMux addB = flagOperandCount < 2 ? mlir::vc4::QPUMux::a
                         : useMirroredSecond     ? mlir::vc4::QPUMux::b
                                                  : mlir::vc4::QPUMux::r1;
  std::optional<int64_t> smallImm;
  if (flagOperandCount == 1) {
    auto kindAttr =
        llvm::dyn_cast_or_null<mlir::ssavc4::FlagKindAttr>(source->getAttr("kind"));
    if (!kindAttr ||
        kindAttr.getValue() != mlir::ssavc4::FlagKind::zero_test)
      return source->emitOpError(
          "one-operand make_flags must be kind zero_test in M5 lowering");
    raddrB = 0;
    addB = mlir::vc4::QPUMux::b;
    smallImm = 0;
  }

  if (makeFlagsNeedsSecondOperandAccumulatorMove(templ, allocator)) {
    createScheduledBundle(builder, source->getLoc(), mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, operandRegs[1],
                          operandRegs[1], mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
  }

  createScheduledBundle(builder, source->getLoc(),
                        smallImm ? mlir::vc4::QPUSignal::small_imm
                                 : mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/31, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::sub,
                        mlir::vc4::MulOpcode::nop, raddrA, raddrB,
                        mlir::vc4::QPUMux::a, addB,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        smallImm, /*setFlags=*/true);
  return success();
}

static LogicalResult emitScheduledBranch(OpBuilder &builder,
                                         const InstructionTemplate &templ,
                                         const LayoutSummary &layout) {
  Operation *source = templ.source;
  auto immediateIt = layout.branchImmediates.find(templ.ordinal);
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

  auto orientation =
      llvm::cast<mlir::ssavc4::VPMOrientationAttr>(
          source->getAttr("orientation"));
  int64_t stride = getI32IntegerAttrOr(source, "stride", 1);
  int64_t x = getI32IntegerAttrOr(source, "x", 0);
  int64_t setupBase =
      (1 << 20) | ((stride & 0x3f) << 12) |
      (orientation.getValue() == mlir::ssavc4::VPMOrientation::vertical
           ? (0x200 | x)
           : 0xa00);

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

  auto orientation =
      llvm::cast<mlir::ssavc4::VPMOrientationAttr>(
          source->getAttr("orientation"));
  int64_t stride = getI32IntegerAttrOr(source, "stride", 1);
  int64_t x = getI32IntegerAttrOr(source, "x", 0);
  int64_t setupBase =
      (1 << 20) | ((stride & 0x3f) << 12) |
      (orientation.getValue() == mlir::ssavc4::VPMOrientation::vertical
           ? (0x200 | x)
           : 0xa00);

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

static void emitRawVDWStoreFromVPM(OpBuilder &builder, Location loc,
                                   int64_t addressReg,
                                   int64_t vpmYReg,
                                   int64_t vpmXReg,
                                   std::optional<int64_t> dynamicActiveLanesReg,
                                   int64_t activeLanes,
                                   int64_t rowLen,
                                   int64_t nrows,
                                   int64_t memoryPitchBytes,
                                   bool vertical,
                                   bool useMutex,
                                   mlir::vc4::QPUMux addressMux =
                                       mlir::vc4::QPUMux::a,
                                   mlir::vc4::QPUMux vpmYMux =
                                       mlir::vc4::QPUMux::a,
                                   mlir::vc4::QPUMux vpmXMux =
                                       mlir::vc4::QPUMux::a) {
  (void)activeLanes;
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

  // VDW store setup: the source coordinate uses the DMA descriptor layout
  // dma_h32/dma_v32(y, x), i.e. y in bits 7.. and x in bits 3...
  // The data is already resident in VPM, so unlike ssavc4.vdw.store there is
  // no register-to-VPM staging write here.
  int64_t strideBytes = memoryPitchBytes - rowLen * 4;

  if (dynamicActiveLanesReg) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, *dynamicActiveLanesReg,
                          *dynamicActiveLanesReg, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::max,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                          /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                          mlir::vc4::QPUMux::b,
                          mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                          /*smallImm=*/0);
    if (rowLen == 16) {
      createSplat32LDI(builder, loc, 16, 32);
      createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                            mlir::vc4::Cond::always,
                            mlir::vc4::Cond::never,
                            /*waddrAdd=*/33, /*waddrMul=*/32,
                            mlir::vc4::AddOpcode::min,
                            mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                            /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                            mlir::vc4::QPUMux::r0,
                            mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
    } else {
      createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                            mlir::vc4::Cond::always,
                            mlir::vc4::Cond::never,
                            /*waddrAdd=*/33, /*waddrMul=*/32,
                            mlir::vc4::AddOpcode::min,
                            mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                            /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                            mlir::vc4::QPUMux::b,
                            mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                            /*smallImm=*/static_cast<int32_t>(rowLen));
    }
  } else {
    createSplat32LDI(builder, loc, rowLen, 33);
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
  uint32_t setupBase = 0x80000000u |
                       ((static_cast<uint32_t>(nrows) & 0x7fu) << 23) |
                       (vertical ? 0u : 0x4000u);
  createSplat32LDI(builder, loc, static_cast<int32_t>(setupBase), 35);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, vpmYReg, vpmYReg,
                        vpmYMux, vpmYMux,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/7);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, vpmXReg, vpmXReg,
                        vpmXMux, vpmXMux,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/3);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                    mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  // Program the extended memory stride after the basic DMA setup.  This is
  // the order used by the handwritten VC4 examples and by VC4C's VPM writer.
  createSplat32LDI(builder, loc,
                   static_cast<int32_t>(0xc0000000u |
                                        (static_cast<uint32_t>(strideBytes) &
                                         0xffffu)),
                   35);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                    mlir::vc4::QPUMux::r3, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createVPMVCDAddr(builder, loc, mlir::vc4::VPMVCDSide::write,
                   mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                   mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                   addressReg, addressReg, addressMux,
                   addressMux, mlir::vc4::QPUMux::r0,
                   mlir::vc4::QPUMux::r1);
  createVPMVCDWait(builder, loc, mlir::vc4::VPMVCDSide::write);
  if (useMutex)
    emitMutexRelease(builder, loc);
}


static uint32_t encodeVDRCount16(int64_t value) {
  return value == 16 ? 0u : static_cast<uint32_t>(value & 0x0f);
}

static uint32_t encodeVDRVPMXY(int64_t row, int64_t col) {
  return ((static_cast<uint32_t>(row) & 0x3fu) << 4) |
         (static_cast<uint32_t>(col) & 0x0fu);
}

static std::optional<uint32_t> encodeVDRMemoryPitchBytes(int64_t bytes) {
  if (bytes < 8)
    return std::nullopt;
  uint32_t pitch = 8;
  for (uint32_t encoded = 0; encoded <= 15; ++encoded, pitch <<= 1) {
    if (static_cast<int64_t>(pitch) == bytes)
      return encoded;
  }
  return std::nullopt;
}

static LogicalResult buildVDRLoadSetupWordFromParts(
    Operation *source, int64_t rowLen, int64_t nrows,
    int64_t memoryPitchBytes, int64_t vpmX, int64_t vpmPitch, bool vertical,
    int64_t &setupWord) {
  if (rowLen < 1 || rowLen > 16 || nrows < 1 || nrows > 16 ||
      vpmX < 0 || vpmX > 15 || vpmPitch < 1 || vpmPitch > 16)
    return source->emitOpError("has invalid VDR setup attributes after verification");

  std::optional<uint32_t> mpitch = encodeVDRMemoryPitchBytes(memoryPitchBytes);
  if (!mpitch)
    return source->emitOpError()
           << "requires memory_pitch_bytes encodable as VDR MPITCH = 8 * 2^n bytes";

  // VPMVCD_RD_SETUP basic 32-bit DMA-load setup:
  //   bit 31 marks the read/DMA setup word, MPITCH encodes 8*2^n byte source
  //   pitch, zero-encoded 16-wide ROWLEN/NROWS fields describe the memory
  //   transfer, VPITCH advances the destination VPM row between source rows,
  //   VERT selects orientation, and low bits encode ADDRA X[3:0].  ADDRA
  //   Y[5:0] is patched in dynamically from the block-local shared tile row
  //   when emitting the scheduled VC4 setup sequence.
  //   This is intentionally narrower than the full
  //   VC4 setup space; M5 only exposes regular 32-bit tile loads.
  uint32_t word = 0x80000000u;
  word |= (*mpitch & 0x0fu) << 24;
  word |= (encodeVDRCount16(rowLen) & 0x0fu) << 20;
  word |= (encodeVDRCount16(nrows) & 0x0fu) << 16;
  word |= (encodeVDRCount16(vpmPitch) & 0x0fu) << 12;
  if (vertical)
    word |= 1u << 11;
  word |= encodeVDRVPMXY(/*row=*/0, vpmX);
  setupWord = static_cast<int32_t>(word);
  return success();
}

static LogicalResult buildVDRLoadSetupWord(Operation *source,
                                           int64_t &setupWord) {
  int64_t rowLen = getI32IntegerAttrOr(source, "row_len", -1);
  int64_t nrows = getI32IntegerAttrOr(source, "nrows", -1);
  int64_t memoryPitchBytes =
      getI32IntegerAttrOr(source, "memory_pitch_bytes", -1);
  int64_t vpmX = getI32IntegerAttrOr(source, "vpm_x", -1);
  int64_t vpmPitch = getI32IntegerAttrOr(source, "vpm_pitch", -1);
  auto orientation =
      llvm::cast<mlir::ssavc4::VPMOrientationAttr>(
          source->getAttr("orientation"));
  bool vertical =
      orientation.getValue() == mlir::ssavc4::VPMOrientation::vertical;
  return buildVDRLoadSetupWordFromParts(source, rowLen, nrows,
                                        memoryPitchBytes, vpmX, vpmPitch,
                                        vertical, setupWord);
}

static void emitRawVDRLoad(OpBuilder &builder, Location loc,
                           int64_t addressReg, int64_t setupWord,
                           int64_t vpmBaseRowReg, bool useMutex,
                           mlir::vc4::QPUMux addressMux =
                               mlir::vc4::QPUMux::a,
                           mlir::vc4::QPUMux vpmBaseRowMux =
                               mlir::vc4::QPUMux::a,
                           std::optional<int64_t> addressAddReg =
                               std::nullopt) {
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
  createSplat32LDI(builder, loc, setupWord, 35);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, vpmBaseRowReg,
                        /*raddrB=*/0, vpmBaseRowMux,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/4);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::read,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                    mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  if (addressAddReg) {
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/33, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, *addressAddReg,
                          *addressAddReg, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
    createVPMVCDAddr(builder, loc, mlir::vc4::VPMVCDSide::read,
                     mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                     mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                     addressReg, /*raddrB=*/0, addressMux,
                     mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r0,
                     mlir::vc4::QPUMux::r1);
  } else {
    createVPMVCDAddr(builder, loc, mlir::vc4::VPMVCDSide::read,
                     mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                     mlir::vc4::AddOpcode::bit_or,
                     mlir::vc4::MulOpcode::nop, addressReg, addressReg,
                     addressMux, addressMux, mlir::vc4::QPUMux::r0,
                     mlir::vc4::QPUMux::r1);
  }
  createVPMVCDWait(builder, loc, mlir::vc4::VPMVCDSide::read);
  if (useMutex)
    emitMutexRelease(builder, loc);
}

static LogicalResult emitVDRLoad(OpBuilder &builder,
                                 const InstructionTemplate &templ,
                                 const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (templ.operands.size() != 2)
    return source->emitError("internal lowering error: VDR load has wrong operand count");
  std::optional<int64_t> addressReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> vpmBaseRowReg =
      allocator.lookup(templ, templ.operands[1]);
  if (!addressReg || !vpmBaseRowReg)
    return source->emitOpError()
           << "uses a VDR base address or VPM base row that is not defined by a lowerable SSAVC4 op";
  int64_t setupWord = 0;
  if (failed(buildVDRLoadSetupWord(source, setupWord)))
    return failure();
  emitRawVDRLoad(builder, source->getLoc(), *addressReg, setupWord,
                 *vpmBaseRowReg,
                 hasStringAttr(source, "serialize", "mutex"));
  return success();
}

static void emitVPMZeroWriteRow(OpBuilder &builder, Location loc,
                                int64_t rowReg, int64_t x, bool vertical,
                                bool useMutex,
                                mlir::vc4::QPUMux rowMux =
                                    mlir::vc4::QPUMux::a) {
  int64_t setupBase =
      (1 << 20) | (1 << 12) | (vertical ? (0x200 | x) : 0xa00);
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
                    rowReg, /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                    rowMux, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createSplat32LDI(builder, loc, 0, 34);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/48, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/34,
                        /*raddrB=*/34, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  if (useMutex) {
    createVPMVCDWait(builder, loc, mlir::vc4::VPMVCDSide::write);
    emitMutexRelease(builder, loc);
  }
}

static void emitAddConstantToScratch(OpBuilder &builder, Location loc,
                                     int64_t baseReg,
                                     mlir::vc4::QPUMux baseMux,
                                     int64_t offset, int64_t scratchWaddr) {
  createSplat32LDI(builder, loc, offset, 35);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        scratchWaddr, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add,
                        mlir::vc4::MulOpcode::nop, baseReg, /*raddrB=*/0,
                        baseMux, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
}

static void emitRuntimeActiveRowsGuard(OpBuilder &builder, Location loc,
                                       int64_t activeRowsReg, int64_t maxRows,
                                       int64_t row,
                                       unsigned guardedPayloadSlots) {
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, activeRowsReg,
                        activeRowsReg, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::max,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/0);
  createSplat32LDI(builder, loc, maxRows, 35);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::min,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r3, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/31, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::sub,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1,
                        /*smallImm=*/static_cast<int32_t>(row + 1),
                        /*setFlags=*/true);
  createScheduledBranch(builder, loc, mlir::vc4::BranchCond::any_c_set,
                        static_cast<int64_t>(7 + guardedPayloadSlots) * 8,
                        /*raddrA=*/0, /*waddrAdd=*/31, /*waddrMul=*/30);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
}

static void emitDynamicVDRLoadOneRow(OpBuilder &builder, Location loc,
                                     int64_t addressReg, int64_t vpmBaseRowReg,
                                     int64_t setupWord, bool useMutex,
                                     mlir::vc4::QPUMux addressMux =
                                         mlir::vc4::QPUMux::a,
                                     mlir::vc4::QPUMux vpmBaseRowMux =
                                         mlir::vc4::QPUMux::a) {
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
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_and,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/15);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/8);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/8);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/4);
  createSplat32LDI(builder, loc, setupWord, 32);
  createNopBundle(builder, loc);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r3, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, vpmBaseRowReg,
                        /*raddrB=*/0, vpmBaseRowMux,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/4);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::read,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                    mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createVPMVCDAddr(builder, loc, mlir::vc4::VPMVCDSide::read,
                   mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                   mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                   addressReg, addressReg, addressMux,
                   addressMux, mlir::vc4::QPUMux::r0,
                   mlir::vc4::QPUMux::r1);
  createVPMVCDWait(builder, loc, mlir::vc4::VPMVCDSide::read);
  if (useMutex)
    emitMutexRelease(builder, loc);
}

static void emitDynamicVDRLoadRows(OpBuilder &builder, Location loc,
                                   int64_t addressReg, int64_t vpmBaseRowReg,
                                   int64_t setupWordWithoutNRows,
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
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/8);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/8);
  createSplat32LDI(builder, loc, setupWordWithoutNRows, 32);
  createNopBundle(builder, loc);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r3, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, vpmBaseRowReg,
                        /*raddrB=*/0, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1, /*smallImm=*/4);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::read,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                    mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  createNopBundle(builder, loc);
  createNopBundle(builder, loc);
  createVPMVCDAddr(builder, loc, mlir::vc4::VPMVCDSide::read,
                   mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                   mlir::vc4::AddOpcode::bit_or, mlir::vc4::MulOpcode::nop,
                   addressReg, addressReg, mlir::vc4::QPUMux::a,
                   mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                   mlir::vc4::QPUMux::r1);
  createVPMVCDWait(builder, loc, mlir::vc4::VPMVCDSide::read);
  if (useMutex)
    emitMutexRelease(builder, loc);
}

static LogicalResult emitVDRLoadRectDynamic(
    OpBuilder &builder, const InstructionTemplate &templ,
    const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (templ.operands.size() != 5)
    return source->emitError(
        "internal lowering error: dynamic VDR rect has wrong operand count");
  std::optional<int64_t> addressReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> vpmBaseRowReg =
      allocator.lookup(templ, templ.operands[1]);
  if (!addressReg || !vpmBaseRowReg)
    return source->emitOpError()
           << "uses a VDR base address or VPM base row that is not defined by "
              "a lowerable SSAVC4 op";

  int64_t maxRows = getI32IntegerAttrOr(source, "max_rows", -1);
  int64_t maxCols = getI32IntegerAttrOr(source, "max_cols", -1);
  int64_t dstX = getI32IntegerAttrOr(source, "dst_x", -1);
  int64_t vpmPitch = getI32IntegerAttrOr(source, "vpm_pitch", -1);
  std::optional<int64_t> activeRows =
      getConstantI32FromLoadImm(templ.operands[2]);
  std::optional<int64_t> activeCols =
      getConstantI32FromLoadImm(templ.operands[3]);
  std::optional<int64_t> pitchBytes =
      getConstantI32FromLoadImm(templ.operands[4]);
  std::optional<int64_t> pitchReg =
      pitchBytes ? std::nullopt : allocator.lookup(templ, templ.operands[4]);
  if (!pitchBytes && !pitchReg)
    return source->emitOpError()
           << "uses a dynamic pitch value that is not defined by a lowerable "
              "SSAVC4 op";

  auto orientation =
      llvm::cast<mlir::ssavc4::VPMOrientationAttr>(
          source->getAttr("orientation"));
  bool vertical =
      orientation.getValue() == mlir::ssavc4::VPMOrientation::vertical;
  int64_t clampedRows =
      activeRows ? std::clamp(*activeRows, int64_t(0), maxRows) : maxRows;
  int64_t clampedCols = activeCols ? std::clamp(*activeCols, int64_t(0), maxCols)
                                   : maxCols;
  bool useMutex = hasStringAttr(source, "serialize", "mutex");
  if (!activeRows) {
    if (maxRows < 1 || maxRows > 2)
      return source->emitOpError()
             << "dynamic rectangular VDR runtime active_rows currently "
                "supports only up to two-row rectangles";
    if (!activeCols || clampedCols != maxCols)
      return source->emitOpError()
             << "dynamic rectangular VDR runtime active_rows currently "
                "requires full static active_cols";
    std::optional<int64_t> activeRowsReg =
        allocator.lookup(templ, templ.operands[2]);
    if (!activeRowsReg)
      return source->emitOpError()
             << "uses a dynamic active row value that is not defined by a "
                "lowerable SSAVC4 op";
    if (!pitchBytes) {
      if (maxRows < 1 || maxRows > 2)
        return source->emitOpError()
               << "dynamic rectangular VDR runtime pitch currently supports "
                  "only up to two-row rectangles";
      int64_t oneRowSetup = 0;
      if (failed(buildVDRLoadSetupWordFromParts(
              source, maxCols, /*nrows=*/1, maxCols * 4, dstX, vpmPitch,
              vertical, oneRowSetup)))
        return failure();
      for (int64_t row = 0; row < maxRows; ++row) {
        if (row == 0) {
          emitVPMZeroWriteRow(builder, source->getLoc(), *vpmBaseRowReg, dstX,
                              vertical, useMutex);
        } else {
          emitAddConstantToScratch(builder, source->getLoc(), *vpmBaseRowReg,
                                   mlir::vc4::QPUMux::a, row * vpmPitch,
                                   /*scratchWaddr=*/32);
          emitVPMZeroWriteRow(builder, source->getLoc(), /*rowReg=*/0, dstX,
                              vertical, useMutex, mlir::vc4::QPUMux::r0);
        }
      }
      for (int64_t row = 0; row < maxRows; ++row) {
        unsigned bodySlots = getDynamicVDRPitchRowSlotCount(useMutex, row);
        emitRuntimeActiveRowsGuard(builder, source->getLoc(), *activeRowsReg,
                                   maxRows, row, bodySlots);
        if (row == 0) {
          emitRawVDRLoad(builder, source->getLoc(), *addressReg, oneRowSetup,
                         *vpmBaseRowReg, useMutex);
        } else {
          emitAddConstantToScratch(builder, source->getLoc(), *vpmBaseRowReg,
                                   mlir::vc4::QPUMux::a, row * vpmPitch,
                                   /*scratchWaddr=*/32);
          emitRawVDRLoad(builder, source->getLoc(), *addressReg, oneRowSetup,
                         /*vpmBaseRowReg=*/0, useMutex,
                         mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                         /*addressAddReg=*/*pitchReg);
        }
      }
      return success();
    }
    int64_t setupWordWithoutNRows = 0;
    if (failed(buildVDRLoadSetupWordFromParts(
            source, maxCols, 16, *pitchBytes, dstX, vpmPitch, vertical,
            setupWordWithoutNRows)))
      return failure();
    for (int64_t row = 0; row < maxRows; ++row) {
      if (row == 0) {
        emitVPMZeroWriteRow(builder, source->getLoc(), *vpmBaseRowReg, dstX,
                            vertical, useMutex);
      } else {
        emitAddConstantToScratch(builder, source->getLoc(), *vpmBaseRowReg,
                                 mlir::vc4::QPUMux::a, row * vpmPitch,
                                 /*scratchWaddr=*/32);
        emitVPMZeroWriteRow(builder, source->getLoc(), /*rowReg=*/0, dstX,
                            vertical, useMutex, mlir::vc4::QPUMux::r0);
      }
    }
    unsigned bodySlots = useMutex ? 16 : 14;
    emitRuntimeActiveRowsGuard(builder, source->getLoc(), *activeRowsReg,
                               maxRows, /*row=*/0, bodySlots);
    emitDynamicVDRLoadRows(builder, source->getLoc(), *addressReg,
                           *vpmBaseRowReg, setupWordWithoutNRows, useMutex);
    return success();
  }
  if (!activeCols) {
    if (maxRows != 1 || clampedRows != 1)
      return source->emitOpError()
             << "dynamic rectangular VDR runtime active_cols currently "
                "supports only one active row";
    std::optional<int64_t> activeColsReg =
        allocator.lookup(templ, templ.operands[3]);
    if (!activeColsReg)
      return source->emitOpError()
             << "uses a dynamic active column value that is not defined by a "
                "lowerable SSAVC4 op";
    emitVPMZeroWriteRow(builder, source->getLoc(), *vpmBaseRowReg, dstX,
                        vertical, useMutex);
    createScheduledBundle(builder, source->getLoc(),
                          mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/34, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::bit_or,
                          mlir::vc4::MulOpcode::nop, *activeColsReg,
                          *activeColsReg, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
    createScheduledBundle(builder, source->getLoc(),
                          mlir::vc4::QPUSignal::small_imm,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/34, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::max,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                          /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                          mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1, /*smallImm=*/0);
    createSplat32LDI(builder, source->getLoc(), maxCols, 35);
    createScheduledBundle(builder, source->getLoc(),
                          mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/34, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::min,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                          /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                          mlir::vc4::QPUMux::r3, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1);
    createScheduledBundle(builder, source->getLoc(),
                          mlir::vc4::QPUSignal::small_imm,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/31, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::sub,
                          mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                          /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                          mlir::vc4::QPUMux::b, mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r1, /*smallImm=*/1,
                          /*setFlags=*/true);
	    unsigned bodySlots = useMutex ? 18 : 16;
	    createScheduledBranch(builder, source->getLoc(),
	                          mlir::vc4::BranchCond::any_c_set,
	                          static_cast<int64_t>(7 + bodySlots) * 8,
	                          /*raddrA=*/0, /*waddrAdd=*/31, /*waddrMul=*/30);
    createNopBundle(builder, source->getLoc());
    createNopBundle(builder, source->getLoc());
    createNopBundle(builder, source->getLoc());
    int64_t setupWord = 0;
    int64_t setupPitchBytes = pitchBytes ? *pitchBytes : maxCols * 4;
    if (failed(buildVDRLoadSetupWordFromParts(
            source, 16, 1, setupPitchBytes, dstX, vpmPitch, vertical,
            setupWord)))
      return failure();
    emitDynamicVDRLoadOneRow(builder, source->getLoc(), *addressReg,
                             *vpmBaseRowReg, setupWord, useMutex);
    return success();
  }

  if (!pitchBytes)
    return source->emitOpError()
           << "dynamic rectangular VDR runtime pitch currently requires "
              "runtime active_rows with full static active_cols";

  if (clampedRows != maxRows || clampedCols != maxCols) {
    if (maxRows != 1)
      return source->emitOpError()
             << "dynamic rectangular VDR partial zero-fill currently "
                "supports only one-row rectangles";
    emitVPMZeroWriteRow(builder, source->getLoc(), *vpmBaseRowReg, dstX,
                        vertical, useMutex);
    if (clampedRows == 0 || clampedCols == 0)
      return success();
  }

  int64_t setupWord = 0;
  if (failed(buildVDRLoadSetupWordFromParts(
          source, clampedCols, clampedRows, *pitchBytes, dstX, vpmPitch,
          vertical, setupWord)))
    return failure();
  emitRawVDRLoad(builder, source->getLoc(), *addressReg, setupWord,
                 *vpmBaseRowReg, useMutex);
  return success();
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

  int64_t activeLanes = getI32IntegerAttrOr(source, "active_lanes", -1);
  int64_t vpmRow = getI32IntegerAttrOr(source, "vpm_row", 0);
  auto width = llvm::cast<mlir::ssavc4::VPMElemWidthAttr>(
      source->getAttr("width"));
  auto subword = llvm::cast<mlir::ssavc4::VPMSubwordAttr>(
      source->getAttr("subword"));
  if (width.getValue() != mlir::ssavc4::VPMElemWidth::w32)
    return source->emitOpError(
        "supports only width = #ssavc4.vpm_elem_width<w32> in executable v1");
  if (subword.getValue() != mlir::ssavc4::VPMSubword::none)
    return source->emitOpError(
        "supports only subword = #ssavc4.vpm_subword<none> in executable v1");
  if (!dynamicActiveLanesReg && activeLanes != 16)
    return source->emitOpError("supports only full 16-lane static VDW stores in M3 lowering");
  if (vpmRow < 0)
    return source->emitOpError("requires a non-negative VPM row in M3 lowering");

  emitRawVDWStore(builder, source->getLoc(), *addressReg, *valueReg,
                  dynamicActiveLanesReg, dynamicVPMRowReg, activeLanes,
                  vpmRow, hasStringAttr(source, "serialize", "mutex"));
  return success();
}

static LogicalResult emitVDWStoreVPM(OpBuilder &builder,
                                     const InstructionTemplate &templ,
                                     const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (templ.operands.size() < 3 || templ.operands.size() > 4)
    return source->emitError("internal lowering error: VPM-source VDW store has wrong operand count");
  std::optional<int64_t> addressReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> vpmYReg = allocator.lookup(templ, templ.operands[1]);
  std::optional<int64_t> vpmXReg = allocator.lookup(templ, templ.operands[2]);
  if (!addressReg || !vpmYReg || !vpmXReg)
    return source->emitOpError()
           << "uses a VDW address/VPM coordinate that is not defined by a lowerable SSAVC4 op";

  std::optional<int64_t> dynamicActiveLanesReg;
  if (templ.operands.size() == 4) {
    dynamicActiveLanesReg = allocator.lookup(templ, templ.operands[3]);
    if (!dynamicActiveLanesReg)
      return source->emitOpError()
             << "uses a dynamic VDW active-lane value that is not defined by a lowerable SSAVC4 op";
  }

  int64_t activeLanes = getI32IntegerAttrOr(source, "active_lanes", -1);
  int64_t rowLen = getI32IntegerAttrOr(source, "row_len", -1);
  int64_t nrows = getI32IntegerAttrOr(source, "nrows", -1);
  int64_t memoryPitchBytes =
      getI32IntegerAttrOr(source, "memory_pitch_bytes", -1);
  auto width = llvm::cast<mlir::ssavc4::VPMElemWidthAttr>(
      source->getAttr("width"));
  auto subword = llvm::cast<mlir::ssavc4::VPMSubwordAttr>(
      source->getAttr("subword"));
  if (width.getValue() != mlir::ssavc4::VPMElemWidth::w32)
    return source->emitOpError(
        "supports only width = #ssavc4.vpm_elem_width<w32> in executable v1");
  if (subword.getValue() != mlir::ssavc4::VPMSubword::none)
    return source->emitOpError(
        "supports only subword = #ssavc4.vpm_subword<none> in executable v1");
  if (rowLen < 1 || rowLen > 16)
    return source->emitOpError("requires row_len in range [1, 16] for M3 lowering");
  if (nrows < 1 || nrows > 16)
    return source->emitOpError("requires nrows in range [1, 16] for M3 lowering");
  int64_t strideBytes = memoryPitchBytes - rowLen * 4;
  if (strideBytes < 0 || strideBytes > 65535 ||
      memoryPitchBytes % 4 != 0)
    return source->emitOpError("requires memory_pitch_bytes to produce an encodable VDW stride");
  if (dynamicActiveLanesReg && nrows != 1)
    return source->emitOpError("supports dynamic active_lanes only for single-row VDW stores");
  if (!dynamicActiveLanesReg && activeLanes != -1 && activeLanes != rowLen)
    return source->emitOpError("requires static active_lanes to match row_len in M3 lowering");

  auto orientation =
      llvm::cast<mlir::ssavc4::VPMOrientationAttr>(
          source->getAttr("orientation"));
  bool vertical =
      orientation.getValue() == mlir::ssavc4::VPMOrientation::vertical;
  emitRawVDWStoreFromVPM(builder, source->getLoc(), *addressReg, *vpmYReg,
                         *vpmXReg,
                         dynamicActiveLanesReg, activeLanes, rowLen, nrows,
                         memoryPitchBytes, vertical,
                         hasStringAttr(source, "serialize", "mutex"));
  return success();
}

static void emitDynamicVDWStoreRowsFromVPM(OpBuilder &builder, Location loc,
                                           int64_t addressReg,
                                           int64_t vpmYReg, int64_t vpmXReg,
                                           int64_t rowLen,
                                           std::optional<int64_t> memoryPitchBytes,
                                           std::optional<int64_t> memoryPitchReg,
                                           bool vertical, bool useMutex) {
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

  int64_t rowBytes = rowLen * 4;
  int64_t strideBytes = memoryPitchBytes ? *memoryPitchBytes - rowBytes : 0;
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/8);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/8);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/7);
  createSplat32LDI(builder, loc, rowLen, 33);
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
  uint32_t setupBase = 0x80000000u | (vertical ? 0u : 0x4000u);
  createSplat32LDI(builder, loc, static_cast<int32_t>(setupBase), 32);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r0,
                        mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/34, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                        mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, vpmYReg, vpmYReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/7);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::bit_or,
                        mlir::vc4::MulOpcode::nop, vpmXReg, vpmXReg,
                        mlir::vc4::QPUMux::a, mlir::vc4::QPUMux::a,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::small_imm,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/35, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::shl,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::b,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1,
                        /*smallImm=*/3);
  createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                        mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                        /*waddrAdd=*/33, /*waddrMul=*/32,
                        mlir::vc4::AddOpcode::add,
                        mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                        /*raddrB=*/0, mlir::vc4::QPUMux::r1,
                        mlir::vc4::QPUMux::r3,
                        mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
  createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                    mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                    mlir::vc4::AddOpcode::add, mlir::vc4::MulOpcode::nop,
                    /*raddrA=*/0, /*raddrB=*/0, mlir::vc4::QPUMux::r2,
                    mlir::vc4::QPUMux::r1, mlir::vc4::QPUMux::r0,
                    mlir::vc4::QPUMux::r1);
  if (memoryPitchReg) {
    createSplat32LDI(builder, loc, rowBytes, 32);
    createScheduledBundle(builder, loc, mlir::vc4::QPUSignal::none,
                          mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                          /*waddrAdd=*/35, /*waddrMul=*/32,
                          mlir::vc4::AddOpcode::sub,
                          mlir::vc4::MulOpcode::nop, *memoryPitchReg,
                          /*raddrB=*/0, mlir::vc4::QPUMux::a,
                          mlir::vc4::QPUMux::r0,
                          mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r1);
    createSplat32LDI(builder, loc, static_cast<int32_t>(0xc0000000u), 32);
    createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                      mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                      mlir::vc4::AddOpcode::bit_or,
                      mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                      /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                      mlir::vc4::QPUMux::r0, mlir::vc4::QPUMux::r0,
                      mlir::vc4::QPUMux::r1);
  } else {
    createSplat32LDI(builder, loc,
                     static_cast<int32_t>(0xc0000000u |
                                          (static_cast<uint32_t>(strideBytes) &
                                           0xffffu)),
                     35);
    createVPMVCDSetup(builder, loc, mlir::vc4::VPMVCDSide::write,
                      mlir::vc4::Cond::always, mlir::vc4::Cond::never,
                      mlir::vc4::AddOpcode::bit_or,
                      mlir::vc4::MulOpcode::nop, /*raddrA=*/0,
                      /*raddrB=*/0, mlir::vc4::QPUMux::r3,
                      mlir::vc4::QPUMux::r3, mlir::vc4::QPUMux::r0,
                      mlir::vc4::QPUMux::r1);
  }
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

static LogicalResult emitVDWStoreRectDynamic(
    OpBuilder &builder, const InstructionTemplate &templ,
    const SpillAwareAllocator &allocator) {
  Operation *source = templ.source;
  if (templ.operands.size() != 5)
    return source->emitError(
        "internal lowering error: dynamic VDW rect has wrong operand count");
  std::optional<int64_t> addressReg = allocator.lookup(templ, templ.operands[0]);
  std::optional<int64_t> vpmSourceRowReg =
      allocator.lookup(templ, templ.operands[1]);
  if (!addressReg || !vpmSourceRowReg)
    return source->emitOpError()
           << "uses a VDW address or VPM source row that is not defined by a "
              "lowerable SSAVC4 op";

  int64_t maxRows = getI32IntegerAttrOr(source, "max_rows", -1);
  int64_t maxCols = getI32IntegerAttrOr(source, "max_cols", -1);
  int64_t srcX = getI32IntegerAttrOr(source, "src_x", -1);
  std::optional<int64_t> srcRow =
      getConstantI32FromLoadImm(templ.operands[1]);
  std::optional<int64_t> activeRows =
      getConstantI32FromLoadImm(templ.operands[2]);
  std::optional<int64_t> activeCols =
      getConstantI32FromLoadImm(templ.operands[3]);
  std::optional<int64_t> strideBytes =
      getConstantI32FromLoadImm(templ.operands[4]);
  std::optional<int64_t> strideReg =
      strideBytes ? std::nullopt : allocator.lookup(templ, templ.operands[4]);
  if (!srcRow)
    return source->emitOpError()
           << "dynamic rectangular VDW lowering currently requires constant "
              "source row";
  if (!strideBytes && !strideReg)
    return source->emitOpError()
           << "uses a dynamic stride value that is not defined by a lowerable "
              "SSAVC4 op";
  if (*srcRow != 0 || srcX != 0)
    return source->emitOpError()
           << "dynamic rectangular VDW lowering currently supports only "
              "source row 0 and src_x = 0";
  if (!activeRows) {
    if (maxRows < 1 || maxRows > 2)
      return source->emitOpError()
             << "dynamic rectangular VDW runtime active_rows currently "
                "supports only up to two-row rectangles";
    if (!activeCols || std::clamp(*activeCols, int64_t(0), maxCols) != maxCols)
      return source->emitOpError()
             << "dynamic rectangular VDW runtime active_rows currently "
                "requires full static active_cols";
    std::optional<int64_t> activeRowsReg =
        allocator.lookup(templ, templ.operands[2]);
    if (!activeRowsReg)
      return source->emitOpError()
             << "uses a dynamic active row value that is not defined by a "
                "lowerable SSAVC4 op";
    auto orientation = llvm::cast<mlir::ssavc4::VPMOrientationAttr>(
        source->getAttr("orientation"));
    bool vertical =
        orientation.getValue() == mlir::ssavc4::VPMOrientation::vertical;
    bool useMutex = hasStringAttr(source, "serialize", "mutex");
    unsigned bodySlots =
        getDynamicVDWStoreRowsSlotCount(useMutex, !strideBytes);
    emitRuntimeActiveRowsGuard(builder, source->getLoc(), *activeRowsReg,
                               maxRows, /*row=*/0, bodySlots);
    emitDynamicVDWStoreRowsFromVPM(
        builder, source->getLoc(), *addressReg, *vpmSourceRowReg,
        *vpmSourceRowReg,
        /*rowLen=*/maxCols, /*memoryPitchBytes=*/strideBytes,
        /*memoryPitchReg=*/strideReg, vertical, useMutex);
    return success();
  }

  int64_t clampedRows = std::clamp(*activeRows, int64_t(0), maxRows);
  if (clampedRows == 0)
    return success();
  int64_t staticCols = activeCols ? std::clamp(*activeCols, int64_t(0), maxCols)
                                  : maxCols;
  if (staticCols == 0)
    return success();
  std::optional<int64_t> dynamicActiveColsReg;
  if (!activeCols) {
    if (clampedRows != 1)
      return source->emitOpError()
             << "dynamic rectangular VDW runtime active_cols currently "
                "supports only one active row";
    dynamicActiveColsReg = allocator.lookup(templ, templ.operands[3]);
    if (!dynamicActiveColsReg)
      return source->emitOpError()
             << "uses a dynamic active column value that is not defined by a "
                "lowerable SSAVC4 op";
  }

  if (!strideBytes && !dynamicActiveColsReg)
    return source->emitOpError()
           << "dynamic rectangular VDW runtime stride currently requires "
              "runtime active_rows with full static active_cols";

  auto orientation =
      llvm::cast<mlir::ssavc4::VPMOrientationAttr>(
          source->getAttr("orientation"));
  bool vertical =
      orientation.getValue() == mlir::ssavc4::VPMOrientation::vertical;
  emitRawVDWStoreFromVPM(
      builder, source->getLoc(), *addressReg, *vpmSourceRowReg,
      *vpmSourceRowReg, dynamicActiveColsReg,
      /*activeLanes=*/staticCols, /*rowLen=*/staticCols,
      /*nrows=*/clampedRows,
      /*memoryPitchBytes=*/strideBytes ? *strideBytes : maxCols * 4, vertical,
      hasStringAttr(source, "serialize", "mutex"));
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
    case InstructionTemplate::Kind::CondSelect:
      if (failed(emitCondSelect(builder, templ, allocator)))
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
    case InstructionTemplate::Kind::EdgeCopy:
      if (failed(emitEdgeCopy(builder, templ, allocator)))
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
      if (templ.kind == InstructionTemplate::Kind::CondBranch &&
          failed(emitMakeFlags(builder, templ, allocator)))
        return failure();
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
    case InstructionTemplate::Kind::VDRLoad:
      if (failed(emitVDRLoad(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::VDRLoadRectDynamic:
      if (failed(emitVDRLoadRectDynamic(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::VDWStore:
      if (failed(emitVDWStore(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::VDWStoreVPM:
      if (failed(emitVDWStoreVPM(builder, templ, allocator)))
        return failure();
      break;
    case InstructionTemplate::Kind::VDWStoreRectDynamic:
      if (failed(emitVDWStoreRectDynamic(builder, templ, allocator)))
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
  MLIRContext *ctx = builder.getContext();
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
      builder.getNamedAttr("name", builder.getStringAttr("spill_frame_base")),
      builder.getNamedAttr("kind",
                           mlir::vc4::BuiltinKindAttr::get(
                               ctx, mlir::vc4::BuiltinKind::spill_frame_base)),
      builder.getNamedAttr("materialization",
                           builder.getStringAttr("uniform_suffix")),
      builder.getNamedAttr("uniform_index",
                           builder.getI32IntegerAttr(oldUniformWords)),
  }));
  builtins.push_back(builder.getDictionaryAttr({
      builder.getNamedAttr("name", builder.getStringAttr("spill_vpm_row")),
      builder.getNamedAttr("kind",
                           mlir::vc4::BuiltinKindAttr::get(
                               ctx, mlir::vc4::BuiltinKind::spill_vpm_row)),
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

static DictionaryAttr attachSpillVPMRows(Operation *func, OpBuilder &builder,
                                         DictionaryAttr resource) {
  if (!resource)
    return resource;

  mlir::vc4::SemanticResourceInfo info;
  if (failed(mlir::vc4::parseSemanticResourceMetadata(
          func, resource, info, /*allowAbsent=*/false)))
    return resource;
  if (info.scheduleMode != "cooperative_block")
    return resource;

  int64_t spillRows = info.warpsPerBlock;
  int64_t totalRows = info.userVPMRowsPerBlock +
                      info.compilerVPMStagingRowsPerBlock +
                      info.warpsPerBlock * info.compilerVPMStagingRowsPerWarp +
                      spillRows;

  SmallVector<NamedAttribute, 12> attrs;
  for (NamedAttribute attr : resource)
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

  replaceAttr("spill_vpm_rows_per_block",
              builder.getI32IntegerAttr(spillRows));
  replaceAttr("total_vpm_rows_per_block", builder.getI32IntegerAttr(totalRows));
  replaceAttr("uses_vpm", builder.getBoolAttr(totalRows > 0 || info.usesVPM));
  replaceAttr("requires_vpm_base_row_builtin",
              builder.getBoolAttr(totalRows > 0));
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

  auto resource =
      llvm::dyn_cast_or_null<DictionaryAttr>(vc4Func->getAttr("vc4.resource"));
  if (resource)
    vc4Func->setAttr("vc4.resource",
                     attachSpillVPMRows(vc4Func, builder, resource));
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
