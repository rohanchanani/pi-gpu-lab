//===- VC4KernelToSSAVC4.cpp - VC4Kernel to SSAVC4 lowering ---------------===//

#include "vc4/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Ops.h"
#include "vc4/Dialect/SSAVC4/IR/SSAVC4Types.h"
#include "vc4/Dialect/VC4/IR/VC4Attrs.h"
#include "vc4/Dialect/VC4/IR/VC4Ops.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelOps.h"
#include "vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <memory>
#include <optional>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kKernelOpName("vc4kernel.kernel");
constexpr llvm::StringLiteral kReturnOpName("vc4kernel.return");
constexpr llvm::StringLiteral kProgramIdOpName("vc4kernel.program_id");
constexpr llvm::StringLiteral kNumProgramsOpName("vc4kernel.num_programs");
constexpr llvm::StringLiteral kWarpIdOpName("vc4kernel.warp_id");
constexpr llvm::StringLiteral kLaneRangeOpName("vc4kernel.lane_range");
constexpr llvm::StringLiteral kPredFullOpName("vc4kernel.pred.full");
constexpr llvm::StringLiteral kPredEmptyOpName("vc4kernel.pred.empty");
constexpr llvm::StringLiteral kPredTailOpName("vc4kernel.pred.tail");
constexpr llvm::StringLiteral kPredRectOpName("vc4kernel.pred.rect");
constexpr llvm::StringLiteral kPredAndOpName("vc4kernel.pred.and");
constexpr llvm::StringLiteral kPredOrOpName("vc4kernel.pred.or");
constexpr llvm::StringLiteral kPredNotOpName("vc4kernel.pred.not");
constexpr llvm::StringLiteral kPredAnyOpName("vc4kernel.pred.any");
constexpr llvm::StringLiteral kPredAllOpName("vc4kernel.pred.all");
constexpr llvm::StringLiteral kSplatOpName("vc4kernel.splat");
constexpr llvm::StringLiteral kFragmentBitcastOpName(
    "vc4kernel.fragment_bitcast");
constexpr llvm::StringLiteral kFragmentConstOpName("vc4kernel.fragment_const");
constexpr llvm::StringLiteral kFragmentALUAddOpName(
    "vc4kernel.fragment_alu.add");
constexpr llvm::StringLiteral kFragmentALUMulOpName(
    "vc4kernel.fragment_alu.mul");
constexpr llvm::StringLiteral kFragmentCmpOpName("vc4kernel.fragment_cmp");
constexpr llvm::StringLiteral kFragmentSelectOpName(
    "vc4kernel.fragment_select");
constexpr llvm::StringLiteral kFragmentRotateOpName("vc4kernel.fragment_rotate");
constexpr llvm::StringLiteral kFragmentReduceOpName("vc4kernel.fragment_reduce");
constexpr llvm::StringLiteral kTMULoadOpName("vc4kernel.tmu_load_fragment");
constexpr llvm::StringLiteral kVDWStoreOpName("vc4kernel.vdw_store_fragment");
constexpr llvm::StringLiteral kVPMAllocOpName("vc4kernel.vpm_alloc");
constexpr llvm::StringLiteral kVPMWriteOpName("vc4kernel.vpm_write_fragment");
constexpr llvm::StringLiteral kVPMReadOpName("vc4kernel.vpm_read_fragment");
constexpr llvm::StringLiteral kVDRLoadOpName("vc4kernel.vdr_load_to_vpm");
constexpr llvm::StringLiteral kVDRLoadRectOpName(
    "vc4kernel.vdr_load_rect_to_vpm");
constexpr llvm::StringLiteral kVDWStoreVPMOpName(
    "vc4kernel.vdw_store_vpm_fragment");
constexpr llvm::StringLiteral kVDWStoreRectOpName(
    "vc4kernel.vdw_store_rect_from_vpm");
constexpr llvm::StringLiteral kBarrierOpName("vc4kernel.barrier");

constexpr llvm::StringLiteral kSSAVC4ModuleOpName("ssavc4.module");
constexpr llvm::StringLiteral kSSAVC4FuncOpName("ssavc4.func");
constexpr llvm::StringLiteral kSSAVC4ThreadEndOpName("ssavc4.thread_end");
constexpr llvm::StringLiteral kSSAVC4LoadImmOpName("ssavc4.load_imm");
constexpr llvm::StringLiteral kSSAVC4ElementNumberOpName(
    "ssavc4.element_number");
constexpr llvm::StringLiteral kSSAVC4UniformReadOpName("ssavc4.uniform.read");
constexpr llvm::StringLiteral kSSAVC4SplatOpName("ssavc4.splat");
constexpr llvm::StringLiteral kSSAVC4MovOpName("ssavc4.mov");
constexpr llvm::StringLiteral kSSAVC4ALUAddOpName("ssavc4.alu.add");
constexpr llvm::StringLiteral kSSAVC4ALUMulOpName("ssavc4.alu.mul");
constexpr llvm::StringLiteral kSSAVC4RotateOpName("ssavc4.rotate");
constexpr llvm::StringLiteral kSSAVC4TMURequestOpName("ssavc4.tmu.request");
constexpr llvm::StringLiteral kSSAVC4TMUReadOpName("ssavc4.tmu.read");
constexpr llvm::StringLiteral kSSAVC4VDWStoreOpName("ssavc4.vdw.store");
constexpr llvm::StringLiteral kSSAVC4VPMWriteOpName("ssavc4.vpm.write");
constexpr llvm::StringLiteral kSSAVC4VPMReadOpName("ssavc4.vpm.read");
constexpr llvm::StringLiteral kSSAVC4VDRLoadOpName("ssavc4.vdr.load");
constexpr llvm::StringLiteral kSSAVC4VDRLoadRectDynamicOpName(
    "ssavc4.vdr.load_rect.dynamic");
constexpr llvm::StringLiteral kSSAVC4VDWStoreVPMOpName("ssavc4.vdw.store_vpm");
constexpr llvm::StringLiteral kSSAVC4VDWStoreRectDynamicOpName(
    "ssavc4.vdw.store_rect.dynamic");
constexpr llvm::StringLiteral kSSAVC4BarrierOpName("ssavc4.barrier");
constexpr llvm::StringLiteral kSSAVC4BranchOpName("ssavc4.br");
constexpr llvm::StringLiteral kSSAVC4CondBranchOpName("ssavc4.cond_br");
constexpr llvm::StringLiteral kSSAVC4CondSelectOpName("ssavc4.cond_select");
constexpr llvm::StringLiteral kSSAVC4MakeFlagsOpName("ssavc4.make_flags");

static bool hasName(Operation *op, StringRef name) {
  return op && op->getName().getStringRef() == name;
}

static StringAttr asStringAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<StringAttr>(attr);
}

static ArrayAttr asArrayAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<ArrayAttr>(attr);
}

static DictionaryAttr asDictionaryAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<DictionaryAttr>(attr);
}

static IntegerAttr asIntegerAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<IntegerAttr>(attr);
}

static NamedAttribute getSSAVC4VPMOrientation(OpBuilder &builder,
                                              Operation *op) {
  auto attr =
      llvm::dyn_cast_or_null<mlir::vc4kernel::VPMOrientationAttr>(
          op->getAttr("orientation"));
  mlir::ssavc4::VPMOrientation value =
      attr && attr.getValue() == mlir::vc4kernel::VPMOrientation::vertical
          ? mlir::ssavc4::VPMOrientation::vertical
          : mlir::ssavc4::VPMOrientation::horizontal;
  return builder.getNamedAttr(
      "orientation",
      mlir::ssavc4::VPMOrientationAttr::get(builder.getContext(), value));
}

static NamedAttribute getSSAVC4VPMWidth(OpBuilder &builder, Operation *op) {
  auto attr =
      llvm::dyn_cast_or_null<mlir::vc4kernel::VPMWidthAttr>(
          op->getAttr("width"));
  mlir::ssavc4::VPMElemWidth value = mlir::ssavc4::VPMElemWidth::w32;
  if (attr && attr.getValue() == mlir::vc4kernel::VPMWidth::w16)
    value = mlir::ssavc4::VPMElemWidth::w16;
  if (attr && attr.getValue() == mlir::vc4kernel::VPMWidth::w8)
    value = mlir::ssavc4::VPMElemWidth::w8;
  return builder.getNamedAttr(
      "width",
      mlir::ssavc4::VPMElemWidthAttr::get(builder.getContext(), value));
}

static NamedAttribute getSSAVC4VPMSubword(OpBuilder &builder, Operation *op) {
  auto attr =
      llvm::dyn_cast_or_null<mlir::vc4kernel::VPMSubwordAttr>(
          op->getAttr("subword"));
  mlir::ssavc4::VPMSubword value = mlir::ssavc4::VPMSubword::none;
  if (attr && attr.getValue() == mlir::vc4kernel::VPMSubword::packed)
    value = mlir::ssavc4::VPMSubword::packed;
  if (attr && attr.getValue() == mlir::vc4kernel::VPMSubword::laned)
    value = mlir::ssavc4::VPMSubword::laned;
  return builder.getNamedAttr(
      "subword",
      mlir::ssavc4::VPMSubwordAttr::get(builder.getContext(), value));
}

static StringAttr getSymbolNameAttr(Operation *op) {
  return asStringAttr(op->getAttr(SymbolTable::getSymbolAttrName()));
}

static bool isAllowedVC4KernelOp(Operation *op) {
  static constexpr llvm::StringLiteral names[] = {
      "vc4kernel.kernel",
      "vc4kernel.return",
      "vc4kernel.program_id",
      "vc4kernel.num_programs",
      "vc4kernel.warp_id",
      "vc4kernel.lane_range",
      "vc4kernel.pred.full",
      "vc4kernel.pred.empty",
      "vc4kernel.pred.tail",
      "vc4kernel.pred.rect",
      "vc4kernel.pred.and",
      "vc4kernel.pred.or",
      "vc4kernel.pred.not",
      "vc4kernel.pred.any",
      "vc4kernel.pred.all",
      "vc4kernel.splat",
      "vc4kernel.fragment_bitcast",
      "vc4kernel.fragment_const",
      "vc4kernel.fragment_alu.add",
      "vc4kernel.fragment_alu.mul",
      "vc4kernel.fragment_cmp",
      "vc4kernel.fragment_select",
      "vc4kernel.fragment_rotate",
      "vc4kernel.fragment_reduce",
      "vc4kernel.tmu_load_fragment",
      "vc4kernel.vdw_store_fragment",
      "vc4kernel.vpm_alloc",
      "vc4kernel.vpm_write_fragment",
      "vc4kernel.vpm_read_fragment",
      "vc4kernel.vdr_load_to_vpm",
      "vc4kernel.vdr_load_rect_to_vpm",
      "vc4kernel.vdw_store_vpm_fragment",
      "vc4kernel.vdw_store_rect_from_vpm",
      "vc4kernel.barrier"};
  return llvm::is_contained(names, op->getName().getStringRef());
}

static bool containsIllegalType(Type type) {
  if (!type)
    return false;
  if (mlir::vc4kernel::isLegalVC4KernelType(type))
    return false;
  return true;
}

static bool isScalarI1(Type type) { return type && type.isInteger(1); }

static bool isScalarI32(Type type) {
  return type && type.isSignlessInteger(32);
}

static bool isScalarF32(Type type) { return type && type.isF32(); }

static bool isAllowedScalarArithType(Type type) {
  return isScalarI1(type) || isScalarI32(type) || isScalarF32(type);
}

struct LaneAffineConstPlan {
  int64_t base = 0;
  int64_t shift = 0;
  bool subtract = false;
};

static SmallVector<int64_t, 16> getDenseI32Values(DenseElementsAttr attr) {
  SmallVector<int64_t, 16> values;
  values.reserve(attr.getNumElements());
  for (APInt value : attr.getValues<APInt>())
    values.push_back(value.getSExtValue());
  return values;
}

static bool isDenseI32Splat(DenseElementsAttr attr) {
  return attr && attr.isSplat() &&
         attr.getElementType().isSignlessInteger(32);
}

static bool isDenseF32Splat(DenseElementsAttr attr) {
  return attr && attr.isSplat() && attr.getElementType().isF32();
}

static bool isI32LaneRange(ArrayRef<int64_t> values) {
  if (values.size() != 16)
    return false;
  for (int64_t lane = 0; lane < 16; ++lane)
    if (values[lane] != lane)
      return false;
  return true;
}

static std::optional<LaneAffineConstPlan>
getI32LaneAffinePlan(ArrayRef<int64_t> values) {
  if (values.size() != 16)
    return std::nullopt;
  int64_t base = values.front();
  for (int64_t shift = 0; shift <= 4; ++shift) {
    bool add = true;
    bool sub = true;
    for (int64_t lane = 0; lane < 16; ++lane) {
      int64_t delta = lane << shift;
      add &= values[lane] == base + delta;
      sub &= values[lane] == base - delta;
    }
    if (add)
      return LaneAffineConstPlan{base, shift, false};
    if (sub)
      return LaneAffineConstPlan{base, shift, true};
  }
  return std::nullopt;
}

static bool isPerLaneU2(ArrayRef<int64_t> values) {
  return values.size() == 16 &&
         llvm::all_of(values, [](int64_t value) {
           return value >= 0 && value <= 3;
         });
}

static bool isPerLaneS2(ArrayRef<int64_t> values) {
  return values.size() == 16 &&
         llvm::all_of(values, [](int64_t value) {
           return value >= -2 && value <= 1;
         });
}

static LogicalResult verifyNoArithVectors(Operation *op) {
  for (Value operand : op->getOperands())
    if (llvm::isa<VectorType>(operand.getType()))
      return op->emitOpError(
          "arith operations may not operate on or produce vectors");
  for (Type type : op->getResultTypes())
    if (llvm::isa<VectorType>(type))
      return op->emitOpError(
          "arith operations may not produce vectors; arith operations may not "
          "operate on or produce vectors");
  return success();
}

static LogicalResult verifyIntegerArithI32(Operation *op) {
  for (Value operand : op->getOperands())
    if (!isScalarI32(operand.getType()))
      return op->emitOpError(
          "integer arith operations in vc4kernel require scalar i32 operands and results");
  for (Type type : op->getResultTypes())
    if (!isScalarI32(type))
      return op->emitOpError(
          "integer arith operations in vc4kernel require scalar i32 operands and results");
  return success();
}

static LogicalResult verifyNoVPMSuccessorOperands(Operation *op,
                                                  ValueRange operands) {
  for (Value operand : operands)
    if (mlir::vc4kernel::isVC4KernelVPMTileType(operand.getType()))
      return op->emitOpError(
          "cf successor operands may not carry !vc4kernel.vpm_tile in Stage 1");
  return success();
}

static LogicalResult verifyArithBoundary(Operation *op) {
  if (failed(verifyNoArithVectors(op)))
    return failure();

  StringRef name = op->getName().getStringRef();
  if (name == "arith.constant") {
    if (op->getNumResults() == 1 &&
        isAllowedScalarArithType(op->getResult(0).getType()))
      return success();
    return op->emitOpError(
        "arith.constant in vc4kernel requires one scalar i1/i32/f32 result");
  }
  if (name == "arith.addi" || name == "arith.subi" ||
      name == "arith.muli" || name == "arith.shli")
    return verifyIntegerArithI32(op);
  if (name == "arith.cmpi") {
    if (op->getNumOperands() == 2 && op->getNumResults() == 1 &&
        isScalarI32(op->getOperand(0).getType()) &&
        isScalarI32(op->getOperand(1).getType()) &&
        isScalarI1(op->getResult(0).getType()))
      return success();
    return op->emitOpError(
        "arith.cmpi in vc4kernel requires scalar i32 operands and scalar i1 result");
  }
  if (name == "arith.select") {
    if (op->getNumOperands() == 3 && op->getNumResults() == 1 &&
        isScalarI1(op->getOperand(0).getType())) {
      Type valueType = op->getOperand(1).getType();
      if (isAllowedScalarArithType(valueType) &&
          op->getOperand(2).getType() == valueType &&
          op->getResult(0).getType() == valueType)
        return success();
    }
    return op->emitOpError(
        "arith.select in vc4kernel requires scalar i1 condition and matching scalar i1/i32/f32 values");
  }
  return op->emitOpError("arith operation is not allowed in vc4kernel");
}

static LogicalResult verifyOperationBoundary(Operation *op) {
  StringRef dialect = op->getName().getDialectNamespace();
  if (dialect == "builtin")
    return op->emitOpError(
        "builtin dialect operations are forbidden inside vc4kernel.kernel");
  if (dialect == "vc4kernel") {
    if (!isAllowedVC4KernelOp(op))
      return op->emitOpError("unknown or forbidden vc4kernel operation");
    return success();
  }
  if (dialect == "arith")
    return verifyArithBoundary(op);
  if (dialect == "cf") {
    if (!hasName(op, "cf.br") && !hasName(op, "cf.cond_br"))
      return op->emitOpError("only cf.br and cf.cond_br are allowed");
    if (auto br = dyn_cast<cf::BranchOp>(op))
      return verifyNoVPMSuccessorOperands(op, br.getDestOperands());
    if (auto cond = dyn_cast<cf::CondBranchOp>(op)) {
      if (!cond.getCondition().getType().isInteger(1))
        return op->emitOpError("cf.cond_br condition must be scalar i1");
      if (failed(verifyNoVPMSuccessorOperands(op, cond.getTrueDestOperands())) ||
          failed(verifyNoVPMSuccessorOperands(op, cond.getFalseDestOperands())))
        return failure();
    }
    return success();
  }
  if (dialect == "vector")
    return op->emitOpError("vector dialect operations are forbidden");
  if (dialect == "memref" || dialect == "tensor")
    return op->emitOpError("memref/tensor operations are forbidden");
  if (dialect == "scf")
    return op->emitOpError("raw scf operations are forbidden");
  if (dialect == "ssavc4" || dialect == "vc4")
    return op->emitOpError("lower-half dialect operations are forbidden");
  return op->emitOpError()
         << "dialect '" << dialect << "' is forbidden inside vc4kernel";
}

static LogicalResult verifyVC4Kernel(ModuleOp module) {
  for (Operation &top : module.getBody()->getOperations()) {
    if (!hasName(&top, kKernelOpName))
      return top.emitOpError(
          "only vc4kernel.kernel operations may appear at module top level");
    WalkResult walk = top.walk([&](Operation *op) {
      if (op != &top && hasName(op, kKernelOpName)) {
        op->emitOpError(
            "vc4kernel.kernel operations may appear only at module top level");
        return WalkResult::interrupt();
      }
      if (op != &top && failed(verifyOperationBoundary(op)))
        return WalkResult::interrupt();
      for (Value operand : op->getOperands())
        if (containsIllegalType(operand.getType())) {
          op->emitOpError("uses illegal vc4kernel type ") << operand.getType();
          return WalkResult::interrupt();
        }
      for (Type type : op->getResultTypes())
        if (containsIllegalType(type)) {
          op->emitOpError("produces illegal vc4kernel type ") << type;
          return WalkResult::interrupt();
        }
      for (Region &region : op->getRegions())
        for (Block &block : region) {
          bool isKernelEntryBlock = op == &top && &region == &top.getRegion(0) &&
                                    &block == &region.front();
          for (BlockArgument arg : block.getArguments()) {
            if (!isKernelEntryBlock &&
                mlir::vc4kernel::isVC4KernelVPMTileType(arg.getType())) {
              op->emitOpError(
                  "cf block arguments may not carry !vc4kernel.vpm_tile in Stage 1");
              return WalkResult::interrupt();
            }
            if (containsIllegalType(arg.getType())) {
              op->emitOpError("region block argument has illegal type ")
                  << arg.getType();
              return WalkResult::interrupt();
            }
          }
        }
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return failure();
  }
  return success();
}

static std::string makeCIdentifier(StringRef value) {
  std::string result;
  for (char c : value) {
    unsigned char uc = static_cast<unsigned char>(c);
    result.push_back(std::isalnum(uc) || c == '_' ? c : '_');
  }
  if (result.empty() || !(std::isalpha(static_cast<unsigned char>(result[0])) ||
                          result[0] == '_'))
    result.insert(result.begin(), '_');
  return result;
}

static StringRef getPublicName(Operation *kernel) {
  if (auto attr = asStringAttr(kernel->getAttr("public_name")))
    return attr.getValue();
  if (StringAttr sym = getSymbolNameAttr(kernel))
    return sym.getValue();
  return "vc4kernel_kernel";
}

static bool kernelContains(Operation *kernel, StringRef opName) {
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

static Attribute getBuiltinKindAttr(OpBuilder &builder, StringRef name) {
  MLIRContext *ctx = builder.getContext();
  if (name == "logical_request")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::logical_request);
  if (name == "total_requests")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::total_requests);
  if (name == "logical_warp_id")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::logical_warp_id);
  if (name == "warps_per_block")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::warps_per_block);
  if (name == "vpm_base_row")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::vpm_base_row);
  if (name == "semaphore_base")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::semaphore_base);
  if (name == "program_id_x")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::program_id_x);
  if (name == "program_id_y")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::program_id_y);
  if (name == "program_id_z")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::program_id_z);
  if (name == "num_programs_x")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::num_programs_x);
  if (name == "num_programs_y")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::num_programs_y);
  if (name == "num_programs_z")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::num_programs_z);
  return {};
}

static int64_t getAxisAttrValue(Operation *op) {
  if (auto axis = op->getAttrOfType<IntegerAttr>("axis"))
    return axis.getInt();
  return 0;
}

static StringRef getAxisSuffix(int64_t axis) {
  switch (axis) {
  case 0:
    return "x";
  case 1:
    return "y";
  case 2:
    return "z";
  default:
    return "x";
  }
}

static std::string getProgramIdBuiltinName(int64_t axis) {
  return (Twine("program_id_") + getAxisSuffix(axis)).str();
}

static std::string getNumProgramsBuiltinName(int64_t axis) {
  return (Twine("num_programs_") + getAxisSuffix(axis)).str();
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

static DictionaryAttr buildArgABIEntry(OpBuilder &builder, DictionaryAttr source,
                                       BlockArgument arg, unsigned index,
                                       int64_t uniformIndex) {
  SmallVector<NamedAttribute, 8> attrs;
  if (source) {
    for (NamedAttribute named : source)
      if (named.getName().getValue() != "uniform_index")
        attrs.push_back(named);
  }
  if (!source || !source.get("name"))
    attrs.push_back(builder.getNamedAttr(
        "name", builder.getStringAttr((Twine("arg") + Twine(index)).str())));
  if (!source || !source.get("kind"))
    attrs.push_back(builder.getNamedAttr("kind", builder.getStringAttr("scalar")));
  if (!source || !source.get("direction"))
    attrs.push_back(builder.getNamedAttr("direction",
                                         builder.getStringAttr("by_value")));
  if (!source || (!source.get("type") && !source.get("elem_type"))) {
    attrs.push_back(builder.getNamedAttr(
        "type", builder.getStringAttr(arg.getType().isF32() ? "f32" : "u32")));
  }
  attrs.push_back(builder.getNamedAttr("uniform_index",
                                       builder.getI32IntegerAttr(uniformIndex)));
  return builder.getDictionaryAttr(attrs);
}

struct VC4KernelResourceSummary {
  StringRef schedule_mode;
  int64_t warps_per_block = 1;
  bool uses_tmu = false;
  bool uses_vpm = false;
  bool uses_vpm_qpu_read = false;
  bool uses_vpm_qpu_write = false;
  bool uses_vdr = false;
  bool uses_vdw = false;
  bool uses_barrier = false;
  int64_t user_vpm_rows_per_block = 0;
  int64_t compiler_vpm_staging_rows_per_warp = 0;
  int64_t compiler_vpm_staging_rows_per_block = 0;
  int64_t total_vpm_rows_per_block = 0;
  int64_t total_vpm_bytes_per_block = 0;
  int64_t semaphore_count_per_block = 0;
  bool requires_vpm_base_row_builtin = false;
  bool requires_semaphore_base_builtin = false;
};

static void appendRequiredBuiltins(Operation *kernel,
                                   const VC4KernelResourceSummary &summary,
                                   llvm::function_ref<void(StringRef)> append) {
  SmallVector<std::string, 8> identityBuiltins;
  auto appendIdentityBuiltin = [&](std::string name) {
    if (!llvm::is_contained(identityBuiltins, name))
      identityBuiltins.push_back(std::move(name));
  };
  bool hasLegacyAxis0ProgramId = false;
  bool needs3DGridIdentity = false;
  kernel->walk([&](Operation *op) {
    if (hasName(op, kProgramIdOpName)) {
      int64_t axis = getAxisAttrValue(op);
      if (axis == 0)
        hasLegacyAxis0ProgramId = true;
      else
        needs3DGridIdentity = true;
    } else if (hasName(op, kNumProgramsOpName)) {
      needs3DGridIdentity = true;
    }
  });
  if (needs3DGridIdentity) {
    kernel->walk([&](Operation *op) {
      if (hasName(op, kProgramIdOpName))
        appendIdentityBuiltin(getProgramIdBuiltinName(getAxisAttrValue(op)));
      else if (hasName(op, kNumProgramsOpName))
        appendIdentityBuiltin(getNumProgramsBuiltinName(getAxisAttrValue(op)));
    });
    for (const std::string &name : identityBuiltins)
      append(name);
  } else if (hasLegacyAxis0ProgramId) {
    append("logical_request");
    append("total_requests");
  }
  bool needsLogicalWarp =
      kernelContains(kernel, kWarpIdOpName) || summary.uses_barrier ||
      (summary.schedule_mode == "cooperative_block" &&
       summary.compiler_vpm_staging_rows_per_warp > 0);
  if (needsLogicalWarp)
    append("logical_warp_id");
  if (summary.uses_barrier)
    append("warps_per_block");
  if (summary.requires_vpm_base_row_builtin)
    append("vpm_base_row");
  if (summary.requires_semaphore_base_builtin)
    append("semaphore_base");
}

static DictionaryAttr buildLaunchABI(Operation *kernel, OpBuilder &builder,
                                     const VC4KernelResourceSummary &summary) {
  SmallVector<Attribute, 8> args;
  int64_t nextUniform = 0;
  ArrayAttr argAttrs = asArrayAttr(kernel->getAttr("arg_attrs"));
  if (!kernel->getRegion(0).empty()) {
    for (BlockArgument arg : kernel->getRegion(0).front().getArguments()) {
      DictionaryAttr source;
      if (argAttrs && arg.getArgNumber() < argAttrs.size())
        source = asDictionaryAttr(argAttrs[arg.getArgNumber()]);
      args.push_back(buildArgABIEntry(builder, source, arg, arg.getArgNumber(),
                                      nextUniform++));
    }
  }

  SmallVector<Attribute, 4> builtins;
  auto appendBuiltin = [&](StringRef name) {
    builtins.push_back(buildBuiltinABIEntry(builder, name, nextUniform++));
  };
  appendRequiredBuiltins(kernel, summary, appendBuiltin);

  StringRef publicName = getPublicName(kernel);
  StringAttr sym = getSymbolNameAttr(kernel);
  return builder.getDictionaryAttr({
      builder.getNamedAttr("public_name", builder.getStringAttr(publicName)),
      builder.getNamedAttr("symbol_name",
                           sym ? sym : builder.getStringAttr(publicName)),
      builder.getNamedAttr(
          "code_symbol",
          builder.getStringAttr(makeCIdentifier(publicName) + "_shader")),
      builder.getNamedAttr("tail_policy",
                           builder.getStringAttr("exact_multiple")),
      builder.getNamedAttr("uniform_words_per_qpu",
                           builder.getI32IntegerAttr(nextUniform)),
      builder.getNamedAttr("args", builder.getArrayAttr(args)),
      builder.getNamedAttr("builtins", builder.getArrayAttr(builtins)),
  });
}

static VC4KernelResourceSummary
computeVC4KernelResourceSummary(Operation *kernel) {
  VC4KernelResourceSummary summary;
  summary.schedule_mode = "independent_vector";
  if (auto mode = llvm::dyn_cast_if_present<mlir::vc4kernel::ScheduleModeAttr>(
          kernel->getAttr("schedule_mode"))) {
    if (mode.getValue() == mlir::vc4kernel::ScheduleMode::cooperative_block)
      summary.schedule_mode = "cooperative_block";
  }
  if (auto warps = asIntegerAttr(kernel->getAttr("warps_per_block")))
    summary.warps_per_block = warps.getInt();

  kernel->walk([&](Operation *op) {
    if (hasName(op, kTMULoadOpName)) {
      summary.uses_tmu = true;
      return;
    }
    if (hasName(op, kVPMAllocOpName)) {
      summary.uses_vpm = true;
      if (auto rows = op->getAttrOfType<IntegerAttr>("rows"))
        summary.user_vpm_rows_per_block += rows.getInt();
      return;
    }
    if (hasName(op, kVPMReadOpName)) {
      summary.uses_vpm = true;
      summary.uses_vpm_qpu_read = true;
      return;
    }
    if (hasName(op, kVPMWriteOpName)) {
      summary.uses_vpm = true;
      summary.uses_vpm_qpu_write = true;
      return;
    }
    if (hasName(op, kVDRLoadOpName)) {
      summary.uses_vdr = true;
      summary.uses_vpm = true;
      return;
    }
    if (hasName(op, kVDRLoadRectOpName)) {
      summary.uses_vdr = true;
      summary.uses_vpm = true;
      return;
    }
    if (hasName(op, kVDWStoreVPMOpName)) {
      summary.uses_vdw = true;
      summary.uses_vpm = true;
      if (op->getNumOperands() > 4) {
        Operation *predDef = op->getOperand(4).getDefiningOp();
        if (!predDef || hasName(predDef, kPredTailOpName))
          summary.compiler_vpm_staging_rows_per_warp =
              std::max<int64_t>(summary.compiler_vpm_staging_rows_per_warp, 1);
      }
      return;
    }
    if (hasName(op, kVDWStoreRectOpName)) {
      summary.uses_vdw = true;
      summary.uses_vpm = true;
      return;
    }
    if (hasName(op, kVDWStoreOpName)) {
      summary.uses_vdw = true;
      summary.uses_vpm = true;
      summary.compiler_vpm_staging_rows_per_warp =
          std::max<int64_t>(summary.compiler_vpm_staging_rows_per_warp, 1);
      return;
    }
    if (hasName(op, kBarrierOpName)) {
      summary.uses_barrier = true;
      summary.semaphore_count_per_block = 4;
      summary.requires_semaphore_base_builtin = true;
      return;
    }
  });

  summary.total_vpm_rows_per_block =
      summary.user_vpm_rows_per_block +
      summary.compiler_vpm_staging_rows_per_block +
      summary.warps_per_block * summary.compiler_vpm_staging_rows_per_warp;
  summary.total_vpm_bytes_per_block = summary.total_vpm_rows_per_block * 16 * 4;
  summary.requires_vpm_base_row_builtin =
      summary.total_vpm_rows_per_block > 0;
  return summary;
}

static DictionaryAttr buildSSAVC4ResourceMetadataFromVC4KernelSummary(
    const VC4KernelResourceSummary &summary, OpBuilder &builder) {
  StringRef scheduleMode = "independent_vector";
  if (summary.schedule_mode == "cooperative_block")
    scheduleMode = "cooperative_block";
  return builder.getDictionaryAttr({
      builder.getNamedAttr("schedule_mode", builder.getStringAttr(scheduleMode)),
      builder.getNamedAttr(
          "warps_per_block",
          builder.getI32IntegerAttr(summary.warps_per_block)),
      builder.getNamedAttr(
          "user_vpm_rows_per_block",
          builder.getI32IntegerAttr(summary.user_vpm_rows_per_block)),
      builder.getNamedAttr(
          "compiler_vpm_staging_rows_per_warp",
          builder.getI32IntegerAttr(
              summary.compiler_vpm_staging_rows_per_warp)),
      builder.getNamedAttr(
          "compiler_vpm_staging_rows_per_block",
          builder.getI32IntegerAttr(
              summary.compiler_vpm_staging_rows_per_block)),
      builder.getNamedAttr(
          "total_vpm_rows_per_block",
          builder.getI32IntegerAttr(summary.total_vpm_rows_per_block)),
      builder.getNamedAttr("uses_tmu", builder.getBoolAttr(summary.uses_tmu)),
      builder.getNamedAttr("uses_vpm", builder.getBoolAttr(summary.uses_vpm)),
      builder.getNamedAttr(
          "uses_vpm_qpu_read",
          builder.getBoolAttr(summary.uses_vpm_qpu_read)),
      builder.getNamedAttr(
          "uses_vpm_qpu_write",
          builder.getBoolAttr(summary.uses_vpm_qpu_write)),
      builder.getNamedAttr("uses_vdr", builder.getBoolAttr(summary.uses_vdr)),
      builder.getNamedAttr("uses_vdw", builder.getBoolAttr(summary.uses_vdw)),
      builder.getNamedAttr(
          "uses_barrier", builder.getBoolAttr(summary.uses_barrier)),
      builder.getNamedAttr(
          "semaphore_count_per_block",
          builder.getI32IntegerAttr(summary.semaphore_count_per_block)),
      builder.getNamedAttr(
          "requires_vpm_base_row_builtin",
          builder.getBoolAttr(summary.requires_vpm_base_row_builtin)),
      builder.getNamedAttr(
          "requires_semaphore_base_builtin",
          builder.getBoolAttr(summary.requires_semaphore_base_builtin)),
  });
}

static Operation *createOp(OpBuilder &builder, Location loc, StringRef name,
                           ValueRange operands, ArrayRef<NamedAttribute> attrs,
                           TypeRange resultTypes = TypeRange{}) {
  OperationState state(loc, name);
  state.addOperands(operands);
  for (NamedAttribute attr : attrs)
    state.addAttribute(attr.getName(), attr.getValue());
  state.addTypes(resultTypes);
  return builder.create(state);
}

static Value createOpWithResult(OpBuilder &builder, Location loc, StringRef name,
                                ValueRange operands,
                                ArrayRef<NamedAttribute> attrs, Type resultType) {
  return createOp(builder, loc, name, operands, attrs, resultType)->getResult(0);
}

static Value createUniformRead(OpBuilder &builder, Location loc, Type type,
                               int64_t index) {
  return createOpWithResult(
      builder, loc, kSSAVC4UniformReadOpName, {},
      {builder.getNamedAttr("index", builder.getI32IntegerAttr(index))}, type);
}

static Value createLoadImm(OpBuilder &builder, Location loc, Type type,
                           Attribute value) {
  SmallVector<NamedAttribute, 2> attrs{
      builder.getNamedAttr("mode", mlir::vc4::LoadImmModeAttr::get(
                                       builder.getContext(),
                                       mlir::vc4::LoadImmMode::splat32))};
  if (value)
    attrs.push_back(builder.getNamedAttr("value", value));
  return createOpWithResult(builder, loc, kSSAVC4LoadImmOpName, {}, attrs, type);
}

static Value createPerLaneLoadImm(OpBuilder &builder, Location loc, Type type,
                                  mlir::vc4::LoadImmMode mode,
                                  ArrayRef<int64_t> values) {
  SmallVector<int32_t, 16> i32Values;
  i32Values.reserve(values.size());
  for (int64_t value : values)
    i32Values.push_back(static_cast<int32_t>(value));
  SmallVector<NamedAttribute, 2> attrs{
      builder.getNamedAttr("mode",
                           mlir::vc4::LoadImmModeAttr::get(builder.getContext(),
                                                           mode)),
      builder.getNamedAttr("values",
                           builder.getDenseI32ArrayAttr(i32Values))};
  return createOpWithResult(builder, loc, kSSAVC4LoadImmOpName, {}, attrs, type);
}

struct LoweredValue {
  Value value;
};

// PredicatePlan and ConditionPlan are compiler lowering plans. VC4 flags are
// mutable machine state, so later passes must emit or consume flags at the
// exact program point instead of treating them as persistent SSA values.
struct PredicatePlan {
  enum class Class { Full, Empty, TailPrefix, RectRow, GeneralMask };

  Class kind = Class::GeneralMask;
  Value base;
  Value limit;
  Value row;
  Value rows;
  Value colBase;
  Value cols;
  Value maskValue;
  Value compareLhs;
  Value compareRhs;
  mlir::vc4::Cond compareCond = mlir::vc4::Cond::zs;

  static PredicatePlan full() {
    PredicatePlan plan;
    plan.kind = Class::Full;
    return plan;
  }
  static PredicatePlan empty() {
    PredicatePlan plan;
    plan.kind = Class::Empty;
    return plan;
  }
  static PredicatePlan tailPrefix(Value base, Value limit) {
    PredicatePlan plan;
    plan.kind = Class::TailPrefix;
    plan.base = base;
    plan.limit = limit;
    return plan;
  }
  static PredicatePlan rectRow(Value row, Value rows, Value colBase,
                               Value cols) {
    PredicatePlan plan;
    plan.kind = Class::RectRow;
    plan.row = row;
    plan.rows = rows;
    plan.colBase = colBase;
    plan.cols = cols;
    return plan;
  }
  static PredicatePlan generalMask() {
    PredicatePlan plan;
    plan.kind = Class::GeneralMask;
    return plan;
  }
  static PredicatePlan generalMaskValue(Value mask) {
    PredicatePlan plan;
    plan.kind = Class::GeneralMask;
    plan.maskValue = mask;
    return plan;
  }
  static PredicatePlan generalMask(Value lhs, Value rhs,
                                   mlir::vc4::Cond cond) {
    PredicatePlan plan;
    plan.kind = Class::GeneralMask;
    plan.compareLhs = lhs;
    plan.compareRhs = rhs;
    plan.compareCond = cond;
    return plan;
  }

  bool isFull() const { return kind == Class::Full; }
};

struct ConditionPlan {
  enum class Class {
    ConstantTrue,
    ConstantFalse,
    ScalarI32Compare,
    PredicateAny,
    PredicateAll,
    FlagsValue
  };

  Class kind = Class::FlagsValue;
  Value lhs;
  Value rhs;
  Value predicateSource;
  Value flags;
  Attribute predicate;
  mlir::vc4::BranchCond branchCond = mlir::vc4::BranchCond::any_z_clear;
  mlir::vc4::Cond selectCond = mlir::vc4::Cond::zc;

  static ConditionPlan constant(bool value) {
    ConditionPlan plan;
    plan.kind = value ? Class::ConstantTrue : Class::ConstantFalse;
    return plan;
  }
  static ConditionPlan scalarI32Compare(Value lhs, Value rhs,
                                        Attribute predicate) {
    ConditionPlan plan;
    plan.kind = Class::ScalarI32Compare;
    plan.lhs = lhs;
    plan.rhs = rhs;
    plan.predicate = predicate;
    return plan;
  }
  static ConditionPlan predicateAny(Value source) {
    ConditionPlan plan;
    plan.kind = Class::PredicateAny;
    plan.predicateSource = source;
    return plan;
  }
  static ConditionPlan predicateAll(Value source) {
    ConditionPlan plan;
    plan.kind = Class::PredicateAll;
    plan.predicateSource = source;
    return plan;
  }
  static ConditionPlan flagsValue(Value flags, mlir::vc4::BranchCond branchCond,
                                  mlir::vc4::Cond selectCond) {
    ConditionPlan plan;
    plan.kind = Class::FlagsValue;
    plan.flags = flags;
    plan.branchCond = branchCond;
    plan.selectCond = selectCond;
    return plan;
  }
};

struct VPMAllocationPlan {
  int64_t baseRowOffset = 0;
  int64_t rows = 0;
  int64_t elemBytes = 4;
};

struct ResourcePlan {
  VC4KernelResourceSummary summary;
};

struct LaunchABIPlan {
  llvm::StringMap<Value> builtinValues;

  void bindBuiltin(StringRef name, Value value) { builtinValues[name] = value; }
  Value lookupBuiltin(StringRef name) const {
    auto it = builtinValues.find(name);
    return it == builtinValues.end() ? Value() : it->second;
  }
};

struct LoweringState {
  explicit LoweringState(ResourcePlan resourcePlan)
      : resourcePlan(resourcePlan) {}

  llvm::DenseMap<Value, LoweredValue> values;
  llvm::DenseMap<Value, PredicatePlan> predicates;
  llvm::DenseMap<Value, ConditionPlan> conditions;
  llvm::DenseMap<Value, VPMAllocationPlan> vpmAllocations;
  llvm::DenseMap<Block *, Block *> blockMap;
  ResourcePlan resourcePlan;
  LaunchABIPlan launchABI;
  int64_t nextVPMRowOffset = 0;
};

static Value mapValue(Operation *op, Value value, LoweringState &state) {
  auto it = state.values.find(value);
  if (it == state.values.end()) {
    op->emitOpError("normal SSA operand has no lowering plan");
    return {};
  }
  return it->second.value;
}

static const PredicatePlan *lookupPredicatePlan(Value value,
                                                const LoweringState &state) {
  auto it = state.predicates.find(value);
  return it == state.predicates.end() ? nullptr : &it->second;
}

static Operation *createSSAVC4Module(Operation *kernel, OpBuilder &builder) {
  OperationState state(kernel->getLoc(), kSSAVC4ModuleOpName);
  state.getOrAddProperties<mlir::ssavc4::ModuleOp::Properties>().sym_name =
      builder.getStringAttr("vc4kernel_lowered");
  state.addRegion();
  Operation *module = builder.create(state);
  module->getRegion(0).push_back(new Block());
  return module;
}

static Operation *createSSAVC4Func(Operation *kernel, Operation *module,
                                   OpBuilder &builder) {
  VC4KernelResourceSummary resourceSummary =
      computeVC4KernelResourceSummary(kernel);
  OperationState state(kernel->getLoc(), kSSAVC4FuncOpName);
  StringAttr sym = getSymbolNameAttr(kernel);
  auto &properties =
      state.getOrAddProperties<mlir::ssavc4::FuncOp::Properties>();
  properties.sym_name = sym ? sym : builder.getStringAttr("kernel");
  properties.kernel = builder.getUnitAttr();
  properties.threading = mlir::vc4::ThreadingModeAttr::get(
      builder.getContext(), mlir::vc4::ThreadingMode::single);
  state.addAttribute(builder.getStringAttr("vc4.launch_abi"),
                     buildLaunchABI(kernel, builder, resourceSummary));
  state.addAttribute(builder.getStringAttr("vc4.resource"),
                     buildSSAVC4ResourceMetadataFromVC4KernelSummary(
                         resourceSummary, builder));
  state.addRegion();
  builder.setInsertionPointToEnd(&module->getRegion(0).front());
  return builder.create(state);
}

static Value createI32Add(OpBuilder &builder, Location loc, Value lhs,
                          Value rhs) {
  return createOpWithResult(
      builder, loc, kSSAVC4ALUAddOpName, {lhs, rhs},
      {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                          builder.getContext(),
                                          mlir::vc4::AddOpcode::add))},
      lhs.getType());
}

static Value createI32Sub(OpBuilder &builder, Location loc, Value lhs,
                          Value rhs) {
  return createOpWithResult(
      builder, loc, kSSAVC4ALUAddOpName, {lhs, rhs},
      {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                          builder.getContext(),
                                          mlir::vc4::AddOpcode::sub))},
      lhs.getType());
}

static bool isVectorF32(Type type) {
  auto vectorType = llvm::dyn_cast<VectorType>(type);
  return vectorType && vectorType.getElementType().isF32();
}

static Value createI32BinaryAddPipe(OpBuilder &builder, Location loc, Value lhs,
                                    Value rhs, mlir::vc4::AddOpcode opcode) {
  return createOpWithResult(
      builder, loc, kSSAVC4ALUAddOpName, {lhs, rhs},
      {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                          builder.getContext(), opcode))},
      lhs.getType());
}

static FailureOr<Value> emitFragmentConst(Operation *op, OpBuilder &builder) {
  auto valueAttr = llvm::dyn_cast_if_present<DenseElementsAttr>(
      op->getAttr("value"));
  if (!valueAttr) {
    op->emitOpError("requires dense value attribute");
    return failure();
  }

  Type resultType = op->getResult(0).getType();
  Location loc = op->getLoc();

  if (mlir::vc4kernel::isVC4KernelVector16F32Type(resultType)) {
    if (!isDenseF32Splat(valueAttr)) {
      op->emitOpError("requires efficient fragment_const materialization");
      return failure();
    }
    APFloat value = *valueAttr.getValues<APFloat>().begin();
    if (!value.isFinite()) {
      op->emitOpError("fragment_const f32 splat must be finite in P2");
      return failure();
    }
    auto elementType = llvm::cast<VectorType>(resultType).getElementType();
    return createLoadImm(builder, loc, resultType,
                         FloatAttr::get(elementType, value));
  }

  if (!mlir::vc4kernel::isVC4KernelVector16I32Type(resultType)) {
    op->emitOpError("requires vector<16xi32> or vector<16xf32> result");
    return failure();
  }

  if (isDenseI32Splat(valueAttr)) {
    APInt value = *valueAttr.getValues<APInt>().begin();
    return createLoadImm(builder, loc, resultType,
                         builder.getI32IntegerAttr(value.getSExtValue()));
  }

  SmallVector<int64_t, 16> values = getDenseI32Values(valueAttr);
  if (isI32LaneRange(values))
    return createOpWithResult(builder, loc, kSSAVC4ElementNumberOpName, {}, {},
                              resultType);

  if (std::optional<LaneAffineConstPlan> plan =
          getI32LaneAffinePlan(values)) {
    Value lanes = createOpWithResult(builder, loc, kSSAVC4ElementNumberOpName,
                                     {}, {}, resultType);
    Value shift = createLoadImm(builder, loc, resultType,
                                builder.getI32IntegerAttr(plan->shift));
    Value delta = createI32BinaryAddPipe(builder, loc, lanes, shift,
                                         mlir::vc4::AddOpcode::shl);
    Value base = createLoadImm(builder, loc, resultType,
                               builder.getI32IntegerAttr(plan->base));
    return createI32BinaryAddPipe(
        builder, loc, base, delta,
        plan->subtract ? mlir::vc4::AddOpcode::sub : mlir::vc4::AddOpcode::add);
  }

  if (isPerLaneS2(values) && llvm::any_of(values, [](int64_t value) {
        return value < 0;
      }))
    return createPerLaneLoadImm(builder, loc, resultType,
                                mlir::vc4::LoadImmMode::per_elem_i2, values);
  if (isPerLaneU2(values))
    return createPerLaneLoadImm(builder, loc, resultType,
                                mlir::vc4::LoadImmMode::per_elem_u2, values);
  if (isPerLaneS2(values))
    return createPerLaneLoadImm(builder, loc, resultType,
                                mlir::vc4::LoadImmMode::per_elem_i2, values);

  op->emitOpError("requires efficient fragment_const materialization");
  return failure();
}

static Value createMul24(OpBuilder &builder, Location loc, Value lhs,
                         Value rhs) {
  return createOpWithResult(
      builder, loc, kSSAVC4ALUMulOpName, {lhs, rhs},
      {builder.getNamedAttr("opcode", mlir::vc4::MulOpcodeAttr::get(
                                          builder.getContext(),
                                          mlir::vc4::MulOpcode::mul24))},
      lhs.getType());
}

static mlir::vc4::AddOpcode
mapAddALUOpcode(mlir::vc4kernel::AddALUOpcode opcode) {
  switch (opcode) {
  case mlir::vc4kernel::AddALUOpcode::nop:
    return mlir::vc4::AddOpcode::nop;
  case mlir::vc4kernel::AddALUOpcode::fadd:
    return mlir::vc4::AddOpcode::fadd;
  case mlir::vc4kernel::AddALUOpcode::fsub:
    return mlir::vc4::AddOpcode::fsub;
  case mlir::vc4kernel::AddALUOpcode::fmin:
    return mlir::vc4::AddOpcode::fmin;
  case mlir::vc4kernel::AddALUOpcode::fmax:
    return mlir::vc4::AddOpcode::fmax;
  case mlir::vc4kernel::AddALUOpcode::fminabs:
    return mlir::vc4::AddOpcode::fminabs;
  case mlir::vc4kernel::AddALUOpcode::fmaxabs:
    return mlir::vc4::AddOpcode::fmaxabs;
  case mlir::vc4kernel::AddALUOpcode::ftoi:
    return mlir::vc4::AddOpcode::ftoi;
  case mlir::vc4kernel::AddALUOpcode::itof:
    return mlir::vc4::AddOpcode::itof;
  case mlir::vc4kernel::AddALUOpcode::add:
    return mlir::vc4::AddOpcode::add;
  case mlir::vc4kernel::AddALUOpcode::sub:
    return mlir::vc4::AddOpcode::sub;
  case mlir::vc4kernel::AddALUOpcode::shr:
    return mlir::vc4::AddOpcode::shr;
  case mlir::vc4kernel::AddALUOpcode::asr:
    return mlir::vc4::AddOpcode::asr;
  case mlir::vc4kernel::AddALUOpcode::ror:
    return mlir::vc4::AddOpcode::ror;
  case mlir::vc4kernel::AddALUOpcode::shl:
    return mlir::vc4::AddOpcode::shl;
  case mlir::vc4kernel::AddALUOpcode::min:
    return mlir::vc4::AddOpcode::min;
  case mlir::vc4kernel::AddALUOpcode::max:
    return mlir::vc4::AddOpcode::max;
  case mlir::vc4kernel::AddALUOpcode::bit_and:
    return mlir::vc4::AddOpcode::bit_and;
  case mlir::vc4kernel::AddALUOpcode::bit_or:
    return mlir::vc4::AddOpcode::bit_or;
  case mlir::vc4kernel::AddALUOpcode::bit_xor:
    return mlir::vc4::AddOpcode::bit_xor;
  case mlir::vc4kernel::AddALUOpcode::bit_not:
    return mlir::vc4::AddOpcode::bit_not;
  case mlir::vc4kernel::AddALUOpcode::clz:
    return mlir::vc4::AddOpcode::clz;
  case mlir::vc4kernel::AddALUOpcode::v8adds:
    return mlir::vc4::AddOpcode::v8adds;
  case mlir::vc4kernel::AddALUOpcode::v8subs:
    return mlir::vc4::AddOpcode::v8subs;
  }
  llvm_unreachable("unhandled VC4Kernel ADD-pipe opcode");
}

static mlir::vc4::MulOpcode
mapMulALUOpcode(mlir::vc4kernel::MulALUOpcode opcode) {
  switch (opcode) {
  case mlir::vc4kernel::MulALUOpcode::nop:
    return mlir::vc4::MulOpcode::nop;
  case mlir::vc4kernel::MulALUOpcode::fmul:
    return mlir::vc4::MulOpcode::fmul;
  case mlir::vc4kernel::MulALUOpcode::mul24:
    return mlir::vc4::MulOpcode::mul24;
  case mlir::vc4kernel::MulALUOpcode::v8muld:
    return mlir::vc4::MulOpcode::v8muld;
  case mlir::vc4kernel::MulALUOpcode::v8min:
    return mlir::vc4::MulOpcode::v8min;
  case mlir::vc4kernel::MulALUOpcode::v8max:
    return mlir::vc4::MulOpcode::v8max;
  case mlir::vc4kernel::MulALUOpcode::v8adds:
    return mlir::vc4::MulOpcode::v8adds;
  case mlir::vc4kernel::MulALUOpcode::v8subs:
    return mlir::vc4::MulOpcode::v8subs;
  }
  llvm_unreachable("unhandled VC4Kernel MUL-pipe opcode");
}

static Value createFragmentAdd(OpBuilder &builder, Location loc, Value lhs,
                               Value rhs) {
  mlir::vc4::AddOpcode opcode =
      isVectorF32(lhs.getType()) ? mlir::vc4::AddOpcode::fadd
                                 : mlir::vc4::AddOpcode::add;
  return createOpWithResult(
      builder, loc, kSSAVC4ALUAddOpName, {lhs, rhs},
      {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                          builder.getContext(), opcode))},
      lhs.getType());
}

static Value createAddPipeBinary(OpBuilder &builder, Location loc, Value lhs,
                                 Value rhs, mlir::vc4::AddOpcode opcode) {
  return createOpWithResult(
      builder, loc, kSSAVC4ALUAddOpName, {lhs, rhs},
      {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                          builder.getContext(), opcode))},
      lhs.getType());
}

static Value createI32SplatConstant(OpBuilder &builder, Location loc, Type type,
                                    int32_t value) {
  return createLoadImm(builder, loc, type, builder.getI32IntegerAttr(value));
}

static Value combineReduceValues(OpBuilder &builder, Location loc, Value lhs,
                                 Value rhs,
                                 mlir::vc4kernel::ReduceKind kind) {
  switch (kind) {
  case mlir::vc4kernel::ReduceKind::add:
    return createFragmentAdd(builder, loc, lhs, rhs);
  case mlir::vc4kernel::ReduceKind::min_s:
  case mlir::vc4kernel::ReduceKind::min_u:
    return createAddPipeBinary(builder, loc, lhs, rhs, mlir::vc4::AddOpcode::min);
  case mlir::vc4kernel::ReduceKind::max_s:
  case mlir::vc4kernel::ReduceKind::max_u:
    return createAddPipeBinary(builder, loc, lhs, rhs, mlir::vc4::AddOpcode::max);
  case mlir::vc4kernel::ReduceKind::bit_and:
    return createAddPipeBinary(builder, loc, lhs, rhs,
                               mlir::vc4::AddOpcode::bit_and);
  case mlir::vc4kernel::ReduceKind::bit_or:
    return createAddPipeBinary(builder, loc, lhs, rhs,
                               mlir::vc4::AddOpcode::bit_or);
  case mlir::vc4kernel::ReduceKind::bit_xor:
    return createAddPipeBinary(builder, loc, lhs, rhs,
                               mlir::vc4::AddOpcode::bit_xor);
  case mlir::vc4kernel::ReduceKind::fmin:
    return createAddPipeBinary(builder, loc, lhs, rhs, mlir::vc4::AddOpcode::fmin);
  case mlir::vc4kernel::ReduceKind::fmax:
    return createAddPipeBinary(builder, loc, lhs, rhs, mlir::vc4::AddOpcode::fmax);
  }
  llvm_unreachable("unhandled VC4Kernel reduce kind");
}

static int32_t getI32ReduceIdentity(mlir::vc4kernel::ReduceKind kind) {
  switch (kind) {
  case mlir::vc4kernel::ReduceKind::add:
  case mlir::vc4kernel::ReduceKind::max_u:
  case mlir::vc4kernel::ReduceKind::bit_or:
  case mlir::vc4kernel::ReduceKind::bit_xor:
    return 0;
  case mlir::vc4kernel::ReduceKind::min_s:
    return std::numeric_limits<int32_t>::max();
  case mlir::vc4kernel::ReduceKind::max_s:
    return std::numeric_limits<int32_t>::min();
  case mlir::vc4kernel::ReduceKind::min_u:
  case mlir::vc4kernel::ReduceKind::bit_and:
    return static_cast<int32_t>(0xffffffffu);
  case mlir::vc4kernel::ReduceKind::fmin:
  case mlir::vc4kernel::ReduceKind::fmax:
    llvm_unreachable("f32 reduce kind has no i32 identity");
  }
  llvm_unreachable("unhandled VC4Kernel reduce kind");
}

static Value emitRotateReduceTree(OpBuilder &builder, Location loc, Value value,
                                  mlir::vc4kernel::ReduceKind kind) {
  Value reduced = value;
  for (int64_t amount : {8, 4, 2, 1}) {
    Value rotated = createOpWithResult(
        builder, loc, kSSAVC4RotateOpName, reduced,
        {builder.getNamedAttr("amount", builder.getI32IntegerAttr(amount))},
        value.getType());
    reduced = combineReduceValues(builder, loc, reduced, rotated, kind);
  }
  return reduced;
}

static std::optional<int64_t> getI32ConstantValue(Value value);

static Value emitI32Mul32Fallback(Operation *op, OpBuilder &builder, Value lhs,
                                  Value rhs) {
  Location loc = op->getLoc();
  Value mask = createLoadImm(builder, loc, lhs.getType(),
                             builder.getI32IntegerAttr(0xffff));
  Value shiftScalar = createLoadImm(builder, loc, builder.getI32Type(),
                                    builder.getI32IntegerAttr(16));
  Value shift = shiftScalar;
  if (llvm::isa<VectorType>(lhs.getType()))
    shift = createOpWithResult(builder, loc, kSSAVC4SplatOpName, shiftScalar,
                               {}, lhs.getType());

  Value lhsLo = createI32BinaryAddPipe(builder, loc, lhs, mask,
                                       mlir::vc4::AddOpcode::bit_and);
  Value rhsLo = createI32BinaryAddPipe(builder, loc, rhs, mask,
                                       mlir::vc4::AddOpcode::bit_and);
  Value lhsHi = createI32BinaryAddPipe(builder, loc, lhs, shift,
                                       mlir::vc4::AddOpcode::shr);
  Value rhsHi = createI32BinaryAddPipe(builder, loc, rhs, shift,
                                       mlir::vc4::AddOpcode::shr);
  Value loLo = createMul24(builder, loc, lhsLo, rhsLo);
  Value loHi = createMul24(builder, loc, lhsLo, rhsHi);
  Value hiLo = createMul24(builder, loc, lhsHi, rhsLo);
  Value cross = createI32BinaryAddPipe(builder, loc, loHi, hiLo,
                                       mlir::vc4::AddOpcode::add);
  Value crossShifted = createI32BinaryAddPipe(builder, loc, cross, shift,
                                              mlir::vc4::AddOpcode::shl);
  return createI32BinaryAddPipe(builder, loc, loLo, crossShifted,
                                mlir::vc4::AddOpcode::add);
}

static Value addI32Constant(OpBuilder &builder, Location loc, Value value,
                            int64_t constant) {
  if (constant == 0)
    return value;
  Value offset =
      createLoadImm(builder, loc, builder.getI32Type(),
                    builder.getI32IntegerAttr(constant));
  return createI32Add(builder, loc, value, offset);
}

static std::optional<int64_t> getI32ConstantValue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto attr = llvm::dyn_cast<IntegerAttr>(constant.getValue());
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static std::optional<int64_t> getSplatI32ConstantValue(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, kSplatOpName) || def->getNumOperands() != 1)
    return std::nullopt;
  return getI32ConstantValue(def->getOperand(0));
}

static std::optional<SmallVector<int64_t, 16>>
getFragmentConstI32Values(Value value) {
  auto constOp = dyn_cast_or_null<mlir::vc4kernel::FragmentConstOp>(
      value.getDefiningOp());
  if (!constOp || !mlir::vc4kernel::isVC4KernelVector16I32Type(value.getType()))
    return std::nullopt;
  auto attr = llvm::dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!attr || !attr.getElementType().isSignlessInteger(32))
    return std::nullopt;
  return getDenseI32Values(attr);
}

static std::optional<int64_t> getFragmentConstSplatI32Value(Value value) {
  auto values = getFragmentConstI32Values(value);
  if (!values || values->empty())
    return std::nullopt;
  int64_t first = values->front();
  if (!llvm::all_of(*values, [first](int64_t value) { return value == first; }))
    return std::nullopt;
  return first;
}

static bool isLaneByteOffsets(Value value) {
  if (auto values = getFragmentConstI32Values(value)) {
    if (values->size() != 16)
      return false;
    for (int64_t lane = 0; lane < 16; ++lane)
      if ((*values)[lane] != lane * 4)
        return false;
    return true;
  }

  Operation *def = value.getDefiningOp();
  if (!def || def->getNumOperands() != 2)
    return false;
  if (!hasName(def->getOperand(0).getDefiningOp(), kLaneRangeOpName))
    return false;
  if (!hasName(def, kFragmentALUAddOpName))
    return false;
  auto opcode =
      llvm::dyn_cast_if_present<mlir::vc4kernel::AddALUOpcodeAttr>(
          def->getAttr("opcode"));
  if (!opcode || opcode.getValue() != mlir::vc4kernel::AddALUOpcode::shl)
    return false;
  std::optional<int64_t> shift = getSplatI32ConstantValue(def->getOperand(1));
  return shift && *shift == 2;
}

static Value getSplatScalar(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, kSplatOpName) || def->getNumOperands() != 1)
    return {};
  return def->getOperand(0);
}

struct FullRowVDWOffsets {
  bool matched = false;
  Value scalarBaseByteOffset;
  std::optional<int64_t> constantBaseByteOffset;
};

static FullRowVDWOffsets matchFullRowVDWByteOffsets(Value value) {
  if (isLaneByteOffsets(value))
    return {/*matched=*/true, /*scalarBaseByteOffset=*/{},
            /*constantBaseByteOffset=*/std::nullopt};

  Operation *def = value.getDefiningOp();
  if (!def || def->getNumOperands() != 2)
    return {};
  if (!hasName(def, kFragmentALUAddOpName))
    return {};
  auto opcode =
      llvm::dyn_cast_if_present<mlir::vc4kernel::AddALUOpcodeAttr>(
          def->getAttr("opcode"));
  if (!opcode || opcode.getValue() != mlir::vc4kernel::AddALUOpcode::add)
    return {};

  auto matchBasePlusLaneBytes = [](Value lhs,
                                   Value rhs) -> FullRowVDWOffsets {
    Value scalarBase = getSplatScalar(lhs);
    if (scalarBase && isLaneByteOffsets(rhs))
      return {/*matched=*/true, scalarBase, std::nullopt};
    std::optional<int64_t> constantBase = getFragmentConstSplatI32Value(lhs);
    if (constantBase && isLaneByteOffsets(rhs))
      return {/*matched=*/true, {}, constantBase};
    return {};
  };
  FullRowVDWOffsets result =
      matchBasePlusLaneBytes(def->getOperand(0), def->getOperand(1));
  if (result.matched)
    return result;
  return matchBasePlusLaneBytes(def->getOperand(1), def->getOperand(0));
}

static Value createSubFlags(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs) {
  return createOpWithResult(
      builder, loc, kSSAVC4MakeFlagsOpName, {lhs, rhs},
      {builder.getNamedAttr("kind", mlir::ssavc4::FlagKindAttr::get(
                                        builder.getContext(),
                                        mlir::ssavc4::FlagKind::sub))},
      mlir::ssavc4::FlagsType::get(builder.getContext()));
}

static Value createFSubFlags(OpBuilder &builder, Location loc, Value lhs,
                             Value rhs) {
  return createOpWithResult(
      builder, loc, kSSAVC4MakeFlagsOpName, {lhs, rhs},
      {builder.getNamedAttr("kind", mlir::ssavc4::FlagKindAttr::get(
                                        builder.getContext(),
                                        mlir::ssavc4::FlagKind::fsub))},
      mlir::ssavc4::FlagsType::get(builder.getContext()));
}

static Value createZeroTestFlags(OpBuilder &builder, Location loc, Value input) {
  return createOpWithResult(
      builder, loc, kSSAVC4MakeFlagsOpName, input,
      {builder.getNamedAttr("kind", mlir::ssavc4::FlagKindAttr::get(
                                        builder.getContext(),
                                        mlir::ssavc4::FlagKind::zero_test))},
      mlir::ssavc4::FlagsType::get(builder.getContext()));
}

static void createBranch(OpBuilder &builder, Location loc, Block *target) {
  OperationState state(loc, kSSAVC4BranchOpName);
  state.addSuccessors(target);
  builder.create(state);
}

static void createBranch(OpBuilder &builder, Location loc, Block *target,
                         ValueRange operands) {
  OperationState state(loc, kSSAVC4BranchOpName);
  state.addOperands(operands);
  state.addSuccessors(target);
  builder.create(state);
}

static void createCondBranch(OpBuilder &builder, Location loc, Value flags,
                             Block *trueDest, Block *falseDest,
                             mlir::vc4::BranchCond cond,
                             ValueRange trueOperands = {},
                             ValueRange falseOperands = {}) {
  OperationState state(loc, kSSAVC4CondBranchOpName);
  state.addOperands(flags);
  state.addOperands(trueOperands);
  state.addOperands(falseOperands);
  state.addSuccessors({trueDest, falseDest});
  state.addAttribute("cond",
                     mlir::vc4::BranchCondAttr::get(builder.getContext(),
                                                    cond));
  state.addAttribute("operandSegmentSizes",
                     builder.getDenseI32ArrayAttr(
                         {1, static_cast<int32_t>(trueOperands.size()),
                          static_cast<int32_t>(falseOperands.size())}));
  builder.create(state);
}

static Value createCondSelect(OpBuilder &builder, Location loc, Value flags,
                              Value trueValue, Value falseValue,
                              mlir::vc4::Cond cond) {
  return createOpWithResult(
      builder, loc, kSSAVC4CondSelectOpName, {flags, trueValue, falseValue},
      {builder.getNamedAttr("cond",
                            mlir::vc4::CondAttr::get(builder.getContext(),
                                                     cond))},
      trueValue.getType());
}

static Value createMaskConstant(OpBuilder &builder, Location loc,
                                int64_t value) {
  return createLoadImm(builder, loc, VectorType::get({16}, builder.getI32Type()),
                       builder.getI32IntegerAttr(value));
}

static bool isSameSSAValue(Value lhs, Value rhs) { return lhs && lhs == rhs; }

static Value createI32Min(OpBuilder &builder, Location loc, Value lhs,
                          Value rhs) {
  Value flags = createSubFlags(builder, loc, lhs, rhs);
  return createCondSelect(builder, loc, flags, lhs, rhs,
                          mlir::vc4::Cond::cs);
}

static Value createI32Max(OpBuilder &builder, Location loc, Value lhs,
                          Value rhs) {
  Value flags = createSubFlags(builder, loc, lhs, rhs);
  return createCondSelect(builder, loc, flags, rhs, lhs,
                          mlir::vc4::Cond::cs);
}

static FailureOr<Value>
materializePredicateMask(Operation *op, OpBuilder &builder,
                         const PredicatePlan &predicate,
                         LoweringState &state) {
  Location loc = op->getLoc();
  Value one = createMaskConstant(builder, loc, 1);
  Value zero = createMaskConstant(builder, loc, 0);

  if (predicate.kind == PredicatePlan::Class::Full)
    return one;
  if (predicate.kind == PredicatePlan::Class::Empty)
    return zero;
  if (predicate.kind == PredicatePlan::Class::GeneralMask) {
    if (predicate.maskValue)
      return predicate.maskValue;
    if (!predicate.compareLhs || !predicate.compareRhs)
      return op->emitOpError(
          "general predicate plan has no materialized mask or lowered compare "
          "operands");
    Value flags =
        createSubFlags(builder, loc, predicate.compareLhs, predicate.compareRhs);
    return createCondSelect(builder, loc, flags, one, zero,
                            predicate.compareCond);
  }

  Value lane =
      createOpWithResult(builder, loc, kSSAVC4ElementNumberOpName, {}, {},
                         one.getType());

  if (predicate.kind == PredicatePlan::Class::TailPrefix) {
    if (!predicate.base || !predicate.limit)
      return op->emitOpError("pred.tail plan is missing base or limit values");
    Value baseVec =
        createOpWithResult(builder, loc, kSSAVC4SplatOpName, predicate.base, {},
                           one.getType());
    Value limitVec =
        createOpWithResult(builder, loc, kSSAVC4SplatOpName, predicate.limit,
                           {}, one.getType());
    Value index = createI32Add(builder, loc, baseVec, lane);
    Value flags = createSubFlags(builder, loc, index, limitVec);
    return createCondSelect(builder, loc, flags, one, zero,
                            mlir::vc4::Cond::cs);
  }

  if (predicate.kind == PredicatePlan::Class::RectRow) {
    if (!predicate.row || !predicate.rows || !predicate.colBase ||
        !predicate.cols)
      return op->emitOpError(
          "pred.rect plan is missing row, rows, col_base, or cols values");
    Value colBaseVec =
        createOpWithResult(builder, loc, kSSAVC4SplatOpName,
                           predicate.colBase, {}, one.getType());
    Value colEnd = createI32Add(builder, loc, predicate.colBase,
                                predicate.cols);
    Value colEndVec =
        createOpWithResult(builder, loc, kSSAVC4SplatOpName, colEnd, {},
                           one.getType());
    Value upperFlags = createSubFlags(builder, loc, lane, colEndVec);
    Value upperMasked =
        createCondSelect(builder, loc, upperFlags, one, zero,
                         mlir::vc4::Cond::cs);
    Value lowerFlags = createSubFlags(builder, loc, lane, colBaseVec);
    Value colMasked =
        createCondSelect(builder, loc, lowerFlags, upperMasked, zero,
                         mlir::vc4::Cond::cc);
    Value rowFlags =
        createSubFlags(builder, loc, predicate.row, predicate.rows);
    return createCondSelect(builder, loc, rowFlags, colMasked, zero,
                            mlir::vc4::Cond::cs);
  }

  return op->emitOpError("unsupported predicate plan for mask materialization");
}

struct CompareMaskLowering {
  Value lhs;
  Value rhs;
  mlir::vc4::Cond cond = mlir::vc4::Cond::zs;
  mlir::ssavc4::FlagKind flagKind = mlir::ssavc4::FlagKind::sub;
};

static Value createCompareFlags(OpBuilder &builder, Location loc,
                                const CompareMaskLowering &compare) {
  if (compare.flagKind == mlir::ssavc4::FlagKind::fsub)
    return createFSubFlags(builder, loc, compare.lhs, compare.rhs);
  return createSubFlags(builder, loc, compare.lhs, compare.rhs);
}

static FailureOr<CompareMaskLowering>
mapUnsignedFragmentCmpPredicate(Operation *op,
                                mlir::vc4kernel::CmpPredicate predicate,
                                Value lhs, Value rhs) {
  CompareMaskLowering result;
  switch (predicate) {
  case mlir::vc4kernel::CmpPredicate::eq:
    result = {lhs, rhs, mlir::vc4::Cond::zs};
    break;
  case mlir::vc4kernel::CmpPredicate::ne:
    result = {lhs, rhs, mlir::vc4::Cond::zc};
    break;
  case mlir::vc4kernel::CmpPredicate::ult:
    result = {lhs, rhs, mlir::vc4::Cond::cs};
    break;
  case mlir::vc4kernel::CmpPredicate::uge:
    result = {lhs, rhs, mlir::vc4::Cond::cc};
    break;
  case mlir::vc4kernel::CmpPredicate::ugt:
    result = {rhs, lhs, mlir::vc4::Cond::cs};
    break;
  case mlir::vc4kernel::CmpPredicate::ule:
    result = {rhs, lhs, mlir::vc4::Cond::cc};
    break;
  default:
    return op->emitOpError("internal lowering error: expected unsigned "
                           "fragment_cmp predicate");
  }
  return result;
}

static FailureOr<CompareMaskLowering>
mapOrderedF32FragmentCmpPredicate(Operation *op,
                                  mlir::vc4kernel::CmpPredicate predicate,
                                  Value lhs, Value rhs) {
  CompareMaskLowering result;
  result.flagKind = mlir::ssavc4::FlagKind::fsub;
  switch (predicate) {
  case mlir::vc4kernel::CmpPredicate::oeq:
    result.lhs = lhs;
    result.rhs = rhs;
    result.cond = mlir::vc4::Cond::zs;
    break;
  case mlir::vc4kernel::CmpPredicate::one:
    result.lhs = lhs;
    result.rhs = rhs;
    result.cond = mlir::vc4::Cond::zc;
    break;
  case mlir::vc4kernel::CmpPredicate::olt:
    result.lhs = lhs;
    result.rhs = rhs;
    result.cond = mlir::vc4::Cond::ns;
    break;
  case mlir::vc4kernel::CmpPredicate::oge:
    result.lhs = lhs;
    result.rhs = rhs;
    result.cond = mlir::vc4::Cond::nc;
    break;
  case mlir::vc4kernel::CmpPredicate::ogt:
    result.lhs = rhs;
    result.rhs = lhs;
    result.cond = mlir::vc4::Cond::ns;
    break;
  case mlir::vc4kernel::CmpPredicate::ole:
    result.lhs = rhs;
    result.rhs = lhs;
    result.cond = mlir::vc4::Cond::nc;
    break;
  default:
    return op->emitOpError("unordered f32 fragment_cmp is not supported in P3");
  }
  return result;
}

static Value createSignBiasedI32(OpBuilder &builder, Location loc, Value value) {
  Value signBit = createLoadImm(
      builder, loc, value.getType(),
      builder.getIntegerAttr(builder.getI32Type(), llvm::APInt(32, 0x80000000u)));
  return createI32BinaryAddPipe(builder, loc, value, signBit,
                                mlir::vc4::AddOpcode::bit_xor);
}

static FailureOr<CompareMaskLowering>
mapFragmentCmpPredicate(Operation *op, OpBuilder &builder, Value lhs,
                        Value rhs) {
  auto attr =
      llvm::dyn_cast_or_null<mlir::vc4kernel::CmpPredicateAttr>(
          op->getAttr("predicate"));
  if (!attr)
    return op->emitOpError("fragment_cmp is missing predicate attribute");

  switch (attr.getValue()) {
  case mlir::vc4kernel::CmpPredicate::eq:
  case mlir::vc4kernel::CmpPredicate::ne:
  case mlir::vc4kernel::CmpPredicate::ult:
  case mlir::vc4kernel::CmpPredicate::ule:
  case mlir::vc4kernel::CmpPredicate::ugt:
  case mlir::vc4kernel::CmpPredicate::uge:
    return mapUnsignedFragmentCmpPredicate(op, attr.getValue(), lhs, rhs);
  case mlir::vc4kernel::CmpPredicate::slt: {
    Value biasedLhs = createSignBiasedI32(builder, op->getLoc(), lhs);
    Value biasedRhs = createSignBiasedI32(builder, op->getLoc(), rhs);
    return mapUnsignedFragmentCmpPredicate(
        op, mlir::vc4kernel::CmpPredicate::ult, biasedLhs, biasedRhs);
  }
  case mlir::vc4kernel::CmpPredicate::sle: {
    Value biasedLhs = createSignBiasedI32(builder, op->getLoc(), lhs);
    Value biasedRhs = createSignBiasedI32(builder, op->getLoc(), rhs);
    return mapUnsignedFragmentCmpPredicate(
        op, mlir::vc4kernel::CmpPredicate::ule, biasedLhs, biasedRhs);
  }
  case mlir::vc4kernel::CmpPredicate::sgt: {
    Value biasedLhs = createSignBiasedI32(builder, op->getLoc(), lhs);
    Value biasedRhs = createSignBiasedI32(builder, op->getLoc(), rhs);
    return mapUnsignedFragmentCmpPredicate(
        op, mlir::vc4kernel::CmpPredicate::ugt, biasedLhs, biasedRhs);
  }
  case mlir::vc4kernel::CmpPredicate::sge: {
    Value biasedLhs = createSignBiasedI32(builder, op->getLoc(), lhs);
    Value biasedRhs = createSignBiasedI32(builder, op->getLoc(), rhs);
    return mapUnsignedFragmentCmpPredicate(
        op, mlir::vc4kernel::CmpPredicate::uge, biasedLhs, biasedRhs);
  }
  case mlir::vc4kernel::CmpPredicate::oeq:
  case mlir::vc4kernel::CmpPredicate::one:
  case mlir::vc4kernel::CmpPredicate::olt:
  case mlir::vc4kernel::CmpPredicate::ole:
  case mlir::vc4kernel::CmpPredicate::ogt:
  case mlir::vc4kernel::CmpPredicate::oge:
    return mapOrderedF32FragmentCmpPredicate(op, attr.getValue(), lhs, rhs);
  case mlir::vc4kernel::CmpPredicate::uno:
  case mlir::vc4kernel::CmpPredicate::ueq:
  case mlir::vc4kernel::CmpPredicate::une:
    return op->emitOpError("unordered f32 fragment_cmp is not supported in P3");
  }
  llvm_unreachable("all VC4Kernel comparison predicates are handled");
}

static Value createZeroValue(OpBuilder &builder, Location loc, Type type) {
  return createLoadImm(builder, loc, type, builder.getI32IntegerAttr(0));
}

struct EmittedCondition {
  bool isConstant = false;
  bool constantValue = false;
  Value flags;
  mlir::vc4::BranchCond branchCond = mlir::vc4::BranchCond::any_z_clear;
  mlir::vc4::Cond selectCond = mlir::vc4::Cond::zc;
};

static EmittedCondition constantCondition(bool value) {
  EmittedCondition emitted;
  emitted.isConstant = true;
  emitted.constantValue = value;
  return emitted;
}

static FailureOr<EmittedCondition>
emitConditionPlan(Operation *op, OpBuilder &builder,
                  const ConditionPlan &condition, LoweringState &state);

static FailureOr<Value>
materializeConditionAsI32(Operation *op, OpBuilder &builder,
                          const ConditionPlan &condition,
                          LoweringState &state);

static FailureOr<EmittedCondition>
emitScalarCompareCondition(Operation *op, OpBuilder &builder, Value lhs,
                           Value rhs, Attribute predicate) {
  auto pred = llvm::dyn_cast_or_null<arith::CmpIPredicateAttr>(predicate);
  if (!pred)
    return op->emitOpError("arith.cmpi condition plan is missing predicate");

  EmittedCondition emitted;
  Value flagsLhs = lhs;
  Value flagsRhs = rhs;
  switch (pred.getValue()) {
  case arith::CmpIPredicate::eq:
    emitted.branchCond = mlir::vc4::BranchCond::any_z_set;
    emitted.selectCond = mlir::vc4::Cond::zs;
    break;
  case arith::CmpIPredicate::ne:
    emitted.branchCond = mlir::vc4::BranchCond::any_z_clear;
    emitted.selectCond = mlir::vc4::Cond::zc;
    break;
  case arith::CmpIPredicate::ult:
    emitted.branchCond = mlir::vc4::BranchCond::any_c_set;
    emitted.selectCond = mlir::vc4::Cond::cs;
    break;
  case arith::CmpIPredicate::uge:
    emitted.branchCond = mlir::vc4::BranchCond::any_c_clear;
    emitted.selectCond = mlir::vc4::Cond::cc;
    break;
  case arith::CmpIPredicate::ugt:
    flagsLhs = rhs;
    flagsRhs = lhs;
    emitted.branchCond = mlir::vc4::BranchCond::any_c_set;
    emitted.selectCond = mlir::vc4::Cond::cs;
    break;
  case arith::CmpIPredicate::ule:
    flagsLhs = rhs;
    flagsRhs = lhs;
    emitted.branchCond = mlir::vc4::BranchCond::any_c_clear;
    emitted.selectCond = mlir::vc4::Cond::cc;
    break;
  default:
    return op->emitOpError(
        "arith.cmpi lowering supports only eq, ne, ult, ule, ugt, and uge");
  }
  emitted.flags = createSubFlags(builder, op->getLoc(), flagsLhs, flagsRhs);
  return emitted;
}

static FailureOr<mlir::vc4::BranchCond>
mapLaneCondToBranchCond(Operation *op, mlir::vc4::Cond cond, bool requireAll) {
  switch (cond) {
  case mlir::vc4::Cond::zs:
    return requireAll ? mlir::vc4::BranchCond::all_z_set
                      : mlir::vc4::BranchCond::any_z_set;
  case mlir::vc4::Cond::zc:
    return requireAll ? mlir::vc4::BranchCond::all_z_clear
                      : mlir::vc4::BranchCond::any_z_clear;
  case mlir::vc4::Cond::ns:
    return requireAll ? mlir::vc4::BranchCond::all_n_set
                      : mlir::vc4::BranchCond::any_n_set;
  case mlir::vc4::Cond::nc:
    return requireAll ? mlir::vc4::BranchCond::all_n_clear
                      : mlir::vc4::BranchCond::any_n_clear;
  case mlir::vc4::Cond::cs:
    return requireAll ? mlir::vc4::BranchCond::all_c_set
                      : mlir::vc4::BranchCond::any_c_set;
  case mlir::vc4::Cond::cc:
    return requireAll ? mlir::vc4::BranchCond::all_c_clear
                      : mlir::vc4::BranchCond::any_c_clear;
  case mlir::vc4::Cond::never:
  case mlir::vc4::Cond::always:
    break;
  }
  return op->emitOpError("predicate condition cannot use never/always");
}

static FailureOr<Value>
materializeAllConditionsAsI32(Operation *op, OpBuilder &builder,
                              ArrayRef<ConditionPlan> conditions,
                              LoweringState &state) {
  Location loc = op->getLoc();
  if (conditions.empty())
    return createLoadImm(builder, loc, builder.getI32Type(),
                         builder.getI32IntegerAttr(1));

  Region *region = builder.getInsertionBlock()->getParent();
  Block *doneBlock = new Block();
  doneBlock->addArgument(builder.getI32Type(), loc);
  Block *falseBlock = new Block();
  SmallVector<Block *, 4> testBlocks;
  for (size_t i = 1; i < conditions.size(); ++i)
    testBlocks.push_back(new Block());
  Block *trueBlock = new Block();

  for (Block *block : testBlocks)
    region->push_back(block);
  region->push_back(trueBlock);
  region->push_back(falseBlock);
  region->push_back(doneBlock);

  auto emitTest = [&](const ConditionPlan &condition, Block *trueDest,
                      Block *falseDest) -> LogicalResult {
    FailureOr<EmittedCondition> emitted =
        emitConditionPlan(op, builder, condition, state);
    if (failed(emitted))
      return failure();
    if (emitted->isConstant) {
      createBranch(builder, loc, emitted->constantValue ? trueDest : falseDest);
      return success();
    }
    createCondBranch(builder, loc, emitted->flags, trueDest, falseDest,
                     emitted->branchCond);
    return success();
  };

  Block *firstTrueDest = conditions.size() == 1 ? trueBlock : testBlocks[0];
  if (failed(emitTest(conditions.front(), firstTrueDest, falseBlock)))
    return failure();

  for (size_t i = 1; i < conditions.size(); ++i) {
    builder.setInsertionPointToEnd(testBlocks[i - 1]);
    Block *nextTrueDest =
        i + 1 == conditions.size() ? trueBlock : testBlocks[i];
    if (failed(emitTest(conditions[i], nextTrueDest, falseBlock)))
      return failure();
  }

  builder.setInsertionPointToEnd(trueBlock);
  Value one = createLoadImm(builder, loc, builder.getI32Type(),
                            builder.getI32IntegerAttr(1));
  createBranch(builder, loc, doneBlock, one);

  builder.setInsertionPointToEnd(falseBlock);
  Value zero = createLoadImm(builder, loc, builder.getI32Type(),
                             builder.getI32IntegerAttr(0));
  createBranch(builder, loc, doneBlock, zero);

  builder.setInsertionPointToEnd(doneBlock);
  return doneBlock->getArgument(0);
}

static FailureOr<EmittedCondition>
emitPredicateCondition(Operation *op, OpBuilder &builder,
                       const PredicatePlan &predicate, bool requireAll,
                       LoweringState &state) {
  Location loc = op->getLoc();
  if (predicate.kind == PredicatePlan::Class::Full)
    return constantCondition(true);
  if (predicate.kind == PredicatePlan::Class::Empty)
    return constantCondition(false);

  if (predicate.kind == PredicatePlan::Class::GeneralMask) {
    if (predicate.maskValue) {
      EmittedCondition emitted;
      emitted.flags = createZeroTestFlags(builder, loc, predicate.maskValue);
      emitted.branchCond = requireAll
                               ? mlir::vc4::BranchCond::all_z_clear
                               : mlir::vc4::BranchCond::any_z_clear;
      emitted.selectCond = mlir::vc4::Cond::zc;
      return emitted;
    }
    if (!predicate.compareLhs || !predicate.compareRhs)
      return op->emitOpError(
          "general predicate plan has no lowered compare operands");
    FailureOr<mlir::vc4::BranchCond> branchCond =
        mapLaneCondToBranchCond(op, predicate.compareCond, requireAll);
    if (failed(branchCond))
      return failure();
    EmittedCondition emitted;
    emitted.flags =
        createSubFlags(builder, loc, predicate.compareLhs, predicate.compareRhs);
    emitted.branchCond = *branchCond;
    emitted.selectCond = predicate.compareCond;
    return emitted;
  }

  if (predicate.kind == PredicatePlan::Class::TailPrefix) {
    if (!predicate.base || !predicate.limit)
      return op->emitOpError("pred.tail plan is missing base or limit values");
    Value threshold =
        addI32Constant(builder, loc, predicate.base, requireAll ? 16 : 1);
    EmittedCondition emitted;
    emitted.flags = createSubFlags(builder, loc, predicate.limit, threshold);
    emitted.branchCond = mlir::vc4::BranchCond::any_c_clear;
    emitted.selectCond = mlir::vc4::Cond::cc;
    return emitted;
  }

  if (predicate.kind == PredicatePlan::Class::RectRow) {
    if (!predicate.row || !predicate.rows || !predicate.colBase ||
        !predicate.cols)
      return op->emitOpError(
          "pred.rect plan is missing row, rows, col_base, or cols values");
    Value zero = createLoadImm(builder, loc, builder.getI32Type(),
                               builder.getI32IntegerAttr(0));
    Value sixteen = createLoadImm(builder, loc, builder.getI32Type(),
                                  builder.getI32IntegerAttr(16));
    SmallVector<ConditionPlan, 4> conditions;
    conditions.push_back(ConditionPlan::scalarI32Compare(
        predicate.row, predicate.rows,
        arith::CmpIPredicateAttr::get(builder.getContext(),
                                      arith::CmpIPredicate::ult)));
    if (requireAll) {
      conditions.push_back(ConditionPlan::scalarI32Compare(
          predicate.colBase, zero,
          arith::CmpIPredicateAttr::get(builder.getContext(),
                                        arith::CmpIPredicate::eq)));
      conditions.push_back(ConditionPlan::scalarI32Compare(
          predicate.cols, sixteen,
          arith::CmpIPredicateAttr::get(builder.getContext(),
                                        arith::CmpIPredicate::uge)));
    } else {
      conditions.push_back(ConditionPlan::scalarI32Compare(
          zero, predicate.cols,
          arith::CmpIPredicateAttr::get(builder.getContext(),
                                        arith::CmpIPredicate::ult)));
      conditions.push_back(ConditionPlan::scalarI32Compare(
          predicate.colBase, sixteen,
          arith::CmpIPredicateAttr::get(builder.getContext(),
                                        arith::CmpIPredicate::ult)));
    }
    FailureOr<Value> value =
        materializeAllConditionsAsI32(op, builder, conditions, state);
    if (failed(value))
      return failure();
    EmittedCondition emitted;
    emitted.flags = createZeroTestFlags(builder, loc, *value);
    emitted.branchCond = mlir::vc4::BranchCond::any_z_clear;
    emitted.selectCond = mlir::vc4::Cond::zc;
    return emitted;
  }

  return op->emitOpError("unsupported predicate condition plan");
}

static FailureOr<EmittedCondition>
emitConditionPlan(Operation *op, OpBuilder &builder,
                  const ConditionPlan &condition, LoweringState &state) {
  switch (condition.kind) {
  case ConditionPlan::Class::ConstantTrue:
    return constantCondition(true);
  case ConditionPlan::Class::ConstantFalse:
    return constantCondition(false);
  case ConditionPlan::Class::ScalarI32Compare:
    return emitScalarCompareCondition(op, builder, condition.lhs, condition.rhs,
                                      condition.predicate);
  case ConditionPlan::Class::PredicateAny:
  case ConditionPlan::Class::PredicateAll: {
    const PredicatePlan *predicate =
        lookupPredicatePlan(condition.predicateSource, state);
    if (!predicate)
      return op->emitOpError("predicate operand has no lowering plan");
    return emitPredicateCondition(
        op, builder, *predicate,
        condition.kind == ConditionPlan::Class::PredicateAll, state);
  }
  case ConditionPlan::Class::FlagsValue:
    if (!condition.flags)
      return op->emitOpError("condition flags plan is missing flags value");
    return EmittedCondition{/*isConstant=*/false,
                            /*constantValue=*/false,
                            condition.flags,
                            condition.branchCond,
                            condition.selectCond};
  }
  return op->emitOpError("unsupported condition plan");
}

static FailureOr<Value>
materializeConditionAsI32(Operation *op, OpBuilder &builder,
                          const ConditionPlan &condition,
                          LoweringState &state) {
  Location loc = op->getLoc();
  FailureOr<EmittedCondition> emitted =
      emitConditionPlan(op, builder, condition, state);
  if (failed(emitted))
    return failure();
  if (emitted->isConstant)
    return createLoadImm(builder, loc, builder.getI32Type(),
                         builder.getI32IntegerAttr(emitted->constantValue ? 1
                                                                          : 0));

  Value one = createLoadImm(builder, loc, builder.getI32Type(),
                            builder.getI32IntegerAttr(1));
  Value zero = createLoadImm(builder, loc, builder.getI32Type(),
                             builder.getI32IntegerAttr(0));
  return createCondSelect(builder, loc, emitted->flags, one, zero,
                          emitted->selectCond);
}

static FailureOr<Value>
emitPredicateSelect(Operation *op, OpBuilder &builder,
                    const PredicatePlan &predicate, Value trueValue,
                    Value falseValue) {
  Location loc = op->getLoc();
  Type valueType = trueValue.getType();
  if (trueValue.getType() != falseValue.getType())
    return op->emitOpError("predicate select values have mismatched types");

  if (predicate.kind == PredicatePlan::Class::Full)
    return trueValue;
  if (predicate.kind == PredicatePlan::Class::Empty)
    return falseValue;
  if (predicate.kind == PredicatePlan::Class::GeneralMask) {
    if (predicate.maskValue) {
      Value flags = createZeroTestFlags(builder, loc, predicate.maskValue);
      return createCondSelect(builder, loc, flags, trueValue, falseValue,
                              mlir::vc4::Cond::zc);
    }
    if (!predicate.compareLhs || !predicate.compareRhs)
      return op->emitOpError(
          "general predicate plan has no lowered compare operands");
    Value flags =
        createSubFlags(builder, loc, predicate.compareLhs, predicate.compareRhs);
    return createCondSelect(builder, loc, flags, trueValue, falseValue,
                            predicate.compareCond);
  }

  auto vectorType = llvm::dyn_cast<VectorType>(valueType);
  if (!vectorType || vectorType.getRank() != 1 ||
      vectorType.getDimSize(0) != 16)
    return op->emitOpError(
        "structured predicate select currently requires vector<16xT> values");
  VectorType laneType = VectorType::get({16}, builder.getI32Type());
  Value lane =
      createOpWithResult(builder, loc, kSSAVC4ElementNumberOpName, {}, {},
                         laneType);

  if (predicate.kind == PredicatePlan::Class::TailPrefix) {
    if (!predicate.base || !predicate.limit)
      return op->emitOpError("pred.tail plan is missing base or limit values");
    Value baseVec =
        createOpWithResult(builder, loc, kSSAVC4SplatOpName, predicate.base, {},
                           laneType);
    Value limitVec =
        createOpWithResult(builder, loc, kSSAVC4SplatOpName, predicate.limit,
                           {}, laneType);
    Value index = createI32Add(builder, loc, baseVec, lane);
    Value flags = createSubFlags(builder, loc, index, limitVec);
    return createCondSelect(builder, loc, flags, trueValue, falseValue,
                            mlir::vc4::Cond::cs);
  }

  if (predicate.kind == PredicatePlan::Class::RectRow) {
    if (!predicate.row || !predicate.rows || !predicate.colBase ||
        !predicate.cols)
      return op->emitOpError(
          "pred.rect plan is missing row, rows, col_base, or cols values");
    Value one =
        createLoadImm(builder, loc, builder.getI32Type(),
                      builder.getI32IntegerAttr(1));
    Value sixteen =
        createLoadImm(builder, loc, builder.getI32Type(),
                      builder.getI32IntegerAttr(16));
    Value colBaseVec =
        createOpWithResult(builder, loc, kSSAVC4SplatOpName,
                           predicate.colBase, {}, laneType);
    Value colEnd = createI32Add(builder, loc, predicate.colBase,
                                predicate.cols);
    Value colEndVec =
        createOpWithResult(builder, loc, kSSAVC4SplatOpName, colEnd, {},
                           laneType);

    Value current = trueValue;
    Value upperFlags = createSubFlags(builder, loc, lane, colEndVec);
    current = createCondSelect(builder, loc, upperFlags, current, falseValue,
                               mlir::vc4::Cond::cs);
    Value lowerFlags = createSubFlags(builder, loc, lane, colBaseVec);
    current = createCondSelect(builder, loc, lowerFlags, current, falseValue,
                               mlir::vc4::Cond::cc);
    Value rowPlusOne = createI32Add(builder, loc, predicate.row, one);
    Value rowActiveFlags =
        createSubFlags(builder, loc, predicate.rows, rowPlusOne);
    current = createCondSelect(builder, loc, rowActiveFlags, current,
                               falseValue, mlir::vc4::Cond::cc);
    Value colsActiveFlags =
        createSubFlags(builder, loc, predicate.cols, one);
    current = createCondSelect(builder, loc, colsActiveFlags, current,
                               falseValue, mlir::vc4::Cond::cc);
    Value colBaseInRangeFlags =
        createSubFlags(builder, loc, predicate.colBase, sixteen);
    current = createCondSelect(builder, loc, colBaseInRangeFlags, current,
                               falseValue, mlir::vc4::Cond::cs);
    return current;
  }

  return op->emitOpError("unsupported predicate plan for select");
}

static Value emitTMULoadFragment(Operation *op, OpBuilder &builder, Value base,
                                 Value offsets, Type resultType) {
  Location loc = op->getLoc();
  Value loadBase = base;
  if (loadBase.getType() != offsets.getType())
    loadBase = createOpWithResult(builder, loc, kSSAVC4SplatOpName, loadBase,
                                  {}, offsets.getType());
  Value address = createOpWithResult(
      builder, loc, kSSAVC4ALUAddOpName, {loadBase, offsets},
      {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                          builder.getContext(),
                                          mlir::vc4::AddOpcode::add))},
      offsets.getType());
  Value token = createOpWithResult(
      builder, loc, kSSAVC4TMURequestOpName, address,
      {builder.getNamedAttr("unit", builder.getStringAttr("tmu0")),
       builder.getNamedAttr("mode", builder.getStringAttr("direct"))},
      mlir::ssavc4::AsyncTokenType::get(builder.getContext()));
  return createOpWithResult(
      builder, loc, kSSAVC4TMUReadOpName, token,
      {builder.getNamedAttr("unit", builder.getStringAttr("tmu0")),
       builder.getNamedAttr("part", builder.getStringAttr("raw32"))},
      resultType);
}

static void emitVDWStore(Operation *op, OpBuilder &builder, Value base,
                         Value value, Value activeLanes, LoweringState &state,
                         std::optional<int64_t> staticActiveLanes) {
  SmallVector<Value, 4> operands{base, value, activeLanes};
  SmallVector<int32_t, 4> segments{1, 1, 1, 0};
  Value stagingRow = state.launchABI.lookupBuiltin("vpm_base_row");
  if (stagingRow) {
    int64_t userRows = state.resourcePlan.summary.user_vpm_rows_per_block;
    int64_t perWarpRows =
        state.resourcePlan.summary.compiler_vpm_staging_rows_per_warp;
    stagingRow = addI32Constant(builder, op->getLoc(), stagingRow, userRows);
    if (state.resourcePlan.summary.schedule_mode == "cooperative_block" &&
        perWarpRows > 0) {
      Value logicalWarp = state.launchABI.lookupBuiltin("logical_warp_id");
      if (logicalWarp) {
        Value warpOffset = logicalWarp;
        if (perWarpRows != 1) {
          Value perWarp = createLoadImm(
              builder, op->getLoc(), builder.getI32Type(),
              builder.getI32IntegerAttr(perWarpRows));
          warpOffset = createMul24(builder, op->getLoc(), logicalWarp, perWarp);
        }
        stagingRow = createI32Add(builder, op->getLoc(), stagingRow, warpOffset);
      }
    }
    operands.push_back(stagingRow);
    segments[3] = 1;
  }
  SmallVector<NamedAttribute, 8> attrs{
      getSSAVC4VPMWidth(builder, op),
      getSSAVC4VPMSubword(builder, op),
      builder.getNamedAttr("vpm_row", builder.getI32IntegerAttr(0)),
      builder.getNamedAttr("serialize", builder.getStringAttr("mutex")),
      builder.getNamedAttr("operandSegmentSizes",
                           builder.getDenseI32ArrayAttr(segments))};
  if (staticActiveLanes)
    attrs.push_back(builder.getNamedAttr(
        "active_lanes", builder.getI32IntegerAttr(*staticActiveLanes)));
  createOp(builder, op->getLoc(), kSSAVC4VDWStoreOpName, operands, attrs);
}

static FailureOr<Value> applyVPMTileBase(Operation *op, OpBuilder &builder,
                                         Value tile, Value row,
                                         LoweringState &state) {
  auto tileIt = state.vpmAllocations.find(tile);
  if (tileIt == state.vpmAllocations.end())
    return op->emitOpError("vpm tile operand has no allocation plan");
  const VPMAllocationPlan &plan = tileIt->second;
  Value result = addI32Constant(builder, op->getLoc(), row, plan.baseRowOffset);
  Value base = state.launchABI.lookupBuiltin("vpm_base_row");
  if (base)
    result = createI32Add(builder, op->getLoc(), base, result);
  return result;
}

static FailureOr<Value>
applyVPMPredicateZeroFill(Operation *op, OpBuilder &builder, Value predicate,
                          Value value, LoweringState &state) {
  const PredicatePlan *plan = lookupPredicatePlan(predicate, state);
  if (!plan)
    return op->emitOpError("predicate operand has no lowering plan");
  Value zero = createZeroValue(builder, op->getLoc(), value.getType());
  FailureOr<Value> selected = emitPredicateSelect(op, builder, *plan, value,
                                                  zero);
  if (failed(selected))
    return failure();
  return *selected;
}

static LogicalResult lowerBodyOp(Operation *op, OpBuilder &builder,
                                 LoweringState &state) {
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    if (cst.getType().isInteger(1)) {
      int64_t intValue = 0;
      if (auto value = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
        intValue = value.getValue().isZero() ? 0 : 1;
      if (auto value = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
        state.conditions[cst.getResult()] =
            ConditionPlan::constant(!value.getValue().isZero());
      state.values[cst.getResult()] = {
          createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                        builder.getI32IntegerAttr(intValue))};
    } else {
      state.values[cst.getResult()] = {createLoadImm(
          builder, op->getLoc(), cst.getType(), cst.getValue())};
    }
    return success();
  }
  if (hasName(op, kProgramIdOpName) || hasName(op, kNumProgramsOpName) ||
      hasName(op, kWarpIdOpName)) {
    std::string ownedName;
    StringRef name;
    if (hasName(op, kProgramIdOpName)) {
      ownedName = getProgramIdBuiltinName(getAxisAttrValue(op));
      name = state.launchABI.lookupBuiltin(ownedName) ? StringRef(ownedName)
                                                      : StringRef("logical_request");
    } else if (hasName(op, kNumProgramsOpName)) {
      ownedName = getNumProgramsBuiltinName(getAxisAttrValue(op));
      name = ownedName;
    } else {
      name = "logical_warp_id";
    }
    Value builtin = state.launchABI.lookupBuiltin(name);
    if (!builtin)
      return op->emitOpError("missing launch builtin uniform for identity op");
    state.values[op->getResult(0)] = {builtin};
    return success();
  }
  if (hasName(op, kLaneRangeOpName)) {
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4ElementNumberOpName, {}, {},
        op->getResult(0).getType())};
    return success();
  }
  if (hasName(op, "arith.addi") || hasName(op, "arith.subi") ||
      hasName(op, "arith.muli") || hasName(op, "arith.shli")) {
    SmallVector<Value, 2> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    if (hasName(op, "arith.muli")) {
      state.values[op->getResult(0)] = {
          emitI32Mul32Fallback(op, builder, operands[0], operands[1])};
      return success();
    }
    mlir::vc4::AddOpcode opcode = mlir::vc4::AddOpcode::add;
    if (hasName(op, "arith.subi"))
      opcode = mlir::vc4::AddOpcode::sub;
    if (hasName(op, "arith.shli"))
      opcode = mlir::vc4::AddOpcode::shl;
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, operands,
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(), opcode))},
        op->getResult(0).getType())};
    return success();
  }
  if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
    Value lhs = mapValue(op, cmp.getLhs(), state);
    Value rhs = mapValue(op, cmp.getRhs(), state);
    if (!lhs || !rhs)
      return failure();
    state.conditions[cmp.getResult()] =
        ConditionPlan::scalarI32Compare(lhs, rhs, cmp.getPredicateAttr());
    return success();
  }
  if (auto select = dyn_cast<arith::SelectOp>(op)) {
    auto conditionIt = state.conditions.find(select.getCondition());
    if (conditionIt == state.conditions.end())
      return op->emitOpError("condition operand has no lowering plan");
    Type resultType = select.getResult().getType();
    if (resultType.isInteger(1)) {
      auto trueIt = state.conditions.find(select.getTrueValue());
      auto falseIt = state.conditions.find(select.getFalseValue());
      if (trueIt == state.conditions.end() ||
          falseIt == state.conditions.end())
        return op->emitOpError(
            "arith.select i1 operands require condition plans");
      FailureOr<Value> trueValue =
          materializeConditionAsI32(op, builder, trueIt->second, state);
      if (failed(trueValue))
        return failure();
      FailureOr<Value> falseValue =
          materializeConditionAsI32(op, builder, falseIt->second, state);
      if (failed(falseValue))
        return failure();
      FailureOr<EmittedCondition> emitted =
          emitConditionPlan(op, builder, conditionIt->second, state);
      if (failed(emitted))
        return failure();
      Value selected = emitted->isConstant
                           ? (emitted->constantValue ? *trueValue : *falseValue)
                           : createCondSelect(builder, op->getLoc(),
                                              emitted->flags, *trueValue,
                                              *falseValue, emitted->selectCond);
      Value flags = createZeroTestFlags(builder, op->getLoc(), selected);
      state.values[select.getResult()] = {selected};
      state.conditions[select.getResult()] = ConditionPlan::flagsValue(
          flags, mlir::vc4::BranchCond::any_z_clear, mlir::vc4::Cond::zc);
      return success();
    }
    Value trueValue = mapValue(op, select.getTrueValue(), state);
    Value falseValue = mapValue(op, select.getFalseValue(), state);
    if (!trueValue || !falseValue)
      return failure();
    FailureOr<EmittedCondition> emitted =
        emitConditionPlan(op, builder, conditionIt->second, state);
    if (failed(emitted))
      return failure();
    state.values[select.getResult()] = {
        emitted->isConstant
            ? (emitted->constantValue ? trueValue : falseValue)
            : createCondSelect(builder, op->getLoc(), emitted->flags, trueValue,
                               falseValue, emitted->selectCond)};
    return success();
  }
  if (hasName(op, kSplatOpName)) {
    Value input = mapValue(op, op->getOperand(0), state);
    if (!input)
      return failure();
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4SplatOpName, input, {},
        op->getResult(0).getType())};
    return success();
  }
  if (hasName(op, kFragmentBitcastOpName)) {
    Value input = mapValue(op, op->getOperand(0), state);
    if (!input)
      return failure();
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4MovOpName, input, {},
        op->getResult(0).getType())};
    return success();
  }
  if (hasName(op, kFragmentConstOpName)) {
    FailureOr<Value> value = emitFragmentConst(op, builder);
    if (failed(value))
      return failure();
    state.values[op->getResult(0)] = {*value};
    return success();
  }
  if (hasName(op, kFragmentALUAddOpName)) {
    SmallVector<Value, 2> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    auto opcodeAttr =
        llvm::dyn_cast_if_present<mlir::vc4kernel::AddALUOpcodeAttr>(
            op->getAttr("opcode"));
    if (!opcodeAttr)
      return op->emitOpError("requires ADD-pipe opcode attribute");
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, operands,
        {builder.getNamedAttr(
            "opcode",
            mlir::vc4::AddOpcodeAttr::get(
                builder.getContext(), mapAddALUOpcode(opcodeAttr.getValue())))},
        op->getResult(0).getType())};
    return success();
  }
  if (hasName(op, kFragmentALUMulOpName)) {
    SmallVector<Value, 2> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    auto opcodeAttr =
        llvm::dyn_cast_if_present<mlir::vc4kernel::MulALUOpcodeAttr>(
            op->getAttr("opcode"));
    if (!opcodeAttr)
      return op->emitOpError("requires MUL-pipe opcode attribute");
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUMulOpName, operands,
        {builder.getNamedAttr(
            "opcode",
            mlir::vc4::MulOpcodeAttr::get(
                builder.getContext(), mapMulALUOpcode(opcodeAttr.getValue())))},
        op->getResult(0).getType())};
    return success();
  }
  if (hasName(op, kFragmentRotateOpName)) {
    Value input = mapValue(op, op->getOperand(0), state);
    if (!input)
      return failure();
    // VC4Kernel uses the SSAVC4/VC4 fixed horizontal vector-rotate direction:
    // amount N lowers to vc4asm's `accumulator << N` lane rotate.
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4RotateOpName, input,
        {builder.getNamedAttr("amount", op->getAttr("amount"))},
        op->getResult(0).getType())};
    return success();
  }
  if (hasName(op, kFragmentReduceOpName)) {
    Value input = mapValue(op, op->getOperand(0), state);
    if (!input)
      return failure();
    const PredicatePlan *predicate =
        lookupPredicatePlan(op->getOperand(1), state);
    if (!predicate)
      return op->emitOpError("predicate operand has no lowering plan");
    auto kind =
        llvm::dyn_cast_or_null<mlir::vc4kernel::ReduceKindAttr>(
            op->getAttr("kind"));
    if (!kind)
      return op->emitOpError("fragment_reduce lowering requires kind");
    Type resultType = op->getResult(0).getType();
    mlir::vc4kernel::ReduceKind reduceKind = kind.getValue();
    bool resultIsI32 = mlir::vc4kernel::isVC4KernelVector16I32Type(resultType);
    bool resultIsF32 = mlir::vc4kernel::isVC4KernelVector16F32Type(resultType);
    if (!resultIsI32 && !resultIsF32)
      return op->emitOpError(
          "fragment_reduce lowering requires vector<16xi32> or vector<16xf32>");

    Value identity;
    bool unsignedMinMax =
        reduceKind == mlir::vc4kernel::ReduceKind::min_u ||
        reduceKind == mlir::vc4kernel::ReduceKind::max_u;
    if (resultIsI32) {
      if (reduceKind == mlir::vc4kernel::ReduceKind::fmin ||
          reduceKind == mlir::vc4kernel::ReduceKind::fmax)
        return op->emitOpError("f32-only reduce kind requires vector<16xf32>");
      int32_t identityValue = getI32ReduceIdentity(reduceKind);
      if (unsignedMinMax)
        identityValue ^= static_cast<int32_t>(0x80000000u);
      identity = createI32SplatConstant(builder, op->getLoc(), resultType,
                                        identityValue);
    } else {
      if (reduceKind != mlir::vc4kernel::ReduceKind::add)
        return op->emitOpError(
            "P4b fragment_reduce lowering supports only f32 add");
      identity = createZeroValue(builder, op->getLoc(), resultType);
    }

    if (unsignedMinMax) {
      Value bias = createI32SplatConstant(
          builder, op->getLoc(), resultType, static_cast<int32_t>(0x80000000u));
      input = createAddPipeBinary(builder, op->getLoc(), input, bias,
                                  mlir::vc4::AddOpcode::bit_xor);
    }

    FailureOr<Value> masked =
        emitPredicateSelect(op, builder, *predicate, input, identity);
    if (failed(masked))
      return failure();

    Value reduced = emitRotateReduceTree(builder, op->getLoc(), *masked,
                                         reduceKind);
    if (unsignedMinMax) {
      Value bias = createI32SplatConstant(
          builder, op->getLoc(), resultType, static_cast<int32_t>(0x80000000u));
      reduced = createAddPipeBinary(builder, op->getLoc(), reduced, bias,
                                    mlir::vc4::AddOpcode::bit_xor);
    }
    state.values[op->getResult(0)] = {reduced};
    return success();
  }
  if (hasName(op, kFragmentCmpOpName)) {
    Value lhs = mapValue(op, op->getOperand(0), state);
    Value rhs = mapValue(op, op->getOperand(1), state);
    if (!lhs || !rhs)
      return failure();
    FailureOr<CompareMaskLowering> compare =
        mapFragmentCmpPredicate(op, builder, lhs, rhs);
    if (failed(compare))
      return failure();
    Value one = createMaskConstant(builder, op->getLoc(), 1);
    Value zero = createMaskConstant(builder, op->getLoc(), 0);
    Value flags = createCompareFlags(builder, op->getLoc(), *compare);
    Value mask = createCondSelect(builder, op->getLoc(), flags, one, zero,
                                  compare->cond);
    state.predicates[op->getResult(0)] = PredicatePlan::generalMaskValue(mask);
    return success();
  }
  if (hasName(op, kFragmentSelectOpName)) {
    const PredicatePlan *predicate =
        lookupPredicatePlan(op->getOperand(0), state);
    if (!predicate)
      return op->emitOpError("predicate operand has no lowering plan");
    Value trueValue = mapValue(op, op->getOperand(1), state);
    Value falseValue = mapValue(op, op->getOperand(2), state);
    if (!trueValue || !falseValue)
      return failure();
    FailureOr<Value> selected =
        emitPredicateSelect(op, builder, *predicate, trueValue, falseValue);
    if (failed(selected))
      return failure();
    state.values[op->getResult(0)] = {*selected};
    return success();
  }
  if (hasName(op, kPredFullOpName)) {
    state.predicates[op->getResult(0)] = PredicatePlan::full();
    return success();
  }
  if (hasName(op, kPredEmptyOpName)) {
    state.predicates[op->getResult(0)] = PredicatePlan::empty();
    return success();
  }
  if (hasName(op, kPredTailOpName)) {
    Value base = mapValue(op, op->getOperand(0), state);
    Value limit = mapValue(op, op->getOperand(1), state);
    if (!base || !limit)
      return failure();
    state.predicates[op->getResult(0)] =
        PredicatePlan::tailPrefix(base, limit);
    return success();
  }
  if (hasName(op, kPredRectOpName)) {
    Value row = mapValue(op, op->getOperand(0), state);
    Value rows = mapValue(op, op->getOperand(1), state);
    Value colBase = mapValue(op, op->getOperand(2), state);
    Value cols = mapValue(op, op->getOperand(3), state);
    if (!row || !rows || !colBase || !cols)
      return failure();
    state.predicates[op->getResult(0)] =
        PredicatePlan::rectRow(row, rows, colBase, cols);
    return success();
  }
  if (hasName(op, kPredAndOpName) || hasName(op, kPredOrOpName)) {
    const PredicatePlan *lhs = lookupPredicatePlan(op->getOperand(0), state);
    const PredicatePlan *rhs = lookupPredicatePlan(op->getOperand(1), state);
    if (!lhs || !rhs)
      return op->emitOpError("predicate operand has no lowering plan");
    PredicatePlan result = PredicatePlan::generalMask();
    if (hasName(op, kPredAndOpName)) {
      if (lhs->kind == PredicatePlan::Class::Empty ||
          rhs->kind == PredicatePlan::Class::Empty)
        result = PredicatePlan::empty();
      else if (lhs->kind == PredicatePlan::Class::Full)
        result = *rhs;
      else if (rhs->kind == PredicatePlan::Class::Full)
        result = *lhs;
      else if (lhs->kind == PredicatePlan::Class::TailPrefix &&
               rhs->kind == PredicatePlan::Class::TailPrefix &&
               isSameSSAValue(lhs->base, rhs->base)) {
        Value limit = createI32Min(builder, op->getLoc(), lhs->limit,
                                   rhs->limit);
        result = PredicatePlan::tailPrefix(lhs->base, limit);
      }
      else {
        FailureOr<Value> lhsMask =
            materializePredicateMask(op, builder, *lhs, state);
        FailureOr<Value> rhsMask =
            materializePredicateMask(op, builder, *rhs, state);
        if (failed(lhsMask) || failed(rhsMask))
          return failure();
        Value mask = createI32BinaryAddPipe(builder, op->getLoc(), *lhsMask,
                                            *rhsMask,
                                            mlir::vc4::AddOpcode::bit_and);
        result = PredicatePlan::generalMaskValue(mask);
      }
    } else {
      if (lhs->kind == PredicatePlan::Class::Full ||
          rhs->kind == PredicatePlan::Class::Full)
        result = PredicatePlan::full();
      else if (lhs->kind == PredicatePlan::Class::Empty)
        result = *rhs;
      else if (rhs->kind == PredicatePlan::Class::Empty)
        result = *lhs;
      else if (lhs->kind == PredicatePlan::Class::TailPrefix &&
               rhs->kind == PredicatePlan::Class::TailPrefix &&
               isSameSSAValue(lhs->base, rhs->base)) {
        Value limit = createI32Max(builder, op->getLoc(), lhs->limit,
                                   rhs->limit);
        result = PredicatePlan::tailPrefix(lhs->base, limit);
      }
      else {
        FailureOr<Value> lhsMask =
            materializePredicateMask(op, builder, *lhs, state);
        FailureOr<Value> rhsMask =
            materializePredicateMask(op, builder, *rhs, state);
        if (failed(lhsMask) || failed(rhsMask))
          return failure();
        Value mask = createI32BinaryAddPipe(builder, op->getLoc(), *lhsMask,
                                            *rhsMask,
                                            mlir::vc4::AddOpcode::bit_or);
        result = PredicatePlan::generalMaskValue(mask);
      }
    }
    state.predicates[op->getResult(0)] = result;
    return success();
  }
  if (hasName(op, kPredNotOpName)) {
    const PredicatePlan *input = lookupPredicatePlan(op->getOperand(0), state);
    if (!input)
      return op->emitOpError("predicate operand has no lowering plan");
    PredicatePlan result = PredicatePlan::generalMask();
    if (input->kind == PredicatePlan::Class::Full)
      result = PredicatePlan::empty();
    else if (input->kind == PredicatePlan::Class::Empty)
      result = PredicatePlan::full();
    else {
      FailureOr<Value> inputMask =
          materializePredicateMask(op, builder, *input, state);
      if (failed(inputMask))
        return failure();
      Value one = createMaskConstant(builder, op->getLoc(), 1);
      Value zero = createMaskConstant(builder, op->getLoc(), 0);
      Value flags = createZeroTestFlags(builder, op->getLoc(), *inputMask);
      Value mask = createCondSelect(builder, op->getLoc(), flags, one, zero,
                                    mlir::vc4::Cond::zs);
      result = PredicatePlan::generalMaskValue(mask);
    }
    state.predicates[op->getResult(0)] = result;
    return success();
  }
  if (hasName(op, kPredAnyOpName) || hasName(op, kPredAllOpName)) {
    if (!lookupPredicatePlan(op->getOperand(0), state))
      return op->emitOpError("predicate operand has no lowering plan");
    state.conditions[op->getResult(0)] =
        hasName(op, kPredAnyOpName)
            ? ConditionPlan::predicateAny(op->getOperand(0))
            : ConditionPlan::predicateAll(op->getOperand(0));
    return success();
  }
  if (hasName(op, kTMULoadOpName)) {
    const PredicatePlan *predicate = lookupPredicatePlan(op->getOperand(2),
                                                         state);
    if (!predicate)
      return op->emitOpError("predicate operand has no lowering plan");
    if (predicate->kind != PredicatePlan::Class::Full &&
        predicate->kind != PredicatePlan::Class::Empty &&
        predicate->kind != PredicatePlan::Class::TailPrefix &&
        predicate->kind != PredicatePlan::Class::RectRow &&
        predicate->kind != PredicatePlan::Class::GeneralMask)
      return op->emitOpError()
             << "tmu_load_fragment lowering currently supports only "
                "pred.full, pred.empty, pred.tail, pred.rect, and "
                "fragment_cmp general masks";
    Value base = mapValue(op, op->getOperand(0), state);
    Value offsets = mapValue(op, op->getOperand(1), state);
    if (!base || !offsets)
      return failure();

    if (predicate->kind == PredicatePlan::Class::Full) {
      state.values[op->getResult(0)] = {
          emitTMULoadFragment(op, builder, base, offsets,
                              op->getResult(0).getType())};
      return success();
    }

    Value zero = createZeroValue(builder, op->getLoc(), op->getResult(0).getType());
    if (predicate->kind == PredicatePlan::Class::Empty) {
      state.values[op->getResult(0)] = {zero};
      return success();
    }

    FullRowVDWOffsets sourceOffsets =
        matchFullRowVDWByteOffsets(op->getOperand(1));
    if (!sourceOffsets.matched)
      return op->emitOpError(
          "tmu_load_fragment predicated lowering currently supports only "
          "contiguous byte_offsets = base_byte_offset + 4*lane_range so "
          "inactive lanes can use a proven in-bounds safe address");
    Value zeroOffset = createLoadImm(builder, op->getLoc(),
                                     builder.getI32Type(),
                                     builder.getI32IntegerAttr(0));
    Value safeScalarOffset = zeroOffset;
    if (sourceOffsets.scalarBaseByteOffset) {
      safeScalarOffset = mapValue(op, sourceOffsets.scalarBaseByteOffset, state);
      if (!safeScalarOffset)
        return failure();
    } else if (sourceOffsets.constantBaseByteOffset) {
      safeScalarOffset = createLoadImm(
          builder, op->getLoc(), builder.getI32Type(),
          builder.getI32IntegerAttr(*sourceOffsets.constantBaseByteOffset));
    }

    Value lane = createOpWithResult(builder, op->getLoc(),
                                    kSSAVC4ElementNumberOpName, {}, {},
                                    offsets.getType());
    Value safeOffsetVec = createOpWithResult(
        builder, op->getLoc(), kSSAVC4SplatOpName, safeScalarOffset, {},
        offsets.getType());

    auto emitTailLoad = [&]() -> LogicalResult {
      Value tailSafeOffsetVec = createOpWithResult(
          builder, op->getLoc(), kSSAVC4SplatOpName, zeroOffset, {},
          offsets.getType());
      FailureOr<Value> safeOffsets =
          emitPredicateSelect(op, builder, *predicate, offsets,
                              tailSafeOffsetVec);
      if (failed(safeOffsets))
        return failure();
      Value loaded = emitTMULoadFragment(op, builder, base, *safeOffsets,
                                         op->getResult(0).getType());
      FailureOr<Value> masked =
          emitPredicateSelect(op, builder, *predicate, loaded, zero);
      if (failed(masked))
        return failure();
      state.values[op->getResult(0)] = {*masked};
      return success();
    };

    auto emitRectLoad = [&]() -> LogicalResult {
      Value row = predicate->row;
      Value rows = predicate->rows;
      Value colBase = predicate->colBase;
      Value cols = predicate->cols;
      if (!row || !rows || !colBase || !cols)
        return op->emitOpError(
            "pred.rect plan is missing row, rows, col_base, or cols values");

      Value one =
          createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                        builder.getI32IntegerAttr(1));
      Value two =
          createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                        builder.getI32IntegerAttr(2));
      Value sixteen =
          createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                        builder.getI32IntegerAttr(16));
      Value rowPlusOne = createI32Add(builder, op->getLoc(), row, one);
      Value rowActiveFlags =
          createSubFlags(builder, op->getLoc(), rows, rowPlusOne);

      Region *region = builder.getInsertionBlock()->getParent();
      Block *colsBlock = new Block();
      Block *colBaseBlock = new Block();
      Block *loadBlock = new Block();
      Block *doneBlock = new Block();
      doneBlock->addArgument(op->getResult(0).getType(), op->getLoc());
      region->push_back(colsBlock);
      region->push_back(colBaseBlock);
      region->push_back(loadBlock);
      region->push_back(doneBlock);

      createCondBranch(builder, op->getLoc(), rowActiveFlags, doneBlock,
                       colsBlock, mlir::vc4::BranchCond::any_c_set, zero);

      builder.setInsertionPointToEnd(colsBlock);
      Value colsActiveFlags =
          createSubFlags(builder, op->getLoc(), cols, one);
      createCondBranch(builder, op->getLoc(), colsActiveFlags, doneBlock,
                       colBaseBlock, mlir::vc4::BranchCond::any_c_set, zero);

      builder.setInsertionPointToEnd(colBaseBlock);
      Value colBaseInRangeFlags =
          createSubFlags(builder, op->getLoc(), colBase, sixteen);
      createCondBranch(builder, op->getLoc(), colBaseInRangeFlags, doneBlock,
                       loadBlock, mlir::vc4::BranchCond::any_c_clear, zero);

      builder.setInsertionPointToEnd(loadBlock);
      Value colBaseBytes =
          createOpWithResult(builder, op->getLoc(), kSSAVC4ALUAddOpName,
                             {colBase, two},
                             {builder.getNamedAttr(
                                 "opcode", mlir::vc4::AddOpcodeAttr::get(
                                               builder.getContext(),
                                               mlir::vc4::AddOpcode::shl))},
                             colBase.getType());
      Value rectSafeScalar =
          createI32Add(builder, op->getLoc(), safeScalarOffset, colBaseBytes);
      Value rectSafeVec =
          createOpWithResult(builder, op->getLoc(), kSSAVC4SplatOpName,
                             rectSafeScalar, {}, offsets.getType());
      Value colBaseVec =
          createOpWithResult(builder, op->getLoc(), kSSAVC4SplatOpName,
                             colBase, {}, offsets.getType());
      Value colEnd = createI32Add(builder, op->getLoc(), colBase, cols);
      Value colEndVec =
          createOpWithResult(builder, op->getLoc(), kSSAVC4SplatOpName, colEnd,
                             {}, offsets.getType());
      Value upperFlags = createSubFlags(builder, op->getLoc(), lane, colEndVec);
      Value upperSafe =
          createCondSelect(builder, op->getLoc(), upperFlags, offsets,
                           rectSafeVec, mlir::vc4::Cond::cs);
      Value lowerFlags =
          createSubFlags(builder, op->getLoc(), lane, colBaseVec);
      Value safeOffsets =
          createCondSelect(builder, op->getLoc(), lowerFlags, upperSafe,
                           rectSafeVec, mlir::vc4::Cond::cc);
      Value loaded = emitTMULoadFragment(op, builder, base, safeOffsets,
                                         op->getResult(0).getType());
      Value resultUpperFlags =
          createSubFlags(builder, op->getLoc(), lane, colEndVec);
      Value upperMasked =
          createCondSelect(builder, op->getLoc(), resultUpperFlags, loaded,
                           zero, mlir::vc4::Cond::cs);
      Value resultLowerFlags =
          createSubFlags(builder, op->getLoc(), lane, colBaseVec);
      Value masked =
          createCondSelect(builder, op->getLoc(), resultLowerFlags,
                           upperMasked, zero, mlir::vc4::Cond::cc);
      createBranch(builder, op->getLoc(), doneBlock, masked);

      builder.setInsertionPointToEnd(doneBlock);
      state.values[op->getResult(0)] = {doneBlock->getArgument(0)};
      return success();
    };

    if (predicate->kind == PredicatePlan::Class::TailPrefix)
      return emitTailLoad();
    if (predicate->kind == PredicatePlan::Class::RectRow)
      return emitRectLoad();
    if (predicate->kind == PredicatePlan::Class::GeneralMask) {
      FailureOr<Value> safeOffsets =
          emitPredicateSelect(op, builder, *predicate, offsets, safeOffsetVec);
      if (failed(safeOffsets))
        return failure();
      Value loaded = emitTMULoadFragment(op, builder, base, *safeOffsets,
                                         op->getResult(0).getType());
      FailureOr<Value> masked =
          emitPredicateSelect(op, builder, *predicate, loaded, zero);
      if (failed(masked))
        return failure();
      state.values[op->getResult(0)] = {*masked};
      return success();
    }
    return op->emitOpError("unsupported tmu_load_fragment predicate plan");
  }
  if (hasName(op, kVDWStoreOpName)) {
    FullRowVDWOffsets offsets = matchFullRowVDWByteOffsets(op->getOperand(1));
    if (!offsets.matched)
      return op->emitOpError(
          "vdw_store_fragment lowering currently supports only contiguous "
          "byte_offsets = base_byte_offset + 4*lane_range");
    const PredicatePlan *predicate = lookupPredicatePlan(op->getOperand(3),
                                                         state);
    if (!predicate)
      return op->emitOpError("predicate operand has no lowering plan");
    if (predicate->kind != PredicatePlan::Class::Full &&
        predicate->kind != PredicatePlan::Class::Empty &&
        predicate->kind != PredicatePlan::Class::TailPrefix &&
        predicate->kind != PredicatePlan::Class::GeneralMask)
      return op->emitOpError()
             << "vdw_store_fragment lowering currently supports only "
                "pred.full, pred.empty, pred.tail, and fragment_cmp general "
                "masks";
    Value base = mapValue(op, op->getOperand(0), state);
    Value mappedOffsets = mapValue(op, op->getOperand(1), state);
    Value value = mapValue(op, op->getOperand(2), state);
    if (!base || !mappedOffsets || !value)
      return failure();
    Value memoryBase = base;
    if (offsets.scalarBaseByteOffset) {
      Value mappedOffset =
          mapValue(op, offsets.scalarBaseByteOffset, state);
      if (!mappedOffset)
        return failure();
      base = createI32Add(builder, op->getLoc(), base, mappedOffset);
    } else if (offsets.constantBaseByteOffset) {
      Value mappedOffset = createLoadImm(
          builder, op->getLoc(), builder.getI32Type(),
          builder.getI32IntegerAttr(*offsets.constantBaseByteOffset));
      base = createI32Add(builder, op->getLoc(), base, mappedOffset);
    }
    if (predicate->kind == PredicatePlan::Class::Empty)
      return success();
    Value sixteen =
        createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                      builder.getI32IntegerAttr(16));
    if (predicate->kind == PredicatePlan::Class::Full) {
      emitVDWStore(op, builder, base, value, sixteen, state,
                   /*staticActiveLanes=*/16);
      return success();
    }
    if (predicate->kind == PredicatePlan::Class::GeneralMask) {
      Value oldValue = emitTMULoadFragment(op, builder, memoryBase,
                                           mappedOffsets, value.getType());
      FailureOr<Value> merged =
          emitPredicateSelect(op, builder, *predicate, value, oldValue);
      if (failed(merged))
        return failure();
      emitVDWStore(op, builder, base, *merged, sixteen, state,
                   /*staticActiveLanes=*/16);
      return success();
    }

    Value baseIndex = predicate->base;
    Value limit = predicate->limit;
    if (!baseIndex || !limit)
      return op->emitOpError("pred.tail plan is missing base or limit values");

    Value one =
        createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                      builder.getI32IntegerAttr(1));
    Value basePlusOne = createI32Add(builder, op->getLoc(), baseIndex, one);
    Value basePlusSixteen =
        createI32Add(builder, op->getLoc(), baseIndex, sixteen);
    Value activeTail = createI32Sub(builder, op->getLoc(), limit, baseIndex);
    Value nonEmptyFlags =
        createSubFlags(builder, op->getLoc(), limit, basePlusOne);

    Region *region = builder.getInsertionBlock()->getParent();
    Block *nonEmptyBlock = new Block();
    Block *tailBlock = new Block();
    Block *fullBlock = new Block();
    Block *doneBlock = new Block();
    region->push_back(nonEmptyBlock);
    region->push_back(tailBlock);
    region->push_back(fullBlock);
    region->push_back(doneBlock);

    createCondBranch(builder, op->getLoc(), nonEmptyFlags, doneBlock,
                     nonEmptyBlock, mlir::vc4::BranchCond::any_c_set);

    builder.setInsertionPointToEnd(nonEmptyBlock);
    Value fullFlags =
        createSubFlags(builder, op->getLoc(), limit, basePlusSixteen);
    createCondBranch(builder, op->getLoc(), fullFlags, fullBlock, tailBlock,
                     mlir::vc4::BranchCond::any_c_clear);

    builder.setInsertionPointToEnd(tailBlock);
    emitVDWStore(op, builder, base, value, activeTail, state,
                 /*staticActiveLanes=*/std::nullopt);
    createBranch(builder, op->getLoc(), doneBlock);

    builder.setInsertionPointToEnd(fullBlock);
    emitVDWStore(op, builder, base, value, sixteen, state,
                 /*staticActiveLanes=*/16);
    createBranch(builder, op->getLoc(), doneBlock);

    builder.setInsertionPointToEnd(doneBlock);
    return success();
  }
  if (hasName(op, kVPMAllocOpName)) {
    int64_t rows = 0;
    if (auto rowsAttr = op->getAttrOfType<IntegerAttr>("rows"))
      rows = rowsAttr.getInt();
    int64_t elemBytes = 4;
    if (auto elemBytesAttr = op->getAttrOfType<IntegerAttr>("elem_bytes"))
      elemBytes = elemBytesAttr.getInt();
    state.vpmAllocations[op->getResult(0)] = {
        state.nextVPMRowOffset, rows, elemBytes};
    state.nextVPMRowOffset += rows;
    return success();
  }
  if (hasName(op, kVPMWriteOpName)) {
    Value row = mapValue(op, op->getOperand(1), state);
    Value value = mapValue(op, op->getOperand(2), state);
    if (!row || !value)
      return failure();
    FailureOr<Value> zeroFilled =
        applyVPMPredicateZeroFill(op, builder, op->getOperand(3), value, state);
    if (failed(zeroFilled))
      return failure();
    value = *zeroFilled;
    FailureOr<Value> plannedRow =
        applyVPMTileBase(op, builder, op->getOperand(0), row, state);
    if (failed(plannedRow))
      return failure();
    row = *plannedRow;
    createOp(builder, op->getLoc(), kSSAVC4VPMWriteOpName, {row, value},
             {getSSAVC4VPMOrientation(builder, op),
              getSSAVC4VPMWidth(builder, op),
              getSSAVC4VPMSubword(builder, op),
              builder.getNamedAttr("x", op->getAttr("x")),
              builder.getNamedAttr("stride", op->getAttr("stride")),
              builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16))});
    return success();
  }
  if (hasName(op, kVPMReadOpName)) {
    Value row = mapValue(op, op->getOperand(1), state);
    if (!row)
      return failure();
    FailureOr<Value> plannedRow =
        applyVPMTileBase(op, builder, op->getOperand(0), row, state);
    if (failed(plannedRow))
      return failure();
    row = *plannedRow;
    Value read = createOpWithResult(
        builder, op->getLoc(), kSSAVC4VPMReadOpName, row,
        {getSSAVC4VPMOrientation(builder, op),
         getSSAVC4VPMWidth(builder, op),
         getSSAVC4VPMSubword(builder, op),
         builder.getNamedAttr("x", op->getAttr("x")),
         builder.getNamedAttr("stride", op->getAttr("stride")),
         builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16))},
        op->getResult(0).getType());
    FailureOr<Value> zeroFilled =
        applyVPMPredicateZeroFill(op, builder, op->getOperand(2), read, state);
    if (failed(zeroFilled))
      return failure();
    state.values[op->getResult(0)] = {*zeroFilled};
    return success();
  }
  if (hasName(op, kVDRLoadOpName)) {
    Value base = mapValue(op, op->getOperand(0), state);
    Value byteOffset = mapValue(op, op->getOperand(1), state);
    Value dstRow = mapValue(op, op->getOperand(3), state);
    if (!base || !byteOffset || !dstRow)
      return failure();
    Value address = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, {base, byteOffset},
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(),
                                            mlir::vc4::AddOpcode::add))},
        base.getType());
    FailureOr<Value> plannedDstRow =
        applyVPMTileBase(op, builder, op->getOperand(2), dstRow, state);
    if (failed(plannedDstRow))
      return failure();
    dstRow = *plannedDstRow;
    createOp(builder, op->getLoc(), kSSAVC4VDRLoadOpName, {address, dstRow},
             {getSSAVC4VPMOrientation(builder, op),
              getSSAVC4VPMWidth(builder, op),
              getSSAVC4VPMSubword(builder, op),
              builder.getNamedAttr("row_len", op->getAttr("cols")),
              builder.getNamedAttr("nrows", op->getAttr("rows")),
              builder.getNamedAttr("memory_pitch_bytes",
                                   op->getAttr("global_stride_bytes")),
              builder.getNamedAttr("vpm_x", op->getAttr("dst_x")),
              builder.getNamedAttr("vpm_pitch", op->getAttr("vpm_pitch")),
              builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))});
    return success();
  }
  if (hasName(op, kVDRLoadRectOpName)) {
    Value base = mapValue(op, op->getOperand(0), state);
    Value byteOffset = mapValue(op, op->getOperand(1), state);
    Value dstRow = mapValue(op, op->getOperand(3), state);
    Value activeRows = mapValue(op, op->getOperand(4), state);
    Value activeCols = mapValue(op, op->getOperand(5), state);
    Value memoryPitchBytes = mapValue(op, op->getOperand(6), state);
    if (!base || !byteOffset || !dstRow || !activeRows || !activeCols ||
        !memoryPitchBytes)
      return failure();
    Value address = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, {base, byteOffset},
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(),
                                            mlir::vc4::AddOpcode::add))},
        base.getType());
    FailureOr<Value> plannedDstRow =
        applyVPMTileBase(op, builder, op->getOperand(2), dstRow, state);
    if (failed(plannedDstRow))
      return failure();
    dstRow = *plannedDstRow;
    createOp(builder, op->getLoc(), kSSAVC4VDRLoadRectDynamicOpName,
             {address, dstRow, activeRows, activeCols, memoryPitchBytes},
             {getSSAVC4VPMOrientation(builder, op),
              getSSAVC4VPMWidth(builder, op),
              getSSAVC4VPMSubword(builder, op),
              builder.getNamedAttr("max_rows", op->getAttr("max_rows")),
              builder.getNamedAttr("max_cols", op->getAttr("max_cols")),
              builder.getNamedAttr("elem_bytes", op->getAttr("elem_bytes")),
              builder.getNamedAttr("dst_x", op->getAttr("dst_x")),
              builder.getNamedAttr("vpm_pitch", op->getAttr("vpm_pitch")),
              builder.getNamedAttr("zero_fill", builder.getBoolAttr(true)),
              builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))});
    return success();
  }
  if (hasName(op, kVDWStoreVPMOpName)) {
    const PredicatePlan *predicate = lookupPredicatePlan(op->getOperand(4),
                                                         state);
    if (!predicate)
      return op->emitOpError("predicate operand has no lowering plan");
    if (predicate->kind == PredicatePlan::Class::Empty)
      return success();
    if (predicate->kind != PredicatePlan::Class::Full &&
        predicate->kind != PredicatePlan::Class::TailPrefix)
      return op->emitOpError()
             << "vdw_store_vpm_fragment lowering supports pred.full, "
                "pred.empty, and pred.tail; general-mask predicates are "
                "rejected by the vc4kernel verifier until a preserve-"
                "destination VPM fallback is implemented";
    Value srcRow = mapValue(op, op->getOperand(1), state);
    Value base = mapValue(op, op->getOperand(2), state);
    Value byteOffset = mapValue(op, op->getOperand(3), state);
    if (!srcRow || !base || !byteOffset)
      return failure();
    Value address = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, {base, byteOffset},
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(),
                                            mlir::vc4::AddOpcode::add))},
        base.getType());
    FailureOr<Value> plannedSrcRow =
        applyVPMTileBase(op, builder, op->getOperand(0), srcRow, state);
    if (failed(plannedSrcRow))
      return failure();
    srcRow = *plannedSrcRow;
    if (predicate->kind == PredicatePlan::Class::TailPrefix) {
      if (!predicate->base || !predicate->limit)
        return op->emitOpError("pred.tail plan is missing base or limit values");
      Type fragmentType = VectorType::get({16}, builder.getI32Type());
      Value fragment = createOpWithResult(
          builder, op->getLoc(), kSSAVC4VPMReadOpName, srcRow,
          {getSSAVC4VPMOrientation(builder, op),
           getSSAVC4VPMWidth(builder, op),
           getSSAVC4VPMSubword(builder, op),
           builder.getNamedAttr("x", op->getAttr("src_x")),
           builder.getNamedAttr("stride", op->getAttr("vpm_pitch")),
           builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16))},
          fragmentType);
      Value activeTail =
          createI32Sub(builder, op->getLoc(), predicate->limit, predicate->base);
      emitVDWStore(op, builder, address, fragment, activeTail, state,
                   std::nullopt);
      return success();
    }
    auto srcX = llvm::dyn_cast_or_null<IntegerAttr>(op->getAttr("src_x"));
    Value vpmX = createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                               builder.getI32IntegerAttr(srcX ? srcX.getInt() : 0));
    SmallVector<Value, 4> operands{address, srcRow, vpmX};
    SmallVector<NamedAttribute, 8> attrs{
        getSSAVC4VPMOrientation(builder, op),
        getSSAVC4VPMWidth(builder, op),
        getSSAVC4VPMSubword(builder, op),
        builder.getNamedAttr("row_len", builder.getI32IntegerAttr(16)),
        builder.getNamedAttr("nrows", builder.getI32IntegerAttr(1)),
        builder.getNamedAttr("memory_pitch_bytes",
                             builder.getI32IntegerAttr(64)),
        builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))};
    if (predicate->kind == PredicatePlan::Class::Full) {
      attrs.push_back(builder.getNamedAttr(
          "active_lanes", builder.getI32IntegerAttr(16)));
    }
    createOp(builder, op->getLoc(), kSSAVC4VDWStoreVPMOpName,
             operands, attrs);
    return success();
  }
  if (hasName(op, kVDWStoreRectOpName)) {
    Value srcRow = mapValue(op, op->getOperand(1), state);
    Value base = mapValue(op, op->getOperand(2), state);
    Value byteOffset = mapValue(op, op->getOperand(3), state);
    Value activeRows = mapValue(op, op->getOperand(4), state);
    Value activeCols = mapValue(op, op->getOperand(5), state);
    Value memoryStrideBytes = mapValue(op, op->getOperand(6), state);
    if (!srcRow || !base || !byteOffset || !activeRows || !activeCols ||
        !memoryStrideBytes)
      return failure();
    Value address = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, {base, byteOffset},
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(),
                                            mlir::vc4::AddOpcode::add))},
        base.getType());
    FailureOr<Value> plannedSrcRow =
        applyVPMTileBase(op, builder, op->getOperand(0), srcRow, state);
    if (failed(plannedSrcRow))
      return failure();
    srcRow = *plannedSrcRow;
    createOp(builder, op->getLoc(), kSSAVC4VDWStoreRectDynamicOpName,
             {address, srcRow, activeRows, activeCols, memoryStrideBytes},
             {getSSAVC4VPMOrientation(builder, op),
              getSSAVC4VPMWidth(builder, op),
              getSSAVC4VPMSubword(builder, op),
              builder.getNamedAttr("max_rows", op->getAttr("max_rows")),
              builder.getNamedAttr("max_cols", op->getAttr("max_cols")),
              builder.getNamedAttr("elem_bytes", op->getAttr("elem_bytes")),
              builder.getNamedAttr("src_x", op->getAttr("src_x")),
              builder.getNamedAttr("vpm_pitch", op->getAttr("vpm_pitch")),
              builder.getNamedAttr("preserve_inactive",
                                   builder.getBoolAttr(true)),
              builder.getNamedAttr("serialize", builder.getStringAttr("mutex"))});
    return success();
  }
  if (hasName(op, kBarrierOpName)) {
    Value logicalWarp = state.launchABI.lookupBuiltin("logical_warp_id");
    Value warpsPerBlock = state.launchABI.lookupBuiltin("warps_per_block");
    Value semaphoreBase = state.launchABI.lookupBuiltin("semaphore_base");
    if (!logicalWarp || !warpsPerBlock || !semaphoreBase)
      return op->emitOpError(
          "missing launch builtin uniforms for cooperative barrier");
    createOp(builder, op->getLoc(), kSSAVC4BarrierOpName,
             {logicalWarp, warpsPerBlock, semaphoreBase},
             {builder.getNamedAttr("arrive_offset", builder.getI32IntegerAttr(0)),
              builder.getNamedAttr("go_offset", builder.getI32IntegerAttr(1)),
              builder.getNamedAttr("depart_offset", builder.getI32IntegerAttr(2)),
              builder.getNamedAttr("reset_offset", builder.getI32IntegerAttr(3))});
    return success();
  }
  return op->emitOpError("is not implemented by vc4kernel -> ssavc4 lowering");
}

static LogicalResult lowerTerminator(Operation *op, OpBuilder &builder,
                                     LoweringState &state) {
  if (hasName(op, kReturnOpName)) {
    createOp(builder, op->getLoc(), kSSAVC4ThreadEndOpName, {}, {});
    return success();
  }
  if (auto br = dyn_cast<cf::BranchOp>(op)) {
    SmallVector<Value, 4> operands;
    for (Value operand : br.getDestOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    OperationState branchState(op->getLoc(), kSSAVC4BranchOpName);
    branchState.addOperands(operands);
    branchState.addSuccessors(state.blockMap.lookup(br.getDest()));
    builder.create(branchState);
    return success();
  }
  if (auto cond = dyn_cast<cf::CondBranchOp>(op)) {
    auto conditionIt = state.conditions.find(cond.getCondition());
    if (conditionIt == state.conditions.end())
      return op->emitOpError("condition operand has no lowering plan");
    SmallVector<Value, 4> trueOperands;
    for (Value operand : cond.getTrueDestOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      trueOperands.push_back(mapped);
    }
    SmallVector<Value, 4> falseOperands;
    for (Value operand : cond.getFalseDestOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      falseOperands.push_back(mapped);
    }
    Block *trueDest = state.blockMap.lookup(cond.getTrueDest());
    Block *falseDest = state.blockMap.lookup(cond.getFalseDest());
    FailureOr<EmittedCondition> emitted =
        emitConditionPlan(op, builder, conditionIt->second, state);
    if (failed(emitted))
      return failure();
    if (emitted->isConstant) {
      createBranch(builder, op->getLoc(),
                   emitted->constantValue ? trueDest : falseDest,
                   emitted->constantValue ? ValueRange(trueOperands)
                                          : ValueRange(falseOperands));
      return success();
    }
    createCondBranch(builder, op->getLoc(), emitted->flags, trueDest, falseDest,
                     emitted->branchCond, trueOperands, falseOperands);
    return success();
  }
  return op->emitOpError("unsupported terminator");
}

static LogicalResult lowerKernel(Operation *kernel, Operation *func) {
  Region &source = kernel->getRegion(0);
  Region &dest = func->getRegion(0);
  LoweringState state({computeVC4KernelResourceSummary(kernel)});
  OpBuilder builder(func->getContext());

  for (Block &sourceBlock : source) {
    Block *destBlock = new Block();
    dest.push_back(destBlock);
    state.blockMap[&sourceBlock] = destBlock;
    if (&sourceBlock != &source.front()) {
      for (BlockArgument arg : sourceBlock.getArguments())
        state.values[arg] = {destBlock->addArgument(arg.getType(),
                                                    arg.getLoc())};
    }
  }

  builder.setInsertionPointToStart(state.blockMap.lookup(&source.front()));
  int64_t nextUniform = 0;
  for (BlockArgument arg : source.front().getArguments())
    state.values[arg] = {createUniformRead(builder, arg.getLoc(),
                                           arg.getType(), nextUniform++)};
  auto materializeBuiltin = [&](StringRef name) {
    state.launchABI.bindBuiltin(
        name, createUniformRead(builder, kernel->getLoc(),
                                builder.getI32Type(), nextUniform++));
  };
  appendRequiredBuiltins(kernel, state.resourcePlan.summary, materializeBuiltin);

  for (Block &sourceBlock : source) {
    Block *destBlock = state.blockMap.lookup(&sourceBlock);
    builder.setInsertionPointToEnd(destBlock);
    for (Operation &nested : sourceBlock) {
      if (nested.hasTrait<OpTrait::IsTerminator>()) {
        if (failed(lowerTerminator(&nested, builder, state)))
          return failure();
        continue;
      }
      if (failed(lowerBodyOp(&nested, builder, state)))
        return failure();
    }
  }
  return success();
}

struct VerifyVC4KernelPass
    : public PassWrapper<VerifyVC4KernelPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyVC4KernelPass)

  StringRef getArgument() const override { return "verify-vc4kernel"; }
  StringRef getDescription() const override {
    return "Verify strict vc4kernel dialect boundary and metadata";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::vc4kernel::VC4KernelDialect>();
  }
  void runOnOperation() override {
    if (failed(verifyVC4Kernel(getOperation())))
      signalPassFailure();
  }
};

struct ConvertVC4KernelToSSAVC4Pass
    : public PassWrapper<ConvertVC4KernelToSSAVC4Pass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVC4KernelToSSAVC4Pass)

  StringRef getArgument() const override {
    return "convert-vc4kernel-to-ssavc4";
  }
  StringRef getDescription() const override {
    return "Lower strict VC4Kernel kernels to SSAVC4";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::vc4::VC4Dialect, mlir::ssavc4::SSAVC4Dialect,
                    mlir::vc4kernel::VC4KernelDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyVC4Kernel(module))) {
      signalPassFailure();
      return;
    }

    SmallVector<Operation *, 4> kernels;
    for (Operation &op : module.getBody()->getOperations())
      if (hasName(&op, kKernelOpName))
        kernels.push_back(&op);
    if (kernels.empty())
      return;

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    Operation *ssavc4Module = createSSAVC4Module(kernels.front(), builder);
    for (Operation *kernel : kernels) {
      Operation *func = createSSAVC4Func(kernel, ssavc4Module, builder);
      if (failed(lowerKernel(kernel, func))) {
        signalPassFailure();
        return;
      }
    }
    for (Operation *kernel : kernels)
      kernel->erase();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vc4::createVerifyVC4KernelPass() {
  return std::make_unique<VerifyVC4KernelPass>();
}

std::unique_ptr<Pass> mlir::vc4::createConvertVC4KernelToSSAVC4Pass() {
  return std::make_unique<ConvertVC4KernelToSSAVC4Pass>();
}

void mlir::vc4::registerConvertVC4KernelToSSAVC4Pass() {
  // This translation unit is linked into vc4-opt for vc4kernel staging, and
  // the file-scope PassRegistration objects below install the pass flags.
}

static PassRegistration<VerifyVC4KernelPass> registerVerifyVC4KernelPass;
static PassRegistration<ConvertVC4KernelToSSAVC4Pass>
    registerVC4KernelToSSAVC4Pass;
