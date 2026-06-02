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

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <string>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kKernelOpName("vc4kernel.kernel");
constexpr llvm::StringLiteral kReturnOpName("vc4kernel.return");
constexpr llvm::StringLiteral kProgramIdOpName("vc4kernel.program_id");
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
constexpr llvm::StringLiteral kFragmentAddOpName("vc4kernel.fragment_add");
constexpr llvm::StringLiteral kFragmentSubOpName("vc4kernel.fragment_sub");
constexpr llvm::StringLiteral kFragmentMulOpName("vc4kernel.fragment_mul");
constexpr llvm::StringLiteral kFragmentShlOpName("vc4kernel.fragment_shl");
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
constexpr llvm::StringLiteral kVDWStoreVPMOpName(
    "vc4kernel.vdw_store_vpm_fragment");
constexpr llvm::StringLiteral kBarrierOpName("vc4kernel.barrier");

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
constexpr llvm::StringLiteral kSSAVC4RotateOpName("ssavc4.rotate");
constexpr llvm::StringLiteral kSSAVC4TMURequestOpName("ssavc4.tmu.request");
constexpr llvm::StringLiteral kSSAVC4TMUReadOpName("ssavc4.tmu.read");
constexpr llvm::StringLiteral kSSAVC4VDWStoreOpName("ssavc4.vdw.store");
constexpr llvm::StringLiteral kSSAVC4VPMWriteOpName("ssavc4.vpm.write");
constexpr llvm::StringLiteral kSSAVC4VPMReadOpName("ssavc4.vpm.read");
constexpr llvm::StringLiteral kSSAVC4VDRLoadOpName("ssavc4.vdr.load");
constexpr llvm::StringLiteral kSSAVC4VDWStoreVPMOpName("ssavc4.vdw.store_vpm");
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
      "vc4kernel.fragment_add",
      "vc4kernel.fragment_sub",
      "vc4kernel.fragment_mul",
      "vc4kernel.fragment_shl",
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
      "vc4kernel.vdw_store_vpm_fragment",
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
  return {};
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
  if (kernelContains(kernel, kProgramIdOpName)) {
    append("logical_request");
    append("total_requests");
  }
  if (kernelContains(kernel, kWarpIdOpName) || summary.uses_barrier)
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
    if (hasName(op, kVDWStoreVPMOpName)) {
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
  static ConditionPlan flagsValue(Value flags) {
    ConditionPlan plan;
    plan.kind = Class::FlagsValue;
    plan.flags = flags;
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

static LogicalResult requireFullPredicate(Operation *op, Value pred,
                                          LoweringState &state,
                                          StringRef consumerName) {
  const PredicatePlan *plan = lookupPredicatePlan(pred, state);
  if (!plan)
    return op->emitOpError("predicate operand has no lowering plan");
  if (!plan->isFull())
    return op->emitOpError()
           << consumerName << " lowering currently supports only pred.full";
  return success();
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

static bool isLaneByteOffsets(Value value) {
  Operation *def = value.getDefiningOp();
  if (!hasName(def, kFragmentShlOpName) || def->getNumOperands() != 2)
    return false;
  if (!hasName(def->getOperand(0).getDefiningOp(), kLaneRangeOpName))
    return false;
  std::optional<int64_t> shift = getI32ConstantValue(def->getOperand(1));
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
};

static FullRowVDWOffsets matchFullRowVDWByteOffsets(Value value) {
  if (isLaneByteOffsets(value))
    return {/*matched=*/true, /*scalarBaseByteOffset=*/{}};

  Operation *def = value.getDefiningOp();
  if (!hasName(def, kFragmentAddOpName) || def->getNumOperands() != 2)
    return {};

  auto matchBasePlusLaneBytes = [](Value lhs,
                                   Value rhs) -> FullRowVDWOffsets {
    Value scalarBase = getSplatScalar(lhs);
    if (scalarBase && isLaneByteOffsets(rhs))
      return {/*matched=*/true, scalarBase};
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

struct CompareMaskLowering {
  Value lhs;
  Value rhs;
  mlir::vc4::Cond cond = mlir::vc4::Cond::zs;
};

static FailureOr<CompareMaskLowering>
mapFragmentCmpPredicate(Operation *op, Value lhs, Value rhs) {
  auto attr =
      llvm::dyn_cast_or_null<mlir::vc4kernel::CmpPredicateAttr>(
          op->getAttr("predicate"));
  if (!attr)
    return op->emitOpError("fragment_cmp is missing predicate attribute");

  CompareMaskLowering result;
  switch (attr.getValue()) {
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
  }
  return result;
}

static Value createZeroValue(OpBuilder &builder, Location loc, Type type) {
  return createLoadImm(builder, loc, type, builder.getI32IntegerAttr(0));
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
  if (Value vpmBase = state.launchABI.lookupBuiltin("vpm_base_row")) {
    operands.push_back(vpmBase);
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

static LogicalResult lowerBodyOp(Operation *op, OpBuilder &builder,
                                 LoweringState &state) {
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    state.values[cst.getResult()] = {createLoadImm(
        builder, op->getLoc(), cst.getType(), cst.getValue())};
    if (cst.getType().isInteger(1)) {
      if (auto value = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
        state.conditions[cst.getResult()] =
            ConditionPlan::constant(!value.getValue().isZero());
    }
    return success();
  }
  if (hasName(op, kProgramIdOpName) || hasName(op, kWarpIdOpName)) {
    StringRef name = hasName(op, kProgramIdOpName)   ? "logical_request"
                                                    : "logical_warp_id";
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
      hasName(op, "arith.shli")) {
    SmallVector<Value, 2> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
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
  if (isa<arith::SelectOp>(op))
    return op->emitOpError(
        "arith.select lowering requires condition plan emission, which is not implemented in this Stage 1 slice");
  if (hasName(op, kSplatOpName)) {
    Value input = mapValue(op, op->getOperand(0), state);
    if (!input)
      return failure();
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4SplatOpName, input, {},
        op->getResult(0).getType())};
    return success();
  }
  if (hasName(op, kFragmentAddOpName) || hasName(op, kFragmentSubOpName) ||
      hasName(op, kFragmentShlOpName)) {
    SmallVector<Value, 2> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    mlir::vc4::AddOpcode opcode =
        isVectorF32(op->getResult(0).getType()) ? mlir::vc4::AddOpcode::fadd
                                                : mlir::vc4::AddOpcode::add;
    if (hasName(op, kFragmentSubOpName))
      opcode = isVectorF32(op->getResult(0).getType())
                   ? mlir::vc4::AddOpcode::fsub
                   : mlir::vc4::AddOpcode::sub;
    if (hasName(op, kFragmentShlOpName)) {
      opcode = mlir::vc4::AddOpcode::shl;
      if (operands[1].getType() != op->getResult(0).getType()) {
        operands[1] = createOpWithResult(
            builder, op->getLoc(), kSSAVC4SplatOpName, operands[1], {},
            op->getResult(0).getType());
      }
    }
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, operands,
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(), opcode))},
        op->getResult(0).getType())};
    return success();
  }
  if (hasName(op, kFragmentMulOpName)) {
    SmallVector<Value, 2> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = mapValue(op, operand, state);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    mlir::vc4::MulOpcode opcode =
        isVectorF32(op->getResult(0).getType()) ? mlir::vc4::MulOpcode::fmul
                                                : mlir::vc4::MulOpcode::mul24;
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUMulOpName, operands,
        {builder.getNamedAttr("opcode", mlir::vc4::MulOpcodeAttr::get(
                                            builder.getContext(), opcode))},
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
    if (!kind || kind.getValue() != mlir::vc4kernel::ReduceKind::add)
      return op->emitOpError("fragment_reduce lowering supports only add");

    Value zero = createZeroValue(builder, op->getLoc(), op->getResult(0).getType());
    FailureOr<Value> masked =
        emitPredicateSelect(op, builder, *predicate, input, zero);
    if (failed(masked))
      return failure();

    Value sum = *masked;
    for (int64_t amount : {8, 4, 2, 1}) {
      Value rotated = createOpWithResult(
          builder, op->getLoc(), kSSAVC4RotateOpName, sum,
          {builder.getNamedAttr("amount", builder.getI32IntegerAttr(amount))},
          op->getResult(0).getType());
      sum = createFragmentAdd(builder, op->getLoc(), sum, rotated);
    }
    state.values[op->getResult(0)] = {sum};
    return success();
  }
  if (hasName(op, kFragmentCmpOpName)) {
    Value lhs = mapValue(op, op->getOperand(0), state);
    Value rhs = mapValue(op, op->getOperand(1), state);
    if (!lhs || !rhs)
      return failure();
    FailureOr<CompareMaskLowering> compare =
        mapFragmentCmpPredicate(op, lhs, rhs);
    if (failed(compare))
      return failure();
    state.predicates[op->getResult(0)] =
        PredicatePlan::generalMask(compare->lhs, compare->rhs, compare->cond);
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
    } else {
      if (lhs->kind == PredicatePlan::Class::Full ||
          rhs->kind == PredicatePlan::Class::Full)
        result = PredicatePlan::full();
      else if (lhs->kind == PredicatePlan::Class::Empty)
        result = *rhs;
      else if (rhs->kind == PredicatePlan::Class::Empty)
        result = *lhs;
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
    Value safeScalarOffset =
        sourceOffsets.scalarBaseByteOffset
            ? mapValue(op, sourceOffsets.scalarBaseByteOffset, state)
            : createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                            builder.getI32IntegerAttr(0));
    if (!safeScalarOffset)
      return failure();

    Value lane = createOpWithResult(builder, op->getLoc(),
                                    kSSAVC4ElementNumberOpName, {}, {},
                                    offsets.getType());
    Value safeOffsetVec = createOpWithResult(
        builder, op->getLoc(), kSSAVC4SplatOpName, safeScalarOffset, {},
        offsets.getType());

    auto emitTailLoad = [&]() -> LogicalResult {
      Value baseIndex = predicate->base;
      Value limit = predicate->limit;
      if (!baseIndex || !limit)
        return op->emitOpError("pred.tail plan is missing base or limit values");
      Value one =
          createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                        builder.getI32IntegerAttr(1));
      Value basePlusOne = createI32Add(builder, op->getLoc(), baseIndex, one);
      Value activeTail = createI32Sub(builder, op->getLoc(), limit, baseIndex);
      Value nonEmptyFlags =
          createSubFlags(builder, op->getLoc(), limit, basePlusOne);

      Region *region = builder.getInsertionBlock()->getParent();
      Block *loadBlock = new Block();
      Block *doneBlock = new Block();
      doneBlock->addArgument(op->getResult(0).getType(), op->getLoc());
      region->push_back(loadBlock);
      region->push_back(doneBlock);

      createCondBranch(builder, op->getLoc(), nonEmptyFlags, doneBlock,
                       loadBlock, mlir::vc4::BranchCond::any_c_set, zero);

      builder.setInsertionPointToEnd(loadBlock);
      Value activeVec = createOpWithResult(
          builder, op->getLoc(), kSSAVC4SplatOpName, activeTail, {},
          offsets.getType());
      Value flags = createSubFlags(builder, op->getLoc(), lane, activeVec);
      Value safeOffsets =
          createCondSelect(builder, op->getLoc(), flags, offsets,
                           safeOffsetVec, mlir::vc4::Cond::cs);
      Value loaded = emitTMULoadFragment(op, builder, base, safeOffsets,
                                         op->getResult(0).getType());
      Value resultFlags =
          createSubFlags(builder, op->getLoc(), lane, activeVec);
      Value masked =
          createCondSelect(builder, op->getLoc(), resultFlags, loaded, zero,
                           mlir::vc4::Cond::cs);
      createBranch(builder, op->getLoc(), doneBlock, masked);

      builder.setInsertionPointToEnd(doneBlock);
      state.values[op->getResult(0)] = {doneBlock->getArgument(0)};
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
    if (failed(requireFullPredicate(op, op->getOperand(3), state,
                                    "vpm_write_fragment")))
      return failure();
    Value row = mapValue(op, op->getOperand(1), state);
    Value value = mapValue(op, op->getOperand(2), state);
    if (!row || !value)
      return failure();
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
    if (failed(requireFullPredicate(op, op->getOperand(2), state,
                                    "vpm_read_fragment")))
      return failure();
    Value row = mapValue(op, op->getOperand(1), state);
    if (!row)
      return failure();
    FailureOr<Value> plannedRow =
        applyVPMTileBase(op, builder, op->getOperand(0), row, state);
    if (failed(plannedRow))
      return failure();
    row = *plannedRow;
    state.values[op->getResult(0)] = {createOpWithResult(
        builder, op->getLoc(), kSSAVC4VPMReadOpName, row,
        {getSSAVC4VPMOrientation(builder, op),
         getSSAVC4VPMWidth(builder, op),
         getSSAVC4VPMSubword(builder, op),
         builder.getNamedAttr("x", op->getAttr("x")),
         builder.getNamedAttr("stride", op->getAttr("stride")),
         builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16))},
        op->getResult(0).getType())};
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
              builder.getNamedAttr("vpm_pitch", op->getAttr("vpm_pitch"))});
    return success();
  }
  if (hasName(op, kVDWStoreVPMOpName)) {
    if (failed(requireFullPredicate(op, op->getOperand(4), state,
                                    "vdw_store_vpm_fragment")))
      return failure();
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
    auto srcX = llvm::dyn_cast_or_null<IntegerAttr>(op->getAttr("src_x"));
    Value vpmX = createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                               builder.getI32IntegerAttr(srcX ? srcX.getInt() : 0));
    createOp(builder, op->getLoc(), kSSAVC4VDWStoreVPMOpName,
             {address, srcRow, vpmX},
             {getSSAVC4VPMOrientation(builder, op),
              getSSAVC4VPMWidth(builder, op),
              getSSAVC4VPMSubword(builder, op),
              builder.getNamedAttr("row_len", builder.getI32IntegerAttr(16)),
              builder.getNamedAttr("nrows", builder.getI32IntegerAttr(1)),
              builder.getNamedAttr("memory_pitch_bytes",
                                   builder.getI32IntegerAttr(64)),
              builder.getNamedAttr("active_lanes", builder.getI32IntegerAttr(16))});
    return success();
  }
  if (hasName(op, kBarrierOpName)) {
    Value logicalWarp = state.launchABI.lookupBuiltin("logical_warp_id");
    Value warpsPerBlock = state.launchABI.lookupBuiltin("warps_per_block");
    if (!logicalWarp || !warpsPerBlock)
      return op->emitOpError(
          "missing launch builtin uniforms for cooperative barrier");
    createOp(builder, op->getLoc(), kSSAVC4BarrierOpName,
             {logicalWarp, warpsPerBlock},
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
    return op->emitOpError(
        "cf.cond_br lowering requires condition plan emission, "
        "which is not implemented in this Stage 1 slice");
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
