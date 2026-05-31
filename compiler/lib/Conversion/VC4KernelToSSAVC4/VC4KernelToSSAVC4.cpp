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
constexpr llvm::StringLiteral kBlockIdOpName("vc4kernel.block_id");
constexpr llvm::StringLiteral kWarpIdOpName("vc4kernel.warp_id");
constexpr llvm::StringLiteral kLaneIdOpName("vc4kernel.lane_id");
constexpr llvm::StringLiteral kLaneRangeOpName("vc4kernel.lane_range");
constexpr llvm::StringLiteral kSplatOpName("vc4kernel.splat");
constexpr llvm::StringLiteral kFragmentAddOpName("vc4kernel.fragment_add");
constexpr llvm::StringLiteral kFragmentSubOpName("vc4kernel.fragment_sub");
constexpr llvm::StringLiteral kFragmentMulOpName("vc4kernel.fragment_mul");
constexpr llvm::StringLiteral kFragmentShlOpName("vc4kernel.fragment_shl");
constexpr llvm::StringLiteral kFragmentRotateOpName("vc4kernel.fragment_rotate");
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

static BoolAttr asBoolAttr(Attribute attr) {
  return llvm::dyn_cast_if_present<BoolAttr>(attr);
}

static StringAttr getSymbolNameAttr(Operation *op) {
  return asStringAttr(op->getAttr(SymbolTable::getSymbolAttrName()));
}

static bool isAllowedVC4KernelOp(Operation *op) {
  static constexpr llvm::StringLiteral names[] = {
      "vc4kernel.kernel",
      "vc4kernel.return",
      "vc4kernel.program_id",
      "vc4kernel.block_id",
      "vc4kernel.warp_id",
      "vc4kernel.lane_id",
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

static LogicalResult verifyOperationBoundary(Operation *op) {
  StringRef dialect = op->getName().getDialectNamespace();
  if (dialect == "builtin")
    return success();
  if (dialect == "vc4kernel") {
    if (!isAllowedVC4KernelOp(op))
      return op->emitOpError("unknown or forbidden vc4kernel operation");
    return success();
  }
  if (dialect == "arith") {
    StringRef name = op->getName().getStringRef();
    if (name != "arith.constant" && name != "arith.addi" &&
        name != "arith.subi" && name != "arith.muli" &&
        name != "arith.shli" && name != "arith.cmpi" &&
        name != "arith.select")
      return op->emitOpError("arith operation is not allowed in vc4kernel");
    for (Value operand : op->getOperands())
      if (llvm::isa<VectorType>(operand.getType()))
        return op->emitOpError("arith operations may not operate on vectors");
    for (Type type : op->getResultTypes())
      if (llvm::isa<VectorType>(type))
        return op->emitOpError("arith operations may not produce vectors");
    return success();
  }
  if (dialect == "cf") {
    if (!hasName(op, "cf.br") && !hasName(op, "cf.cond_br"))
      return op->emitOpError("only cf.br and cf.cond_br are allowed");
    if (auto cond = dyn_cast<cf::CondBranchOp>(op))
      if (!cond.getCondition().getType().isInteger(1))
        return op->emitOpError("cf.cond_br condition must be scalar i1");
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
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            if (containsIllegalType(arg.getType())) {
              op->emitOpError("region block argument has illegal type ")
                  << arg.getType();
              return WalkResult::interrupt();
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

static DictionaryAttr getResource(Operation *kernel) {
  return asDictionaryAttr(kernel->getAttr("resource"));
}

static std::optional<int64_t> getI32Attr(DictionaryAttr dict, StringRef name) {
  auto attr = dict ? asIntegerAttr(dict.get(name)) : nullptr;
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static std::optional<bool> getBoolAttr(DictionaryAttr dict, StringRef name) {
  auto attr = dict ? asBoolAttr(dict.get(name)) : nullptr;
  if (!attr)
    return std::nullopt;
  return attr.getValue();
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

static bool isFullPredicate(Value pred) {
  return hasName(pred.getDefiningOp(), "vc4kernel.pred.full");
}

static Attribute getBuiltinKindAttr(OpBuilder &builder, StringRef name) {
  MLIRContext *ctx = builder.getContext();
  if (name == "logical_request")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::logical_request);
  if (name == "total_requests")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::total_requests);
  if (name == "logical_block_id")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::logical_block_id);
  if (name == "logical_warp_id")
    return mlir::vc4::BuiltinKindAttr::get(
        ctx, mlir::vc4::BuiltinKind::logical_warp_id);
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

static DictionaryAttr buildLaunchABI(Operation *kernel, OpBuilder &builder) {
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
  if (kernelContains(kernel, kProgramIdOpName)) {
    appendBuiltin("logical_request");
    appendBuiltin("total_requests");
  }
  if (kernelContains(kernel, kBlockIdOpName))
    appendBuiltin("logical_block_id");
  if (kernelContains(kernel, kWarpIdOpName))
    appendBuiltin("logical_warp_id");

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

static DictionaryAttr buildResource(Operation *kernel, OpBuilder &builder) {
  DictionaryAttr resource = getResource(kernel);
  StringRef scheduleMode = "independent_vector";
  auto mode = llvm::dyn_cast_if_present<mlir::vc4kernel::ScheduleModeAttr>(
      kernel->getAttr("schedule_mode"));
  if (mode &&
      mode.getValue() == mlir::vc4kernel::ScheduleMode::cooperative_block)
    scheduleMode = "cooperative_block";
  return builder.getDictionaryAttr({
      builder.getNamedAttr("schedule_mode", builder.getStringAttr(scheduleMode)),
      builder.getNamedAttr(
          "warps_per_block_max",
          builder.getI32IntegerAttr(
              getI32Attr(resource, "warps_per_block_max").value_or(1))),
      builder.getNamedAttr(
          "uses_shared_vpm",
          builder.getBoolAttr(getBoolAttr(resource, "uses_vpm").value_or(false))),
      builder.getNamedAttr(
          "uses_barrier",
          builder.getBoolAttr(
              getBoolAttr(resource, "uses_barrier").value_or(false))),
      builder.getNamedAttr(
          "require_full_block_residency",
          builder.getBoolAttr(
              getBoolAttr(resource, "require_full_block_residency")
                  .value_or(false))),
      builder.getNamedAttr(
          "vpm_rows_per_block",
          builder.getI32IntegerAttr(
              getI32Attr(resource, "vpm_rows_per_block").value_or(0))),
      builder.getNamedAttr(
          "vpm_bytes_per_block",
          builder.getI32IntegerAttr(
              getI32Attr(resource, "vpm_bytes_per_block").value_or(0))),
      builder.getNamedAttr(
          "semaphores_per_block",
          builder.getI32IntegerAttr(
              getI32Attr(resource, "semaphores_per_block").value_or(0))),
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

static Value mapValue(Operation *op, Value value,
                      llvm::DenseMap<Value, Value> &valueMap) {
  auto it = valueMap.find(value);
  if (it == valueMap.end()) {
    op->emitOpError("operand has not been lowered");
    return {};
  }
  return it->second;
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
  OperationState state(kernel->getLoc(), kSSAVC4FuncOpName);
  StringAttr sym = getSymbolNameAttr(kernel);
  auto &properties =
      state.getOrAddProperties<mlir::ssavc4::FuncOp::Properties>();
  properties.sym_name = sym ? sym : builder.getStringAttr("kernel");
  properties.kernel = builder.getUnitAttr();
  properties.threading = mlir::vc4::ThreadingModeAttr::get(
      builder.getContext(), mlir::vc4::ThreadingMode::single);
  state.addAttribute(builder.getStringAttr("vc4.launch_abi"),
                     buildLaunchABI(kernel, builder));
  state.addAttribute(builder.getStringAttr("vc4.resource"),
                     buildResource(kernel, builder));
  state.addRegion();
  builder.setInsertionPointToEnd(&module->getRegion(0).front());
  return builder.create(state);
}

static LogicalResult lowerBodyOp(Operation *op, OpBuilder &builder,
                                 llvm::DenseMap<Value, Value> &valueMap,
                                 llvm::StringMap<Value> &builtinMap,
                                 llvm::DenseMap<Value, int64_t> &vpmRows) {
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    valueMap[cst.getResult()] =
        createLoadImm(builder, op->getLoc(), cst.getType(), cst.getValue());
    return success();
  }
  if (hasName(op, kProgramIdOpName) || hasName(op, kBlockIdOpName) ||
      hasName(op, kWarpIdOpName)) {
    StringRef name = hasName(op, kProgramIdOpName)   ? "logical_request"
                     : hasName(op, kBlockIdOpName) ? "logical_block_id"
                                                    : "logical_warp_id";
    Value builtin = builtinMap.lookup(name);
    if (!builtin)
      return op->emitOpError("missing launch builtin uniform for identity op");
    valueMap[op->getResult(0)] = builtin;
    return success();
  }
  if (hasName(op, kLaneIdOpName))
    return op->emitOpError(
        "lane_id scalar lowering is not implemented; lower-half SSAVC4 only "
        "has vector element_number today");
  if (hasName(op, kLaneRangeOpName)) {
    valueMap[op->getResult(0)] = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ElementNumberOpName, {}, {},
        op->getResult(0).getType());
    return success();
  }
  if (hasName(op, kSplatOpName)) {
    Value input = mapValue(op, op->getOperand(0), valueMap);
    if (!input)
      return failure();
    valueMap[op->getResult(0)] = createOpWithResult(
        builder, op->getLoc(), kSSAVC4SplatOpName, input, {},
        op->getResult(0).getType());
    return success();
  }
  if (hasName(op, kFragmentAddOpName) || hasName(op, kFragmentSubOpName) ||
      hasName(op, kFragmentShlOpName)) {
    SmallVector<Value, 2> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = mapValue(op, operand, valueMap);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    mlir::vc4::AddOpcode opcode = mlir::vc4::AddOpcode::add;
    if (hasName(op, kFragmentSubOpName))
      opcode = mlir::vc4::AddOpcode::sub;
    if (hasName(op, kFragmentShlOpName)) {
      opcode = mlir::vc4::AddOpcode::shl;
      if (operands[1].getType() != op->getResult(0).getType()) {
        operands[1] = createOpWithResult(
            builder, op->getLoc(), kSSAVC4SplatOpName, operands[1], {},
            op->getResult(0).getType());
      }
    }
    valueMap[op->getResult(0)] = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, operands,
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(), opcode))},
        op->getResult(0).getType());
    return success();
  }
  if (hasName(op, kFragmentMulOpName)) {
    SmallVector<Value, 2> operands;
    for (Value operand : op->getOperands()) {
      Value mapped = mapValue(op, operand, valueMap);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    valueMap[op->getResult(0)] = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUMulOpName, operands,
        {builder.getNamedAttr("opcode", mlir::vc4::MulOpcodeAttr::get(
                                            builder.getContext(),
                                            mlir::vc4::MulOpcode::mul24))},
        op->getResult(0).getType());
    return success();
  }
  if (hasName(op, kFragmentRotateOpName)) {
    Value input = mapValue(op, op->getOperand(0), valueMap);
    if (!input)
      return failure();
    valueMap[op->getResult(0)] = createOpWithResult(
        builder, op->getLoc(), kSSAVC4RotateOpName, input,
        {builder.getNamedAttr("amount", op->getAttr("amount"))},
        op->getResult(0).getType());
    return success();
  }
  if (op->getName().getStringRef().starts_with("vc4kernel.pred.")) {
    if (op->getNumResults() != 0)
      valueMap[op->getResult(0)] = Value();
    return success();
  }
  if (hasName(op, kTMULoadOpName)) {
    if (!isFullPredicate(op->getOperand(2)))
      return op->emitOpError(
          "tmu_load_fragment lowering currently supports only pred.full");
    Value base = mapValue(op, op->getOperand(0), valueMap);
    Value offsets = mapValue(op, op->getOperand(1), valueMap);
    if (!base || !offsets)
      return failure();
    if (base.getType() != offsets.getType())
      base = createOpWithResult(builder, op->getLoc(), kSSAVC4SplatOpName,
                                base, {}, offsets.getType());
    Value address = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, {base, offsets},
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(),
                                            mlir::vc4::AddOpcode::add))},
        offsets.getType());
    Value token = createOpWithResult(
        builder, op->getLoc(), kSSAVC4TMURequestOpName, address,
        {builder.getNamedAttr("unit", builder.getStringAttr("tmu0")),
         builder.getNamedAttr("mode", builder.getStringAttr("direct"))},
        mlir::ssavc4::AsyncTokenType::get(builder.getContext()));
    valueMap[op->getResult(0)] = createOpWithResult(
        builder, op->getLoc(), kSSAVC4TMUReadOpName, token,
        {builder.getNamedAttr("unit", builder.getStringAttr("tmu0")),
         builder.getNamedAttr("part", builder.getStringAttr("raw32"))},
        op->getResult(0).getType());
    return success();
  }
  if (hasName(op, kVDWStoreOpName)) {
    if (!isFullPredicate(op->getOperand(3)))
      return op->emitOpError(
          "vdw_store_fragment lowering currently supports only pred.full");
    Value base = mapValue(op, op->getOperand(0), valueMap);
    Value value = mapValue(op, op->getOperand(2), valueMap);
    if (!base || !value)
      return failure();
    createOp(builder, op->getLoc(), kSSAVC4VDWStoreOpName, {base, value},
             {builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4)),
              builder.getNamedAttr("vpm_row", builder.getI32IntegerAttr(0)),
              builder.getNamedAttr("operandSegmentSizes",
                                   builder.getDenseI32ArrayAttr({1, 1, 0, 0}))});
    return success();
  }
  if (hasName(op, kVPMAllocOpName)) {
    vpmRows[op->getResult(0)] = 0;
    return success();
  }
  if (hasName(op, kVPMWriteOpName)) {
    if (!isFullPredicate(op->getOperand(3)))
      return op->emitOpError(
          "vpm_write_fragment lowering currently supports only pred.full");
    Value row = mapValue(op, op->getOperand(1), valueMap);
    Value value = mapValue(op, op->getOperand(2), valueMap);
    if (!row || !value)
      return failure();
    createOp(builder, op->getLoc(), kSSAVC4VPMWriteOpName, {row, value},
             {builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4)),
              builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16)),
              builder.getNamedAttr("orientation", builder.getStringAttr("row"))});
    return success();
  }
  if (hasName(op, kVPMReadOpName)) {
    if (!isFullPredicate(op->getOperand(2)))
      return op->emitOpError(
          "vpm_read_fragment lowering currently supports only pred.full");
    Value row = mapValue(op, op->getOperand(1), valueMap);
    if (!row)
      return failure();
    valueMap[op->getResult(0)] = createOpWithResult(
        builder, op->getLoc(), kSSAVC4VPMReadOpName, row,
        {builder.getNamedAttr("elem_bytes", builder.getI32IntegerAttr(4)),
         builder.getNamedAttr("lanes", builder.getI32IntegerAttr(16)),
         builder.getNamedAttr("orientation", builder.getStringAttr("row"))},
        op->getResult(0).getType());
    return success();
  }
  if (hasName(op, kVDRLoadOpName)) {
    Value base = mapValue(op, op->getOperand(0), valueMap);
    Value byteOffset = mapValue(op, op->getOperand(1), valueMap);
    Value dstRow = mapValue(op, op->getOperand(3), valueMap);
    if (!base || !byteOffset || !dstRow)
      return failure();
    Value address = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, {base, byteOffset},
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(),
                                            mlir::vc4::AddOpcode::add))},
        base.getType());
    createOp(builder, op->getLoc(), kSSAVC4VDRLoadOpName, {address, dstRow},
             {builder.getNamedAttr("elem_bytes", op->getAttr("elem_bytes")),
              builder.getNamedAttr("row_len", op->getAttr("cols")),
              builder.getNamedAttr("nrows", op->getAttr("rows")),
              builder.getNamedAttr("memory_pitch_bytes",
                                   op->getAttr("global_stride_bytes")),
              builder.getNamedAttr("vpm_base_col", builder.getI32IntegerAttr(0)),
              builder.getNamedAttr("orientation", builder.getStringAttr("row")),
              builder.getNamedAttr("vpitch", builder.getI32IntegerAttr(1))});
    return success();
  }
  if (hasName(op, kVDWStoreVPMOpName)) {
    if (!isFullPredicate(op->getOperand(4)))
      return op->emitOpError(
          "vdw_store_vpm_fragment lowering currently supports only pred.full");
    Value srcRow = mapValue(op, op->getOperand(1), valueMap);
    Value base = mapValue(op, op->getOperand(2), valueMap);
    Value byteOffset = mapValue(op, op->getOperand(3), valueMap);
    if (!srcRow || !base || !byteOffset)
      return failure();
    Value address = createOpWithResult(
        builder, op->getLoc(), kSSAVC4ALUAddOpName, {base, byteOffset},
        {builder.getNamedAttr("opcode", mlir::vc4::AddOpcodeAttr::get(
                                            builder.getContext(),
                                            mlir::vc4::AddOpcode::add))},
        base.getType());
    Value zero = createLoadImm(builder, op->getLoc(), builder.getI32Type(),
                               builder.getI32IntegerAttr(0));
    createOp(builder, op->getLoc(), kSSAVC4VDWStoreVPMOpName,
             {address, srcRow, zero},
             {builder.getNamedAttr("elem_bytes", op->getAttr("elem_bytes")),
              builder.getNamedAttr("row_len", builder.getI32IntegerAttr(16)),
              builder.getNamedAttr("nrows", builder.getI32IntegerAttr(1)),
              builder.getNamedAttr("memory_pitch_bytes",
                                   builder.getI32IntegerAttr(64)),
              builder.getNamedAttr("orientation", builder.getStringAttr("row"))});
    return success();
  }
  if (hasName(op, kBarrierOpName)) {
    createOp(builder, op->getLoc(), kSSAVC4BarrierOpName, {},
             {builder.getNamedAttr("arrive_offset", builder.getI32IntegerAttr(0)),
              builder.getNamedAttr("go_offset", builder.getI32IntegerAttr(1)),
              builder.getNamedAttr("depart_offset", builder.getI32IntegerAttr(2)),
              builder.getNamedAttr("reset_offset", builder.getI32IntegerAttr(3))});
    return success();
  }
  return op->emitOpError("is not implemented by vc4kernel -> ssavc4 lowering");
}

static LogicalResult lowerTerminator(Operation *op, OpBuilder &builder,
                                     llvm::DenseMap<Value, Value> &valueMap,
                                     llvm::DenseMap<Block *, Block *> &blockMap) {
  if (hasName(op, kReturnOpName)) {
    createOp(builder, op->getLoc(), kSSAVC4ThreadEndOpName, {}, {});
    return success();
  }
  if (auto br = dyn_cast<cf::BranchOp>(op)) {
    SmallVector<Value, 4> operands;
    for (Value operand : br.getDestOperands()) {
      Value mapped = mapValue(op, operand, valueMap);
      if (!mapped)
        return failure();
      operands.push_back(mapped);
    }
    OperationState state(op->getLoc(), kSSAVC4BranchOpName);
    state.addOperands(operands);
    state.addSuccessors(blockMap.lookup(br.getDest()));
    builder.create(state);
    return success();
  }
  if (isa<cf::CondBranchOp>(op))
    return op->emitOpError(
        "cf.cond_br lowering requires flag-producing pred.any/all support, "
        "which is not implemented in this Stage 1 slice");
  return op->emitOpError("unsupported terminator");
}

static LogicalResult lowerKernel(Operation *kernel, Operation *func) {
  Region &source = kernel->getRegion(0);
  Region &dest = func->getRegion(0);
  llvm::DenseMap<Value, Value> valueMap;
  llvm::StringMap<Value> builtinMap;
  llvm::DenseMap<Block *, Block *> blockMap;
  llvm::DenseMap<Value, int64_t> vpmRows;
  OpBuilder builder(func->getContext());

  for (Block &sourceBlock : source) {
    Block *destBlock = new Block();
    dest.push_back(destBlock);
    blockMap[&sourceBlock] = destBlock;
    if (&sourceBlock != &source.front()) {
      for (BlockArgument arg : sourceBlock.getArguments())
        valueMap[arg] = destBlock->addArgument(arg.getType(), arg.getLoc());
    }
  }

  builder.setInsertionPointToStart(blockMap.lookup(&source.front()));
  int64_t nextUniform = 0;
  for (BlockArgument arg : source.front().getArguments())
    valueMap[arg] = createUniformRead(builder, arg.getLoc(), arg.getType(),
                                      nextUniform++);
  auto materializeBuiltin = [&](StringRef name) {
    builtinMap[name] =
        createUniformRead(builder, kernel->getLoc(), builder.getI32Type(),
                          nextUniform++);
  };
  if (kernelContains(kernel, kProgramIdOpName)) {
    materializeBuiltin("logical_request");
    materializeBuiltin("total_requests");
  }
  if (kernelContains(kernel, kBlockIdOpName))
    materializeBuiltin("logical_block_id");
  if (kernelContains(kernel, kWarpIdOpName))
    materializeBuiltin("logical_warp_id");

  for (Block &sourceBlock : source) {
    Block *destBlock = blockMap.lookup(&sourceBlock);
    builder.setInsertionPointToEnd(destBlock);
    for (Operation &nested : sourceBlock) {
      if (nested.hasTrait<OpTrait::IsTerminator>()) {
        if (failed(lowerTerminator(&nested, builder, valueMap, blockMap)))
          return failure();
        continue;
      }
      if (failed(lowerBodyOp(&nested, builder, valueMap, builtinMap, vpmRows)))
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
