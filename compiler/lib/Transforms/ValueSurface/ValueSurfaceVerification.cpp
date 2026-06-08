//===- ValueSurfaceVerification.cpp - VC4 value surface verifier ----------===//

#include "vc4/Transforms/ValueSurface/ValueSurfacePasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cctype>
#include <memory>
#include <optional>

using namespace mlir;

namespace {

static StringRef getDialectNamespace(Operation *op) {
  return op->getName().getStringRef().split('.').first;
}

static bool isForbiddenTargetDialect(StringRef dialect) {
  return dialect == "vc4kernel" || dialect == "ssavc4" || dialect == "vc4";
}

static bool isForbiddenProducerDialect(StringRef dialect) {
  return dialect == "tt" || dialect == "ttg" || dialect == "gpu" ||
         dialect == "linalg" || dialect == "tensor" || dialect == "nvgpu" ||
         dialect == "nvvm" || dialect == "rocdl" || dialect == "spirv" ||
         dialect == "iree" || dialect == "stablehlo" || dialect == "mhlo";
}

static bool isAllowedDialect(StringRef dialect) {
  return dialect == "builtin" || dialect == "func" || dialect == "vc4value" ||
         dialect == "vector" || dialect == "memref" || dialect == "arith" ||
         dialect == "math" || dialect == "scf" || dialect == "cf";
}

static bool isLegalVC4ValueOp(StringRef name) {
  return name == "vc4value.program_id" || name == "vc4value.num_programs";
}

static bool isForbiddenMemRefSideEffectOp(StringRef name) {
  return name == "memref.load" || name == "memref.store" ||
         name == "memref.atomic_rmw" || name == "memref.generic_atomic_rmw" ||
         name == "memref.copy" || name == "memref.dma_start" ||
         name == "memref.dma_wait";
}

static bool isForbiddenSparseStoreVectorOp(StringRef name) {
  return name == "vector.scatter" || name == "vector.compressstore";
}

static bool isAllowedSCFOp(StringRef name) {
  return name == "scf.for" || name == "scf.if" || name == "scf.yield";
}

static bool isAllowedCFOp(StringRef name) {
  return name == "cf.br" || name == "cf.cond_br";
}

static bool isSupportedScalarType(Type type) {
  if (type.isIndex())
    return true;
  if (auto integerType = dyn_cast<IntegerType>(type))
    return integerType.getWidth() == 1 || integerType.getWidth() == 8 ||
           integerType.getWidth() == 16 || integerType.getWidth() == 32;
  return type.isF16() || type.isF32();
}

static bool isSupportedVectorElementType(Type type) {
  return isSupportedScalarType(type);
}

static bool isSupportedMemRefElementType(Type type) {
  if (auto integerType = dyn_cast<IntegerType>(type))
    return integerType.getWidth() == 8 || integerType.getWidth() == 16 ||
           integerType.getWidth() == 32;
  return type.isF16() || type.isF32();
}

static bool isPublicAbiScalarArgType(Type type) {
  if (type.isIndex() || type.isF32())
    return true;
  if (auto integerType = dyn_cast<IntegerType>(type))
    return integerType.getWidth() == 32;
  return false;
}

static bool hasF16Type(Type type) {
  if (type.isF16())
    return true;
  if (auto vectorType = dyn_cast<VectorType>(type))
    return hasF16Type(vectorType.getElementType());
  if (auto memRefType = dyn_cast<MemRefType>(type))
    return hasF16Type(memRefType.getElementType());
  return false;
}

static bool isNativeF16ArithmeticOp(StringRef name) {
  return name == "arith.addf" || name == "arith.subf" ||
         name == "arith.mulf" || name == "arith.divf" ||
         name == "arith.maximumf" || name == "arith.minimumf" ||
         name == "arith.maxnumf" || name == "arith.minnumf";
}

static Operation *getEnclosingFunc(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() == "func.func")
      return parent;
  }
  return nullptr;
}

static std::optional<int64_t> getI64Attr(Operation *op, StringRef attrName) {
  auto attr = dyn_cast_or_null<IntegerAttr>(op->getAttr(attrName));
  if (!attr)
    return std::nullopt;
  return attr.getValue().getSExtValue();
}

static bool isIdentifierStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

static bool isIdentifierBody(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static bool isLegalArgNameSyntax(StringRef name) {
  if (name.empty() || !isIdentifierStart(name.front()))
    return false;
  return llvm::all_of(name.drop_front(), isIdentifierBody);
}

static bool isReservedArgName(StringRef name) {
  return llvm::is_contained(
      ArrayRef<StringRef>{"program_id", "num_programs", "lane_id", "qpu_id",
                          "physical_qpu_id", "warp_id", "thread_id",
                          "num_qpus", "vpm_base_row", "semaphore_base",
                          "uniform_index", "uniform_offset", "tmu", "vdr",
                          "vdw", "vpm", "ssavc4", "vc4kernel", "vc4"},
      name);
}

static bool isAllowedDirection(StringRef value) {
  return value == "in" || value == "out" || value == "inout";
}

static bool isAllowedScalarRole(StringRef value) {
  return value == "value" || value == "extent" || value == "stride" ||
         value == "grid_dim" || value == "policy";
}

static bool isAllowedValueArgAttr(StringRef name) {
  return name == "vc4value.arg_name" || name == "vc4value.direction" ||
         name == "vc4value.scalar_role" || name == "vc4value.shape_args" ||
         name == "vc4value.stride_args";
}

static bool isArrayOfStringAttr(Attribute attr) {
  auto arrayAttr = dyn_cast_or_null<ArrayAttr>(attr);
  return arrayAttr && llvm::all_of(arrayAttr, [](Attribute element) {
           return isa<StringAttr>(element);
         });
}

static DictionaryAttr getArgAttrs(func::FuncOp func, unsigned argIndex) {
  std::optional<ArrayAttr> argAttrs = func.getArgAttrs();
  if (!argAttrs || argIndex >= argAttrs->size())
    return {};
  return dyn_cast<DictionaryAttr>((*argAttrs)[argIndex]);
}

static InFlightDiagnostic emitArgError(func::FuncOp func, unsigned argIndex) {
  InFlightDiagnostic diag = func.emitError()
                            << "public value kernel @" << func.getName()
                            << " argument #" << argIndex;
  if (auto argName =
          dyn_cast_or_null<StringAttr>(
              func.getArgAttr(argIndex, "vc4value.arg_name"))) {
    diag << " '" << argName.getValue() << "'";
  }
  diag << ": ";
  return diag;
}

static bool checkType(Operation *owner, Type type, bool &sawError) {
  if (isa<ComplexType>(type)) {
    owner->emitError() << "complex types are not legal in the VC4 value surface";
    sawError = true;
    return false;
  }

  if (isa<RankedTensorType, UnrankedTensorType>(type)) {
    owner->emitError()
        << "tensor types are not legal in the initial VC4 value surface";
    sawError = true;
    return false;
  }

  if (auto vectorType = dyn_cast<VectorType>(type)) {
    if (vectorType.isScalable()) {
      owner->emitError()
          << "scalable vector types are not legal in the VC4 value surface";
      sawError = true;
      return false;
    }
    if (vectorType.getRank() == 0) {
      owner->emitError()
          << "vector rank 0 is not legal in the VC4 value surface";
      sawError = true;
      return false;
    }
    if (vectorType.getRank() > 2) {
      owner->emitError() << "vector rank greater than 2 is not legal in the "
                         << "Phase 3.5 VC4 value surface";
      sawError = true;
      return false;
    }
    if (llvm::any_of(vectorType.getShape(), ShapedType::isDynamic)) {
      owner->emitError() << "vector dimensions must be static positive "
                         << "integers in the VC4 value surface";
      sawError = true;
      return false;
    }
    if (llvm::any_of(vectorType.getShape(),
                     [](int64_t dim) { return dim <= 0; })) {
      owner->emitError() << "vector dimensions must be static positive "
                         << "integers in the VC4 value surface";
      sawError = true;
      return false;
    }
    if (!isSupportedVectorElementType(vectorType.getElementType())) {
      owner->emitError()
          << "vector element type is not legal in the VC4 value surface";
      sawError = true;
      return false;
    }
    return true;
  }

  if (isa<UnrankedMemRefType>(type)) {
    owner->emitError()
        << "unranked memrefs are not legal in the VC4 value surface";
    sawError = true;
    return false;
  }

  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    if (memRefType.getRank() > 2) {
      owner->emitError() << "memref rank greater than 2 is not legal in the "
                         << "VC4 value surface";
      sawError = true;
      return false;
    }
    if (!isSupportedMemRefElementType(memRefType.getElementType())) {
      owner->emitError()
          << "unsupported memref element type in VC4 value surface";
      sawError = true;
      return false;
    }
    return true;
  }

  if (!isSupportedScalarType(type)) {
    owner->emitError()
        << "scalar type is not legal in the VC4 value surface";
    sawError = true;
    return false;
  }

  return true;
}

static void checkPublicKernelArgumentSchema(func::FuncOp func, bool &sawError) {
  llvm::SmallDenseSet<StringRef> seenArgNames;

  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    Type argType = func.getArgument(i).getType();
    DictionaryAttr argAttrs = getArgAttrs(func, i);

    if (argAttrs) {
      for (NamedAttribute attr : argAttrs) {
        StringRef name = attr.getName().getValue();
        if (name.starts_with("vc4value.") && !isAllowedValueArgAttr(name)) {
          emitArgError(func, i)
              << "unknown vc4value public argument ABI attribute '" << name
              << "'";
          sawError = true;
        }
      }
    }

    auto argNameAttr =
        dyn_cast_or_null<StringAttr>(func.getArgAttr(i, "vc4value.arg_name"));
    if (!argNameAttr) {
      emitArgError(func, i)
          << "requires StringAttr vc4value.arg_name matching "
          << "^[A-Za-z_][A-Za-z0-9_]*$";
      sawError = true;
    } else {
      StringRef argName = argNameAttr.getValue();
      if (!isLegalArgNameSyntax(argName)) {
        emitArgError(func, i)
            << "vc4value.arg_name must match "
            << "^[A-Za-z_][A-Za-z0-9_]*$";
        sawError = true;
      }
      if (isReservedArgName(argName)) {
        emitArgError(func, i)
            << "vc4value.arg_name '" << argName
            << "' is reserved for lower-half/runtime/builtin metadata";
        sawError = true;
      }
      if (!seenArgNames.insert(argName).second) {
        emitArgError(func, i)
            << "duplicate vc4value.arg_name '" << argName << "'";
        sawError = true;
      }
    }

    auto directionAttr =
        dyn_cast_or_null<StringAttr>(func.getArgAttr(i, "vc4value.direction"));
    if (isa<MemRefType>(argType)) {
      if (!directionAttr) {
        emitArgError(func, i)
            << "memref argument requires vc4value.direction = \"in\", "
            << "\"out\", or \"inout\"";
        sawError = true;
      } else if (!isAllowedDirection(directionAttr.getValue())) {
        emitArgError(func, i)
            << "vc4value.direction must be \"in\", \"out\", or \"inout\"";
        sawError = true;
      }
    } else if (directionAttr) {
      emitArgError(func, i)
          << "non-memref scalar argument must not have vc4value.direction";
      sawError = true;
    }

    if (auto scalarRoleAttr = dyn_cast_or_null<StringAttr>(
            func.getArgAttr(i, "vc4value.scalar_role"))) {
      if (isa<MemRefType>(argType)) {
        emitArgError(func, i)
            << "memref argument must not have vc4value.scalar_role";
        sawError = true;
      }
      if (!isAllowedScalarRole(scalarRoleAttr.getValue())) {
        emitArgError(func, i)
            << "vc4value.scalar_role must be \"value\", \"extent\", "
            << "\"stride\", \"grid_dim\", or \"policy\"";
        sawError = true;
      }
    }

    for (StringRef attrName :
         {"vc4value.shape_args", "vc4value.stride_args"}) {
      if (Attribute attr = func.getArgAttr(i, attrName);
          attr && !isArrayOfStringAttr(attr)) {
        emitArgError(func, i) << attrName << " must be an ArrayAttr of "
                              << "StringAttr";
        sawError = true;
      }
    }

    if (isa<VectorType>(argType)) {
      emitArgError(func, i)
          << "vector public arguments are not legal in the VC4 value ABI";
      sawError = true;
      continue;
    }
    if (isa<RankedTensorType, UnrankedTensorType>(argType)) {
      emitArgError(func, i)
          << "tensor public arguments are not legal in the VC4 value ABI";
      sawError = true;
      continue;
    }
    if (isa<MemRefType>(argType)) {
      continue;
    }
    if (!isPublicAbiScalarArgType(argType)) {
      emitArgError(func, i)
          << "public scalar argument type must be index, i32, or f32";
      sawError = true;
    }
  }
}

static void checkOperationTypes(Operation *op, bool &sawError) {
  for (Type type : op->getOperandTypes())
    checkType(op, type, sawError);
  for (Type type : op->getResultTypes())
    checkType(op, type, sawError);

  if (auto typeAttr = dyn_cast_or_null<TypeAttr>(op->getAttr("function_type"))) {
    if (auto functionType = dyn_cast<FunctionType>(typeAttr.getValue())) {
      for (Type type : functionType.getInputs())
        checkType(op, type, sawError);
      for (Type type : functionType.getResults())
        checkType(op, type, sawError);
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument arg : block.getArguments())
        checkType(op, arg.getType(), sawError);
    }
  }
}

struct VerifyValueSurfacePass
    : public PassWrapper<VerifyValueSurfacePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyValueSurfacePass)

  StringRef getArgument() const final { return "vc4-verify-value-surface"; }

  StringRef getDescription() const final {
    return "verify VC4 standard value-surface boundary admissibility";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    unsigned kernelCount = 0;
    bool sawError = false;

    module.walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();
      StringRef dialect = getDialectNamespace(op);
      bool isFunc = opName == "func.func";

      checkOperationTypes(op, sawError);

      if (isNativeF16ArithmeticOp(opName) &&
          (llvm::any_of(op->getOperandTypes(), hasF16Type) ||
           llvm::any_of(op->getResultTypes(), hasF16Type))) {
        op->emitError() << "native f16 arithmetic is not legal in the "
                        << "Phase 3.5 VC4 value surface";
        sawError = true;
      }

      if ((op->hasAttr("vc4value.kernel") ||
           op->hasAttr("vc4value.grid_rank")) &&
          !isFunc) {
        op->emitError()
            << "vc4value kernel metadata is only legal on func.func";
        sawError = true;
      }

      if (isFunc && op->hasAttr("vc4value.kernel")) {
        ++kernelCount;
        auto gridRank = getI64Attr(op, "vc4value.grid_rank");
        if (!gridRank || *gridRank < 1 || *gridRank > 3) {
          op->emitError() << "vc4value.kernel requires integer attr "
                          << "vc4value.grid_rank in [1, 3]";
          sawError = true;
        }
        if (auto func = dyn_cast<func::FuncOp>(op))
          checkPublicKernelArgumentSchema(func, sawError);
      }

      if (isForbiddenTargetDialect(dialect)) {
        op->emitError() << "target/lower-half dialect is not legal in the "
                        << "VC4 value surface: " << dialect;
        sawError = true;
        return WalkResult::advance();
      }

      if (isForbiddenProducerDialect(dialect)) {
        op->emitError() << "producer dialect is not legal in the VC4 value "
                        << "surface: " << dialect;
        sawError = true;
        return WalkResult::advance();
      }

      if (!isAllowedDialect(dialect)) {
        op->emitError() << "dialect '" << dialect
                        << "' is not legal in the VC4 value surface";
        sawError = true;
        return WalkResult::advance();
      }

      if (dialect == "vc4value" && !isLegalVC4ValueOp(opName)) {
        op->emitError()
            << "only vc4value.program_id and vc4value.num_programs are "
            << "legal in Phase 3; got " << opName;
        sawError = true;
      }

      if (isLegalVC4ValueOp(opName)) {
        Operation *func = getEnclosingFunc(op);
        if (!func || !func->hasAttr("vc4value.kernel")) {
          op->emitError() << "vc4value launch identity ops require an "
                          << "enclosing vc4value.kernel function";
          sawError = true;
        } else {
          auto axis = getI64Attr(op, "axis");
          auto gridRank = getI64Attr(func, "vc4value.grid_rank");
          if (axis && gridRank && (*axis < 0 || *axis >= *gridRank)) {
            op->emitError() << "vc4value launch axis must be within the "
                            << "enclosing vc4value.grid_rank";
            sawError = true;
          }
        }
      }

      if (isForbiddenMemRefSideEffectOp(opName)) {
        op->emitError()
            << "direct memref side-effect operation is not legal in the "
            << "VC4 value surface; use structured vector transfer/planning "
            << "forms";
        sawError = true;
      }

      if (isForbiddenSparseStoreVectorOp(opName)) {
        op->emitError() << "sparse-store-shaped vector operation is not legal "
                        << "in the VC4 value surface";
        sawError = true;
      }

      if (dialect == "scf" && !isAllowedSCFOp(opName)) {
        op->emitError() << "scf operation is not in the Phase 3 VC4 "
                        << "value-surface control subset";
        sawError = true;
      }

      if (dialect == "cf" && !isAllowedCFOp(opName)) {
        op->emitError() << "cf operation is not in the Phase 3 VC4 "
                        << "value-surface control subset";
        sawError = true;
      }

      return WalkResult::advance();
    });

    if (kernelCount == 0) {
      module.emitError()
          << "expected at least one func.func marked with vc4value.kernel";
      sawError = true;
    }

    if (sawError)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vc4::createVerifyValueSurfacePass() {
  return std::make_unique<VerifyValueSurfacePass>();
}

void mlir::vc4::registerValueSurfacePasses() {
  // The file-scope PassRegistration below installs
  // --vc4-verify-value-surface when this library is linked into vc4-opt.
}

static PassRegistration<VerifyValueSurfacePass> registerVerifyValueSurfacePass;
