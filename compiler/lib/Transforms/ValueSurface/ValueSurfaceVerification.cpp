//===- ValueSurfaceVerification.cpp - VC4 value surface verifier ----------===//

#include "vc4/Transforms/ValueSurface/ValueSurfacePasses.h"

#include "vc4/Dialect/VC4Value/IR/VC4ValueAttrs.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
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
using namespace mlir::vc4value;

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
         name == "memref.atomic_yield" || name == "memref.copy" ||
         name == "memref.dma_start" || name == "memref.dma_wait";
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

static bool isMemRefValueSurfaceCFGType(Type type) {
  return isa<MemRefType, UnrankedMemRefType>(type);
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
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.isSignlessInteger() &&
           (integerType.getWidth() == 8 || integerType.getWidth() == 16 ||
            integerType.getWidth() == 32);
  }
  return type.isF16() || type.isF32();
}

static bool isPublicAbiScalarArgType(Type type) {
  if (type.isIndex() || type.isF32())
    return true;
  if (auto integerType = dyn_cast<IntegerType>(type))
    return integerType.getWidth() == 32;
  return false;
}

static bool isIndexOrI32(Type type) {
  if (type.isIndex())
    return true;
  if (auto integerType = dyn_cast<IntegerType>(type))
    return integerType.isSignlessInteger(32);
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

static SmallVector<StringRef> getStringArrayValues(Attribute attr) {
  SmallVector<StringRef> values;
  auto arrayAttr = dyn_cast_or_null<ArrayAttr>(attr);
  if (!arrayAttr)
    return values;
  values.reserve(arrayAttr.size());
  for (Attribute element : arrayAttr)
    values.push_back(cast<StringAttr>(element).getValue());
  return values;
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

static InFlightDiagnostic emitMemRefDimError(Operation *op) {
  return op->emitError() << "memref.dim in Phase 4 value ABI: ";
}

static bool hasGlobalMemorySpace(MemRefType type) {
  return isa_and_nonnull<GlobalMemorySpaceAttr>(type.getMemorySpace());
}

static unsigned countDynamicDims(MemRefType type) {
  return llvm::count_if(type.getShape(), ShapedType::isDynamic);
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

static bool hasShapeArgForDynamicDim(func::FuncOp func, unsigned argIndex,
                                     MemRefType type, unsigned dim) {
  std::optional<unsigned> ordinal = getDynamicDimOrdinal(type, dim);
  if (!ordinal)
    return true;
  Attribute attr = func.getArgAttr(argIndex, "vc4value.shape_args");
  if (!isArrayOfStringAttr(attr))
    return false;
  SmallVector<StringRef> names = getStringArrayValues(attr);
  return *ordinal < names.size() && !names[*ordinal].empty();
}

static std::optional<unsigned> getPublicMemRefArgIndex(func::FuncOp func,
                                                       Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return std::nullopt;
  if (blockArg.getOwner() != &func.getBody().front())
    return std::nullopt;
  unsigned argIndex = blockArg.getArgNumber();
  if (argIndex >= func.getNumArguments() ||
      !isa<MemRefType>(func.getArgument(argIndex).getType()))
    return std::nullopt;
  return argIndex;
}

static bool getStridesAndOffset(MemRefType type, SmallVectorImpl<int64_t> &strides,
                                int64_t &offset) {
  MemRefLayoutAttrInterface layout = type.getLayout();
  if (!layout || layout.isIdentity()) {
    strides.clear();
    offset = 0;
    return true;
  }
  return succeeded(layout.getStridesAndOffset(type.getShape(), strides, offset));
}

struct PublicArgInfo {
  Type type;
  std::optional<unsigned> index;
  bool isScalar = false;
  bool isExtentCompatible = false;
  bool isStrideCompatible = false;
};

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
  llvm::DenseMap<StringRef, PublicArgInfo> argsByName;

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
      PublicArgInfo info;
      info.type = argType;
      info.index = i;
      info.isScalar = !isa<MemRefType, UnrankedMemRefType, VectorType,
                           RankedTensorType, UnrankedTensorType>(argType);
      if (info.isScalar && isPublicAbiScalarArgType(argType)) {
        info.isExtentCompatible = true;
        info.isStrideCompatible = true;
      }
      if (auto scalarRoleAttr = dyn_cast_or_null<StringAttr>(
              func.getArgAttr(i, "vc4value.scalar_role"))) {
        info.isExtentCompatible = scalarRoleAttr.getValue() == "extent" ||
                                  scalarRoleAttr.getValue() == "value";
        info.isStrideCompatible = scalarRoleAttr.getValue() == "stride" ||
                                  scalarRoleAttr.getValue() == "value";
      }
      argsByName.try_emplace(argName, info);
    }

    auto directionAttr =
        dyn_cast_or_null<StringAttr>(func.getArgAttr(i, "vc4value.direction"));
    if (isa<MemRefType, UnrankedMemRefType>(argType)) {
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
    if (isa<MemRefType, UnrankedMemRefType>(argType)) {
      continue;
    }
    if (!isPublicAbiScalarArgType(argType)) {
      emitArgError(func, i)
          << "public scalar argument type must be index, i32, or f32";
      sawError = true;
    }
  }

  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    Type argType = func.getArgument(i).getType();
    if (isa<UnrankedMemRefType>(argType)) {
      emitArgError(func, i) << "public memref argument must be a ranked "
                            << "memref with rank 1 or 2";
      sawError = true;
      continue;
    }
    auto memRefType = dyn_cast<MemRefType>(argType);
    if (!memRefType)
      continue;

    if (memRefType.getRank() == 0 || memRefType.getRank() > 2) {
      emitArgError(func, i) << "public memref argument rank must be 1 or 2";
      sawError = true;
    }
    if (!hasGlobalMemorySpace(memRefType)) {
      emitArgError(func, i)
          << "public memref argument memory space must be #vc4value.global";
      sawError = true;
    }
    if (!isSupportedMemRefElementType(memRefType.getElementType())) {
      emitArgError(func, i)
          << "public memref element type must be signless i8, signless i16, "
          << "signless i32, f16, or f32";
      sawError = true;
    }

    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (!getStridesAndOffset(memRefType, strides, offset)) {
      emitArgError(func, i)
          << "public memref layout must be identity or strided";
      sawError = true;
    } else if (ShapedType::isDynamic(offset) || offset != 0) {
      emitArgError(func, i)
          << "public memref layout offset must be static 0";
      sawError = true;
    }

    Attribute shapeAttr = func.getArgAttr(i, "vc4value.shape_args");
    unsigned dynamicDimCount = countDynamicDims(memRefType);
    SmallVector<StringRef> shapeNames;
    bool shapeAttrWellFormed = !shapeAttr || isArrayOfStringAttr(shapeAttr);
    if (shapeAttr && !shapeAttrWellFormed) {
      emitArgError(func, i)
          << "vc4value.shape_args must be an ArrayAttr of StringAttr";
      sawError = true;
    } else {
      shapeNames = getStringArrayValues(shapeAttr);
      if (dynamicDimCount == 0 && !shapeNames.empty()) {
        emitArgError(func, i)
            << "static public memref argument must not have nonempty "
            << "vc4value.shape_args";
        sawError = true;
      }
      if (dynamicDimCount > 0 && !shapeAttr) {
        emitArgError(func, i)
            << "dynamic public memref argument requires vc4value.shape_args";
        sawError = true;
      }
      if (shapeAttr && shapeNames.size() != dynamicDimCount) {
        emitArgError(func, i)
            << "vc4value.shape_args length must equal the number of dynamic "
            << "memref dimensions";
        sawError = true;
      }
      llvm::SmallDenseSet<StringRef> seenShapeNames;
      for (StringRef name : shapeNames) {
        if (!seenShapeNames.insert(name).second) {
          emitArgError(func, i)
              << "vc4value.shape_args must not contain duplicate names";
          sawError = true;
        }
        auto it = argsByName.find(name);
        if (it == argsByName.end()) {
          emitArgError(func, i)
              << "vc4value.shape_args entry '" << name
              << "' must name an existing scalar argument";
          sawError = true;
          continue;
        }
        if (!it->second.isScalar || !isIndexOrI32(it->second.type)) {
          emitArgError(func, i)
              << "vc4value.shape_args entry '" << name
              << "' must name an index or i32 scalar argument";
          sawError = true;
        }
        if (!it->second.isExtentCompatible) {
          emitArgError(func, i)
              << "vc4value.shape_args entry '" << name
              << "' must name a scalar with vc4value.scalar_role \"extent\" "
              << "or \"value\"";
          sawError = true;
        }
      }
    }

    Attribute strideAttr = func.getArgAttr(i, "vc4value.stride_args");
    SmallVector<StringRef> strideNames;
    bool strideAttrWellFormed = !strideAttr || isArrayOfStringAttr(strideAttr);
    if (strideAttr && !strideAttrWellFormed) {
      emitArgError(func, i)
          << "vc4value.stride_args must be an ArrayAttr of StringAttr";
      sawError = true;
    } else {
      strideNames = getStringArrayValues(strideAttr);
      unsigned dynamicStrideCount = llvm::count_if(
          strides, [](int64_t stride) { return ShapedType::isDynamic(stride); });
      if (dynamicStrideCount == 0 && !strideNames.empty()) {
        emitArgError(func, i)
            << "public memref argument without dynamic strides must not have "
            << "nonempty vc4value.stride_args";
        sawError = true;
      }
      if (dynamicStrideCount > 0 && strideNames.size() != dynamicStrideCount) {
        emitArgError(func, i)
            << "vc4value.stride_args length must equal the number of dynamic "
            << "memref strides";
        sawError = true;
      }
      llvm::SmallDenseSet<StringRef> seenStrideNames;
      for (StringRef name : strideNames) {
        if (!seenStrideNames.insert(name).second) {
          emitArgError(func, i)
              << "vc4value.stride_args must not contain duplicate names";
          sawError = true;
        }
        auto it = argsByName.find(name);
        if (it == argsByName.end()) {
          emitArgError(func, i)
              << "vc4value.stride_args entry '" << name
              << "' must name an existing scalar argument";
          sawError = true;
          continue;
        }
        if (!it->second.isScalar || !isIndexOrI32(it->second.type)) {
          emitArgError(func, i)
              << "vc4value.stride_args entry '" << name
              << "' must name an index or i32 scalar argument";
          sawError = true;
        }
        if (!it->second.isStrideCompatible) {
          emitArgError(func, i)
              << "vc4value.stride_args entry '" << name
              << "' must name a scalar with vc4value.scalar_role \"stride\" "
              << "or \"value\"";
          sawError = true;
        }
      }
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
      bool isFuncEntryBlock =
          op->getName().getStringRef() == "func.func" && &block == &region.front();
      for (BlockArgument arg : block.getArguments()) {
        checkType(op, arg.getType(), sawError);
        if (!isFuncEntryBlock && isMemRefValueSurfaceCFGType(arg.getType())) {
          op->emitError() << "memref block arguments are not in the Phase 8 "
                          << "VC4 value control-flow subset";
          sawError = true;
        }
      }
    }
  }
}

static void checkCFOp(Operation *op, bool &sawError) {
  if (auto cond = dyn_cast<cf::CondBranchOp>(op)) {
    if (!cond.getCondition().getType().isInteger(1)) {
      op->emitError()
          << "cf.cond_br condition must be scalar i1 in the Phase 8 "
          << "VC4 value control-flow subset";
      sawError = true;
    }
    for (Value operand : cond.getTrueDestOperands()) {
      if (isMemRefValueSurfaceCFGType(operand.getType())) {
        op->emitError()
            << "memref successor operands are not in the Phase 8 "
            << "VC4 value control-flow subset";
        sawError = true;
      }
    }
    for (Value operand : cond.getFalseDestOperands()) {
      if (isMemRefValueSurfaceCFGType(operand.getType())) {
        op->emitError()
            << "memref successor operands are not in the Phase 8 "
            << "VC4 value control-flow subset";
        sawError = true;
      }
    }
    return;
  }

  if (auto br = dyn_cast<cf::BranchOp>(op)) {
    for (Value operand : br.getDestOperands()) {
      if (isMemRefValueSurfaceCFGType(operand.getType())) {
        op->emitError()
            << "memref successor operands are not in the Phase 8 "
            << "VC4 value control-flow subset";
        sawError = true;
      }
    }
  }
}

static void checkMemRefDimOp(memref::DimOp dimOp, bool &sawError) {
  auto func = dyn_cast_or_null<func::FuncOp>(getEnclosingFunc(dimOp));
  if (!func || !func->hasAttr("vc4value.kernel")) {
    emitMemRefDimError(dimOp) << "operation requires an enclosing public "
                              << "vc4value.kernel function";
    sawError = true;
    return;
  }

  auto memRefType = dyn_cast<MemRefType>(dimOp.getSource().getType());
  if (!memRefType || !hasGlobalMemorySpace(memRefType)) {
    emitMemRefDimError(dimOp)
        << "source must be a public #vc4value.global memref argument";
    sawError = true;
    return;
  }

  std::optional<unsigned> argIndex = getPublicMemRefArgIndex(func, dimOp.getSource());
  if (!argIndex) {
    emitMemRefDimError(dimOp)
        << "source must be a public #vc4value.global memref argument";
    sawError = true;
    return;
  }

  APInt dimValue;
  if (!matchPattern(dimOp.getIndex(), m_ConstantInt(&dimValue))) {
    emitMemRefDimError(dimOp) << "dimension index must be a constant index";
    sawError = true;
    return;
  }
  int64_t dim = dimValue.getSExtValue();
  if (dim < 0 || dim >= memRefType.getRank()) {
    emitMemRefDimError(dimOp) << "dimension index is out of range";
    sawError = true;
    return;
  }
  if (!hasShapeArgForDynamicDim(func, *argIndex, memRefType,
                                static_cast<unsigned>(dim))) {
    emitMemRefDimError(dimOp)
        << "dynamic dimension requires vc4value.shape_args mapping";
    sawError = true;
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

      if (auto dimOp = dyn_cast<memref::DimOp>(op))
        checkMemRefDimOp(dimOp, sawError);

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

      if (dialect == "cf" && isAllowedCFOp(opName))
        checkCFOp(op, sawError);

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
