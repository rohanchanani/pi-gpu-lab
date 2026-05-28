//===- VC4TileOps.cpp - VC4Tile operation implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vc4/Dialect/VC4Tile/IR/VC4TileOps.h"
#include "vc4/Dialect/VC4Tile/IR/VC4TileTypes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace mlir;
using namespace mlir::vc4tile;

namespace {

static bool getBoolAttr(Operation *op, StringRef name);
static std::optional<int64_t> getI32Attr(Operation *op, StringRef name);
static LogicalResult verifySharedKernelContract(Operation *op);

static LogicalResult emitTypeError(Operation *op, Type type, StringRef role,
                                   StringRef expected) {
  return op->emitOpError() << role << " type must be " << expected << "; got "
                           << type;
}

static LogicalResult verifyScalarId(Operation *op, Type type, StringRef role) {
  if (isVC4TileScalarIdType(type))
    return success();
  return emitTypeError(op, type, role, "i32 or index");
}

static LogicalResult verifyNoUniformIndex(Operation *op, StringRef identityName) {
  if (!op->getAttr("uniform_index"))
    return success();
  return op->emitOpError()
         << identityName
         << " is a semantic identity op and must not carry uniform_index; "
            "must not carry 'uniform_index'; physical uniform slots are assigned "
            "only after VC4Tile lowering";
}

static LogicalResult verifyVector16I32(Operation *op, Type type,
                                       StringRef role) {
  if (isVC4TileVector16I32Type(type))
    return success();
  return emitTypeError(op, type, role, "vector<16xi32>");
}

static LogicalResult verifyVector16I1(Operation *op, Type type,
                                      StringRef role) {
  if (isVC4TileVector16I1Type(type))
    return success();
  return emitTypeError(op, type, role, "vector<16xi1>");
}

static LogicalResult verifyVector16Data(Operation *op, Type type,
                                        StringRef role) {
  if (isVC4TileVector16DataType(type))
    return success();
  return emitTypeError(op, type, role,
                       "vector<16xi32> or vector<16xf32>");
}

static LogicalResult verifySharedTile(Operation *op, Type type,
                                      StringRef role) {
  if (isVC4TileSharedTileType(type))
    return success();
  return emitTypeError(op, type, role, "!vc4tile.shared_tile");
}

static LogicalResult verifyElemBytes4(Operation *op, IntegerAttr attr) {
  if (attr && attr.getInt() == 4)
    return success();
  return op->emitOpError("elem_bytes must be 4 for the M4 32-bit contract");
}

static LogicalResult verifyRowsAttr(Operation *op, IntegerAttr attr,
                                    StringRef name) {
  if (!attr)
    return op->emitOpError() << name << " attribute is required";
  int64_t rows = attr.getInt();
  if (rows < 1 || rows > 64)
    return op->emitOpError() << name << " must be in range [1, 64]";
  return success();
}

static LogicalResult collectPositiveI64Array(Operation *op, ArrayAttr attr,
                                             StringRef name,
                                             SmallVectorImpl<int64_t> &values) {
  if (!attr)
    return op->emitOpError() << name << " attribute is required";
  if (attr.empty())
    return op->emitOpError() << name << " must not be empty";
  for (Attribute element : attr) {
    auto intAttr = llvm::dyn_cast<IntegerAttr>(element);
    if (!intAttr)
      return op->emitOpError() << name << " entries must be integers";
    int64_t value = intAttr.getInt();
    if (value <= 0)
      return op->emitOpError() << name << " entries must be positive";
    values.push_back(value);
  }
  return success();
}

static LogicalResult verifyM532BitTypeAttr(Operation *op, StringRef name,
                                           StringRef diagnosticRole,
                                           Type &type) {
  auto attr = op->getAttrOfType<TypeAttr>(name);
  if (!attr)
    return op->emitOpError() << name << " attribute is required";
  type = attr.getValue();
  if (isVC4TileSupportedM532BitElementType(type))
    return success();
  if (name == "storage_type") {
    return op->emitOpError()
           << "M5 supports only 32-bit executable tile storage; got "
           << "storage_type = " << type;
  }
  return op->emitOpError() << "M5 supports only 32-bit executable tile "
                           << diagnosticRole << "; got " << name << " = "
                           << type;
}

static LogicalResult verifyTileDescriptorLayout(Operation *op,
                                                LayoutAttr layout,
                                                ArrayAttr stridesAttr,
                                                unsigned rank) {
  if (!layout)
    return op->emitOpError("layout attribute is required");
  if (layout.getValue() != Layout::affine_2d) {
    if (stridesAttr)
      return op->emitOpError(
          "strides are only supported for affine_2d layout");
    return success();
  }

  SmallVector<int64_t, 2> strides;
  if (failed(collectPositiveI64Array(op, stridesAttr, "strides", strides)))
    return failure();
  if (strides.size() != rank)
    return op->emitOpError("affine_2d strides length must match rank");
  if (!op->getAttrOfType<OffsetUnitAttr>("offset_unit"))
    return op->emitOpError("affine_2d layout requires offset_unit");
  return success();
}

static LogicalResult verifyOptionalPositiveI64Array(Operation *op,
                                                     StringRef name) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return success();
  SmallVector<int64_t, 4> values;
  return collectPositiveI64Array(op, attr, name, values);
}

static LogicalResult verifyOptionalNonNegativeI64Array(Operation *op,
                                                       StringRef name) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return success();
  for (Attribute element : attr) {
    auto intAttr = llvm::dyn_cast<IntegerAttr>(element);
    if (!intAttr)
      return op->emitOpError() << name << " entries must be integers";
    if (intAttr.getInt() < 0)
      return op->emitOpError() << name << " entries must be non-negative";
  }
  return success();
}

static LogicalResult verifyRequiredNonNegativeI64Array(Operation *op,
                                                       StringRef name) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return op->emitOpError() << name << " attribute is required";
  if (attr.empty())
    return op->emitOpError() << name << " must not be empty";
  for (Attribute element : attr) {
    auto intAttr = llvm::dyn_cast<IntegerAttr>(element);
    if (!intAttr)
      return op->emitOpError() << name << " entries must be integers";
    if (intAttr.getInt() < 0)
      return op->emitOpError() << name << " entries must be non-negative";
  }
  return success();
}

static LogicalResult verifyTileSurfaceShape(Operation *op,
                                            SmallVectorImpl<int64_t> &shape) {
  auto shapeAttr = op->getAttrOfType<ArrayAttr>("shape");
  if (failed(collectPositiveI64Array(op, shapeAttr, "shape", shape)))
    return failure();
  if (shape.size() > 2)
    return op->emitOpError("M5 copy planner v1 supports only rank <= 2 tile shapes");
  return success();
}

static bool isVC4TileSurfaceCarrierType(Type type) {
  return isVC4TileVector16DataType(type) || isVC4TileTileType(type) ||
         isVC4TileSharedTileType(type);
}

static LogicalResult verifySurfaceCarrier(Operation *op, Type type,
                                          StringRef role) {
  if (isVC4TileSurfaceCarrierType(type))
    return success();
  return emitTypeError(op, type, role,
                       "vector<16xi32>, vector<16xf32>, !vc4tile.tile, or !vc4tile.shared_tile");
}

static bool isVC4TileCopyOperandType(Type type) {
  return isVC4TileSurfaceCarrierType(type) || isVC4TileScalarIdType(type) ||
         isVC4TileVector16I1Type(type);
}

static LogicalResult verifyM5Exact32Metadata(Operation *op) {
  SmallVector<int64_t, 4> shape;
  if (failed(verifyTileSurfaceShape(op, shape)))
    return failure();

  Type elementType;
  Type storageType;
  if (failed(verifyM532BitTypeAttr(op, "element_type", "element types",
                                   elementType)) ||
      failed(verifyM532BitTypeAttr(op, "storage_type", "storage",
                                   storageType)))
    return failure();

  if (auto expressed = op->getAttrOfType<TypeAttr>("expressed_type")) {
    Type expressedType = expressed.getValue();
    if (!isVC4TileSupportedM532BitElementType(expressedType)) {
      return op->emitOpError()
             << "M5 supports only 32-bit executable tile expressed types; got "
             << "expressed_type = " << expressedType;
    }
  }
  if (auto accumulator = op->getAttrOfType<TypeAttr>("accumulator_type")) {
    Type accumulatorType = accumulator.getValue();
    if (!isVC4TileSupportedM532BitElementType(accumulatorType)) {
      return op->emitOpError()
             << "M5 supports only 32-bit executable tile accumulator types; got "
             << "accumulator_type = " << accumulatorType;
    }
  }

  auto precision = op->getAttrOfType<PrecisionAttr>("precision");
  if (!precision)
    return op->emitOpError("precision attribute is required");
  if (precision.getValue() != Precision::exact_32)
    return op->emitOpError("M5 executable precision_policy must be exact_32");

  if (auto packing = op->getAttrOfType<PackingAttr>("packing")) {
    if (packing.getValue() != Packing::none)
      return op->emitOpError("M5 supports only packing = none");
  }

  if (auto boundary = op->getAttrOfType<BoundaryPolicyAttr>("boundary")) {
    if (boundary.getValue() == BoundaryPolicy::zero ||
        boundary.getValue() == BoundaryPolicy::clamp ||
        boundary.getValue() == BoundaryPolicy::reject)
      return op->emitOpError(
          "boundary policy zero/clamp/reject is not implemented in M5");
  }

  for (StringRef forbidden : {"quantization", "scale", "zero_point",
                             "scale_granularity", "zero_point_policy"}) {
    if (op->getAttr(forbidden))
      return op->emitOpError(
          "quantization scales and zero points are not executable in M5");
  }
  return success();
}


static LogicalResult verifyTileComputeMetadata(Operation *op) {
  if (failed(verifyM5Exact32Metadata(op)))
    return failure();

  SmallVector<int64_t, 4> shape;
  if (failed(verifyTileSurfaceShape(op, shape)))
    return failure();
  int64_t elements = 1;
  for (int64_t extent : shape)
    elements *= extent;
  if (elements != 16)
    return op->emitOpError(
        "elementwise tile compute supports exactly 16 lanes in M5");

  auto memorySpace = op->getAttrOfType<MemorySpaceAttr>("memory_space");
  if (!memorySpace)
    return op->emitOpError("memory_space attribute is required");
  if (memorySpace.getValue() != MemorySpace::register_space) {
    return op->emitOpError(
        "tile compute operations require memory_space = "
        "#vc4tile.memory_space<register>");
  }

  Type resultType = op->getResult(0).getType();
  if (!isVC4TileVector16DataType(resultType))
    return emitTypeError(op, resultType, "result",
                         "vector<16xi32> or vector<16xf32>");

  auto vectorType = llvm::cast<VectorType>(resultType);
  auto elementType = op->getAttrOfType<TypeAttr>("element_type");
  if (!elementType)
    return op->emitOpError("element_type attribute is required");
  if (vectorType.getElementType() != elementType.getValue()) {
    return op->emitOpError()
           << "result element type must match element_type; got result "
           << resultType << " and element_type = " << elementType.getValue();
  }
  return success();
}

static bool isVC4TileScalarComputeType(Type type) {
  return type.isSignlessInteger(32) || type.isF32();
}

static LogicalResult verifyTileScalarComputeType(Operation *op, Type type,
                                                 StringRef role) {
  if (isVC4TileScalarComputeType(type))
    return success();
  return emitTypeError(op, type, role, "i32 or f32");
}

static LogicalResult verifyTileComputeValueMatchesElement(Operation *op,
                                                          Attribute value,
                                                          Type elementType) {
  if (elementType.isSignlessInteger(32)) {
    auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
    if (!intAttr)
      return op->emitOpError("tile_fill i32 value must be an integer attribute");
    if (!intAttr.getType().isSignlessInteger(32))
      return op->emitOpError("tile_fill integer value must be i32");
    return success();
  }
  if (elementType.isF32()) {
    auto floatAttr = llvm::dyn_cast<FloatAttr>(value);
    if (!floatAttr)
      return op->emitOpError("tile_fill f32 value must be a float attribute");
    if (!floatAttr.getType().isF32())
      return op->emitOpError("tile_fill float value must be f32");
    return success();
  }
  return op->emitOpError("tile_fill supports only i32 or f32 element_type");
}

static LogicalResult verifyTileBinaryComputeOperands(Operation *op,
                                                     Value lhs, Value rhs,
                                                     Value result) {
  if (failed(verifySurfaceCarrier(op, lhs.getType(), "lhs")) ||
      failed(verifySurfaceCarrier(op, rhs.getType(), "rhs")) ||
      failed(verifySurfaceCarrier(op, result.getType(), "result")))
    return failure();
  if (lhs.getType() != result.getType() || rhs.getType() != result.getType())
    return op->emitOpError(
        "tile binary operands and result must have identical types in M5");
  if (!isVC4TileVector16DataType(result.getType()))
    return emitTypeError(op, result.getType(), "result",
                         "vector<16xi32> or vector<16xf32>");
  return success();
}

static LogicalResult verifyI32AttrInRange(Operation *op, StringRef name,
                                          int64_t minValue,
                                          int64_t maxValue) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return op->emitOpError() << name << " attribute is required";
  int64_t value = attr.getInt();
  if (value < minValue || value > maxValue)
    return op->emitOpError() << name << " must be in range [" << minValue
                             << ", " << maxValue << "]";
  return success();
}

static LogicalResult verifyContractionLayoutAttr(Operation *op, StringRef name,
                                                 bool allowRhsTransposed) {
  auto layout = op->getAttrOfType<LayoutAttr>(name);
  if (!layout)
    return op->emitOpError() << name << " attribute is required";
  Layout value = layout.getValue();
  if (value == Layout::row_major)
    return success();
  if (allowRhsTransposed &&
      (value == Layout::col_major || value == Layout::transposed_view))
    return success();
  return op->emitOpError()
         << "M5 tile contractions support only row_major lhs/acc layouts and "
            "row_major, col_major, or transposed_view rhs layouts";
}

static LogicalResult verifyContractionDimsAttr(Operation *op) {
  auto dims = op->getAttrOfType<ArrayAttr>("contracting_dims");
  if (!dims)
    return success();
  if (dims.size() != 2)
    return op->emitOpError(
        "contracting_dims must be [[1], [0]] for M5 matrix contractions");
  auto lhsDims = llvm::dyn_cast<ArrayAttr>(dims[0]);
  auto rhsDims = llvm::dyn_cast<ArrayAttr>(dims[1]);
  if (!lhsDims || !rhsDims || lhsDims.size() != 1 || rhsDims.size() != 1)
    return op->emitOpError(
        "contracting_dims must be [[1], [0]] for M5 matrix contractions");
  auto lhsDim = llvm::dyn_cast<IntegerAttr>(lhsDims[0]);
  auto rhsDim = llvm::dyn_cast<IntegerAttr>(rhsDims[0]);
  if (!lhsDim || !rhsDim || lhsDim.getInt() != 1 || rhsDim.getInt() != 0)
    return op->emitOpError(
        "contracting_dims must be [[1], [0]] for M5 matrix contractions");
  return success();
}

static LogicalResult verifyIteratorTypesAttr(Operation *op) {
  auto iterators = op->getAttrOfType<ArrayAttr>("iterator_types");
  if (!iterators)
    return success();
  if (iterators.size() != 3)
    return op->emitOpError(
        "iterator_types must be [\"parallel\", \"parallel\", \"reduction\"] "
        "for M5 matrix contractions");
  constexpr llvm::StringLiteral expected[3] = {"parallel", "parallel",
                                               "reduction"};
  for (auto [index, attr] : llvm::enumerate(iterators)) {
    auto stringAttr = llvm::dyn_cast<StringAttr>(attr);
    if (!stringAttr || stringAttr.getValue() != expected[index])
      return op->emitOpError(
          "iterator_types must be [\"parallel\", \"parallel\", \"reduction\"] "
          "for M5 matrix contractions");
  }
  return success();
}

static LogicalResult verifyTileDotContractMetadata(Operation *op,
                                                   bool matrixForm) {
  if (failed(verifyM5Exact32Metadata(op)))
    return failure();

  auto memorySpace = op->getAttrOfType<MemorySpaceAttr>("memory_space");
  if (!memorySpace)
    return op->emitOpError("memory_space attribute is required");
  if (memorySpace.getValue() != MemorySpace::register_space)
    return op->emitOpError(
        "tile contraction operations require memory_space = #vc4tile.memory_space<register>");

  Type resultType = op->getResult(0).getType();
  if (!isVC4TileVector16DataType(resultType))
    return emitTypeError(op, resultType, "result",
                         "vector<16xi32> or vector<16xf32>");
  auto vectorType = llvm::cast<VectorType>(resultType);
  auto elementType = op->getAttrOfType<TypeAttr>("element_type");
  if (!elementType)
    return op->emitOpError("element_type attribute is required");
  if (vectorType.getElementType() != elementType.getValue())
    return op->emitOpError() << "result element type must match element_type; got result "
                             << resultType << " and element_type = "
                             << elementType.getValue();

  auto accumulatorType = op->getAttrOfType<TypeAttr>("accumulator_type");
  if (!accumulatorType)
    return op->emitOpError("accumulator_type attribute is required");
  if (accumulatorType.getValue() != elementType.getValue())
    return op->emitOpError(
        "M5 tile contractions require accumulator_type to match element_type");

  SmallVector<int64_t, 4> shape;
  if (failed(verifyTileSurfaceShape(op, shape)))
    return failure();
  if (matrixForm) {
    if (shape.size() != 2)
      return op->emitOpError(
          "tile_contract/tile_matmul require rank-2 static tile shapes in M5");
  } else if (shape.size() != 2 || shape[0] != 1 || shape[1] != 16) {
    return op->emitOpError("tile_dot supports only shape = [1, 16] in M5");
  }

  int64_t elements = 1;
  for (int64_t extent : shape)
    elements *= extent;
  if (elements != 16)
    return op->emitOpError(
        "M5 tile contractions support exactly sixteen 32-bit lanes");

  if (failed(verifyI32AttrInRange(op, "k", 1, 16)))
    return failure();

  if (matrixForm) {
    if (failed(verifyI32AttrInRange(op, "m", 1, 16)) ||
        failed(verifyI32AttrInRange(op, "n", 1, 16)))
      return failure();
    auto mAttr = op->getAttrOfType<IntegerAttr>("m");
    auto nAttr = op->getAttrOfType<IntegerAttr>("n");
    auto kAttr = op->getAttrOfType<IntegerAttr>("k");
    if (mAttr && nAttr && mAttr.getInt() * nAttr.getInt() != 16)
      return op->emitOpError("M5 tile_contract/tile_matmul require m * n = 16");
    if (mAttr && kAttr && mAttr.getInt() * kAttr.getInt() != 16)
      return op->emitOpError(
          "M5 tile_contract/tile_matmul require m * k = 16 for single-vector lhs carriers");
    if (kAttr && nAttr && kAttr.getInt() * nAttr.getInt() != 16)
      return op->emitOpError(
          "M5 tile_contract/tile_matmul require k * n = 16 for single-vector rhs carriers");
    if (shape[0] != mAttr.getInt() || shape[1] != nAttr.getInt())
      return op->emitOpError(
          "tile_contract/tile_matmul shape must match [m, n] in M5");
    if (failed(verifyContractionLayoutAttr(op, "lhs_layout",
                                           /*allowRhsTransposed=*/false)) ||
        failed(verifyContractionLayoutAttr(op, "rhs_layout",
                                           /*allowRhsTransposed=*/true)) ||
        failed(verifyContractionLayoutAttr(op, "acc_layout",
                                           /*allowRhsTransposed=*/false)))
      return failure();
    if (failed(verifyContractionDimsAttr(op)) ||
        failed(verifyIteratorTypesAttr(op)))
      return failure();
  } else {
    auto layout = op->getAttrOfType<LayoutAttr>("layout");
    if (!layout)
      return op->emitOpError("layout attribute is required");
    if (layout.getValue() != Layout::row_major)
      return op->emitOpError("tile_dot supports only row_major layout in M5");
  }
  return success();
}

static LogicalResult verifyTileDotOperands(Operation *op, Value lhs, Value rhs,
                                           Value mask, Value result) {
  if (failed(verifySurfaceCarrier(op, lhs.getType(), "lhs")) ||
      failed(verifySurfaceCarrier(op, rhs.getType(), "rhs")) ||
      failed(verifyVector16I1(op, mask.getType(), "mask")) ||
      failed(verifySurfaceCarrier(op, result.getType(), "result")))
    return failure();
  if (lhs.getType() != result.getType() || rhs.getType() != result.getType())
    return op->emitOpError(
        "tile_dot lhs, rhs, and result must have identical vector types in M5");
  if (!isVC4TileVector16DataType(result.getType()))
    return emitTypeError(op, result.getType(), "result",
                         "vector<16xi32> or vector<16xf32>");
  return success();
}

static LogicalResult verifyTileContractOperands(Operation *op, Value lhs,
                                                Value rhs, Value acc,
                                                Value mask, Value result) {
  if (failed(verifySurfaceCarrier(op, lhs.getType(), "lhs")) ||
      failed(verifySurfaceCarrier(op, rhs.getType(), "rhs")) ||
      failed(verifySurfaceCarrier(op, acc.getType(), "acc")) ||
      failed(verifyVector16I1(op, mask.getType(), "mask")) ||
      failed(verifySurfaceCarrier(op, result.getType(), "result")))
    return failure();
  if (lhs.getType() != result.getType() || rhs.getType() != result.getType() ||
      acc.getType() != result.getType())
    return op->emitOpError(
        "tile contraction lhs, rhs, acc, and result must have identical vector types in M5");
  if (!isVC4TileVector16DataType(result.getType()))
    return emitTypeError(op, result.getType(), "result",
                         "vector<16xi32> or vector<16xf32>");
  return success();
}

static LogicalResult verifyReductionAxis(Operation *op, bool requireAxis) {
  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  if (!axisAttr)
    return requireAxis ? op->emitOpError("tile_reduce requires an axis attribute")
                       : success();

  SmallVector<int64_t, 4> shape;
  if (failed(verifyTileSurfaceShape(op, shape)))
    return failure();
  int64_t axis = axisAttr.getInt();
  if (axis < 0 || axis >= static_cast<int64_t>(shape.size()))
    return op->emitOpError("reduction axis must name a dimension of shape");
  return success();
}

static LogicalResult verifyTileReductionCommon(Operation *op, Value input,
                                               Value mask, Value result,
                                               bool requireAxis) {
  if (failed(verifyTileComputeMetadata(op)) ||
      failed(verifySurfaceCarrier(op, input.getType(), "input")) ||
      failed(verifyVector16I1(op, mask.getType(), "mask")) ||
      failed(verifySurfaceCarrier(op, result.getType(), "result")))
    return failure();
  if (input.getType() != result.getType())
    return op->emitOpError(
        "reduction input and result must have identical types in M5");
  if (!isVC4TileVector16DataType(result.getType()))
    return emitTypeError(op, result.getType(), "result",
                         "vector<16xi32> or vector<16xf32>");
  auto kind = op->getAttrOfType<ReduceKindAttr>("kind");
  if (!kind)
    return op->emitOpError("reduction kind attribute is required");
  if (kind.getValue() != ReduceKind::add)
    return op->emitOpError(
        "tile reductions currently support only kind = #vc4tile.reduce_kind<add> in M5");
  return verifyReductionAxis(op, requireAxis);
}

static LogicalResult verifyBlockReductionKernelContract(Operation *op) {
  if (failed(verifySharedKernelContract(op)))
    return failure();
  auto kernel = op->getParentOfType<KernelOp>();
  if (!getBoolAttr(kernel.getOperation(), "uses_barrier"))
    return op->emitOpError(
        "block_reduce requires parent vc4tile.kernel to set uses_barrier = true");
  if (!getBoolAttr(kernel.getOperation(), "require_full_block_residency"))
    return op->emitOpError(
        "block_reduce requires parent vc4tile.kernel to set require_full_block_residency = true");
  if (getI32Attr(kernel.getOperation(), "semaphores_per_block").value_or(0) != 4)
    return op->emitOpError(
        "block_reduce requires parent semaphores_per_block = 4");
  if (getI32Attr(kernel.getOperation(), "vpm_rows_per_block").value_or(0) < 1)
    return op->emitOpError(
        "block_reduce requires at least one parent VPM row");
  return success();
}

static LogicalResult verifyTileMovementLayout(Operation *op, StringRef name) {
  auto layout = op->getAttrOfType<LayoutAttr>(name);
  if (!layout)
    return op->emitOpError() << name << " attribute is required";
  return success();
}

static LogicalResult verifyStaticIndexArray(Operation *op, StringRef name,
                                            unsigned expectedSize) {
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr)
    return op->emitOpError() << name << " attribute is required";
  SmallVector<int64_t, 4> values;
  if (failed(collectPositiveI64Array(op, attr, name, values)))
    return failure();
  if (expectedSize && values.size() != expectedSize)
    return op->emitOpError() << name << " length must be " << expectedSize;
  return success();
}

static LogicalResult verifyTransposePermutation(Operation *op,
                                                ArrayAttr permutation) {
  if (!permutation)
    return op->emitOpError("permutation attribute is required");
  if (permutation.size() != 2)
    return op->emitOpError("transpose_view supports only rank-2 permutations in M5");
  SmallVector<int64_t, 2> values;
  for (Attribute attr : permutation) {
    auto intAttr = llvm::dyn_cast<IntegerAttr>(attr);
    if (!intAttr)
      return op->emitOpError("permutation entries must be integers");
    values.push_back(intAttr.getInt());
  }
  if (values[0] != 1 || values[1] != 0)
    return op->emitOpError("M5 transpose_view supports only permutation [1, 0]");
  return success();
}



static bool isMaskAllOp(Value value) {
  Operation *def = value.getDefiningOp();
  return def && def->getName().getStringRef() == "vc4tile.mask_all";
}

static bool isTailMaskOp(Value value) {
  Operation *def = value.getDefiningOp();
  return def && def->getName().getStringRef() == "vc4tile.tail_mask";
}

static LogicalResult verifyBoundaryPolicyMatchesMask(Operation *op,
                                                      Value mask) {
  auto boundary = op->getAttrOfType<BoundaryPolicyAttr>("boundary");
  if (!boundary)
    return success();
  switch (boundary.getValue()) {
  case BoundaryPolicy::exact:
    if (!isMaskAllOp(mask))
      return op->emitOpError(
          "boundary policy exact requires a vc4tile.mask_all mask");
    return success();
  case BoundaryPolicy::tail_predicated:
    if (!isTailMaskOp(mask))
      return op->emitOpError(
          "boundary policy tail_predicated requires a vc4tile.tail_mask mask");
    return success();
  case BoundaryPolicy::zero:
  case BoundaryPolicy::clamp:
  case BoundaryPolicy::reject:
    return op->emitOpError(
        "boundary policy zero/clamp/reject is not implemented in M5");
  }
  return success();
}

static LogicalResult verifyCopyTileBoundaryPolicy(Operation *op) {
  auto boundary = op->getAttrOfType<BoundaryPolicyAttr>("boundary");
  if (!boundary)
    return success();
  switch (boundary.getValue()) {
  case BoundaryPolicy::exact:
    return success();
  case BoundaryPolicy::tail_predicated:
    for (Value operand : op->getOperands()) {
      if (isVC4TileVector16I1Type(operand.getType()))
        return success();
    }
    return op->emitOpError(
        "boundary policy tail_predicated requires a vector<16xi1> mask operand");
  case BoundaryPolicy::zero:
  case BoundaryPolicy::clamp:
  case BoundaryPolicy::reject:
    return op->emitOpError(
        "boundary policy zero/clamp/reject is not implemented in M5");
  }
  return success();
}

static DictionaryAttr getResourceIntent(Operation *op) {
  return op->getAttrOfType<DictionaryAttr>("resource_intent");
}

static std::optional<bool> getResourceIntentBool(Operation *op,
                                                 StringRef name) {
  DictionaryAttr intent = getResourceIntent(op);
  if (!intent)
    return std::nullopt;
  auto attr = intent.getAs<BoolAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

static std::optional<int64_t> getResourceIntentI32(Operation *op,
                                                   StringRef name) {
  DictionaryAttr intent = getResourceIntent(op);
  if (!intent)
    return std::nullopt;
  auto attr = intent.getAs<IntegerAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static LogicalResult verifyResourceIntent(Operation *op) {
  DictionaryAttr intent = getResourceIntent(op);
  if (!intent)
    return success();

  for (NamedAttribute named : intent) {
    StringRef key = named.getName().getValue();
    Attribute value = named.getValue();
    if (key == "uses_shared_vpm" || key == "uses_barrier") {
      if (!llvm::isa<BoolAttr>(value))
        return op->emitOpError() << "resource_intent." << key
                                 << " must be a bool attribute";
      continue;
    }
    if (key == "vpm_rows" || key == "vpm_bytes" || key == "semaphores" ||
        key == "warps_per_block") {
      auto intAttr = llvm::dyn_cast<IntegerAttr>(value);
      if (!intAttr)
        return op->emitOpError() << "resource_intent." << key
                                 << " must be an integer attribute";
      if (intAttr.getInt() < 0)
        return op->emitOpError() << "resource_intent." << key
                                 << " must be non-negative";
      continue;
    }
    return op->emitOpError() << "unsupported resource_intent key '" << key
                             << "'";
  }

  auto checkBool = [&](StringRef intentName, StringRef attrName)
      -> LogicalResult {
    std::optional<bool> value = getResourceIntentBool(op, intentName);
    if (!value)
      return success();
    if (auto direct = op->getAttrOfType<BoolAttr>(attrName)) {
      if (direct.getValue() != *value)
        return op->emitOpError() << "resource_intent." << intentName
                                 << " must match " << attrName;
    }
    return success();
  };
  auto checkI32 = [&](StringRef intentName, StringRef attrName)
      -> LogicalResult {
    std::optional<int64_t> value = getResourceIntentI32(op, intentName);
    if (!value)
      return success();
    if (auto direct = op->getAttrOfType<IntegerAttr>(attrName)) {
      if (direct.getInt() != *value)
        return op->emitOpError() << "resource_intent." << intentName
                                 << " must match " << attrName;
    }
    return success();
  };

  if (failed(checkBool("uses_shared_vpm", "uses_shared_vpm")) ||
      failed(checkBool("uses_barrier", "uses_barrier")) ||
      failed(checkI32("vpm_rows", "vpm_rows_per_block")) ||
      failed(checkI32("vpm_bytes", "vpm_bytes_per_block")) ||
      failed(checkI32("semaphores", "semaphores_per_block")) ||
      failed(checkI32("warps_per_block", "warps_per_block_max")))
    return failure();
  return success();
}

static bool getBoolAttrOrResourceIntent(Operation *op, StringRef attrName,
                                        bool fallback = false) {
  if (auto attr = op->getAttrOfType<BoolAttr>(attrName))
    return attr.getValue();
  if (std::optional<bool> value = getResourceIntentBool(op, attrName))
    return *value;
  return fallback;
}

static std::optional<int64_t> getI32AttrOrResourceIntent(Operation *op,
                                                         StringRef attrName,
                                                         StringRef intentName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return getResourceIntentI32(op, intentName);
}

static std::optional<int64_t> getI32Attr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static bool getBoolAttr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<BoolAttr>(name);
  return attr && attr.getValue();
}

static std::optional<ScheduleMode> getKernelScheduleMode(KernelOp kernel) {
  if (auto attr = kernel.getScheduleModeAttr())
    return attr.getValue();
  return std::nullopt;
}

static bool isKernelCooperative(KernelOp kernel) {
  std::optional<ScheduleMode> mode = getKernelScheduleMode(kernel);
  return mode && *mode == ScheduleMode::cooperative_block;
}

static bool isKernelIndependent(KernelOp kernel) {
  std::optional<ScheduleMode> mode = getKernelScheduleMode(kernel);
  return !mode || *mode == ScheduleMode::independent_vector;
}

static LogicalResult verifyInsideKernel(Operation *op) {
  if (op->getParentOfType<KernelOp>())
    return success();
  return op->emitOpError("must appear inside vc4tile.kernel");
}

static LogicalResult verifySharedKernelContract(Operation *op) {
  auto kernel = op->getParentOfType<KernelOp>();
  if (!kernel)
    return op->emitOpError("must appear inside vc4tile.kernel");
  if (!getBoolAttr(kernel.getOperation(), "uses_shared_vpm")) {
    return op->emitOpError(
        "requires parent vc4tile.kernel to set uses_shared_vpm = true");
  }
  if (!isKernelCooperative(kernel)) {
    return op->emitOpError(
        "requires parent vc4tile.kernel schedule_mode = "
        "#vc4tile.schedule_mode<cooperative_block>");
  }
  return success();
}

static LogicalResult verifyMemoryAccessNotGeneric(Operation *op,
                                                  MemoryAccessAttr accessAttr) {
  if (accessAttr && accessAttr.getValue() == MemoryAccess::generic) {
    return op->emitOpError(
        "generic per-lane store access is outside the M4 hardware-lowered "
        "contract; use coalesced or affine_contiguous");
  }
  return success();
}

static LogicalResult verifyLoadAccessNotGeneric(Operation *op,
                                                MemoryAccessAttr accessAttr) {
  if (accessAttr && accessAttr.getValue() == MemoryAccess::generic) {
    return op->emitOpError(
        "generic per-lane load access is outside the M4 hardware-lowered "
        "contract; use coalesced or affine_contiguous");
  }
  return success();
}

static LogicalResult verifyMemorySpace(Operation *op, MemorySpaceAttr spaceAttr,
                                       MemorySpace expected,
                                       StringRef expectedName) {
  if (!spaceAttr || spaceAttr.getValue() == expected)
    return success();
  return op->emitOpError() << "memory_space must be #vc4tile.memory_space<"
                           << expectedName << ">";
}

static bool isValidABIName(StringRef name) {
  if (name.empty())
    return false;
  auto isAlpha = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  };
  auto isDigit = [](char c) { return c >= '0' && c <= '9'; };
  if (!(isAlpha(name.front()) || name.front() == '_'))
    return false;
  for (char c : name.drop_front()) {
    if (!(isAlpha(c) || isDigit(c) || c == '_'))
      return false;
  }
  return true;
}

static bool isReservedABIName(StringRef name) {
  return name == "logical_request" || name == "total_requests" ||
         name == "logical_block_id" || name == "logical_warp_id" ||
         name == "warps_per_block";
}

static StringAttr getABINameAttr(DictionaryAttr dict) {
  if (!dict)
    return {};
  if (auto name = dict.getAs<StringAttr>("name"))
    return name;
  return dict.getAs<StringAttr>("abi_name");
}

static LogicalResult verifyKernelFormalArgumentContract(KernelOp kernel) {
  Operation *op = kernel.getOperation();
  if (kernel.getBody().empty())
    return success();

  Block &entry = kernel.getBody().front();
  auto argAttrs = op->getAttrOfType<ArrayAttr>("arg_attrs");

  if (entry.getNumArguments() > 0 && op->getAttr("launch_abi")) {
    return kernel.emitOpError(
        "kernels with formal arguments must not carry manual launch_abi");
  }

  if (auto typeAttr = op->getAttrOfType<TypeAttr>("function_type")) {
    if (auto fnType = llvm::dyn_cast<FunctionType>(typeAttr.getValue())) {
      if (fnType.getNumResults() != 0)
        return kernel.emitOpError("function_type must have zero results");
    }
  }

  for (auto indexedArg : llvm::enumerate(entry.getArguments())) {
    Type type = indexedArg.value().getType();
    if (!type.isSignlessInteger(32) && !type.isF32()) {
      return kernel.emitOpError()
             << "formal argument " << indexedArg.index()
             << " type must be i32 or f32";
    }
  }

  if (!argAttrs)
    return success();

  if (argAttrs.size() != entry.getNumArguments())
    return kernel.emitOpError("arg_attrs length must match formal argument count");

  llvm::StringSet<> seenNames;
  for (auto indexedAttr : llvm::enumerate(argAttrs)) {
    auto dict = llvm::dyn_cast<DictionaryAttr>(indexedAttr.value());
    if (!dict)
      return kernel.emitOpError("arg_attrs entries must be dictionaries");
    if (dict.get("uniform_index"))
      return kernel.emitOpError("must not contain uniform_index");

    StringAttr nameAttr = getABINameAttr(dict);
    if (!nameAttr || nameAttr.getValue().empty())
      return kernel.emitOpError("requires non-empty abi_name");
    StringRef name = nameAttr.getValue();
    if (!isValidABIName(name))
      return kernel.emitOpError() << "has invalid abi_name '" << name << "'";
    if (isReservedABIName(name))
      return kernel.emitOpError() << "abi_name '" << name << "' is reserved";
    if (!seenNames.insert(name).second)
      return kernel.emitOpError() << "duplicate abi_name '" << name
                                  << "' in arg_attrs";
  }

  return success();
}

} // namespace

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symNameAttr;
  if (parser.parseSymbolName(symNameAttr, "sym_name", result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      do {
        OpAsmParser::Argument arg;
        if (parser.parseArgument(arg, /*allowType=*/true))
          return failure();
        entryArgs.push_back(arg);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, entryArgs, /*enableNameShadowing=*/true))
    return failure();

  SmallVector<Type, 4> argTypes;
  if (!entryArgs.empty()) {
    argTypes.reserve(entryArgs.size());
    for (const OpAsmParser::Argument &arg : entryArgs)
      argTypes.push_back(arg.type);
  } else if (!body->empty()) {
    argTypes.reserve(body->front().getNumArguments());
    for (BlockArgument arg : body->front().getArguments())
      argTypes.push_back(arg.getType());
  }

  if (!result.attributes.get("function_type")) {
    FunctionType functionType =
        parser.getBuilder().getFunctionType(argTypes, TypeRange{});
    result.attributes.set("function_type", TypeAttr::get(functionType));
  }
  return success();
}

void KernelOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());

  Region &body = getBody();
  if (!body.empty() && body.front().getNumArguments() != 0) {
    p << '(';
    llvm::interleaveComma(body.front().getArguments(), p,
                          [&](BlockArgument arg) {
                            p.printOperand(arg);
                            p << " : " << arg.getType();
                          });
    p << ')';
  }

  SmallVector<StringRef, 2> elidedAttrs = {"sym_name", "function_type"};
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

LogicalResult KernelOp::verify() {
  Operation *op = getOperation();
  std::optional<ScheduleMode> mode = getKernelScheduleMode(*this);
  bool usesShared = getBoolAttrOrResourceIntent(op, "uses_shared_vpm");
  bool usesBarrier = getBoolAttrOrResourceIntent(op, "uses_barrier");
  int64_t warpsPerBlock = getI32AttrOrResourceIntent(
      op, "warps_per_block_max", "warps_per_block").value_or(
      mode && *mode == ScheduleMode::cooperative_block ? 12 : 1);
  int64_t vpmRows = getI32AttrOrResourceIntent(
      op, "vpm_rows_per_block", "vpm_rows").value_or(0);
  int64_t vpmBytes = getI32AttrOrResourceIntent(
      op, "vpm_bytes_per_block", "vpm_bytes").value_or(0);
  int64_t semaphores = getI32AttrOrResourceIntent(
      op, "semaphores_per_block", "semaphores").value_or(0);
  bool requireFullResidency = getBoolAttr(op, "require_full_block_residency");

  if (failed(verifyResourceIntent(op)) ||
      failed(verifyKernelFormalArgumentContract(*this)))
    return failure();

  if (warpsPerBlock < 1 || warpsPerBlock > 12)
    return emitOpError("warps_per_block_max must be in range [1, 12]");
  if (vpmRows < 0 || vpmRows > 64)
    return emitOpError("vpm_rows_per_block must be in range [0, 64]");
  if (vpmBytes < 0 || vpmBytes > 4096)
    return emitOpError("vpm_bytes_per_block must be in range [0, 4096]");
  if (semaphores < 0 || semaphores > 16)
    return emitOpError("semaphores_per_block must be in range [0, 16]");
  if (vpmBytes > vpmRows * 64)
    return emitOpError("vpm_bytes_per_block must fit vpm_rows_per_block");

  if (mode && *mode == ScheduleMode::independent_vector) {
    if (warpsPerBlock != 1)
      return emitOpError(
          "independent_vector kernels require warps_per_block_max = 1");
    if (usesShared)
      return emitOpError("independent_vector kernels must not use shared VPM");
    if (usesBarrier)
      return emitOpError("independent_vector kernels must not use barriers");
  }

  if (usesShared && (vpmRows == 0 || vpmBytes == 0)) {
    return emitOpError(
        "uses_shared_vpm requires positive vpm_rows_per_block and "
        "vpm_bytes_per_block");
  }

  if (usesBarrier) {
    if (!isKernelCooperative(*this)) {
      return emitOpError(
          "uses_barrier requires schedule_mode = "
          "#vc4tile.schedule_mode<cooperative_block>");
    }
    if (semaphores != 4)
      return emitOpError("uses_barrier requires semaphores_per_block = 4");
    if (!requireFullResidency) {
      return emitOpError(
          "uses_barrier requires require_full_block_residency = true");
    }
  }

  bool sawBarrier = false;
  bool sawSharedOp = false;
  Operation *markedRawSCFOp = nullptr;
  getBody().walk([&](Operation *nested) {
    if (nested == op)
      return WalkResult::advance();
    if (isa<BarrierOp>(nested))
      sawBarrier = true;
    if (isa<SharedAllocOp, SharedLoadOp, SharedStoreOp, VDRLoadTileOp>(nested))
      sawSharedOp = true;
    if (!markedRawSCFOp && getBoolAttr(op, "requires_core_legalize") &&
        nested->getName().getDialectNamespace() == "scf") {
      markedRawSCFOp = nested;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (markedRawSCFOp) {
    markedRawSCFOp->emitOpError(
        "contains raw scf surface operation in a vc4tile.kernel marked "
        "requires_core_legalize; run --legalize-vc4tile-core-cfg before "
        "verifying or lowering "
        "VC4Tile core");
    return failure();
  }
  if (sawBarrier && !usesBarrier)
    return emitOpError("contains vc4tile.barrier but uses_barrier is not true");
  if (sawSharedOp && !usesShared)
    return emitOpError("contains shared VPM ops but uses_shared_vpm is not true");

  return success();
}

LogicalResult ReturnOp::verify() { return verifyInsideKernel(getOperation()); }

LogicalResult ProgramIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyNoUniformIndex(op, "vc4tile.program_id")))
    return failure();
  return verifyScalarId(op, getResult().getType(), "result");
}

LogicalResult BlockIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyNoUniformIndex(op, "vc4tile.block_id")))
    return failure();
  auto kernel = op->getParentOfType<KernelOp>();
  if (isKernelIndependent(kernel))
    return emitOpError("is only legal in cooperative_block kernels");
  return verifyScalarId(op, getResult().getType(), "result");
}

LogicalResult WarpIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyNoUniformIndex(op, "vc4tile.warp_id")))
    return failure();
  auto kernel = op->getParentOfType<KernelOp>();
  if (isKernelIndependent(kernel))
    return emitOpError("is only legal in cooperative_block kernels");
  return verifyScalarId(op, getResult().getType(), "result");
}

LogicalResult LaneIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyNoUniformIndex(op, "vc4tile.lane_id")))
    return failure();
  return verifyScalarId(op, getResult().getType(), "result");
}

LogicalResult LaneRangeOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyNoUniformIndex(op, "vc4tile.lane_range")))
    return failure();
  return verifyVector16I32(op, getResult().getType(), "result");
}

LogicalResult ThreadIdOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyNoUniformIndex(op, "vc4tile.thread_id")))
    return failure();
  return verifyVector16I32(op, getResult().getType(), "result");
}

LogicalResult MaskAllOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyVector16I1(op, getResult().getType(), "result");
}

LogicalResult TailMaskOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyScalarId(op, getBase().getType(), "base")) ||
      failed(verifyScalarId(op, getLimit().getType(), "limit")))
    return failure();
  return verifyVector16I1(op, getResult().getType(), "result");
}

LogicalResult TileDescriptorOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();

  SmallVector<int64_t, 4> shape;
  if (failed(collectPositiveI64Array(op, getShapeAttr(), "logical shape",
                                     shape)))
    return failure();
  if (shape.size() > 4)
    return emitOpError("logical shape rank greater than 4 is unsupported in M5");
  if (getRankAttr().getInt() != static_cast<int64_t>(shape.size()))
    return emitOpError("rank attribute must match logical shape rank");

  Type elementType;
  Type storageType;
  Type expressedType;
  Type accumulatorType;
  if (failed(verifyM532BitTypeAttr(op, "element_type", "element types",
                                   elementType)) ||
      failed(verifyM532BitTypeAttr(op, "storage_type", "storage",
                                   storageType)) ||
      failed(verifyM532BitTypeAttr(op, "expressed_type", "expressed types",
                                   expressedType)) ||
      failed(verifyM532BitTypeAttr(op, "accumulator_type", "accumulator types",
                                   accumulatorType)))
    return failure();

  if (failed(verifyTileDescriptorLayout(op, getLayoutAttr(), getStridesAttr(),
                                        shape.size())))
    return failure();

  if (getPrecisionAttr().getValue() != Precision::exact_32)
    return emitOpError("M5 executable precision_policy must be exact_32");
  if (getPackingAttr().getValue() != Packing::none)
    return emitOpError("M5 supports only packing = none");

  if (auto boundary = getBoundaryAttr()) {
    if (boundary.getValue() == BoundaryPolicy::zero ||
        boundary.getValue() == BoundaryPolicy::clamp ||
        boundary.getValue() == BoundaryPolicy::reject)
      return emitOpError(
          "boundary policy zero/clamp/reject is not implemented in M5");
  }

  for (StringRef forbidden : {"quantization", "scale", "zero_point",
                             "scale_granularity", "zero_point_policy"}) {
    if (op->getAttr(forbidden))
      return emitOpError(
          "quantization scales and zero points are not executable in M5");
  }

  if (!isVC4TileTileType(getTile().getType()))
    return emitOpError("result type must be !vc4tile.tile");
  return success();
}

LogicalResult TileLoadOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyScalarId(op, getOffset().getType(), "offset")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifySurfaceCarrier(op, getTile().getType(), "result")) ||
      failed(verifyM5Exact32Metadata(op)) ||
      failed(verifyBoundaryPolicyMatchesMask(op, getMask())) ||
      failed(verifyTileMovementLayout(op, "layout")))
    return failure();
  if (op->getAttr("lane_stride") || op->getAttr("stride"))
    return emitOpError(
        "raw affine lane_stride shorthand must be canonicalized by "
        "--canonicalize-vc4tile-surface before tile copy planning");

  MemorySpace space = getMemorySpaceAttr().getValue();
  if (space == MemorySpace::global)
    return verifyScalarId(op, getBase().getType(), "base");
  if (space == MemorySpace::shared_vpm)
    return verifySharedTile(op, getBase().getType(), "base");
  return emitOpError("tile_load source memory_space must be global or shared_vpm");
}

LogicalResult TileStoreOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySurfaceCarrier(op, getTile().getType(), "tile")) ||
      failed(verifyScalarId(op, getOffset().getType(), "offset")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyM5Exact32Metadata(op)) ||
      failed(verifyBoundaryPolicyMatchesMask(op, getMask())) ||
      failed(verifyTileMovementLayout(op, "layout")))
    return failure();
  if (op->getAttr("lane_stride") || op->getAttr("stride"))
    return emitOpError(
        "raw affine lane_stride shorthand must be canonicalized by "
        "--canonicalize-vc4tile-surface before tile copy planning");

  MemorySpace space = getMemorySpaceAttr().getValue();
  if (space == MemorySpace::global)
    return verifyScalarId(op, getBase().getType(), "base");
  if (space == MemorySpace::shared_vpm)
    return verifySharedTile(op, getBase().getType(), "base");
  return emitOpError("tile_store destination memory_space must be global or shared_vpm");
}

LogicalResult CopyTileOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) || failed(verifyM5Exact32Metadata(op)) ||
      failed(verifyTileMovementLayout(op, "src_layout")) ||
      failed(verifyTileMovementLayout(op, "dst_layout")))
    return failure();
  if (getNumOperands() == 0)
    return emitOpError("requires at least one operand");
  for (Value operand : getOperands()) {
    if (!isVC4TileCopyOperandType(operand.getType()))
      return emitTypeError(op, operand.getType(), "operand",
                           "tile carrier, i32/index row, or vector<16xi1> mask");
  }
  for (Value result : getResults()) {
    if (failed(verifySurfaceCarrier(op, result.getType(), "result")))
      return failure();
  }
  if (auto elemBytes = getElemBytesAttr()) {
    if (elemBytes.getInt() != 4)
      return emitOpError("elem_bytes must be 4 for M5 copy planner v1");
  }
  if (failed(verifyCopyTileBoundaryPolicy(op)))
    return failure();
  return success();
}

LogicalResult TileViewOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySurfaceCarrier(op, getSource().getType(), "source")) ||
      failed(verifySurfaceCarrier(op, getResult().getType(), "result")) ||
      failed(verifyOptionalNonNegativeI64Array(op, "offsets")) ||
      failed(verifyOptionalPositiveI64Array(op, "sizes")) ||
      failed(verifyOptionalPositiveI64Array(op, "strides")))
    return failure();
  if (getSource().getType() != getResult().getType())
    return emitOpError("tile_view result type must match source type in M5");
  return success();
}

LogicalResult TileSubviewOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySurfaceCarrier(op, getSource().getType(), "source")) ||
      failed(verifySurfaceCarrier(op, getResult().getType(), "result")) ||
      failed(verifyRequiredNonNegativeI64Array(op, "offsets")) ||
      failed(verifyStaticIndexArray(op, "sizes", 0)) ||
      failed(verifyOptionalPositiveI64Array(op, "strides")))
    return failure();
  if (getSource().getType() != getResult().getType())
    return emitOpError("tile_subview result type must match source type in M5");
  return success();
}

LogicalResult TransposeViewOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySurfaceCarrier(op, getSource().getType(), "source")) ||
      failed(verifySurfaceCarrier(op, getResult().getType(), "result")) ||
      failed(verifyTransposePermutation(op, getPermutationAttr())))
    return failure();
  if (getSource().getType() != getResult().getType())
    return emitOpError("transpose_view result type must match source type in M5");
  return success();
}

LogicalResult SharedTileAllocOp::verify() {
  Operation *op = getOperation();
  Type elementType;
  Type storageType;
  if (failed(verifyInsideKernel(op)) || failed(verifySharedKernelContract(op)) ||
      failed(verifyRowsAttr(op, getRowsAttr(), "rows")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyM532BitTypeAttr(op, "element_type", "element types", elementType)) ||
      failed(verifyM532BitTypeAttr(op, "storage_type", "storage", storageType)) ||
      failed(verifySurfaceCarrier(op, getResult().getType(), "result")))
    return failure();
  if (getMemorySpaceAttr() &&
      getMemorySpaceAttr().getValue() != MemorySpace::shared_vpm)
    return emitOpError("memory_space must be #vc4tile.memory_space<shared_vpm>");
  if (auto precision = getPrecisionAttr()) {
    if (precision.getValue() != Precision::exact_32)
      return emitOpError("M5 executable precision_policy must be exact_32");
  }
  if (auto packing = getPackingAttr()) {
    if (packing.getValue() != Packing::none)
      return emitOpError("M5 supports only packing = none");
  }
  SmallVector<int64_t, 4> shape;
  if (failed(collectPositiveI64Array(op, getShapeAttr(), "shape", shape)))
    return failure();
  if (shape.size() > 2)
    return emitOpError("shared_tile_alloc supports only rank <= 2 in M5");
  return success();
}


LogicalResult TileFillOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) || failed(verifyTileComputeMetadata(op)))
    return failure();
  auto elementType = op->getAttrOfType<TypeAttr>("element_type");
  if (!elementType)
    return emitOpError("element_type attribute is required");
  return verifyTileComputeValueMatchesElement(op, getValueAttr(),
                                              elementType.getValue());
}

LogicalResult TileBroadcastOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) || failed(verifyTileComputeMetadata(op)) ||
      failed(verifyTileScalarComputeType(op, getSource().getType(), "source")))
    return failure();
  auto elementType = op->getAttrOfType<TypeAttr>("element_type");
  if (!elementType)
    return emitOpError("element_type attribute is required");
  if (getSource().getType() != elementType.getValue())
    return emitOpError("tile_broadcast source type must match element_type");
  return success();
}

LogicalResult TileAddOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) || failed(verifyTileComputeMetadata(op)))
    return failure();
  return verifyTileBinaryComputeOperands(op, getLhs(), getRhs(), getResult());
}

LogicalResult TileSubOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) || failed(verifyTileComputeMetadata(op)))
    return failure();
  return verifyTileBinaryComputeOperands(op, getLhs(), getRhs(), getResult());
}

LogicalResult TileMulOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) || failed(verifyTileComputeMetadata(op)))
    return failure();
  return verifyTileBinaryComputeOperands(op, getLhs(), getRhs(), getResult());
}

LogicalResult TileSelectOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) || failed(verifyTileComputeMetadata(op)) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifySurfaceCarrier(op, getTrueValue().getType(), "true_value")) ||
      failed(verifySurfaceCarrier(op, getFalseValue().getType(), "false_value")) ||
      failed(verifySurfaceCarrier(op, getResult().getType(), "result")))
    return failure();
  if (getTrueValue().getType() != getResult().getType() ||
      getFalseValue().getType() != getResult().getType())
    return emitOpError(
        "tile_select true_value, false_value, and result must have identical types in M5");
  if (!isVC4TileVector16DataType(getResult().getType()))
    return emitTypeError(op, getResult().getType(), "result",
                         "vector<16xi32> or vector<16xf32>");
  return success();
}

LogicalResult TileReduceOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyTileReductionCommon(op, getInput(), getMask(), getResult(),
                                   /*requireAxis=*/true);
}

LogicalResult RowReduceOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyTileReductionCommon(op, getInput(), getMask(), getResult(),
                                   /*requireAxis=*/false);
}

LogicalResult WarpReduceOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  return verifyTileReductionCommon(op, getInput(), getMask(), getResult(),
                                   /*requireAxis=*/false);
}

LogicalResult BlockReduceOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyTileReductionCommon(op, getInput(), getMask(), getResult(),
                                       /*requireAxis=*/false)) ||
      failed(verifyBlockReductionKernelContract(op)))
    return failure();
  return success();
}


LogicalResult TileDotOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyTileDotContractMetadata(op, /*matrixForm=*/false)) ||
      failed(verifyTileDotOperands(op, getLhs(), getRhs(), getMask(), getResult())))
    return failure();
  return success();
}

LogicalResult TileContractOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyTileDotContractMetadata(op, /*matrixForm=*/true)) ||
      failed(verifyTileContractOperands(op, getLhs(), getRhs(), getAcc(),
                                        getMask(), getResult())))
    return failure();
  return success();
}

LogicalResult TileMatmulOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyTileDotContractMetadata(op, /*matrixForm=*/true)) ||
      failed(verifyTileContractOperands(op, getLhs(), getRhs(), getAcc(),
                                        getMask(), getResult())))
    return failure();
  return success();
}

LogicalResult SurfacePlaceholderOp::verify() {
  return verifyInsideKernel(getOperation());
}

LogicalResult MaskedLoadGlobalOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyScalarId(op, getBase().getType(), "base")) ||
      failed(verifyVector16I32(op, getOffsets().getType(), "offsets")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyVector16Data(op, getResult().getType(), "result")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(), MemorySpace::global,
                               "global")) ||
      failed(verifyLoadAccessNotGeneric(op, getAccessAttr())))
    return failure();
  return success();
}

LogicalResult MaskedStoreGlobalOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyScalarId(op, getBase().getType(), "base")) ||
      failed(verifyVector16I32(op, getOffsets().getType(), "offsets")) ||
      failed(verifyVector16Data(op, getValue().getType(), "value")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(), MemorySpace::global,
                               "global")) ||
      failed(verifyMemoryAccessNotGeneric(op, getAccessAttr())))
    return failure();
  return success();
}

LogicalResult RotateOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyVector16Data(op, inputType, "input")) ||
      failed(verifyVector16Data(op, resultType, "result")))
    return failure();
  if (inputType != resultType)
    return emitOpError() << "input and result types must match; got "
                         << inputType << " and " << resultType;
  int64_t amount = getAmountAttr().getInt();
  if (amount < 0 || amount > 15)
    return emitOpError("amount must be in range [0, 15]");
  return success();
}

LogicalResult ReduceOp::verify() {
  Operation *op = getOperation();
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifyVector16Data(op, inputType, "input")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyVector16Data(op, resultType, "result")))
    return failure();
  if (inputType != resultType)
    return emitOpError() << "input and result types must match; got "
                         << inputType << " and " << resultType;
  bool isInteger = isVC4TileVector16I32Type(inputType);
  switch (getKind()) {
  case ReduceKind::add:
  case ReduceKind::min:
  case ReduceKind::max:
    return success();
  case ReduceKind::bit_and:
  case ReduceKind::bit_or:
  case ReduceKind::bit_xor:
    if (isInteger)
      return success();
    return emitOpError("bitwise reductions require vector<16xi32>");
  }
  llvm_unreachable("unhandled VC4Tile reduction kind");
}

LogicalResult SharedAllocOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySharedKernelContract(op)) ||
      failed(verifyRowsAttr(op, getRowsAttr(), "rows")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(),
                               MemorySpace::shared_vpm, "shared_vpm")) ||
      failed(verifySharedTile(op, getResult().getType(), "result")))
    return failure();
  return success();
}

LogicalResult SharedLoadOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySharedKernelContract(op)) ||
      failed(verifySharedTile(op, getHandle().getType(), "handle")) ||
      failed(verifyScalarId(op, getRow().getType(), "row")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyVector16Data(op, getResult().getType(), "result")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(),
                               MemorySpace::shared_vpm, "shared_vpm")))
    return failure();
  return success();
}

LogicalResult SharedStoreOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) ||
      failed(verifySharedKernelContract(op)) ||
      failed(verifySharedTile(op, getHandle().getType(), "handle")) ||
      failed(verifyScalarId(op, getRow().getType(), "row")) ||
      failed(verifyVector16Data(op, getValue().getType(), "value")) ||
      failed(verifyVector16I1(op, getMask().getType(), "mask")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())) ||
      failed(verifyMemorySpace(op, getMemorySpaceAttr(),
                               MemorySpace::shared_vpm, "shared_vpm")))
    return failure();
  return success();
}

LogicalResult VDRLoadTileOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)) || failed(verifySharedKernelContract(op)) ||
      failed(verifyScalarId(op, getAddress().getType(), "address")) ||
      failed(verifySharedTile(op, getHandle().getType(), "handle")) ||
      failed(verifyElemBytes4(op, getElemBytesAttr())))
    return failure();

  int64_t rowLen = getRowLenAttr().getInt();
  if (rowLen < 1 || rowLen > 16)
    return emitOpError("row_len must be in range [1, 16]");
  int64_t nrows = getNrowsAttr().getInt();
  if (nrows < 1 || nrows > 16)
    return emitOpError("nrows must be in range [1, 16]");
  int64_t memoryPitchBytes = getMemoryPitchBytesAttr().getInt();
  if (memoryPitchBytes <= 0 || memoryPitchBytes % 4 != 0)
    return emitOpError("memory_pitch_bytes must be a positive multiple of 4");
  if (memoryPitchBytes < rowLen * 4)
    return emitOpError("memory_pitch_bytes must cover row_len elements");
  int64_t vpmBaseRow = getVpmBaseRowAttr().getInt();
  if (vpmBaseRow < 0 || vpmBaseRow > 63)
    return emitOpError("vpm_base_row must be in range [0, 63]");
  if (getVpmBaseColAttr().getInt() != 0)
    return emitOpError("M5 VDR tile loads support only vpm_base_col = 0");
  int64_t vpitch = getVpitchAttr().getInt();
  if (vpitch < 1 || vpitch > 16)
    return emitOpError("vpitch must be in range [1, 16]");
  if (auto layout = getLayoutAttr()) {
    if (layout.getValue() != Layout::row_major &&
        layout.getValue() != Layout::vpm_row &&
        layout.getValue() != Layout::vpm_col &&
        layout.getValue() != Layout::col_major)
      return emitOpError("VDR tile load layout must be row/column VPM compatible");
  }
  if (auto serialize = getSerializeAttr()) {
    if (serialize.getValue() != "mutex" && serialize.getValue() != "none")
      return emitOpError("serialize must be \"mutex\" or \"none\"");
  }
  return success();
}

LogicalResult BarrierOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyInsideKernel(op)))
    return failure();
  auto kernel = op->getParentOfType<KernelOp>();
  if (!isKernelCooperative(kernel)) {
    return emitOpError(
        "requires parent vc4tile.kernel schedule_mode = "
        "#vc4tile.schedule_mode<cooperative_block>");
  }
  if (!getBoolAttr(kernel.getOperation(), "uses_barrier")) {
    return emitOpError(
        "requires parent vc4tile.kernel to set uses_barrier = true");
  }
  if (!getBoolAttr(kernel.getOperation(), "require_full_block_residency")) {
    return emitOpError(
        "requires parent vc4tile.kernel to set "
        "require_full_block_residency = true");
  }
  int64_t semaphores =
      getI32Attr(kernel.getOperation(), "semaphores_per_block").value_or(0);
  if (semaphores != 4)
    return emitOpError("requires parent semaphores_per_block = 4");
  return success();
}

#define GET_OP_CLASSES
#include "vc4/Dialect/VC4Tile/IR/VC4TileOps.cpp.inc"
