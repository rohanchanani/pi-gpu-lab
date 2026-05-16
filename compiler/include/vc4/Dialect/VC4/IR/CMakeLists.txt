set(LLVM_TARGET_DEFINITIONS VC4Enums.td)
mlir_tablegen(VC4Enums.h.inc -gen-enum-decls)
mlir_tablegen(VC4Enums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(VC4EnumsIncGen)

set(LLVM_TARGET_DEFINITIONS VC4Attrs.td)
mlir_tablegen(VC4AttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=vc4)
mlir_tablegen(VC4AttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=vc4)
add_mlir_dialect_tablegen_target(VC4AttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS VC4Types.td)
mlir_tablegen(VC4Types.h.inc -gen-typedef-decls -typedefs-dialect=vc4)
mlir_tablegen(VC4Types.cpp.inc -gen-typedef-defs -typedefs-dialect=vc4)
add_mlir_dialect_tablegen_target(VC4TypesIncGen)

set(LLVM_TARGET_DEFINITIONS VC4Ops.td)
mlir_tablegen(VC4Ops.h.inc -gen-op-decls)
mlir_tablegen(VC4Ops.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(VC4OpsIncGen)

set(LLVM_TARGET_DEFINITIONS VC4Dialect.td)
mlir_tablegen(VC4Dialect.h.inc -gen-dialect-decls -dialect=vc4)
mlir_tablegen(VC4Dialect.cpp.inc -gen-dialect-defs -dialect=vc4)
add_mlir_dialect_tablegen_target(VC4DialectIncGen)
