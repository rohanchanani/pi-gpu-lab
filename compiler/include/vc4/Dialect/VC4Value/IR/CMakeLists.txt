set(LLVM_TARGET_DEFINITIONS VC4ValueDialect.td)
mlir_tablegen(VC4ValueDialect.h.inc -gen-dialect-decls -dialect=vc4value)
mlir_tablegen(VC4ValueDialect.cpp.inc -gen-dialect-defs -dialect=vc4value)
add_public_tablegen_target(VC4ValueDialectIncGen)

set(LLVM_TARGET_DEFINITIONS VC4ValueAttrs.td)
mlir_tablegen(VC4ValueAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=vc4value)
mlir_tablegen(VC4ValueAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=vc4value)
add_public_tablegen_target(VC4ValueAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS VC4ValueOps.td)
mlir_tablegen(VC4ValueOps.h.inc -gen-op-decls)
mlir_tablegen(VC4ValueOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(VC4ValueOpsIncGen)
