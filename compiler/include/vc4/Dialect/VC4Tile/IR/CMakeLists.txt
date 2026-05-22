set(LLVM_TARGET_DEFINITIONS VC4TileDialect.td)
mlir_tablegen(VC4TileDialect.h.inc -gen-dialect-decls -dialect=vc4tile)
mlir_tablegen(VC4TileDialect.cpp.inc -gen-dialect-defs -dialect=vc4tile)
add_public_tablegen_target(VC4TileDialectIncGen)
