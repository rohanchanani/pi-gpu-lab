// RUN: not vc4-opt %s --verify-vc4tile-core 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: surface operation
// ERR: --canonicalize-vc4tile-surface
vc4tile.kernel @verify_core_reject_surface_placeholder attributes {
  public_name = "verify_core_reject_surface_placeholder"
} {
  vc4tile.surface_placeholder
  vc4tile.return
}
