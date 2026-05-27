// RUN: not vc4-opt %s --convert-vc4tile-to-ssavc4 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: surface operation
// ERR: --canonicalize-vc4tile-surface
vc4tile.kernel @reject_surface_before_canonicalize attributes {
  public_name = "reject_surface_before_canonicalize"
} {
  vc4tile.surface_placeholder
  vc4tile.return
}
