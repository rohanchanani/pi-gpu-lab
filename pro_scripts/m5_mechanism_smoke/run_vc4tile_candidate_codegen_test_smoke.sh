#!/usr/bin/env bash
set -euo pipefail
input.vc4tile.mlir
surface.vc4tile.mlir
planned.vc4tile.mlir
core.vc4tile.mlir
lowered.ssavc4.mlir
scheduled.vc4.mlir
vc4-opt "$1" --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 --convert-ssavc4-to-vc4
vc4-codegen --emit-bundle
