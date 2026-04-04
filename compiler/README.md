# Compiler

This directory contains the restricted-CUDA-to-VC4 compiler work.

## Intended pipeline

tiny CUDA subset
-> MLIR scalar dialects
-> MLIR `vector` dialect with width-16 execution groups
-> custom `vc4` dialect
-> `vc4asm`

## Current priorities
- establish MLIR infrastructure
- define a minimal custom `vc4` dialect
- build the first end-to-end lowering path for SAXPY

## Non-goals right now
- full CUDA parsing
- full runtime design
- broad optimization
- support for `__shared__` or synchronization