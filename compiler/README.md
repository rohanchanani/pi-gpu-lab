# Compiler

This directory contains the restricted-CUDA-to-VC4 compiler work.

## Intended pipeline

tiny CUDA subset
-> MLIR scalar dialects
-> MLIR `vector` dialect with width-16 execution groups
-> custom `vc4` dialect
-> `vc4asm`

## VC4 dialect seed

The `vc4` dialect is the target-facing structured IR that sits after the
width-16 vector stage. It is intentionally tiny right now, but it is no longer
just a generic target marker: the seed dialect is meant to expose staged memory
movement and simple backend arithmetic after vectorization.

The current seed focuses on a very small set of general backend concepts:
- uniform values
- DMA movement into or out of VPM
- VPM reads
- simple arithmetic over staged vec16 data

This keeps the post-vectorization boundary inspectable without adding real
hardware scheduling, register allocation, or `vc4asm` emission yet.

## Current priorities
- establish MLIR infrastructure
- define a minimal custom `vc4` dialect
- build the first end-to-end lowering path for SAXPY

## Non-goals right now
- full CUDA parsing
- full runtime design
- broad optimization
- support for `__shared__` or synchronization
