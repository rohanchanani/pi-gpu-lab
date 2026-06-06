// RUN: python3 -c "from pathlib import Path; base=Path(r'%S'); cands=[p/'lib/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.cpp' for p in (base, *base.parents)]; s=next(p for p in cands if p.exists()).read_text(); assert 'isMul24FastPathProven' in s; assert 'emitI32Mul32Fallback' in s; assert 'isVectorF32(resultType)' in s; assert 'mlir::vc4::MulOpcode::fmul\\n                                                : mlir::vc4::MulOpcode::mul24' not in s"

// This is a source guard: arbitrary i32 legacy multiply must not lower directly
// to VC4 mul24 without a proof or a full i32 fallback.
