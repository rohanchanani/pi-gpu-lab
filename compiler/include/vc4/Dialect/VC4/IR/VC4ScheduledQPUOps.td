#ifndef VC4_SCHEDULED_QPU_OPS_TD
#define VC4_SCHEDULED_QPU_OPS_TD

include "vc4/Dialect/VC4/IR/VC4CoreOps.td"

def VC4_QPULDIOp : VC4_Op<"qpu.ldi"> {
  let summary = "Scheduled VC4 load-immediate instruction";
  let arguments = (ins
    VC4_LoadImmModeAttr:$mode,
    AnyAttr:$value,
    BoolAttr:$pm,
    OptionalAttr<AnyAttr>:$pack,
    VC4_CondAttr:$cond_add,
    VC4_CondAttr:$cond_mul,
    OptionalAttr<UnitAttr>:$set_flags,
    OptionalAttr<UnitAttr>:$write_swap,
    I32Attr:$waddr_add,
    I32Attr:$waddr_mul
  );
  let assemblyFormat = "$mode attr-dict";
  let hasVerifier = 1;
}

def VC4_QPUSemaOp : VC4_MemoryEffectOp<"qpu.sema"> {
  let summary = "Scheduled VC4 semaphore instruction";
  let arguments = (ins
    VC4_SemaphoreModeAttr:$mode,
    I32Attr:$id,
    BoolAttr:$pm,
    OptionalAttr<AnyAttr>:$pack,
    VC4_CondAttr:$cond_add,
    VC4_CondAttr:$cond_mul,
    OptionalAttr<UnitAttr>:$set_flags,
    OptionalAttr<UnitAttr>:$write_swap,
    I32Attr:$waddr_add,
    I32Attr:$waddr_mul
  );
  let assemblyFormat = "$mode attr-dict";
  let hasVerifier = 1;
}

def VC4_QPUBundleOp : VC4_Op<"qpu.bundle"> {
  let summary = "Scheduled VC4 ALU or small-immediate instruction";
  let arguments = (ins
    VC4_QPUSignalAttr:$sig,
    OptionalAttr<AnyAttr>:$unpack,
    BoolAttr:$pm,
    OptionalAttr<AnyAttr>:$pack,
    VC4_CondAttr:$cond_add,
    VC4_CondAttr:$cond_mul,
    OptionalAttr<UnitAttr>:$set_flags,
    OptionalAttr<UnitAttr>:$write_swap,
    I32Attr:$waddr_add,
    I32Attr:$waddr_mul,
    VC4_AddOpcodeAttr:$op_add,
    VC4_MulOpcodeAttr:$op_mul,
    I32Attr:$raddr_a,
    OptionalAttr<I32Attr>:$raddr_b,
    OptionalAttr<I32Attr>:$small_imm,
    VC4_QPUMuxAttr:$add_a,
    VC4_QPUMuxAttr:$add_b,
    VC4_QPUMuxAttr:$mul_a,
    VC4_QPUMuxAttr:$mul_b
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def VC4_QPUBranchOp : VC4_Op<"qpu.branch", GraphRegionNoTerminator.traits> {
  let summary = "Scheduled VC4 branch instruction with explicit delay slots";
  let arguments = (ins
    VC4_BranchCondAttr:$cond,
    BoolAttr:$relative,
    BoolAttr:$use_reg,
    I32Attr:$raddr_a,
    I32Attr:$immediate,
    OptionalAttr<UnitAttr>:$write_swap,
    I32Attr:$waddr_add,
    I32Attr:$waddr_mul
  );
  let regions = (region SizedRegion<1>:$delay_slots);
  let assemblyFormat = "attr-dict-with-keyword $delay_slots";
  let hasVerifier = 1;
}

#endif // VC4_SCHEDULED_QPU_OPS_TD
