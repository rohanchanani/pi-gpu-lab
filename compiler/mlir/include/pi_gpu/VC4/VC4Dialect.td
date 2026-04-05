include "mlir/IR/OpBase.td"

def VC4_Dialect : Dialect {
  let name = "vc4";
  let cppNamespace = "::pi_gpu::vc4";
  let summary = "Minimal VC4 staged-memory backend dialect scaffold";
  let description = [{
    Minimal post-vectorization backend IR for staged memory movement on VC4.
    This starts intentionally small so the vec16-to-vc4 boundary is easy to
    inspect before real hardware modeling, scheduling, or code emission is
    added.
  }];
}

def VC4_UniformOp : Op<VC4_Dialect, "uniform"> {
  let summary = "Read one backend uniform value";
  let description = [{
    General backend op for values supplied by the launch environment, such as
    buffer base addresses or scalar parameters.
  }];

  let arguments = (ins I32Attr:$slot);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$slot attr-dict `:` type($value)";
}

def VC4_DmaLoadVPMOp : Op<VC4_Dialect, "dma_load_vpm"> {
  let summary = "Stage a vector from memory into VPM";
  let arguments = (ins AnyType:$base);
  let results = (outs AnyType:$token);
  let assemblyFormat = "$base attr-dict `:` type($base) `->` type($token)";
}

def VC4_VPMReadOp : Op<VC4_Dialect, "vpm_read"> {
  let summary = "Read one staged vector from VPM";
  let arguments = (ins AnyType:$token);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$token attr-dict `:` type($token) `->` type($value)";
}

def VC4_BinOp : Op<VC4_Dialect, "bin"> {
  let summary = "Apply a simple vector arithmetic operation";
  let arguments = (ins StrAttr:$kind, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$kind $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($value)";
}

def VC4_DmaStoreVPMOp : Op<VC4_Dialect, "dma_store_vpm"> {
  let summary = "Stage a vector write from VPM back to memory";
  let arguments = (ins AnyType:$base, AnyType:$value);
  let results = (outs AnyType:$token);
  let assemblyFormat = "$base `,` $value attr-dict `:` type($base) `,` type($value) `->` type($token)";
}
