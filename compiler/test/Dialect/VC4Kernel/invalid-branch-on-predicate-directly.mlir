// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %pred = vc4kernel.pred.full : !vc4kernel.pred<16>
    // CHECK: expects different type than prior uses: 'i1' vs '!vc4kernel.pred<16>'
    cf.cond_br %pred, ^then, ^else
  ^then:
    vc4kernel.return
  ^else:
    vc4kernel.return
  }
}
