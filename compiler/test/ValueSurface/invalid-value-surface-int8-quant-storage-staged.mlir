// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @int8_quant_storage_staged() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %a = arith.constant dense<1> : vector<16xi8>
    %b = arith.constant dense<2> : vector<16xi8>
    // expected-error @+1 {{int8/int16 quantized arithmetic and storage policy are staged in the Phase 14 VC4 value surface}}
    %sum = arith.addi %a, %b : vector<16xi8>
    return
  }
}
