module {
  tt.func public @dot_probe() attributes {noinline = false} {
    %a = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %b = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return
  }
}
