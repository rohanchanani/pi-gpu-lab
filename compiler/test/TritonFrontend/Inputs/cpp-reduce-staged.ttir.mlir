module {
  tt.func public @reduce_probe() attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %0 = "tt.reduce"(%cst) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<16xf32>) -> f32
    tt.return
  }
}
