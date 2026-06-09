#loc = loc(unknown)

module {
  tt.func public @ptr_rank2_probe(%x_ptr: !tt.ptr<f32> loc("x_ptr"(#loc))) attributes {noinline = false} {
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

