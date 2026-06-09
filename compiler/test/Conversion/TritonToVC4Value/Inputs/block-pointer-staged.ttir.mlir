#loc = loc(unknown)

module {
  tt.func public @block_ptr_probe(%x_ptr: !tt.ptr<tensor<16xf32>> loc("x_ptr"(#loc))) attributes {noinline = false} {
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

