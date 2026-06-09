#loc = loc(unknown)

module {
  tt.func public @num_programs_axis_y_probe(%x_ptr: !tt.ptr<f32> loc("x_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32 loc(#loc)
    %num = tt.get_num_programs y : i32 loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

