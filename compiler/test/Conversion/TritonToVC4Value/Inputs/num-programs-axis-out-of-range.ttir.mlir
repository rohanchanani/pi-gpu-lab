#loc = loc(unknown)

module {
  tt.func public @num_programs_axis_out_of_range_probe(%x_ptr: !tt.ptr<f32> loc("x_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %pid = tt.get_program_id x : i32 loc(#loc)
    %num = "tt.get_num_programs"() {axis = 3 : i32} : () -> i32
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
