// RUN: vc4-opt %s -split-input-file -verify-diagnostics

ssavc4.module @br_too_few {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    // expected-error@+1 {{successor operand count does not match target block argument count}}
    ssavc4.br ^dest(%zero : i32)
  ^dest(%a: i32, %b: i32):
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @br_too_many {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // expected-error@+1 {{successor operand count does not match target block argument count}}
    ssavc4.br ^dest(%zero, %one : i32, i32)
  ^dest(%a: i32):
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @br_wrong_type {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %f = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : f32
    // expected-error@+1 {{successor operand type does not match target block argument type}}
    ssavc4.br ^dest(%f : f32)
  ^dest(%a: i32):
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @cond_true_count {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    // expected-error@+1 {{true successor operand count does not match target block argument count}}
    ssavc4.cond_br %flags, ^true_dest, ^false_dest {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
  ^true_dest(%a: i32):
    ssavc4.thread_end
  ^false_dest:
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @cond_false_count {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    // expected-error@+1 {{false successor operand count does not match target block argument count}}
    ssavc4.cond_br %flags, ^true_dest, ^false_dest {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
  ^true_dest:
    ssavc4.thread_end
  ^false_dest(%a: i32):
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @cond_true_type {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    // expected-error@+1 {{true successor operand type does not match target block argument type}}
    ssavc4.cond_br %flags, ^true_dest(%zero : i32), ^false_dest {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
  ^true_dest(%a: f32):
    ssavc4.thread_end
  ^false_dest:
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @cond_false_type {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    // expected-error@+1 {{false successor operand type does not match target block argument type}}
    ssavc4.cond_br %flags, ^true_dest, ^false_dest(%zero : i32) {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
  ^true_dest:
    ssavc4.thread_end
  ^false_dest(%a: f32):
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @cond_always {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    // expected-error@+1 {{cond_br with always condition is invalid; use ssavc4.br}}
    ssavc4.cond_br %flags, ^true_dest, ^false_dest {cond = #vc4.branch_cond<always>} : !ssavc4.flags
  ^true_dest:
    ssavc4.thread_end
  ^false_dest:
    ssavc4.thread_end
  }
}
