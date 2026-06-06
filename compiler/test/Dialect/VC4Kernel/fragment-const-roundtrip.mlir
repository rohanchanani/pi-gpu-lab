// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @fragment_const() attributes {
    public_name = "fragment_const",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: vc4kernel.fragment_const {{.*}}dense<0> : vector<16xi32>
    %i32_zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %i32_allones = vc4kernel.fragment_const {value = dense<-1> : vector<16xi32>} : vector<16xi32>
    %i32_any = vc4kernel.fragment_const {value = dense<-123456789> : vector<16xi32>} : vector<16xi32>
    %f32_zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %f32_negzero = vc4kernel.fragment_const {value = dense<-0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %f32_one = vc4kernel.fragment_const {value = dense<1.000000e+00> : vector<16xf32>} : vector<16xf32>
    %f32_neg = vc4kernel.fragment_const {value = dense<-2.500000e+00> : vector<16xf32>} : vector<16xf32>
    %lane_range = vc4kernel.fragment_const {value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>} : vector<16xi32>
    %lane_affine_add = vc4kernel.fragment_const {value = dense<[7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37]> : vector<16xi32>} : vector<16xi32>
    %lane_affine_sub = vc4kernel.fragment_const {value = dense<[100, 92, 84, 76, 68, 60, 52, 44, 36, 28, 20, 12, 4, -4, -12, -20]> : vector<16xi32>} : vector<16xi32>
    %u2 = vc4kernel.fragment_const {value = dense<[0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0, 0, 2, 1, 3]> : vector<16xi32>} : vector<16xi32>
    %s2 = vc4kernel.fragment_const {value = dense<[-2, -1, 0, 1, -2, -1, 0, 1, 1, 0, -1, -2, -2, 0, -1, 1]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.return
  }
}
