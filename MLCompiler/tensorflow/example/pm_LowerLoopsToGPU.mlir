// -----// IR Dump After EmbedTFFrameworkPass (embed-tf-framework) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c1536 = arith.constant 1536 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    scf.parallel (%arg2) = (%c0) to (%5) step (%c1) {
      %6 = arith.muli %arg2, %c2048 : index
      %7 = arith.cmpi slt, %arg2, %4 : index
      scf.if %7 {
        scf.parallel (%arg3) = (%c0) to (%c512) step (%c1) {
          %8 = arith.addi %arg3, %c512 : index
          %9 = arith.addi %arg3, %c1024 : index
          %10 = arith.addi %arg3, %c1536 : index
          %11 = arith.addi %arg3, %6 : index
          %12 = arith.addi %8, %6 : index
          %13 = arith.addi %9, %6 : index
          %14 = arith.addi %10, %6 : index
          %15 = arith.remsi %11, %c1024 : index
          %16 = arith.remsi %12, %c1024 : index
          %17 = arith.remsi %13, %c1024 : index
          %18 = arith.remsi %14, %c1024 : index
          %19 = arith.divsi %11, %c1024 : index
          %20 = arith.divsi %12, %c1024 : index
          %21 = arith.divsi %13, %c1024 : index
          %22 = arith.divsi %14, %c1024 : index
          %23 = arith.remsi %19, %c8 : index
          %24 = arith.remsi %20, %c8 : index
          %25 = arith.remsi %21, %c8 : index
          %26 = arith.remsi %22, %c8 : index
          %27 = arith.divsi %19, %c8 : index
          %28 = arith.divsi %20, %c8 : index
          %29 = arith.divsi %21, %c8 : index
          %30 = arith.divsi %22, %c8 : index
          %31 = memref.load %arg1[%27, %23, %15] : memref<?x8x1024xf32>
          %32 = memref.load %arg1[%28, %24, %16] : memref<?x8x1024xf32>
          %33 = memref.load %arg1[%29, %25, %17] : memref<?x8x1024xf32>
          %34 = memref.load %arg1[%30, %26, %18] : memref<?x8x1024xf32>
          %35 = arith.mulf %31, %cst_0 : f32
          %36 = arith.mulf %32, %cst_0 : f32
          %37 = arith.mulf %33, %cst_0 : f32
          %38 = arith.mulf %34, %cst_0 : f32
          %39 = arith.maxf %35, %cst : f32
          %40 = arith.maxf %36, %cst : f32
          %41 = arith.maxf %37, %cst : f32
          %42 = arith.maxf %38, %cst : f32
          memref.store %39, %0[%27, %23, %15] : memref<?x8x1024xf32>
          memref.store %40, %0[%28, %24, %16] : memref<?x8x1024xf32>
          memref.store %41, %0[%29, %25, %17] : memref<?x8x1024xf32>
          memref.store %42, %0[%30, %26, %18] : memref<?x8x1024xf32>
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
      } else {
        %8 = affine.min #map(%6)[%3]
        scf.parallel (%arg3) = (%c0) to (%c512) step (%c1) {
          %15 = arith.cmpi slt, %arg3, %8 : index
          scf.if %15 {
            %16 = arith.addi %arg3, %6 : index
            %17 = arith.remsi %16, %c1024 : index
            %18 = arith.divsi %16, %c1024 : index
            %19 = arith.remsi %18, %c8 : index
            %20 = arith.divsi %18, %c8 : index
            %21 = memref.load %arg1[%20, %19, %17] : memref<?x8x1024xf32>
            %22 = arith.mulf %21, %cst_0 : f32
            %23 = arith.maxf %22, %cst : f32
            memref.store %23, %0[%20, %19, %17] : memref<?x8x1024xf32>
          }
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
        %9 = arith.addi %6, %c512 : index
        %10 = affine.min #map(%9)[%3]
        scf.parallel (%arg3) = (%c0) to (%c512) step (%c1) {
          %15 = arith.addi %arg3, %c512 : index
          %16 = arith.cmpi slt, %arg3, %10 : index
          scf.if %16 {
            %17 = arith.addi %15, %6 : index
            %18 = arith.remsi %17, %c1024 : index
            %19 = arith.divsi %17, %c1024 : index
            %20 = arith.remsi %19, %c8 : index
            %21 = arith.divsi %19, %c8 : index
            %22 = memref.load %arg1[%21, %20, %18] : memref<?x8x1024xf32>
            %23 = arith.mulf %22, %cst_0 : f32
            %24 = arith.maxf %23, %cst : f32
            memref.store %24, %0[%21, %20, %18] : memref<?x8x1024xf32>
          }
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
        %11 = arith.addi %6, %c1024 : index
        %12 = affine.min #map(%11)[%3]
        scf.parallel (%arg3) = (%c0) to (%c512) step (%c1) {
          %15 = arith.addi %arg3, %c1024 : index
          %16 = arith.cmpi slt, %arg3, %12 : index
          scf.if %16 {
            %17 = arith.addi %15, %6 : index
            %18 = arith.remsi %17, %c1024 : index
            %19 = arith.divsi %17, %c1024 : index
            %20 = arith.remsi %19, %c8 : index
            %21 = arith.divsi %19, %c8 : index
            %22 = memref.load %arg1[%21, %20, %18] : memref<?x8x1024xf32>
            %23 = arith.mulf %22, %cst_0 : f32
            %24 = arith.maxf %23, %cst : f32
            memref.store %24, %0[%21, %20, %18] : memref<?x8x1024xf32>
          }
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
        %13 = arith.addi %6, %c1536 : index
        %14 = affine.min #map(%13)[%3]
        scf.parallel (%arg3) = (%c0) to (%c512) step (%c1) {
          %15 = arith.addi %arg3, %c1536 : index
          %16 = arith.cmpi slt, %arg3, %14 : index
          scf.if %16 {
            %17 = arith.addi %15, %6 : index
            %18 = arith.remsi %17, %c1024 : index
            %19 = arith.divsi %17, %c1024 : index
            %20 = arith.remsi %19, %c8 : index
            %21 = arith.divsi %19, %c8 : index
            %22 = memref.load %arg1[%21, %20, %18] : memref<?x8x1024xf32>
            %23 = arith.mulf %22, %cst_0 : f32
            %24 = arith.maxf %23, %cst : f32
            memref.store %24, %0[%21, %20, %18] : memref<?x8x1024xf32>
          }
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
      }
      scf.yield
    } {_sony_af_cwise_unroll = 512 : si32, _sony_af_gpu_processor = 0 : i64, mapping}
    return %0 : memref<?x8x1024xf32>
  }
}


// -----// IR Dump After CustomConvertParallelLoopsToGpu (custom-convert-parallel-loops-to-gpu) //----- //
func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
  %c1536 = arith.constant 1536 : index
  %c2048 = arith.constant 2048 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
  %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
  %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
  tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
  %2 = arith.muli %dim, %c8 : index
  %3 = arith.muli %2, %c1024 : index
  %4 = arith.divsi %3, %c2048 : index
  %5 = arith.ceildivsi %3, %c2048 : index
  gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %5, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c512, %arg12 = %c1, %arg13 = %c1) {
    %6 = arith.muli %arg2, %c2048 : index
    %7 = arith.cmpi slt, %arg2, %4 : index
    scf.if %7 {
      %8 = arith.addi %arg5, %c512 : index
      %9 = arith.addi %arg5, %c1024 : index
      %10 = arith.addi %arg5, %c1536 : index
      %11 = arith.addi %arg5, %6 : index
      %12 = arith.addi %8, %6 : index
      %13 = arith.addi %9, %6 : index
      %14 = arith.addi %10, %6 : index
      %15 = arith.remsi %11, %c1024 : index
      %16 = arith.remsi %12, %c1024 : index
      %17 = arith.remsi %13, %c1024 : index
      %18 = arith.remsi %14, %c1024 : index
      %19 = arith.divsi %11, %c1024 : index
      %20 = arith.divsi %12, %c1024 : index
      %21 = arith.divsi %13, %c1024 : index
      %22 = arith.divsi %14, %c1024 : index
      %23 = arith.remsi %19, %c8 : index
      %24 = arith.remsi %20, %c8 : index
      %25 = arith.remsi %21, %c8 : index
      %26 = arith.remsi %22, %c8 : index
      %27 = arith.divsi %19, %c8 : index
      %28 = arith.divsi %20, %c8 : index
      %29 = arith.divsi %21, %c8 : index
      %30 = arith.divsi %22, %c8 : index
      %31 = memref.load %arg1[%27, %23, %15] : memref<?x8x1024xf32>
      %32 = memref.load %arg1[%28, %24, %16] : memref<?x8x1024xf32>
      %33 = memref.load %arg1[%29, %25, %17] : memref<?x8x1024xf32>
      %34 = memref.load %arg1[%30, %26, %18] : memref<?x8x1024xf32>
      %35 = arith.mulf %31, %cst_0 : f32
      %36 = arith.mulf %32, %cst_0 : f32
      %37 = arith.mulf %33, %cst_0 : f32
      %38 = arith.mulf %34, %cst_0 : f32
      %39 = arith.maxf %35, %cst : f32
      %40 = arith.maxf %36, %cst : f32
      %41 = arith.maxf %37, %cst : f32
      %42 = arith.maxf %38, %cst : f32
      memref.store %39, %0[%27, %23, %15] : memref<?x8x1024xf32>
      memref.store %40, %0[%28, %24, %16] : memref<?x8x1024xf32>
      memref.store %41, %0[%29, %25, %17] : memref<?x8x1024xf32>
      memref.store %42, %0[%30, %26, %18] : memref<?x8x1024xf32>
    } else {
      %8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%6)[%3]
      %9 = arith.cmpi slt, %arg5, %8 : index
      scf.if %9 {
        %22 = arith.addi %arg5, %6 : index
        %23 = arith.remsi %22, %c1024 : index
        %24 = arith.divsi %22, %c1024 : index
        %25 = arith.remsi %24, %c8 : index
        %26 = arith.divsi %24, %c8 : index
        %27 = memref.load %arg1[%26, %25, %23] : memref<?x8x1024xf32>
        %28 = arith.mulf %27, %cst_0 : f32
        %29 = arith.maxf %28, %cst : f32
        memref.store %29, %0[%26, %25, %23] : memref<?x8x1024xf32>
      }
      %10 = arith.addi %6, %c512 : index
      %11 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%10)[%3]
      %12 = arith.addi %arg5, %c512 : index
      %13 = arith.cmpi slt, %arg5, %11 : index
      scf.if %13 {
        %22 = arith.addi %12, %6 : index
        %23 = arith.remsi %22, %c1024 : index
        %24 = arith.divsi %22, %c1024 : index
        %25 = arith.remsi %24, %c8 : index
        %26 = arith.divsi %24, %c8 : index
        %27 = memref.load %arg1[%26, %25, %23] : memref<?x8x1024xf32>
        %28 = arith.mulf %27, %cst_0 : f32
        %29 = arith.maxf %28, %cst : f32
        memref.store %29, %0[%26, %25, %23] : memref<?x8x1024xf32>
      }
      %14 = arith.addi %6, %c1024 : index
      %15 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%14)[%3]
      %16 = arith.addi %arg5, %c1024 : index
      %17 = arith.cmpi slt, %arg5, %15 : index
      scf.if %17 {
        %22 = arith.addi %16, %6 : index
        %23 = arith.remsi %22, %c1024 : index
        %24 = arith.divsi %22, %c1024 : index
        %25 = arith.remsi %24, %c8 : index
        %26 = arith.divsi %24, %c8 : index
        %27 = memref.load %arg1[%26, %25, %23] : memref<?x8x1024xf32>
        %28 = arith.mulf %27, %cst_0 : f32
        %29 = arith.maxf %28, %cst : f32
        memref.store %29, %0[%26, %25, %23] : memref<?x8x1024xf32>
      }
      %18 = arith.addi %6, %c1536 : index
      %19 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%18)[%3]
      %20 = arith.addi %arg5, %c1536 : index
      %21 = arith.cmpi slt, %arg5, %19 : index
      scf.if %21 {
        %22 = arith.addi %20, %6 : index
        %23 = arith.remsi %22, %c1024 : index
        %24 = arith.divsi %22, %c1024 : index
        %25 = arith.remsi %24, %c8 : index
        %26 = arith.divsi %24, %c8 : index
        %27 = memref.load %arg1[%26, %25, %23] : memref<?x8x1024xf32>
        %28 = arith.mulf %27, %cst_0 : f32
        %29 = arith.maxf %28, %cst : f32
        memref.store %29, %0[%26, %25, %23] : memref<?x8x1024xf32>
      }
    }
    gpu.terminator
  }
  return %0 : memref<?x8x1024xf32>
}

// -----// IR Dump After GpuLaunchSinkIndexComputations (gpu-launch-sink-index-computations) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c1536 = arith.constant 1536 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %5, %arg9 = %c1, %arg10 = %c1) threads(%arg5, %arg6, %arg7) in (%arg11 = %c512, %arg12 = %c1, %arg13 = %c1) {
      %c2048_1 = arith.constant 2048 : index
      %c512_2 = arith.constant 512 : index
      %c1024_3 = arith.constant 1024 : index
      %c1536_4 = arith.constant 1536 : index
      %c8_5 = arith.constant 8 : index
      %cst_6 = arith.constant 5.000000e-01 : f32
      %cst_7 = arith.constant 0.000000e+00 : f32
      %6 = arith.muli %arg2, %c2048_1 : index
      %7 = arith.cmpi slt, %arg2, %4 : index
      scf.if %7 {
        %8 = arith.addi %arg5, %c512_2 : index
        %9 = arith.addi %arg5, %c1024_3 : index
        %10 = arith.addi %arg5, %c1536_4 : index
        %11 = arith.addi %arg5, %6 : index
        %12 = arith.addi %8, %6 : index
        %13 = arith.addi %9, %6 : index
        %14 = arith.addi %10, %6 : index
        %15 = arith.remsi %11, %c1024_3 : index
        %16 = arith.remsi %12, %c1024_3 : index
        %17 = arith.remsi %13, %c1024_3 : index
        %18 = arith.remsi %14, %c1024_3 : index
        %19 = arith.divsi %11, %c1024_3 : index
        %20 = arith.divsi %12, %c1024_3 : index
        %21 = arith.divsi %13, %c1024_3 : index
        %22 = arith.divsi %14, %c1024_3 : index
        %23 = arith.remsi %19, %c8_5 : index
        %24 = arith.remsi %20, %c8_5 : index
        %25 = arith.remsi %21, %c8_5 : index
        %26 = arith.remsi %22, %c8_5 : index
        %27 = arith.divsi %19, %c8_5 : index
        %28 = arith.divsi %20, %c8_5 : index
        %29 = arith.divsi %21, %c8_5 : index
        %30 = arith.divsi %22, %c8_5 : index
        %31 = memref.load %arg1[%27, %23, %15] : memref<?x8x1024xf32>
        %32 = memref.load %arg1[%28, %24, %16] : memref<?x8x1024xf32>
        %33 = memref.load %arg1[%29, %25, %17] : memref<?x8x1024xf32>
        %34 = memref.load %arg1[%30, %26, %18] : memref<?x8x1024xf32>
        %35 = arith.mulf %31, %cst_6 : f32
        %36 = arith.mulf %32, %cst_6 : f32
        %37 = arith.mulf %33, %cst_6 : f32
        %38 = arith.mulf %34, %cst_6 : f32
        %39 = arith.maxf %35, %cst_7 : f32
        %40 = arith.maxf %36, %cst_7 : f32
        %41 = arith.maxf %37, %cst_7 : f32
        %42 = arith.maxf %38, %cst_7 : f32
        memref.store %39, %0[%27, %23, %15] : memref<?x8x1024xf32>
        memref.store %40, %0[%28, %24, %16] : memref<?x8x1024xf32>
        memref.store %41, %0[%29, %25, %17] : memref<?x8x1024xf32>
        memref.store %42, %0[%30, %26, %18] : memref<?x8x1024xf32>
      } else {
        %8 = affine.min #map(%6)[%3]
        %9 = arith.cmpi slt, %arg5, %8 : index
        scf.if %9 {
          %22 = arith.addi %arg5, %6 : index
          %23 = arith.remsi %22, %c1024_3 : index
          %24 = arith.divsi %22, %c1024_3 : index
          %25 = arith.remsi %24, %c8_5 : index
          %26 = arith.divsi %24, %c8_5 : index
          %27 = memref.load %arg1[%26, %25, %23] : memref<?x8x1024xf32>
          %28 = arith.mulf %27, %cst_6 : f32
          %29 = arith.maxf %28, %cst_7 : f32
          memref.store %29, %0[%26, %25, %23] : memref<?x8x1024xf32>
        }
        %10 = arith.addi %6, %c512_2 : index
        %11 = affine.min #map(%10)[%3]
        %12 = arith.addi %arg5, %c512_2 : index
        %13 = arith.cmpi slt, %arg5, %11 : index
        scf.if %13 {
          %22 = arith.addi %12, %6 : index
          %23 = arith.remsi %22, %c1024_3 : index
          %24 = arith.divsi %22, %c1024_3 : index
          %25 = arith.remsi %24, %c8_5 : index
          %26 = arith.divsi %24, %c8_5 : index
          %27 = memref.load %arg1[%26, %25, %23] : memref<?x8x1024xf32>
          %28 = arith.mulf %27, %cst_6 : f32
          %29 = arith.maxf %28, %cst_7 : f32
          memref.store %29, %0[%26, %25, %23] : memref<?x8x1024xf32>
        }
        %14 = arith.addi %6, %c1024_3 : index
        %15 = affine.min #map(%14)[%3]
        %16 = arith.addi %arg5, %c1024_3 : index
        %17 = arith.cmpi slt, %arg5, %15 : index
        scf.if %17 {
          %22 = arith.addi %16, %6 : index
          %23 = arith.remsi %22, %c1024_3 : index
          %24 = arith.divsi %22, %c1024_3 : index
          %25 = arith.remsi %24, %c8_5 : index
          %26 = arith.divsi %24, %c8_5 : index
          %27 = memref.load %arg1[%26, %25, %23] : memref<?x8x1024xf32>
          %28 = arith.mulf %27, %cst_6 : f32
          %29 = arith.maxf %28, %cst_7 : f32
          memref.store %29, %0[%26, %25, %23] : memref<?x8x1024xf32>
        }
        %18 = arith.addi %6, %c1536_4 : index
        %19 = affine.min #map(%18)[%3]
        %20 = arith.addi %arg5, %c1536_4 : index
        %21 = arith.cmpi slt, %arg5, %19 : index
        scf.if %21 {
          %22 = arith.addi %20, %6 : index
          %23 = arith.remsi %22, %c1024_3 : index
          %24 = arith.divsi %22, %c1024_3 : index
          %25 = arith.remsi %24, %c8_5 : index
          %26 = arith.divsi %24, %c8_5 : index
          %27 = memref.load %arg1[%26, %25, %23] : memref<?x8x1024xf32>
          %28 = arith.mulf %27, %cst_6 : f32
          %29 = arith.maxf %28, %cst_7 : f32
          memref.store %29, %0[%26, %25, %23] : memref<?x8x1024xf32>
        }
      }
      gpu.terminator
    }
    return %0 : memref<?x8x1024xf32>
  }
}


// -----// IR Dump After GpuKernelOutlining (gpu-kernel-outlining) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module attributes {gpu.container_module} {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c1536 = arith.constant 1536 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch_func  @predict_online_5603970_0_kernel::@predict_online_5603970_0_kernel blocks in (%5, %c1, %c1) threads in (%c512, %c1, %c1) args(%4 : index, %arg1 : memref<?x8x1024xf32>, %0 : memref<?x8x1024xf32>, %3 : index)
    return %0 : memref<?x8x1024xf32>
  }
  gpu.module @predict_online_5603970_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_5603970_0_kernel(%arg0: index, %arg1: memref<?x8x1024xf32>, %arg2: memref<?x8x1024xf32>, %arg3: index) kernel {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c2048 = arith.constant 2048 : index
      %c512 = arith.constant 512 : index
      %c1024 = arith.constant 1024 : index
      %c1536 = arith.constant 1536 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant 5.000000e-01 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %12 = arith.muli %0, %c2048 : index
      %13 = arith.cmpi slt, %0, %arg0 : index
      scf.if %13 {
        %14 = arith.addi %3, %c512 : index
        %15 = arith.addi %3, %c1024 : index
        %16 = arith.addi %3, %c1536 : index
        %17 = arith.addi %3, %12 : index
        %18 = arith.addi %14, %12 : index
        %19 = arith.addi %15, %12 : index
        %20 = arith.addi %16, %12 : index
        %21 = arith.remsi %17, %c1024 : index
        %22 = arith.remsi %18, %c1024 : index
        %23 = arith.remsi %19, %c1024 : index
        %24 = arith.remsi %20, %c1024 : index
        %25 = arith.divsi %17, %c1024 : index
        %26 = arith.divsi %18, %c1024 : index
        %27 = arith.divsi %19, %c1024 : index
        %28 = arith.divsi %20, %c1024 : index
        %29 = arith.remsi %25, %c8 : index
        %30 = arith.remsi %26, %c8 : index
        %31 = arith.remsi %27, %c8 : index
        %32 = arith.remsi %28, %c8 : index
        %33 = arith.divsi %25, %c8 : index
        %34 = arith.divsi %26, %c8 : index
        %35 = arith.divsi %27, %c8 : index
        %36 = arith.divsi %28, %c8 : index
        %37 = memref.load %arg1[%33, %29, %21] : memref<?x8x1024xf32>
        %38 = memref.load %arg1[%34, %30, %22] : memref<?x8x1024xf32>
        %39 = memref.load %arg1[%35, %31, %23] : memref<?x8x1024xf32>
        %40 = memref.load %arg1[%36, %32, %24] : memref<?x8x1024xf32>
        %41 = arith.mulf %37, %cst : f32
        %42 = arith.mulf %38, %cst : f32
        %43 = arith.mulf %39, %cst : f32
        %44 = arith.mulf %40, %cst : f32
        %45 = arith.maxf %41, %cst_0 : f32
        %46 = arith.maxf %42, %cst_0 : f32
        %47 = arith.maxf %43, %cst_0 : f32
        %48 = arith.maxf %44, %cst_0 : f32
        memref.store %45, %arg2[%33, %29, %21] : memref<?x8x1024xf32>
        memref.store %46, %arg2[%34, %30, %22] : memref<?x8x1024xf32>
        memref.store %47, %arg2[%35, %31, %23] : memref<?x8x1024xf32>
        memref.store %48, %arg2[%36, %32, %24] : memref<?x8x1024xf32>
      } else {
        %14 = affine.min #map(%12)[%arg3]
        %15 = arith.cmpi slt, %3, %14 : index
        scf.if %15 {
          %28 = arith.addi %3, %12 : index
          %29 = arith.remsi %28, %c1024 : index
          %30 = arith.divsi %28, %c1024 : index
          %31 = arith.remsi %30, %c8 : index
          %32 = arith.divsi %30, %c8 : index
          %33 = memref.load %arg1[%32, %31, %29] : memref<?x8x1024xf32>
          %34 = arith.mulf %33, %cst : f32
          %35 = arith.maxf %34, %cst_0 : f32
          memref.store %35, %arg2[%32, %31, %29] : memref<?x8x1024xf32>
        }
        %16 = arith.addi %12, %c512 : index
        %17 = affine.min #map(%16)[%arg3]
        %18 = arith.addi %3, %c512 : index
        %19 = arith.cmpi slt, %3, %17 : index
        scf.if %19 {
          %28 = arith.addi %18, %12 : index
          %29 = arith.remsi %28, %c1024 : index
          %30 = arith.divsi %28, %c1024 : index
          %31 = arith.remsi %30, %c8 : index
          %32 = arith.divsi %30, %c8 : index
          %33 = memref.load %arg1[%32, %31, %29] : memref<?x8x1024xf32>
          %34 = arith.mulf %33, %cst : f32
          %35 = arith.maxf %34, %cst_0 : f32
          memref.store %35, %arg2[%32, %31, %29] : memref<?x8x1024xf32>
        }
        %20 = arith.addi %12, %c1024 : index
        %21 = affine.min #map(%20)[%arg3]
        %22 = arith.addi %3, %c1024 : index
        %23 = arith.cmpi slt, %3, %21 : index
        scf.if %23 {
          %28 = arith.addi %22, %12 : index
          %29 = arith.remsi %28, %c1024 : index
          %30 = arith.divsi %28, %c1024 : index
          %31 = arith.remsi %30, %c8 : index
          %32 = arith.divsi %30, %c8 : index
          %33 = memref.load %arg1[%32, %31, %29] : memref<?x8x1024xf32>
          %34 = arith.mulf %33, %cst : f32
          %35 = arith.maxf %34, %cst_0 : f32
          memref.store %35, %arg2[%32, %31, %29] : memref<?x8x1024xf32>
        }
        %24 = arith.addi %12, %c1536 : index
        %25 = affine.min #map(%24)[%arg3]
        %26 = arith.addi %3, %c1536 : index
        %27 = arith.cmpi slt, %3, %25 : index
        scf.if %27 {
          %28 = arith.addi %26, %12 : index
          %29 = arith.remsi %28, %c1024 : index
          %30 = arith.divsi %28, %c1024 : index
          %31 = arith.remsi %30, %c8 : index
          %32 = arith.divsi %30, %c8 : index
          %33 = memref.load %arg1[%32, %31, %29] : memref<?x8x1024xf32>
          %34 = arith.mulf %33, %cst : f32
          %35 = arith.maxf %34, %cst_0 : f32
          memref.store %35, %arg2[%32, %31, %29] : memref<?x8x1024xf32>
        }
      }
      gpu.return
    }
  }
}


// -----// IR Dump After AFGPUAllocaToGPUMem (af-gpu-alloca-to-gpu-mem) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module attributes {gpu.container_module} {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch_func  @predict_online_5603970_0_kernel::@predict_online_5603970_0_kernel blocks in (%5, %c1, %c1) threads in (%c512, %c1, %c1) args(%4 : index, %arg1 : memref<?x8x1024xf32>, %0 : memref<?x8x1024xf32>, %3 : index)
    return %0 : memref<?x8x1024xf32>
  }
  gpu.module @predict_online_5603970_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_5603970_0_kernel(%arg0: index, %arg1: memref<?x8x1024xf32>, %arg2: memref<?x8x1024xf32>, %arg3: index) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 5.000000e-01 : f32
      %c8 = arith.constant 8 : index
      %c1536 = arith.constant 1536 : index
      %c1024 = arith.constant 1024 : index
      %c512 = arith.constant 512 : index
      %c2048 = arith.constant 2048 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %2 = arith.muli %0, %c2048 : index
      %3 = arith.cmpi slt, %0, %arg0 : index
      scf.if %3 {
        %4 = arith.addi %1, %c512 : index
        %5 = arith.addi %1, %c1024 : index
        %6 = arith.addi %1, %c1536 : index
        %7 = arith.addi %1, %2 : index
        %8 = arith.addi %4, %2 : index
        %9 = arith.addi %5, %2 : index
        %10 = arith.addi %6, %2 : index
        %11 = arith.remsi %7, %c1024 : index
        %12 = arith.remsi %8, %c1024 : index
        %13 = arith.remsi %9, %c1024 : index
        %14 = arith.remsi %10, %c1024 : index
        %15 = arith.divsi %7, %c1024 : index
        %16 = arith.divsi %8, %c1024 : index
        %17 = arith.divsi %9, %c1024 : index
        %18 = arith.divsi %10, %c1024 : index
        %19 = arith.remsi %15, %c8 : index
        %20 = arith.remsi %16, %c8 : index
        %21 = arith.remsi %17, %c8 : index
        %22 = arith.remsi %18, %c8 : index
        %23 = arith.divsi %15, %c8 : index
        %24 = arith.divsi %16, %c8 : index
        %25 = arith.divsi %17, %c8 : index
        %26 = arith.divsi %18, %c8 : index
        %27 = memref.load %arg1[%23, %19, %11] : memref<?x8x1024xf32>
        %28 = memref.load %arg1[%24, %20, %12] : memref<?x8x1024xf32>
        %29 = memref.load %arg1[%25, %21, %13] : memref<?x8x1024xf32>
        %30 = memref.load %arg1[%26, %22, %14] : memref<?x8x1024xf32>
        %31 = arith.mulf %27, %cst_0 : f32
        %32 = arith.mulf %28, %cst_0 : f32
        %33 = arith.mulf %29, %cst_0 : f32
        %34 = arith.mulf %30, %cst_0 : f32
        %35 = arith.maxf %31, %cst : f32
        %36 = arith.maxf %32, %cst : f32
        %37 = arith.maxf %33, %cst : f32
        %38 = arith.maxf %34, %cst : f32
        memref.store %35, %arg2[%23, %19, %11] : memref<?x8x1024xf32>
        memref.store %36, %arg2[%24, %20, %12] : memref<?x8x1024xf32>
        memref.store %37, %arg2[%25, %21, %13] : memref<?x8x1024xf32>
        memref.store %38, %arg2[%26, %22, %14] : memref<?x8x1024xf32>
      } else {
        %4 = affine.min #map(%2)[%arg3]
        %5 = arith.cmpi slt, %1, %4 : index
        scf.if %5 {
          %18 = arith.addi %1, %2 : index
          %19 = arith.remsi %18, %c1024 : index
          %20 = arith.divsi %18, %c1024 : index
          %21 = arith.remsi %20, %c8 : index
          %22 = arith.divsi %20, %c8 : index
          %23 = memref.load %arg1[%22, %21, %19] : memref<?x8x1024xf32>
          %24 = arith.mulf %23, %cst_0 : f32
          %25 = arith.maxf %24, %cst : f32
          memref.store %25, %arg2[%22, %21, %19] : memref<?x8x1024xf32>
        }
        %6 = arith.addi %2, %c512 : index
        %7 = affine.min #map(%6)[%arg3]
        %8 = arith.addi %1, %c512 : index
        %9 = arith.cmpi slt, %1, %7 : index
        scf.if %9 {
          %18 = arith.addi %8, %2 : index
          %19 = arith.remsi %18, %c1024 : index
          %20 = arith.divsi %18, %c1024 : index
          %21 = arith.remsi %20, %c8 : index
          %22 = arith.divsi %20, %c8 : index
          %23 = memref.load %arg1[%22, %21, %19] : memref<?x8x1024xf32>
          %24 = arith.mulf %23, %cst_0 : f32
          %25 = arith.maxf %24, %cst : f32
          memref.store %25, %arg2[%22, %21, %19] : memref<?x8x1024xf32>
        }
        %10 = arith.addi %2, %c1024 : index
        %11 = affine.min #map(%10)[%arg3]
        %12 = arith.addi %1, %c1024 : index
        %13 = arith.cmpi slt, %1, %11 : index
        scf.if %13 {
          %18 = arith.addi %12, %2 : index
          %19 = arith.remsi %18, %c1024 : index
          %20 = arith.divsi %18, %c1024 : index
          %21 = arith.remsi %20, %c8 : index
          %22 = arith.divsi %20, %c8 : index
          %23 = memref.load %arg1[%22, %21, %19] : memref<?x8x1024xf32>
          %24 = arith.mulf %23, %cst_0 : f32
          %25 = arith.maxf %24, %cst : f32
          memref.store %25, %arg2[%22, %21, %19] : memref<?x8x1024xf32>
        }
        %14 = arith.addi %2, %c1536 : index
        %15 = affine.min #map(%14)[%arg3]
        %16 = arith.addi %1, %c1536 : index
        %17 = arith.cmpi slt, %1, %15 : index
        scf.if %17 {
          %18 = arith.addi %16, %2 : index
          %19 = arith.remsi %18, %c1024 : index
          %20 = arith.divsi %18, %c1024 : index
          %21 = arith.remsi %20, %c8 : index
          %22 = arith.divsi %20, %c8 : index
          %23 = memref.load %arg1[%22, %21, %19] : memref<?x8x1024xf32>
          %24 = arith.mulf %23, %cst_0 : f32
          %25 = arith.maxf %24, %cst : f32
          memref.store %25, %arg2[%22, %21, %19] : memref<?x8x1024xf32>
        }
      }
      gpu.return
    }
  }
}


// -----// IR Dump After ConvertAffineToStandard (lower-affine) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch_func  @predict_online_5603970_0_kernel::@predict_online_5603970_0_kernel blocks in (%5, %c1, %c1) threads in (%c512, %c1, %c1) args(%4 : index, %arg1 : memref<?x8x1024xf32>, %0 : memref<?x8x1024xf32>, %3 : index)
    return %0 : memref<?x8x1024xf32>
  }
  gpu.module @predict_online_5603970_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_5603970_0_kernel(%arg0: index, %arg1: memref<?x8x1024xf32>, %arg2: memref<?x8x1024xf32>, %arg3: index) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 5.000000e-01 : f32
      %c8 = arith.constant 8 : index
      %c1536 = arith.constant 1536 : index
      %c1024 = arith.constant 1024 : index
      %c512 = arith.constant 512 : index
      %c2048 = arith.constant 2048 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %2 = arith.muli %0, %c2048 : index
      %3 = arith.cmpi slt, %0, %arg0 : index
      scf.if %3 {
        %4 = arith.addi %1, %c512 : index
        %5 = arith.addi %1, %c1024 : index
        %6 = arith.addi %1, %c1536 : index
        %7 = arith.addi %1, %2 : index
        %8 = arith.addi %4, %2 : index
        %9 = arith.addi %5, %2 : index
        %10 = arith.addi %6, %2 : index
        %11 = arith.remsi %7, %c1024 : index
        %12 = arith.remsi %8, %c1024 : index
        %13 = arith.remsi %9, %c1024 : index
        %14 = arith.remsi %10, %c1024 : index
        %15 = arith.divsi %7, %c1024 : index
        %16 = arith.divsi %8, %c1024 : index
        %17 = arith.divsi %9, %c1024 : index
        %18 = arith.divsi %10, %c1024 : index
        %19 = arith.remsi %15, %c8 : index
        %20 = arith.remsi %16, %c8 : index
        %21 = arith.remsi %17, %c8 : index
        %22 = arith.remsi %18, %c8 : index
        %23 = arith.divsi %15, %c8 : index
        %24 = arith.divsi %16, %c8 : index
        %25 = arith.divsi %17, %c8 : index
        %26 = arith.divsi %18, %c8 : index
        %27 = memref.load %arg1[%23, %19, %11] : memref<?x8x1024xf32>
        %28 = memref.load %arg1[%24, %20, %12] : memref<?x8x1024xf32>
        %29 = memref.load %arg1[%25, %21, %13] : memref<?x8x1024xf32>
        %30 = memref.load %arg1[%26, %22, %14] : memref<?x8x1024xf32>
        %31 = arith.mulf %27, %cst_0 : f32
        %32 = arith.mulf %28, %cst_0 : f32
        %33 = arith.mulf %29, %cst_0 : f32
        %34 = arith.mulf %30, %cst_0 : f32
        %35 = arith.maxf %31, %cst : f32
        %36 = arith.maxf %32, %cst : f32
        %37 = arith.maxf %33, %cst : f32
        %38 = arith.maxf %34, %cst : f32
        memref.store %35, %arg2[%23, %19, %11] : memref<?x8x1024xf32>
        memref.store %36, %arg2[%24, %20, %12] : memref<?x8x1024xf32>
        memref.store %37, %arg2[%25, %21, %13] : memref<?x8x1024xf32>
        memref.store %38, %arg2[%26, %22, %14] : memref<?x8x1024xf32>
      } else {
        %c-1 = arith.constant -1 : index
        %4 = arith.muli %2, %c-1 : index
        %5 = arith.addi %4, %arg3 : index
        %c512_1 = arith.constant 512 : index
        %6 = arith.cmpi slt, %5, %c512_1 : index
        %7 = arith.select %6, %5, %c512_1 : index
        %8 = arith.cmpi slt, %1, %7 : index
        scf.if %8 {
          %30 = arith.addi %1, %2 : index
          %31 = arith.remsi %30, %c1024 : index
          %32 = arith.divsi %30, %c1024 : index
          %33 = arith.remsi %32, %c8 : index
          %34 = arith.divsi %32, %c8 : index
          %35 = memref.load %arg1[%34, %33, %31] : memref<?x8x1024xf32>
          %36 = arith.mulf %35, %cst_0 : f32
          %37 = arith.maxf %36, %cst : f32
          memref.store %37, %arg2[%34, %33, %31] : memref<?x8x1024xf32>
        }
        %9 = arith.addi %2, %c512 : index
        %c-1_2 = arith.constant -1 : index
        %10 = arith.muli %9, %c-1_2 : index
        %11 = arith.addi %10, %arg3 : index
        %c512_3 = arith.constant 512 : index
        %12 = arith.cmpi slt, %11, %c512_3 : index
        %13 = arith.select %12, %11, %c512_3 : index
        %14 = arith.addi %1, %c512 : index
        %15 = arith.cmpi slt, %1, %13 : index
        scf.if %15 {
          %30 = arith.addi %14, %2 : index
          %31 = arith.remsi %30, %c1024 : index
          %32 = arith.divsi %30, %c1024 : index
          %33 = arith.remsi %32, %c8 : index
          %34 = arith.divsi %32, %c8 : index
          %35 = memref.load %arg1[%34, %33, %31] : memref<?x8x1024xf32>
          %36 = arith.mulf %35, %cst_0 : f32
          %37 = arith.maxf %36, %cst : f32
          memref.store %37, %arg2[%34, %33, %31] : memref<?x8x1024xf32>
        }
        %16 = arith.addi %2, %c1024 : index
        %c-1_4 = arith.constant -1 : index
        %17 = arith.muli %16, %c-1_4 : index
        %18 = arith.addi %17, %arg3 : index
        %c512_5 = arith.constant 512 : index
        %19 = arith.cmpi slt, %18, %c512_5 : index
        %20 = arith.select %19, %18, %c512_5 : index
        %21 = arith.addi %1, %c1024 : index
        %22 = arith.cmpi slt, %1, %20 : index
        scf.if %22 {
          %30 = arith.addi %21, %2 : index
          %31 = arith.remsi %30, %c1024 : index
          %32 = arith.divsi %30, %c1024 : index
          %33 = arith.remsi %32, %c8 : index
          %34 = arith.divsi %32, %c8 : index
          %35 = memref.load %arg1[%34, %33, %31] : memref<?x8x1024xf32>
          %36 = arith.mulf %35, %cst_0 : f32
          %37 = arith.maxf %36, %cst : f32
          memref.store %37, %arg2[%34, %33, %31] : memref<?x8x1024xf32>
        }
        %23 = arith.addi %2, %c1536 : index
        %c-1_6 = arith.constant -1 : index
        %24 = arith.muli %23, %c-1_6 : index
        %25 = arith.addi %24, %arg3 : index
        %c512_7 = arith.constant 512 : index
        %26 = arith.cmpi slt, %25, %c512_7 : index
        %27 = arith.select %26, %25, %c512_7 : index
        %28 = arith.addi %1, %c1536 : index
        %29 = arith.cmpi slt, %1, %27 : index
        scf.if %29 {
          %30 = arith.addi %28, %2 : index
          %31 = arith.remsi %30, %c1024 : index
          %32 = arith.divsi %30, %c1024 : index
          %33 = arith.remsi %32, %c8 : index
          %34 = arith.divsi %32, %c8 : index
          %35 = memref.load %arg1[%34, %33, %31] : memref<?x8x1024xf32>
          %36 = arith.mulf %35, %cst_0 : f32
          %37 = arith.maxf %36, %cst : f32
          memref.store %37, %arg2[%34, %33, %31] : memref<?x8x1024xf32>
        }
      }
      gpu.return
    }
  }
}


// -----// IR Dump After SCFToControlFlow (convert-scf-to-cf) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    tf_framework.assert %arg0, %1, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch_func  @predict_online_5603970_0_kernel::@predict_online_5603970_0_kernel blocks in (%5, %c1, %c1) threads in (%c512, %c1, %c1) args(%4 : index, %arg1 : memref<?x8x1024xf32>, %0 : memref<?x8x1024xf32>, %3 : index)
    return %0 : memref<?x8x1024xf32>
  }
  gpu.module @predict_online_5603970_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_5603970_0_kernel(%arg0: index, %arg1: memref<?x8x1024xf32>, %arg2: memref<?x8x1024xf32>, %arg3: index) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 5.000000e-01 : f32
      %c8 = arith.constant 8 : index
      %c1536 = arith.constant 1536 : index
      %c1024 = arith.constant 1024 : index
      %c512 = arith.constant 512 : index
      %c2048 = arith.constant 2048 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %2 = arith.muli %0, %c2048 : index
      %3 = arith.cmpi slt, %0, %arg0 : index
      cf.cond_br %3, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %4 = arith.addi %1, %c512 : index
      %5 = arith.addi %1, %c1024 : index
      %6 = arith.addi %1, %c1536 : index
      %7 = arith.addi %1, %2 : index
      %8 = arith.addi %4, %2 : index
      %9 = arith.addi %5, %2 : index
      %10 = arith.addi %6, %2 : index
      %11 = arith.remsi %7, %c1024 : index
      %12 = arith.remsi %8, %c1024 : index
      %13 = arith.remsi %9, %c1024 : index
      %14 = arith.remsi %10, %c1024 : index
      %15 = arith.divsi %7, %c1024 : index
      %16 = arith.divsi %8, %c1024 : index
      %17 = arith.divsi %9, %c1024 : index
      %18 = arith.divsi %10, %c1024 : index
      %19 = arith.remsi %15, %c8 : index
      %20 = arith.remsi %16, %c8 : index
      %21 = arith.remsi %17, %c8 : index
      %22 = arith.remsi %18, %c8 : index
      %23 = arith.divsi %15, %c8 : index
      %24 = arith.divsi %16, %c8 : index
      %25 = arith.divsi %17, %c8 : index
      %26 = arith.divsi %18, %c8 : index
      %27 = memref.load %arg1[%23, %19, %11] : memref<?x8x1024xf32>
      %28 = memref.load %arg1[%24, %20, %12] : memref<?x8x1024xf32>
      %29 = memref.load %arg1[%25, %21, %13] : memref<?x8x1024xf32>
      %30 = memref.load %arg1[%26, %22, %14] : memref<?x8x1024xf32>
      %31 = arith.mulf %27, %cst_0 : f32
      %32 = arith.mulf %28, %cst_0 : f32
      %33 = arith.mulf %29, %cst_0 : f32
      %34 = arith.mulf %30, %cst_0 : f32
      %35 = arith.maxf %31, %cst : f32
      %36 = arith.maxf %32, %cst : f32
      %37 = arith.maxf %33, %cst : f32
      %38 = arith.maxf %34, %cst : f32
      memref.store %35, %arg2[%23, %19, %11] : memref<?x8x1024xf32>
      memref.store %36, %arg2[%24, %20, %12] : memref<?x8x1024xf32>
      memref.store %37, %arg2[%25, %21, %13] : memref<?x8x1024xf32>
      memref.store %38, %arg2[%26, %22, %14] : memref<?x8x1024xf32>
      cf.br ^bb12
    ^bb3:  // pred: ^bb1
      %c-1 = arith.constant -1 : index
      %39 = arith.muli %2, %c-1 : index
      %40 = arith.addi %39, %arg3 : index
      %c512_1 = arith.constant 512 : index
      %41 = arith.cmpi slt, %40, %c512_1 : index
      %42 = arith.select %41, %40, %c512_1 : index
      %43 = arith.cmpi slt, %1, %42 : index
      cf.cond_br %43, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %44 = arith.addi %1, %2 : index
      %45 = arith.remsi %44, %c1024 : index
      %46 = arith.divsi %44, %c1024 : index
      %47 = arith.remsi %46, %c8 : index
      %48 = arith.divsi %46, %c8 : index
      %49 = memref.load %arg1[%48, %47, %45] : memref<?x8x1024xf32>
      %50 = arith.mulf %49, %cst_0 : f32
      %51 = arith.maxf %50, %cst : f32
      memref.store %51, %arg2[%48, %47, %45] : memref<?x8x1024xf32>
      cf.br ^bb5
    ^bb5:  // 2 preds: ^bb3, ^bb4
      %52 = arith.addi %2, %c512 : index
      %c-1_2 = arith.constant -1 : index
      %53 = arith.muli %52, %c-1_2 : index
      %54 = arith.addi %53, %arg3 : index
      %c512_3 = arith.constant 512 : index
      %55 = arith.cmpi slt, %54, %c512_3 : index
      %56 = arith.select %55, %54, %c512_3 : index
      %57 = arith.addi %1, %c512 : index
      %58 = arith.cmpi slt, %1, %56 : index
      cf.cond_br %58, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %59 = arith.addi %57, %2 : index
      %60 = arith.remsi %59, %c1024 : index
      %61 = arith.divsi %59, %c1024 : index
      %62 = arith.remsi %61, %c8 : index
      %63 = arith.divsi %61, %c8 : index
      %64 = memref.load %arg1[%63, %62, %60] : memref<?x8x1024xf32>
      %65 = arith.mulf %64, %cst_0 : f32
      %66 = arith.maxf %65, %cst : f32
      memref.store %66, %arg2[%63, %62, %60] : memref<?x8x1024xf32>
      cf.br ^bb7
    ^bb7:  // 2 preds: ^bb5, ^bb6
      %67 = arith.addi %2, %c1024 : index
      %c-1_4 = arith.constant -1 : index
      %68 = arith.muli %67, %c-1_4 : index
      %69 = arith.addi %68, %arg3 : index
      %c512_5 = arith.constant 512 : index
      %70 = arith.cmpi slt, %69, %c512_5 : index
      %71 = arith.select %70, %69, %c512_5 : index
      %72 = arith.addi %1, %c1024 : index
      %73 = arith.cmpi slt, %1, %71 : index
      cf.cond_br %73, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %74 = arith.addi %72, %2 : index
      %75 = arith.remsi %74, %c1024 : index
      %76 = arith.divsi %74, %c1024 : index
      %77 = arith.remsi %76, %c8 : index
      %78 = arith.divsi %76, %c8 : index
      %79 = memref.load %arg1[%78, %77, %75] : memref<?x8x1024xf32>
      %80 = arith.mulf %79, %cst_0 : f32
      %81 = arith.maxf %80, %cst : f32
      memref.store %81, %arg2[%78, %77, %75] : memref<?x8x1024xf32>
      cf.br ^bb9
    ^bb9:  // 2 preds: ^bb7, ^bb8
      %82 = arith.addi %2, %c1536 : index
      %c-1_6 = arith.constant -1 : index
      %83 = arith.muli %82, %c-1_6 : index
      %84 = arith.addi %83, %arg3 : index
      %c512_7 = arith.constant 512 : index
      %85 = arith.cmpi slt, %84, %c512_7 : index
      %86 = arith.select %85, %84, %c512_7 : index
      %87 = arith.addi %1, %c1536 : index
      %88 = arith.cmpi slt, %1, %86 : index
      cf.cond_br %88, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %89 = arith.addi %87, %2 : index
      %90 = arith.remsi %89, %c1024 : index
      %91 = arith.divsi %89, %c1024 : index
      %92 = arith.remsi %91, %c8 : index
      %93 = arith.divsi %91, %c8 : index
      %94 = memref.load %arg1[%93, %92, %90] : memref<?x8x1024xf32>
      %95 = arith.mulf %94, %cst_0 : f32
      %96 = arith.maxf %95, %cst : f32
      memref.store %96, %arg2[%93, %92, %90] : memref<?x8x1024xf32>
      cf.br ^bb11
    ^bb11:  // 2 preds: ^bb9, ^bb10
      cf.br ^bb12
    ^bb12:  // 2 preds: ^bb2, ^bb11
      gpu.return
    }
  }
}


// -----// IR Dump After RewriteTFFrameworkAssert (rewrite-tf-framework-assert) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    cf.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch_func  @predict_online_5603970_0_kernel::@predict_online_5603970_0_kernel blocks in (%5, %c1, %c1) threads in (%c512, %c1, %c1) args(%4 : index, %arg1 : memref<?x8x1024xf32>, %0 : memref<?x8x1024xf32>, %3 : index)
    return %0 : memref<?x8x1024xf32>
  ^bb2:  // pred: ^bb0
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %6 = tf_framework.null_memref : memref<?x8x1024xf32>
    return %6 : memref<?x8x1024xf32>
  }
  gpu.module @predict_online_5603970_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_5603970_0_kernel(%arg0: index, %arg1: memref<?x8x1024xf32>, %arg2: memref<?x8x1024xf32>, %arg3: index) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 5.000000e-01 : f32
      %c8 = arith.constant 8 : index
      %c1536 = arith.constant 1536 : index
      %c1024 = arith.constant 1024 : index
      %c512 = arith.constant 512 : index
      %c2048 = arith.constant 2048 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %2 = arith.muli %0, %c2048 : index
      %3 = arith.cmpi slt, %0, %arg0 : index
      cf.cond_br %3, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %4 = arith.addi %1, %c512 : index
      %5 = arith.addi %1, %c1024 : index
      %6 = arith.addi %1, %c1536 : index
      %7 = arith.addi %1, %2 : index
      %8 = arith.addi %4, %2 : index
      %9 = arith.addi %5, %2 : index
      %10 = arith.addi %6, %2 : index
      %11 = arith.remsi %7, %c1024 : index
      %12 = arith.remsi %8, %c1024 : index
      %13 = arith.remsi %9, %c1024 : index
      %14 = arith.remsi %10, %c1024 : index
      %15 = arith.divsi %7, %c1024 : index
      %16 = arith.divsi %8, %c1024 : index
      %17 = arith.divsi %9, %c1024 : index
      %18 = arith.divsi %10, %c1024 : index
      %19 = arith.remsi %15, %c8 : index
      %20 = arith.remsi %16, %c8 : index
      %21 = arith.remsi %17, %c8 : index
      %22 = arith.remsi %18, %c8 : index
      %23 = arith.divsi %15, %c8 : index
      %24 = arith.divsi %16, %c8 : index
      %25 = arith.divsi %17, %c8 : index
      %26 = arith.divsi %18, %c8 : index
      %27 = memref.load %arg1[%23, %19, %11] : memref<?x8x1024xf32>
      %28 = memref.load %arg1[%24, %20, %12] : memref<?x8x1024xf32>
      %29 = memref.load %arg1[%25, %21, %13] : memref<?x8x1024xf32>
      %30 = memref.load %arg1[%26, %22, %14] : memref<?x8x1024xf32>
      %31 = arith.mulf %27, %cst_0 : f32
      %32 = arith.mulf %28, %cst_0 : f32
      %33 = arith.mulf %29, %cst_0 : f32
      %34 = arith.mulf %30, %cst_0 : f32
      %35 = arith.maxf %31, %cst : f32
      %36 = arith.maxf %32, %cst : f32
      %37 = arith.maxf %33, %cst : f32
      %38 = arith.maxf %34, %cst : f32
      memref.store %35, %arg2[%23, %19, %11] : memref<?x8x1024xf32>
      memref.store %36, %arg2[%24, %20, %12] : memref<?x8x1024xf32>
      memref.store %37, %arg2[%25, %21, %13] : memref<?x8x1024xf32>
      memref.store %38, %arg2[%26, %22, %14] : memref<?x8x1024xf32>
      cf.br ^bb12
    ^bb3:  // pred: ^bb1
      %c-1 = arith.constant -1 : index
      %39 = arith.muli %2, %c-1 : index
      %40 = arith.addi %39, %arg3 : index
      %c512_1 = arith.constant 512 : index
      %41 = arith.cmpi slt, %40, %c512_1 : index
      %42 = arith.select %41, %40, %c512_1 : index
      %43 = arith.cmpi slt, %1, %42 : index
      cf.cond_br %43, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %44 = arith.addi %1, %2 : index
      %45 = arith.remsi %44, %c1024 : index
      %46 = arith.divsi %44, %c1024 : index
      %47 = arith.remsi %46, %c8 : index
      %48 = arith.divsi %46, %c8 : index
      %49 = memref.load %arg1[%48, %47, %45] : memref<?x8x1024xf32>
      %50 = arith.mulf %49, %cst_0 : f32
      %51 = arith.maxf %50, %cst : f32
      memref.store %51, %arg2[%48, %47, %45] : memref<?x8x1024xf32>
      cf.br ^bb5
    ^bb5:  // 2 preds: ^bb3, ^bb4
      %52 = arith.addi %2, %c512 : index
      %c-1_2 = arith.constant -1 : index
      %53 = arith.muli %52, %c-1_2 : index
      %54 = arith.addi %53, %arg3 : index
      %c512_3 = arith.constant 512 : index
      %55 = arith.cmpi slt, %54, %c512_3 : index
      %56 = arith.select %55, %54, %c512_3 : index
      %57 = arith.addi %1, %c512 : index
      %58 = arith.cmpi slt, %1, %56 : index
      cf.cond_br %58, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %59 = arith.addi %57, %2 : index
      %60 = arith.remsi %59, %c1024 : index
      %61 = arith.divsi %59, %c1024 : index
      %62 = arith.remsi %61, %c8 : index
      %63 = arith.divsi %61, %c8 : index
      %64 = memref.load %arg1[%63, %62, %60] : memref<?x8x1024xf32>
      %65 = arith.mulf %64, %cst_0 : f32
      %66 = arith.maxf %65, %cst : f32
      memref.store %66, %arg2[%63, %62, %60] : memref<?x8x1024xf32>
      cf.br ^bb7
    ^bb7:  // 2 preds: ^bb5, ^bb6
      %67 = arith.addi %2, %c1024 : index
      %c-1_4 = arith.constant -1 : index
      %68 = arith.muli %67, %c-1_4 : index
      %69 = arith.addi %68, %arg3 : index
      %c512_5 = arith.constant 512 : index
      %70 = arith.cmpi slt, %69, %c512_5 : index
      %71 = arith.select %70, %69, %c512_5 : index
      %72 = arith.addi %1, %c1024 : index
      %73 = arith.cmpi slt, %1, %71 : index
      cf.cond_br %73, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %74 = arith.addi %72, %2 : index
      %75 = arith.remsi %74, %c1024 : index
      %76 = arith.divsi %74, %c1024 : index
      %77 = arith.remsi %76, %c8 : index
      %78 = arith.divsi %76, %c8 : index
      %79 = memref.load %arg1[%78, %77, %75] : memref<?x8x1024xf32>
      %80 = arith.mulf %79, %cst_0 : f32
      %81 = arith.maxf %80, %cst : f32
      memref.store %81, %arg2[%78, %77, %75] : memref<?x8x1024xf32>
      cf.br ^bb9
    ^bb9:  // 2 preds: ^bb7, ^bb8
      %82 = arith.addi %2, %c1536 : index
      %c-1_6 = arith.constant -1 : index
      %83 = arith.muli %82, %c-1_6 : index
      %84 = arith.addi %83, %arg3 : index
      %c512_7 = arith.constant 512 : index
      %85 = arith.cmpi slt, %84, %c512_7 : index
      %86 = arith.select %85, %84, %c512_7 : index
      %87 = arith.addi %1, %c1536 : index
      %88 = arith.cmpi slt, %1, %86 : index
      cf.cond_br %88, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %89 = arith.addi %87, %2 : index
      %90 = arith.remsi %89, %c1024 : index
      %91 = arith.divsi %89, %c1024 : index
      %92 = arith.remsi %91, %c8 : index
      %93 = arith.divsi %91, %c8 : index
      %94 = memref.load %arg1[%93, %92, %90] : memref<?x8x1024xf32>
      %95 = arith.mulf %94, %cst_0 : f32
      %96 = arith.maxf %95, %cst : f32
      memref.store %96, %arg2[%93, %92, %90] : memref<?x8x1024xf32>
      cf.br ^bb11
    ^bb11:  // 2 preds: ^bb9, ^bb10
      cf.br ^bb12
    ^bb12:  // 2 preds: ^bb2, ^bb11
      gpu.return
    }
  }
}


// -----// IR Dump After InterleaveLoadAndCompute (interleave-load-and-compute) //----- //
module attributes {gpu.container_module} {
  func.func @predict_online_5603970_0(%arg0: !tf_framework.op_kernel_context {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true}, %arg1: memref<?x8x1024xf32>) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg1, %c0 : memref<?x8x1024xf32>
    %0 = tf_framework.alloc(%arg0, %dim) : memref<?x8x1024xf32>
    %1 = tf_framework.is_valid_memref(%0) : memref<?x8x1024xf32> -> i1
    cf.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = arith.muli %dim, %c8 : index
    %3 = arith.muli %2, %c1024 : index
    %4 = arith.divsi %3, %c2048 : index
    %5 = arith.ceildivsi %3, %c2048 : index
    gpu.launch_func  @predict_online_5603970_0_kernel::@predict_online_5603970_0_kernel blocks in (%5, %c1, %c1) threads in (%c512, %c1, %c1) args(%4 : index, %arg1 : memref<?x8x1024xf32>, %0 : memref<?x8x1024xf32>, %3 : index)
    return %0 : memref<?x8x1024xf32>
  ^bb2:  // pred: ^bb0
    tf_framework.report_error %arg0, RESOURCE_EXHAUSTED, "failed to allocate memory"
    %6 = tf_framework.null_memref : memref<?x8x1024xf32>
    return %6 : memref<?x8x1024xf32>
  }
  gpu.module @predict_online_5603970_0_kernel attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
    gpu.func @predict_online_5603970_0_kernel(%arg0: index, %arg1: memref<?x8x1024xf32>, %arg2: memref<?x8x1024xf32>, %arg3: index) kernel {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 5.000000e-01 : f32
      %c8 = arith.constant 8 : index
      %c1536 = arith.constant 1536 : index
      %c1024 = arith.constant 1024 : index
      %c512 = arith.constant 512 : index
      %c2048 = arith.constant 2048 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %2 = arith.muli %0, %c2048 : index
      %3 = arith.cmpi slt, %0, %arg0 : index
      cf.cond_br %3, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %4 = arith.addi %1, %c512 : index
      %5 = arith.addi %1, %c1024 : index
      %6 = arith.addi %1, %c1536 : index
      %7 = arith.addi %1, %2 : index
      %8 = arith.addi %4, %2 : index
      %9 = arith.addi %5, %2 : index
      %10 = arith.addi %6, %2 : index
      %11 = arith.remsi %7, %c1024 : index
      %12 = arith.remsi %8, %c1024 : index
      %13 = arith.remsi %9, %c1024 : index
      %14 = arith.remsi %10, %c1024 : index
      %15 = arith.divsi %7, %c1024 : index
      %16 = arith.divsi %8, %c1024 : index
      %17 = arith.divsi %9, %c1024 : index
      %18 = arith.divsi %10, %c1024 : index
      %19 = arith.remsi %15, %c8 : index
      %20 = arith.remsi %16, %c8 : index
      %21 = arith.remsi %17, %c8 : index
      %22 = arith.remsi %18, %c8 : index
      %23 = arith.divsi %15, %c8 : index
      %24 = arith.divsi %16, %c8 : index
      %25 = arith.divsi %17, %c8 : index
      %26 = arith.divsi %18, %c8 : index
      %27 = memref.load %arg1[%23, %19, %11] : memref<?x8x1024xf32>
      %28 = arith.mulf %27, %cst_0 : f32
      %29 = memref.load %arg1[%24, %20, %12] : memref<?x8x1024xf32>
      %30 = arith.mulf %29, %cst_0 : f32
      %31 = memref.load %arg1[%25, %21, %13] : memref<?x8x1024xf32>
      %32 = arith.mulf %31, %cst_0 : f32
      %33 = memref.load %arg1[%26, %22, %14] : memref<?x8x1024xf32>
      %34 = arith.mulf %33, %cst_0 : f32
      %35 = arith.maxf %28, %cst : f32
      memref.store %35, %arg2[%23, %19, %11] : memref<?x8x1024xf32>
      %36 = arith.maxf %30, %cst : f32
      memref.store %36, %arg2[%24, %20, %12] : memref<?x8x1024xf32>
      %37 = arith.maxf %32, %cst : f32
      memref.store %37, %arg2[%25, %21, %13] : memref<?x8x1024xf32>
      %38 = arith.maxf %34, %cst : f32
      memref.store %38, %arg2[%26, %22, %14] : memref<?x8x1024xf32>
      cf.br ^bb12
    ^bb3:  // pred: ^bb1
      %c-1 = arith.constant -1 : index
      %39 = arith.muli %2, %c-1 : index
      %40 = arith.addi %39, %arg3 : index
      %c512_1 = arith.constant 512 : index
      %41 = arith.cmpi slt, %40, %c512_1 : index
      %42 = arith.select %41, %40, %c512_1 : index
      %43 = arith.cmpi slt, %1, %42 : index
      cf.cond_br %43, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %44 = arith.addi %1, %2 : index
      %45 = arith.remsi %44, %c1024 : index
      %46 = arith.divsi %44, %c1024 : index
      %47 = arith.remsi %46, %c8 : index
      %48 = arith.divsi %46, %c8 : index
      %49 = memref.load %arg1[%48, %47, %45] : memref<?x8x1024xf32>
      %50 = arith.mulf %49, %cst_0 : f32
      %51 = arith.maxf %50, %cst : f32
      memref.store %51, %arg2[%48, %47, %45] : memref<?x8x1024xf32>
      cf.br ^bb5
    ^bb5:  // 2 preds: ^bb3, ^bb4
      %52 = arith.addi %2, %c512 : index
      %c-1_2 = arith.constant -1 : index
      %53 = arith.muli %52, %c-1_2 : index
      %54 = arith.addi %53, %arg3 : index
      %c512_3 = arith.constant 512 : index
      %55 = arith.cmpi slt, %54, %c512_3 : index
      %56 = arith.select %55, %54, %c512_3 : index
      %57 = arith.addi %1, %c512 : index
      %58 = arith.cmpi slt, %1, %56 : index
      cf.cond_br %58, ^bb6, ^bb7
    ^bb6:  // pred: ^bb5
      %59 = arith.addi %57, %2 : index
      %60 = arith.remsi %59, %c1024 : index
      %61 = arith.divsi %59, %c1024 : index
      %62 = arith.remsi %61, %c8 : index
      %63 = arith.divsi %61, %c8 : index
      %64 = memref.load %arg1[%63, %62, %60] : memref<?x8x1024xf32>
      %65 = arith.mulf %64, %cst_0 : f32
      %66 = arith.maxf %65, %cst : f32
      memref.store %66, %arg2[%63, %62, %60] : memref<?x8x1024xf32>
      cf.br ^bb7
    ^bb7:  // 2 preds: ^bb5, ^bb6
      %67 = arith.addi %2, %c1024 : index
      %c-1_4 = arith.constant -1 : index
      %68 = arith.muli %67, %c-1_4 : index
      %69 = arith.addi %68, %arg3 : index
      %c512_5 = arith.constant 512 : index
      %70 = arith.cmpi slt, %69, %c512_5 : index
      %71 = arith.select %70, %69, %c512_5 : index
      %72 = arith.addi %1, %c1024 : index
      %73 = arith.cmpi slt, %1, %71 : index
      cf.cond_br %73, ^bb8, ^bb9
    ^bb8:  // pred: ^bb7
      %74 = arith.addi %72, %2 : index
      %75 = arith.remsi %74, %c1024 : index
      %76 = arith.divsi %74, %c1024 : index
      %77 = arith.remsi %76, %c8 : index
      %78 = arith.divsi %76, %c8 : index
      %79 = memref.load %arg1[%78, %77, %75] : memref<?x8x1024xf32>
      %80 = arith.mulf %79, %cst_0 : f32
      %81 = arith.maxf %80, %cst : f32
      memref.store %81, %arg2[%78, %77, %75] : memref<?x8x1024xf32>
      cf.br ^bb9
    ^bb9:  // 2 preds: ^bb7, ^bb8
      %82 = arith.addi %2, %c1536 : index
      %c-1_6 = arith.constant -1 : index
      %83 = arith.muli %82, %c-1_6 : index
      %84 = arith.addi %83, %arg3 : index
      %c512_7 = arith.constant 512 : index
      %85 = arith.cmpi slt, %84, %c512_7 : index
      %86 = arith.select %85, %84, %c512_7 : index
      %87 = arith.addi %1, %c1536 : index
      %88 = arith.cmpi slt, %1, %86 : index
      cf.cond_br %88, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %89 = arith.addi %87, %2 : index
      %90 = arith.remsi %89, %c1024 : index
      %91 = arith.divsi %89, %c1024 : index
      %92 = arith.remsi %91, %c8 : index
      %93 = arith.divsi %91, %c8 : index
      %94 = memref.load %arg1[%93, %92, %90] : memref<?x8x1024xf32>
      %95 = arith.mulf %94, %cst_0 : f32
      %96 = arith.maxf %95, %cst : f32
      memref.store %96, %arg2[%93, %92, %90] : memref<?x8x1024xf32>
      cf.br ^bb11
    ^bb11:  // 2 preds: ^bb9, ^bb10
      cf.br ^bb12
    ^bb12:  // 2 preds: ^bb2, ^bb11
      gpu.return
    }
  }
}


