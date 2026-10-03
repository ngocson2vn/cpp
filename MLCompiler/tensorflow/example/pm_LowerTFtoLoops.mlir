// -----// IR Dump After LegalizeTFNoFallback (xla-legalize-tf-no-fallback) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<5.000000e-01> : tensor<f32>
  %1 = chlo.broadcast_multiply %arg0, %0 {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<?x8x1024xf32>, tensor<f32>) -> tensor<?x8x1024xf32>
  return %1 : tensor<?x8x1024xf32>
}

// -----// IR Dump After ChloLegalizeToHloPass (chlo-legalize-to-hlo) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<5.000000e-01> : tensor<f32>
  %1 = shape.shape_of %arg0 : tensor<?x8x1024xf32> -> tensor<3xindex>
  %2 = shape.shape_of %0 : tensor<f32> -> tensor<0xindex>
  %3 = shape.cstr_broadcastable %1, %2 : tensor<3xindex>, tensor<0xindex>
  %4 = shape.assuming %3 -> (tensor<?x8x1024xf32>) {
    %5 = shape.shape_of %arg0 : tensor<?x8x1024xf32> -> tensor<3xindex>
    %6 = shape.const_shape [] : tensor<0xindex>
    %7 = shape.broadcast %5, %6 : tensor<3xindex>, tensor<0xindex> -> tensor<3xindex>
    %8 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %7) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x8x1024xf32>, tensor<3xindex>) -> tensor<?x8x1024xf32>
    %9 = "mhlo.dynamic_broadcast_in_dim"(%0, %7) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<3xindex>) -> tensor<?x8x1024xf32>
    %10 = mhlo.multiply %8, %9 : tensor<?x8x1024xf32>
    shape.assuming_yield %10 : tensor<?x8x1024xf32>
  }
  return %4 : tensor<?x8x1024xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<5.000000e-01> : tensor<f32>
  %1 = shape.shape_of %arg0 : tensor<?x8x1024xf32> -> tensor<3xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<3xindex>) -> tensor<?x8x1024xf32>
  %3 = mhlo.multiply %arg0, %2 : tensor<?x8x1024xf32>
  return %3 : tensor<?x8x1024xf32>
}

// -----// IR Dump After AddFakeSymbolicShape (add-fake-symbolic-shape) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32, #tf_type.shape<137x8x1024>> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32, #tf_type.shape<137x8x1024>> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<5.000000e-01> : tensor<f32, #tf_type.shape<>>
  %1 = shape.shape_of %arg0 : tensor<?x8x1024xf32, #tf_type.shape<137x8x1024>> -> tensor<3xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32, #tf_type.shape<>>, tensor<3xindex>) -> tensor<?x8x1024xf32, #tf_type.shape<137x8x1024>>
  %3 = mhlo.multiply %arg0, %2 : tensor<?x8x1024xf32, #tf_type.shape<137x8x1024>>
  return %3 : tensor<?x8x1024xf32, #tf_type.shape<137x8x1024>>
}

// -----// IR Dump After DelFakeSymbolicShape (del-fake-symbolic-shape) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %0 = mhlo.constant dense<5.000000e-01> : tensor<f32>
  %1 = shape.shape_of %arg0 : tensor<?x8x1024xf32> -> tensor<3xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<3xindex>) -> tensor<?x8x1024xf32>
  %3 = mhlo.multiply %arg0, %2 : tensor<?x8x1024xf32>
  return %3 : tensor<?x8x1024xf32>
}

// -----// IR Dump After HloLegalizeToLinalgPass (hlo-legalize-to-linalg) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant dense<5.000000e-01> : tensor<f32>
  %0 = shape.shape_of %arg0 : tensor<?x8x1024xf32> -> tensor<3xindex>
  %c0 = arith.constant 0 : index
  %extracted = tensor.extract %0[%c0] : tensor<3xindex>
  %1 = tensor.empty(%extracted) : tensor<?x8x1024xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%1 : tensor<?x8x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<?x8x1024xf32>
  %3 = shape.shape_of %arg0 : tensor<?x8x1024xf32> -> tensor<3xindex>
  %c0_0 = arith.constant 0 : index
  %extracted_1 = tensor.extract %3[%c0_0] : tensor<3xindex>
  %4 = tensor.empty(%extracted_1) : tensor<?x8x1024xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %2 : tensor<?x8x1024xf32>, tensor<?x8x1024xf32>) outs(%4 : tensor<?x8x1024xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %6 = arith.mulf %in, %in_2 : f32
    linalg.yield %6 : f32
  } -> tensor<?x8x1024xf32>
  return %5 : tensor<?x8x1024xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
#map = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<5.000000e-01> : tensor<f32>
    %dim = tensor.dim %arg0, %c0 : tensor<?x8x1024xf32>
    %0 = tensor.empty(%dim) : tensor<?x8x1024xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<?x8x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x8x1024xf32>
    %dim_0 = tensor.dim %arg0, %c0 : tensor<?x8x1024xf32>
    %2 = tensor.empty(%dim_0) : tensor<?x8x1024xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %1 : tensor<?x8x1024xf32>, tensor<?x8x1024xf32>) outs(%2 : tensor<?x8x1024xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %4 = arith.mulf %in, %in_1 : f32
      linalg.yield %4 : f32
    } -> tensor<?x8x1024xf32>
    return %3 : tensor<?x8x1024xf32>
  }
}


// -----// IR Dump After CSE (cse) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<5.000000e-01> : tensor<f32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x8x1024xf32>
  %0 = tensor.empty(%dim) : tensor<?x8x1024xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<?x8x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<?x8x1024xf32>
  %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %1 : tensor<?x8x1024xf32>, tensor<?x8x1024xf32>) outs(%0 : tensor<?x8x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    linalg.yield %3 : f32
  } -> tensor<?x8x1024xf32>
  return %2 : tensor<?x8x1024xf32>
}

// -----// IR Dump After CustomLinalgElementwiseOpFusion (custom-linalg-cwise-fusion) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<5.000000e-01> : tensor<f32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x8x1024xf32>
  %0 = tensor.empty(%dim) : tensor<?x8x1024xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst : tensor<?x8x1024xf32>, tensor<f32>) outs(%0 : tensor<?x8x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.mulf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<?x8x1024xf32>
  return %1 : tensor<?x8x1024xf32>
}

// -----// IR Dump After EmptyTensorToAllocTensor (empty-tensor-to-alloc-tensor) //----- //
func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<5.000000e-01> : tensor<f32>
  %dim = tensor.dim %arg0, %c0 : tensor<?x8x1024xf32>
  %0 = bufferization.alloc_tensor(%dim) : tensor<?x8x1024xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst : tensor<?x8x1024xf32>, tensor<f32>) outs(%0 : tensor<?x8x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.mulf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<?x8x1024xf32>
  return %1 : tensor<?x8x1024xf32>
}

// -----// IR Dump After CustomComputeOpAndFuncBufferizePass (custom-computeop-and-func-bufferize) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> ()>
module {
  func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
    %0 = bufferization.to_tensor %arg0 : memref<?x8x1024xf32>
    %1 = bufferization.to_memref %0 : memref<?x8x1024xf32>
    %2 = bufferization.to_memref %0 : memref<?x8x1024xf32>
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<5.000000e-01> : tensor<f32>
    %3 = bufferization.to_memref %cst : memref<f32>
    %dim = memref.dim %2, %c0 : memref<?x8x1024xf32>
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
    %4 = bufferization.to_tensor %alloc : memref<?x8x1024xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %3 : memref<?x8x1024xf32>, memref<f32>) outs(%alloc : memref<?x8x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %7 = arith.mulf %in, %in_0 : f32
      linalg.yield %7 : f32
    }
    %5 = bufferization.to_tensor %alloc : memref<?x8x1024xf32>
    %6 = bufferization.to_memref %5 : memref<?x8x1024xf32>
    return %6 : memref<?x8x1024xf32>
  }
}


// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant dense<5.000000e-01> : tensor<f32>
  %c0 = arith.constant 0 : index
  %0 = bufferization.to_memref %cst : memref<f32>
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %0 : memref<?x8x1024xf32>, memref<f32>) outs(%alloc : memref<?x8x1024xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    linalg.yield %1 : f32
  }
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After LinalgLowerToParallelLoops (convert-linalg-to-parallel-loops) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %cst = arith.constant dense<5.000000e-01> : tensor<f32>
  %0 = bufferization.to_memref %cst : memref<f32>
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  %dim_0 = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  scf.parallel (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%dim_0, %c8, %c1024) step (%c1, %c1, %c1) {
    %1 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<?x8x1024xf32>
    %2 = memref.load %0[] : memref<f32>
    %3 = arith.mulf %1, %2 : f32
    memref.store %3, %alloc[%arg1, %arg2, %arg3] : memref<?x8x1024xf32>
    scf.yield
  }
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  %dim_0 = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  scf.parallel (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%dim_0, %c8, %c1024) step (%c1, %c1, %c1) {
    %0 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<?x8x1024xf32>
    %1 = arith.mulf %0, %cst : f32
    memref.store %1, %alloc[%arg1, %arg2, %arg3] : memref<?x8x1024xf32>
    scf.yield
  }
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After CSE (cse) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  scf.parallel (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%dim, %c8, %c1024) step (%c1, %c1, %c1) {
    %0 = memref.load %arg0[%arg1, %arg2, %arg3] : memref<?x8x1024xf32>
    %1 = arith.mulf %0, %cst : f32
    memref.store %1, %alloc[%arg1, %arg2, %arg3] : memref<?x8x1024xf32>
    scf.yield
  }
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After CollapseParallelLoopsTo1DPass (collapse-parallel-loops-to-1d) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  %c0_0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %c1_2 = arith.constant 1 : index
  %0 = arith.muli %c1_2, %dim : index
  %1 = arith.muli %0, %c8 : index
  %2 = arith.muli %1, %c1024 : index
  scf.parallel (%arg1) = (%c0_0) to (%2) step (%c1_1) {
    %3 = arith.remsi %arg1, %c1024 : index
    %4 = arith.divsi %arg1, %c1024 : index
    %5 = arith.remsi %4, %c8 : index
    %6 = arith.divsi %4, %c8 : index
    %7 = memref.load %arg0[%6, %5, %3] : memref<?x8x1024xf32>
    %8 = arith.mulf %7, %cst : f32
    memref.store %8, %alloc[%6, %5, %3] : memref<?x8x1024xf32>
    scf.yield
  }
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %cst = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  %0 = arith.muli %dim, %c8 : index
  %1 = arith.muli %0, %c1024 : index
  scf.parallel (%arg1) = (%c0) to (%1) step (%c1) {
    %2 = arith.remsi %arg1, %c1024 : index
    %3 = arith.divsi %arg1, %c1024 : index
    %4 = arith.remsi %3, %c8 : index
    %5 = arith.divsi %3, %c8 : index
    %6 = memref.load %arg0[%5, %4, %2] : memref<?x8x1024xf32>
    %7 = arith.mulf %6, %cst : f32
    memref.store %7, %alloc[%5, %4, %2] : memref<?x8x1024xf32>
    scf.yield
  }
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After TileLoopsPass (tile-loops) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %cst = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  %0 = arith.muli %dim, %c8 : index
  %1 = arith.muli %0, %c1024 : index
  scf.parallel (%arg1) = (%c0) to (%1) step (%c512) {
    %2 = affine.min affine_map<(d0, d1, d2) -> (512, d1 - d2)>(%c512, %1, %arg1)
    scf.parallel (%arg2) = (%c0) to (%2) step (%c1) {
      %3 = arith.addi %arg2, %arg1 : index
      %4 = arith.remsi %3, %c1024 : index
      %5 = arith.divsi %3, %c1024 : index
      %6 = arith.remsi %5, %c8 : index
      %7 = arith.divsi %5, %c8 : index
      %8 = memref.load %arg0[%7, %6, %4] : memref<?x8x1024xf32>
      %9 = arith.mulf %8, %cst : f32
      memref.store %9, %alloc[%7, %6, %4] : memref<?x8x1024xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %c512 = arith.constant 512 : index
  %cst = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  %0 = arith.muli %dim, %c8 : index
  %1 = arith.muli %0, %c1024 : index
  scf.parallel (%arg1) = (%c0) to (%1) step (%c512) {
    %2 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%arg1)[%1]
    scf.parallel (%arg2) = (%c0) to (%2) step (%c1) {
      %3 = arith.addi %arg2, %arg1 : index
      %4 = arith.remsi %3, %c1024 : index
      %5 = arith.divsi %3, %c1024 : index
      %6 = arith.remsi %5, %c8 : index
      %7 = arith.divsi %5, %c8 : index
      %8 = memref.load %arg0[%7, %6, %4] : memref<?x8x1024xf32>
      %9 = arith.mulf %8, %cst : f32
      memref.store %9, %alloc[%7, %6, %4] : memref<?x8x1024xf32>
      scf.yield
    }
    scf.yield
  }
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After MergeSCFPass (merge-scf) //----- //
func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %c1536 = arith.constant 1536 : index
  %c2048 = arith.constant 2048 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %cst = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
  %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
  %0 = arith.muli %dim, %c8 : index
  %1 = arith.muli %0, %c1024 : index
  %2 = arith.divsi %1, %c2048 : index
  %3 = arith.ceildivsi %1, %c2048 : index
  scf.parallel (%arg1) = (%c0) to (%3) step (%c1) {
    %4 = arith.muli %arg1, %c2048 : index
    %5 = arith.cmpi slt, %arg1, %2 : index
    scf.if %5 {
      scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
        %6 = arith.addi %arg2, %c512 : index
        %7 = arith.addi %arg2, %c1024 : index
        %8 = arith.addi %arg2, %c1536 : index
        %9 = arith.addi %arg2, %4 : index
        %10 = arith.addi %6, %4 : index
        %11 = arith.addi %7, %4 : index
        %12 = arith.addi %8, %4 : index
        %13 = arith.remsi %9, %c1024 : index
        %14 = arith.remsi %10, %c1024 : index
        %15 = arith.remsi %11, %c1024 : index
        %16 = arith.remsi %12, %c1024 : index
        %17 = arith.divsi %9, %c1024 : index
        %18 = arith.divsi %10, %c1024 : index
        %19 = arith.divsi %11, %c1024 : index
        %20 = arith.divsi %12, %c1024 : index
        %21 = arith.remsi %17, %c8 : index
        %22 = arith.remsi %18, %c8 : index
        %23 = arith.remsi %19, %c8 : index
        %24 = arith.remsi %20, %c8 : index
        %25 = arith.divsi %17, %c8 : index
        %26 = arith.divsi %18, %c8 : index
        %27 = arith.divsi %19, %c8 : index
        %28 = arith.divsi %20, %c8 : index
        %29 = memref.load %arg0[%25, %21, %13] : memref<?x8x1024xf32>
        %30 = memref.load %arg0[%26, %22, %14] : memref<?x8x1024xf32>
        %31 = memref.load %arg0[%27, %23, %15] : memref<?x8x1024xf32>
        %32 = memref.load %arg0[%28, %24, %16] : memref<?x8x1024xf32>
        %33 = arith.mulf %29, %cst : f32
        %34 = arith.mulf %30, %cst : f32
        %35 = arith.mulf %31, %cst : f32
        %36 = arith.mulf %32, %cst : f32
        memref.store %33, %alloc[%25, %21, %13] : memref<?x8x1024xf32>
        memref.store %34, %alloc[%26, %22, %14] : memref<?x8x1024xf32>
        memref.store %35, %alloc[%27, %23, %15] : memref<?x8x1024xf32>
        memref.store %36, %alloc[%28, %24, %16] : memref<?x8x1024xf32>
        scf.yield
      } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
    } else {
      %6 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%4)[%1]
      scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
        %13 = arith.cmpi slt, %arg2, %6 : index
        scf.if %13 {
          %14 = arith.addi %arg2, %4 : index
          %15 = arith.remsi %14, %c1024 : index
          %16 = arith.divsi %14, %c1024 : index
          %17 = arith.remsi %16, %c8 : index
          %18 = arith.divsi %16, %c8 : index
          %19 = memref.load %arg0[%18, %17, %15] : memref<?x8x1024xf32>
          %20 = arith.mulf %19, %cst : f32
          memref.store %20, %alloc[%18, %17, %15] : memref<?x8x1024xf32>
        }
        scf.yield
      } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
      %7 = arith.addi %4, %c512 : index
      %8 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%7)[%1]
      scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
        %13 = arith.addi %arg2, %c512 : index
        %14 = arith.cmpi slt, %arg2, %8 : index
        scf.if %14 {
          %15 = arith.addi %13, %4 : index
          %16 = arith.remsi %15, %c1024 : index
          %17 = arith.divsi %15, %c1024 : index
          %18 = arith.remsi %17, %c8 : index
          %19 = arith.divsi %17, %c8 : index
          %20 = memref.load %arg0[%19, %18, %16] : memref<?x8x1024xf32>
          %21 = arith.mulf %20, %cst : f32
          memref.store %21, %alloc[%19, %18, %16] : memref<?x8x1024xf32>
        }
        scf.yield
      } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
      %9 = arith.addi %4, %c1024 : index
      %10 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%9)[%1]
      scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
        %13 = arith.addi %arg2, %c1024 : index
        %14 = arith.cmpi slt, %arg2, %10 : index
        scf.if %14 {
          %15 = arith.addi %13, %4 : index
          %16 = arith.remsi %15, %c1024 : index
          %17 = arith.divsi %15, %c1024 : index
          %18 = arith.remsi %17, %c8 : index
          %19 = arith.divsi %17, %c8 : index
          %20 = memref.load %arg0[%19, %18, %16] : memref<?x8x1024xf32>
          %21 = arith.mulf %20, %cst : f32
          memref.store %21, %alloc[%19, %18, %16] : memref<?x8x1024xf32>
        }
        scf.yield
      } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
      %11 = arith.addi %4, %c1536 : index
      %12 = affine.min affine_map<(d0)[s0] -> (-d0 + s0, 512)>(%11)[%1]
      scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
        %13 = arith.addi %arg2, %c1536 : index
        %14 = arith.cmpi slt, %arg2, %12 : index
        scf.if %14 {
          %15 = arith.addi %13, %4 : index
          %16 = arith.remsi %15, %c1024 : index
          %17 = arith.divsi %15, %c1024 : index
          %18 = arith.remsi %17, %c8 : index
          %19 = arith.divsi %17, %c8 : index
          %20 = memref.load %arg0[%19, %18, %16] : memref<?x8x1024xf32>
          %21 = arith.mulf %20, %cst : f32
          memref.store %21, %alloc[%19, %18, %16] : memref<?x8x1024xf32>
        }
        scf.yield
      } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
    }
    scf.yield
  } {_sony_af_cwise_unroll = 512 : si32, _sony_af_gpu_processor = 0 : i64, mapping}
  return %alloc : memref<?x8x1024xf32>
}

// -----// IR Dump After FuseKernelLaunchPass (fuse-kernel-launch) //----- //
#map = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
module {
  func.func @predict_online_5603970_0(%arg0: memref<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "2:0", input.has_one_use = true}) -> memref<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_kernel_size = 1 : i64, _sony_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
    %c1536 = arith.constant 1536 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %cst = arith.constant 5.000000e-01 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1024 = arith.constant 1024 : index
    %dim = memref.dim %arg0, %c0 : memref<?x8x1024xf32>
    %alloc = memref.alloc(%dim) {alignment = 128 : i64} : memref<?x8x1024xf32>
    %0 = arith.muli %dim, %c8 : index
    %1 = arith.muli %0, %c1024 : index
    %2 = arith.divsi %1, %c2048 : index
    %3 = arith.ceildivsi %1, %c2048 : index
    scf.parallel (%arg1) = (%c0) to (%3) step (%c1) {
      %4 = arith.muli %arg1, %c2048 : index
      %5 = arith.cmpi slt, %arg1, %2 : index
      scf.if %5 {
        scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
          %6 = arith.addi %arg2, %c512 : index
          %7 = arith.addi %arg2, %c1024 : index
          %8 = arith.addi %arg2, %c1536 : index
          %9 = arith.addi %arg2, %4 : index
          %10 = arith.addi %6, %4 : index
          %11 = arith.addi %7, %4 : index
          %12 = arith.addi %8, %4 : index
          %13 = arith.remsi %9, %c1024 : index
          %14 = arith.remsi %10, %c1024 : index
          %15 = arith.remsi %11, %c1024 : index
          %16 = arith.remsi %12, %c1024 : index
          %17 = arith.divsi %9, %c1024 : index
          %18 = arith.divsi %10, %c1024 : index
          %19 = arith.divsi %11, %c1024 : index
          %20 = arith.divsi %12, %c1024 : index
          %21 = arith.remsi %17, %c8 : index
          %22 = arith.remsi %18, %c8 : index
          %23 = arith.remsi %19, %c8 : index
          %24 = arith.remsi %20, %c8 : index
          %25 = arith.divsi %17, %c8 : index
          %26 = arith.divsi %18, %c8 : index
          %27 = arith.divsi %19, %c8 : index
          %28 = arith.divsi %20, %c8 : index
          %29 = memref.load %arg0[%25, %21, %13] : memref<?x8x1024xf32>
          %30 = memref.load %arg0[%26, %22, %14] : memref<?x8x1024xf32>
          %31 = memref.load %arg0[%27, %23, %15] : memref<?x8x1024xf32>
          %32 = memref.load %arg0[%28, %24, %16] : memref<?x8x1024xf32>
          %33 = arith.mulf %29, %cst : f32
          %34 = arith.mulf %30, %cst : f32
          %35 = arith.mulf %31, %cst : f32
          %36 = arith.mulf %32, %cst : f32
          memref.store %33, %alloc[%25, %21, %13] : memref<?x8x1024xf32>
          memref.store %34, %alloc[%26, %22, %14] : memref<?x8x1024xf32>
          memref.store %35, %alloc[%27, %23, %15] : memref<?x8x1024xf32>
          memref.store %36, %alloc[%28, %24, %16] : memref<?x8x1024xf32>
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
      } else {
        %6 = affine.min #map(%4)[%1]
        scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
          %13 = arith.cmpi slt, %arg2, %6 : index
          scf.if %13 {
            %14 = arith.addi %arg2, %4 : index
            %15 = arith.remsi %14, %c1024 : index
            %16 = arith.divsi %14, %c1024 : index
            %17 = arith.remsi %16, %c8 : index
            %18 = arith.divsi %16, %c8 : index
            %19 = memref.load %arg0[%18, %17, %15] : memref<?x8x1024xf32>
            %20 = arith.mulf %19, %cst : f32
            memref.store %20, %alloc[%18, %17, %15] : memref<?x8x1024xf32>
          }
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
        %7 = arith.addi %4, %c512 : index
        %8 = affine.min #map(%7)[%1]
        scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
          %13 = arith.addi %arg2, %c512 : index
          %14 = arith.cmpi slt, %arg2, %8 : index
          scf.if %14 {
            %15 = arith.addi %13, %4 : index
            %16 = arith.remsi %15, %c1024 : index
            %17 = arith.divsi %15, %c1024 : index
            %18 = arith.remsi %17, %c8 : index
            %19 = arith.divsi %17, %c8 : index
            %20 = memref.load %arg0[%19, %18, %16] : memref<?x8x1024xf32>
            %21 = arith.mulf %20, %cst : f32
            memref.store %21, %alloc[%19, %18, %16] : memref<?x8x1024xf32>
          }
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
        %9 = arith.addi %4, %c1024 : index
        %10 = affine.min #map(%9)[%1]
        scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
          %13 = arith.addi %arg2, %c1024 : index
          %14 = arith.cmpi slt, %arg2, %10 : index
          scf.if %14 {
            %15 = arith.addi %13, %4 : index
            %16 = arith.remsi %15, %c1024 : index
            %17 = arith.divsi %15, %c1024 : index
            %18 = arith.remsi %17, %c8 : index
            %19 = arith.divsi %17, %c8 : index
            %20 = memref.load %arg0[%19, %18, %16] : memref<?x8x1024xf32>
            %21 = arith.mulf %20, %cst : f32
            memref.store %21, %alloc[%19, %18, %16] : memref<?x8x1024xf32>
          }
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
        %11 = arith.addi %4, %c1536 : index
        %12 = affine.min #map(%11)[%1]
        scf.parallel (%arg2) = (%c0) to (%c512) step (%c1) {
          %13 = arith.addi %arg2, %c1536 : index
          %14 = arith.cmpi slt, %arg2, %12 : index
          scf.if %14 {
            %15 = arith.addi %13, %4 : index
            %16 = arith.remsi %15, %c1024 : index
            %17 = arith.divsi %15, %c1024 : index
            %18 = arith.remsi %17, %c8 : index
            %19 = arith.divsi %17, %c8 : index
            %20 = memref.load %arg0[%19, %18, %16] : memref<?x8x1024xf32>
            %21 = arith.mulf %20, %cst : f32
            memref.store %21, %alloc[%19, %18, %16] : memref<?x8x1024xf32>
          }
          scf.yield
        } {_sony_af_cwise_unroll, _sony_af_gpu_processor = 3 : i64, mapping}
      }
      scf.yield
    } {_sony_af_cwise_unroll = 512 : si32, _sony_af_gpu_processor = 0 : i64, mapping}
    return %alloc : memref<?x8x1024xf32>
  }
}


