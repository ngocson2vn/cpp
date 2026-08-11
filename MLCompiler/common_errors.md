# Common Errors
## CUDA_ERROR_ILLEGAL_ADDRESS
Given the following MLIR function:
```mlir
func.func @fused_ops_5603970_3349(%arg0: tensor<?x5xi32> {input.fake_symbolic_shape = #tf_type.shape<137x5>} loc(unknown), %arg1: tensor<?x1xi32> {input.fake_symbolic_shape = #tf_type.shape<137x1>} loc(unknown)) -> tensor<?x5xi32> attributes {llvm.emit_c_interface, tf_entry} {
  %cst = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<>], device = "", value = dense<1> : tensor<i32>} : () -> tensor<i32> loc(fused["Const:", "Sum_92/reduction_indices"])
  %cst_0 = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<1x4>], device = "", value = dense<[[0, 416, 544, 1568]]> : tensor<1x4xi32>} : () -> tensor<1x4xi32> loc(fused["Const:", "Const_434"])
  %cst_1 = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<2>], device = "", value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32> loc(fused["Const:", "bias_slice_339/begin"])
  %cst_2 = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<2>], device = "", value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32> loc(fused["Const:", "strided_slice_2/stack_2"])
  %cst_3 = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<2>], device = "", value = dense<[0, 4]> : tensor<2xi32>} : () -> tensor<2xi32> loc(fused["Const:", "strided_slice_124/stack_1"])
  %0 = "tf.StridedSlice"(%arg0, %cst_1, %cst_3, %cst_2) {_symbolic_output_shapes = [#tf_type.shape<137x4>], begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?x5xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xi32> loc(fused["StridedSlice:", "strided_slice_124"])
  %1 = "tf.AddV2"(%0, %cst_0) {_symbolic_output_shapes = [#tf_type.shape<137x4>], device = ""} : (tensor<?x4xi32>, tensor<1x4xi32>) -> tensor<?x4xi32> loc(fused["AddV2:", "add_369"])
  %2 = "tf.ConcatV2"(%arg1, %1, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x5>], device = ""} : (tensor<?x1xi32>, tensor<?x4xi32>, tensor<i32>) -> tensor<?x5xi32> loc(fused["ConcatV2:", "concat_145"])
  return %2 : tensor<?x5xi32> loc(unknown)
} loc(unknown)
```

Look at `%cst_0 = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<1x4>], device = "", value = dense<[[0, 416, 544, 1568]]> : tensor<1x4xi32>} : () -> tensor<1x4xi32> loc(fused["Const:", "Const_434"])`
Later, this const will be lowered to a CPU buffer which is then passed to a GPU kernel.
Therefore, this leads to CUDA_ERROR_ILLEGAL_ADDRESS

Fix: exclude the non-trivial const from the cluster
