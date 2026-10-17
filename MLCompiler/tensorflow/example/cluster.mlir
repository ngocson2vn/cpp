func.func @predict_online_5603970_0(%arg0: tensor<?x8x1024xf32> {input.fake_symbolic_shape = #tf_type.shape<137x8x1024>, input.from = "3:0", input.has_one_use = true} loc(unknown)) -> tensor<?x8x1024xf32> attributes {SimpleFusion, _sony_af_group_idx = 0 : i64, _sony_af_op_idx = "4", llvm.emit_c_interface, tf_entry} {
  %cst = "tf.Const"() {_symbolic_output_shapes = [#tf_type.shape<>], device = "", value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32> loc(fused["Const:", "Const"])
  %0 = "tf.Mul"(%arg0, %cst) {_symbolic_output_shapes = [#tf_type.shape<137x8x1024>], device = ""} : (tensor<?x8x1024xf32>, tensor<f32>) -> tensor<?x8x1024xf32> loc(fused["Mul:", "Mul"])
  %1 = "tf.Relu"(%0) : (tensor<?x8x1024xf32>) -> tensor<?x8x1024xf32> loc(fused["Maximum:", "Maximum"])
  return %1 : tensor<?x8x1024xf32> loc(unknown)
} loc(unknown)
