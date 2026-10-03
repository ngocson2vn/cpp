#loc = loc(unknown)
#loc1 = loc("Placeholder:")
#loc2 = loc("Placeholder")
#loc3 = loc("Mul:")
#loc4 = loc("Mul")
#loc5 = loc("Identity:")
#loc6 = loc("Output_1")
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 716 : i32}} {
  func.func @main() -> tensor<?x8x1024xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "Output_1:0"}} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island {
        %1 = "tf.Placeholder"() {_symbolic_output_shapes = [#tf_type.shape<137x8x1024>], device = "", shape = #tf_type.shape<?x8x1024>} : () -> tensor<?x8x1024xf32> loc(#loc7)
        tf_executor.yield %1 : tensor<?x8x1024xf32> loc(#loc7)
      } {_sony_af_op_idx = "2"} loc(#loc7)
      %outputs_0, %control_1 = tf_executor.island {
        %1 = "tf.FusedCwise"(%outputs) {_symbolic_output_shapes = [#tf_type.shape<137x8x1024>], metadata = "predict_online_5603970_0"} : (tensor<?x8x1024xf32>) -> tensor<?x8x1024xf32> loc(#loc8)
        tf_executor.yield %1 : tensor<?x8x1024xf32> loc(#loc8)
      } {_sony_af_group_idx = 0 : i64, _sony_af_op_idx = "3"} loc(#loc8)
      %outputs_2, %control_3 = tf_executor.island {
        %1 = "tf.Identity"(%outputs_0) {_symbolic_output_shapes = [#tf_type.shape<137x8x1024>], device = ""} : (tensor<?x8x1024xf32>) -> tensor<?x8x1024xf32> loc(#loc9)
        tf_executor.yield %1 : tensor<?x8x1024xf32> loc(#loc9)
      } {_sony_af_op_idx = "4"} loc(#loc9)
      tf_executor.fetch %outputs_2 : tensor<?x8x1024xf32> {_sony_af_op_idx = "5"} loc(#loc)
    } loc(#loc)
    return %0 : tensor<?x8x1024xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc7 = loc(fused["Placeholder:", "Placeholder"])
#loc8 = loc(fused["Mul:", "Mul"])
#loc9 = loc(fused["Identity:", "Output_1"])

