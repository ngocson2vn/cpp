# MLIR-HLO
https://github.com/tensorflow/mlir-hlo

- A Standalone "HLO" MLIR-based Compiler
- This implements a self-contained compiler for a linear algebra set of operations inspired by XLA HLO IR using MLIR components.
- It is designed to provide an end-to-end flow independent of TensorFlow and XLA, but usable inside of these projects.
- MLIR-HLO aims to provide an end-to-end compiler for CPU and GPU, as well as building reusable blocks for other accelerators. This is heavily inspired by the success of XLA.
<br/><br/>

# Workflow
When TensorFlow integrates with MLIR-HLO, the workflow is as follows:

## 1. Graph Lowering to MLIR-TF Dialect
- TensorFlow translates its computation graph into the MLIR-TF dialect, a representation designed to capture TensorFlow-specific semantics in MLIR.
- This step ensures that all TensorFlow operations are represented in a form compatible with the MLIR infrastructure.

## 2. **MLIR-TF dialect** → **MLIR-HLO dialect**.

## 3. Optimizations are applied at the **HLO** level.

## 4. **HLO dialect** → Lower-level MLIR dialects (e.g., Linalg, GPU).

## 5. Code generation for the target hardware.

This approach leverages MLIR's infrastructure for flexibility and extensibility while ensuring high performance through hardware-specific optimizations. MLIR-HLO thus enables TensorFlow to scale across diverse platforms efficiently.

## Laniakea example
### 1. TensorFlow computation graph → **MLIR-TF dialect** file `graph.mlir`
Graph:
```pbtxt
node {
  name: "X1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "X2"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: -1
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "I1"
  op: "Add"
  input: "X1"
  input: "X2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Sigmoid"
  op: "Sigmoid"
  input: "I1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 134
}
```

is translated to **MLIR-TF dialect**
```MLIR
#loc = loc(unknown)
#loc1 = loc("Placeholder:")
#loc2 = loc("X1")
#loc3 = loc("X2")
#loc4 = loc("Add:")
#loc5 = loc("I1")
#loc6 = loc("Sigmoid:")
#loc7 = loc("Sigmoid")
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func.func @main() -> tensor<?x4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "Sigmoid:0"}} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Placeholder"() {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = "", shape = #tf_type.shape<?x4>} : () -> tensor<?x4xf32> loc(#loc8)
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Placeholder"() {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = "", shape = #tf_type.shape<?x4>} : () -> tensor<?x4xf32> loc(#loc9)
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Add"(%outputs, %outputs_0) {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = ""} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(#loc10)
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Sigmoid"(%outputs_2) {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32> loc(#loc11)
      tf_executor.fetch %outputs_4 : tensor<?x4xf32> loc(#loc)
    } loc(#loc)
    return %0 : tensor<?x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc8 = loc(fused["Placeholder:", "X1"])
#loc9 = loc(fused["Placeholder:", "X2"])
#loc10 = loc(fused["Add:", "I1"])
#loc11 = loc(fused["Sigmoid:", "Sigmoid"])
```

### 2. Clustering
Heavily relies on MLIR Pass Infrastructure https://mlir.llvm.org/docs/PassManagement/

`graph.mlir` -> Cluster Algorithms -> {`fused_graph.mlir`, `cluster.mlir`}

**fused_graph.mlir in MLIR-TF dialect:**
```MLIR
#loc = loc(unknown)
#loc1 = loc("Placeholder:")
#loc2 = loc("X1")
#loc3 = loc("X2")
#loc4 = loc("Sigmoid:")
#loc5 = loc("Sigmoid")
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 134 : i32}} {
  func.func @main() -> tensor<?x4xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "Sigmoid:0"}} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island {
        %1 = "tf.Placeholder"() {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = "", shape = #tf_type.shape<?x4>} : () -> tensor<?x4xf32> loc(#loc6)
        tf_executor.yield %1 : tensor<?x4xf32> loc(#loc6)
      } {_byted_af_op_idx = "1"} loc(#loc6)
      %outputs_0, %control_1 = tf_executor.island {
        %1 = "tf.Placeholder"() {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = "", shape = #tf_type.shape<?x4>} : () -> tensor<?x4xf32> loc(#loc7)
        tf_executor.yield %1 : tensor<?x4xf32> loc(#loc7)
      } {_byted_af_op_idx = "2"} loc(#loc7)
      %outputs_2, %control_3 = tf_executor.island {
        %1 = "tf.FusedCwise"(%outputs, %outputs_0) {_symbolic_output_shapes = [#tf_type.shape<3x4>], metadata = "predict_online_0"} : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(#loc8)
        tf_executor.yield %1 : tensor<?x4xf32> loc(#loc8)
      } {_byted_af_group_idx = 0 : i64, _byted_af_op_idx = "3"} loc(#loc8)
      tf_executor.fetch %outputs_2 : tensor<?x4xf32> {_byted_af_op_idx = "4"} loc(#loc)
    } loc(#loc)
    return %0 : tensor<?x4xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc6 = loc(fused["Placeholder:", "X1"])
#loc7 = loc(fused["Placeholder:", "X2"])
#loc8 = loc(fused["Sigmoid:", "Sigmoid"])
```

**cluster.mlir in plain MLIR**

https://mlir.llvm.org/docs/LangRef/
```MLIR
func.func @predict_online_0(%arg0: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<3x4>, input.from = "1:0", input.has_one_use = true} loc(unknown), %arg1: tensor<?x4xf32> {input.fake_symbolic_shape = #tf_type.shape<3x4>, input.from = "2:0", input.has_one_use = true} loc(unknown)) -> tensor<?x4xf32> attributes {SimpleFusion, _byted_af_group_idx = 0 : i64, _byted_af_op_idx = "3", llvm.emit_c_interface, tf_entry} {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["Add:", "I1"])
  %1 = "tf.Sigmoid"(%0) {_symbolic_output_shapes = [#tf_type.shape<3x4>], device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32> loc(fused["Sigmoid:", "Sigmoid"])
  return %1 : tensor<?x4xf32> loc(unknown)
} loc(unknown)
```

Question: How to lower from **MLIR-TF dialect** to this representation?
<br/><br/>




