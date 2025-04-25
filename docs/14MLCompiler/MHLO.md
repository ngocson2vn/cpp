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




---
# **📌 TensorFlow to MLIR-HLO Optimization Pipeline with XLA HLO Understanding**
Here's an **updated full optimization pipeline diagram** for TensorFlow models using **MLIR-HLO**, now explicitly showing how **XLA understands MHLO** and performs **MHLO → XLA HLO conversion** before final optimizations.  

---
```
+--------------------------------------------------+
|   TensorFlow Model (SavedModel, Keras, Function)|
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   Convert to MLIR-TF Dialect                    |  (MLIR Frontend)
|   - Converts TensorFlow graph to MLIR-TF        |
|   - Basic optimizations (constant folding, etc.)|
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   Lower to MLIR-HLO (MHLO Dialect)              |  (MLIR Transformation)
|   - Converts TensorFlow ops into MHLO ops       |
|   - Simplifies MLIR for XLA compatibility       |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   **MHLO → XLA HLO Conversion**                 |  (XLA Frontend)
|   - LowerMHLOToHLO pass (MHLO → XLA HLO)        |
|   - Converts MLIR-friendly MHLO to XLA's format |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   XLA HLO Optimizations                         |  (XLA Compiler)
|   - Canonicalization (simplify expressions)     |
|   - Shape Inference (propagate tensor shapes)   |
|   - Constant Folding (precompute constants)     |
|   - Element-wise Fusion (merge simple ops)      |
|   - Algebraic Simplifications                   |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   Convert to StableHLO (Optional)               |  (For portability outside XLA)
|   - Used when targeting non-XLA backends        |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   Lower to MLIR Linalg / TOSA                   |  (MLIR Mid-Level Dialects)
|   - Linalg (For CPU/GPU)                        |
|   - TOSA  (For Mobile/Embedded Devices)         |
|   - Applies bufferization, tiling, etc.         |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   XLA Target-Specific Lowering                  |  (XLA Backend)
|   - Further optimizations for TPU/GPU/CPU       |
|   - Tiling & Vectorization                      |
|   - Parallelization & Bufferization             |
|   - Converts to platform-specific instructions  |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   Lower to LLVM IR / GPU Dialect                |  (MLIR Backend)
|   - LLVM IR for CPU execution                   |
|   - GPU Dialect for CUDA/ROCm execution         |
|   - Final hardware-specific codegen             |
+--------------------------------------------------+
```

---

### **🔹 Key Updates in This Diagram**
1️⃣ **Explicitly Shows the MHLO → XLA HLO Conversion**
   - **New stage added:** "MHLO → XLA HLO Conversion"  
   - **Uses `LowerMHLOToHLO` pass** to convert MHLO ops into XLA’s native HLO format.  
   - This is the bridge where **XLA compiler starts understanding MHLO**.  

2️⃣ **Separates XLA Frontend (HLO Conversion) from Backend (Optimizations & Codegen)**
   - XLA **first converts MHLO to its internal HLO format**.  
   - Then, XLA **optimizes HLO before hardware-specific lowering**.  

3️⃣ **Clarifies How MLIR-HLO Relates to XLA**
   - MLIR-HLO (MHLO) is a higher-level representation, **not directly executable by XLA**.  
   - Conversion to **XLA HLO makes it usable inside XLA’s compiler**.  

---

### **💡 Key Takeaways**
✅ **XLA does not execute MHLO directly—it first converts it to XLA HLO.**  
✅ **The `LowerMHLOToHLO` pass bridges MLIR-HLO with XLA.**  
✅ **Once converted, XLA applies optimizations before lowering to hardware code.**  
✅ **StableHLO enables portability for non-XLA backends (IREE, OneDNN, etc.).**  

### **🔹 How Does the MLIR Backend Understand the Output of the XLA Backend?**  

The **MLIR backend and XLA backend do not directly interact in most cases** because they serve **different roles in the compilation pipeline**. However, when **MLIR needs to process XLA-generated code**, it understands it through **lowering to MLIR-compatible formats like LLVM IR or GPU Dialect**.  

---

### **📌 Understanding the Flow: XLA Backend to MLIR Backend**
The **XLA backend generates target-specific IR** (e.g., LLVM IR, GPU code) instead of passing an intermediate representation back to MLIR. However, **if MLIR needs to process XLA's output**, it does so via **the following mechanisms**:

1️⃣ **XLA Backend Lowers to LLVM IR (for CPU targets)**
   - **XLA’s backend can generate LLVM IR**, which can be **processed by MLIR's LLVM Dialect**.
   - MLIR provides an **LLVM dialect (`mlir::LLVM`)** that can read and optimize LLVM IR.
   - **MLIR Backend can then perform additional optimizations** before final machine code generation.

2️⃣ **XLA Backend Lowers to GPU Dialects (for CUDA/ROCm)**
   - XLA’s backend can generate **GPU-specific operations**.
   - MLIR has a **GPU dialect** (`mlir::gpu`) that understands GPU-specific IR.
   - **This allows MLIR to further optimize CUDA or ROCm kernels** before execution.

3️⃣ **MLIR Uses StableHLO (Optional, for Non-XLA Backends)**
   - If targeting **IREE, OneDNN, or other MLIR-based runtimes**, **StableHLO can be used instead of XLA**.
   - StableHLO allows **porting XLA-optimized HLO graphs into other MLIR-based compilers**.

---

### **📌 Where Does This Fit in the Pipeline?**
```
+--------------------------------------------------+
|   XLA Target-Specific Lowering                  |  (XLA Backend)
|   - Converts XLA HLO to platform-specific code  |
|   - CPU (LLVM IR) / GPU (CUDA, ROCm) / TPU Ops  |
+--------------------------------------------------+
                          |
                          v
+--------------------------------------------------+
|   MLIR Backend (LLVM IR / GPU Dialect)          |  (MLIR Final Processing)
|   - Reads LLVM IR from XLA for CPU              |
|   - Reads GPU IR from XLA for CUDA/ROCm         |
|   - Final optimizations before machine code     |
+--------------------------------------------------+
```

---

### **🔹 Summary of How MLIR Understands XLA's Output**
| XLA Backend Output | MLIR Backend Handling |
|--------------------|----------------------|
| **LLVM IR (CPU)** | **MLIR LLVM Dialect (`mlir::LLVM`)** reads and optimizes it |
| **GPU IR (CUDA, ROCm)** | **MLIR GPU Dialect (`mlir::gpu`)** can optimize GPU kernels |
| **StableHLO (Optional)** | Allows further optimization in MLIR-based backends (e.g., IREE, OneDNN) |

---

### **💡 Key Takeaways**
✅ **XLA backend generates LLVM IR or GPU-specific code, which MLIR can process using the LLVM or GPU dialects.**  
✅ **MLIR’s LLVM Dialect can further optimize XLA-generated LLVM IR before final codegen.**  
✅ **StableHLO allows transferring optimized XLA HLO graphs to non-XLA MLIR runtimes.**  
✅ **MLIR and XLA are separate, but they integrate well through shared IR formats.**  

