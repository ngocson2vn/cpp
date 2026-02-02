# AMX
```Bash
g++ -mamx-tile -mamx-bf16 -mamx-int8
```
These `g++` options enable specific instruction sets for **Intel Advanced Matrix Extensions (AMX)**.

AMX is a dedicated accelerator block found on modern Intel server CPUs (starting with 4th Gen Xeon "Sapphire Rapids") designed to drastically speed up matrix multiplication for AI and Machine Learning workloads.

Unlike traditional vector extensions (like AVX-512) that operate on 1D vectors, AMX operates on **2D matrices called "tiles."**

Here is the detailed breakdown of each option:

### 1. `-mamx-tile`

* **Purpose:** Enables the foundational **Tile Architecture** instructions.
* **What it does:** This flag allows the compiler to generate instructions that manage the state of the AMX unit. Before you can perform any math, you must configure the hardware to understand the dimensions of the matrices you are working with.
* **Key Instructions Enabled:**
* `ldtilecfg` / `sttilecfg`: Load and store the tile configuration. This tells the CPU the number of rows and bytes-per-row for the matrices you are about to process.
* `tileloadd`: Load data from system memory into the high-speed internal tile registers (`tmm0` through `tmm7`).
* `tilestored`: Store results from tile registers back to system memory.
* `tilerelease`: Clears the AMX state and returns the hardware to a clean slate (essential for context switching).


* **Dependency:** This is the prerequisite for all other AMX operations. You cannot perform BF16 or INT8 math without first enabling the tile architecture.

### 2. `-mamx-bf16`

* **Purpose:** Enables matrix multiplication instructions for **BFloat16** (Brain Floating Point) data.
* **What it does:** BFloat16 is a floating-point format widely used in Deep Learning (AI) training. It keeps the same dynamic range as a standard 32-bit float but reduces precision to 16 bits to speed up calculation. This flag allows the CPU to perform dot-products on these 16-bit tiles.
* **Key Instruction Enabled:**
* `tdpbf16ps` (Tile Dot Product BFloat16 into Packed Single): This single instruction takes two tiles of BF16 numbers, multiplies them, and accumulates the result into a 32-bit floating-point (FP32) tile.


* **Use Case:** Accelerating AI training and inference where speed is critical and the slight precision loss of BF16 (vs FP32) is acceptable.

### 3. `-mamx-int8`

* **Purpose:** Enables matrix multiplication instructions for **8-bit Integers** (INT8).
* **What it does:** This is used for "Quantized" AI models. By converting neural network weights from floating-point to 8-bit integers, models run much faster and use less memory. This flag enables the hardware to multiply these 8-bit tiles.
* **Key Instructions Enabled:**
* `tdpbuud`: Unsigned  Unsigned.
* `tdpbsud`: Signed  Unsigned.
* `tdpbusd`: Unsigned  Signed.
* `tdpbssd`: Signed  Signed.
* *Note: All these instructions accumulate the result into a 32-bit Integer (INT32) tile to prevent overflow.*


* **Use Case:** High-performance AI inference (running a pre-trained model) where throughput is the highest priority.

---

### Summary Table

| Option | Full Name | Operates On | Output Format | Primary Use Case |
| --- | --- | --- | --- | --- |
| **`-mamx-tile`** | AMX Tile | Memory / Config | N/A | Managing data movement & setup |
| **`-mamx-bf16`** | AMX BFloat16 | `bfloat16` (16-bit) | `float` (32-bit) | AI Training & Inference |
| **`-mamx-int8`** | AMX Int8 | `int8` / `uint8` | `int32` | Quantized AI Inference |

### Important Implementation Notes

**1. Hardware Requirements**
You generally cannot use these flags on standard consumer CPUs (like Intel Core i9 or AMD Ryzen). They require **Intel Sapphire Rapids (4th Gen Xeon)** or newer server-class processors.

**2. Intrinsics vs. Auto-vectorization**
Simply adding these flags to `g++` will likely **not** make your standard C++ `for` loops magically use AMX. The compiler's auto-vectorizer rarely targets AMX automatically because the setup overhead (loading tiles) is high.
To use these instructions, developers typically write code using **C-style Intrinsics** defined in `<immintrin.h>`, such as:

* `_tile_loadconfig(...)`
* `_tile_dpbf16ps(...)`

**3. Enabling All at Once**
Instead of specifying them individually, you can often use `-march=sapphirerapids` (or your specific CPU architecture), which automatically enables all three flags if the target CPU supports them.

### Quick Verification Step

Would you like me to provide a small C++ code snippet showing how to use the `_tile_dpbf16ps` intrinsic to see how these flags are used in practice?