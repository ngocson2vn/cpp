# Grappler 
## 1. Constant Folding
Computes sub-graphs containing purely static operations during compile time. 
This replaces nodes like `1 + 2` directly with a constant `3` node to skip math execution at runtime.


## 2. Arithmetic Optimizer
### Removing Redundant Operations (Identity Nodes)

The most basic function of the arithmetic optimizer is catching mathematically pointless operations that can sneak into large models, especially during automated gradient calculations or complex broadcasting.

* **Addition of Zero:** $x + 0 \rightarrow x$
* **Multiplication by One:** $x \times 1 \rightarrow x$
* **Logical Double Negation:** `Not(Not(x))` $\rightarrow x$

### Common Subexpression Elimination (CSE)
Imagine you define a calculation where the exact same tensor addition `(a + b)` is used in two separate places:
```python
import tensorflow as tf

# Define two input tensors
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

# Common subexpression (a + b) appears twice
x = (a + b) * 2.0
y = (a + b) + 5.0
```
**Before Optimization:** <br/>
The graph builds two distinct addition nodes—one to compute `(a + b)` for `x`, and a separate identical addition node to compute `(a + b)` for `y`. [1] (https://github.com/samjabrahams/tensorflow-white-paper-notes)

**After CSE Optimization:** <br/>
TensorFlow scans the dataflow graph, detects that both addition nodes take the exact same inputs (`a` and `b`) with the same operation (`+`), and merges them. [1] (https://weishungchung.com/2018/07/16/tensorflow-data-flow-graph-optimization/), [2] (https://github.com/samjabrahams/tensorflow-white-paper-notes)

**The Result:** <br/>
The graph computes `t1 = a + b` just once, and passes the output of `t1` into both the multiplication by `2.0` and the addition of `5.0`.