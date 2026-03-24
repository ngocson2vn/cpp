In C++17, the compiler skips calling IntegerType and TypeBase constructors, right?

You have hit the nail on the head regarding the *function calls*, but there is a small, crucial distinction to make between "calling a constructor function" and "initializing the object."

To answer directly: **Yes, it skips the *function calls* for the derived constructors, but it does *not* skip the initialization of those derived layers.**

Here is exactly what happens in C++17 and later when you use an inherited constructor:

### 1. The Function Call is Skipped
In terms of the actual runtime call stack, you are completely right. The compiler does not generate or call an `IntegerType(...)` function, nor does it generate or call a `TypeBase(...)` function. It routes the arguments directly into the `Type(TypeStorage*)` constructor. 

### 2. Initialization is NOT Skipped
Even though the intermediate constructor functions are bypassed, the compiler still has to ensure the whole object is built properly. If `IntegerType` or `TypeBase` had their own member variables, ignoring them would result in uninitialized memory. 

So, in C++17, when the compiler directly invokes the base `Type` constructor, it essentially bundles in the following steps into a single, highly optimized initialization sequence:
1.  **Construct Base:** Execute `Type(TypeStorage*)` using the provided arguments.
2.  **Initialize Intermediate:** Default-initialize any member variables belonging to `TypeBase`.
3.  **Initialize Derived:** Default-initialize any member variables belonging to `IntegerType`.

Because your specific classes (`TypeBase` and `IntegerType`) do not have any member variables, steps 2 and 3 effectively compile down to doing absolutely nothing. The compiler optimizes it away, leaving only the execution of the `Type` constructor. 

### Why This Distinction Matters
This C++17 rule is brilliant because it gives you the best of both worlds. You get the performance of skipping useless wrapper functions (no bloated call stack), but you still get the safety of C++ guaranteeing that all member variables in your derived classes are properly initialized.

