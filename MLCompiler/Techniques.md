# Hoisting
Hoisting, often called **loop-invariant code motion**, is a compiler optimization that moves computations out of a loop body to a position earlier in the code (pre-header) if the expression's result does not change within the loop. This technique reduces execution time by computing values once instead of repeating the calculation in every iteration. 

Example:
```C++
// Original: 
for(...) { x = y + z; a[i] = x; }

// Hoisted:
temp = y + z; 
for(...) { a[i] = temp; }
```
