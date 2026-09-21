# Signed Integers and Two's Complement Representation
In PTX (and most modern computing systems), signed 32-bit integers (.s32) use two's complement to represent negative numbers. This is a binary system where:
- Positive numbers (including zero) have the most significant bit (MSB, bit 31 in a 32-bit integer) set to 0.
- Negative numbers have the MSB set to 1.
- The value of a negative number is calculated from its positive counterpart by inverting all bits (one's complement) of the positive counterpart and adding 1.

For example:
- A positive number like `+5` in binary: `0000 0000 0000 0000 0000 0000 0000 0101 (0x00000005)`.
- Its negative counterpart, `-5`: Invert bits to `1111 1111 1111 1111 1111 1111 1111 1010`, then add `1` = `1111 1111 1111 1111 1111 1111 1111 1011 (0xFFFFFFFB)`.
- `-1` in two's complement is all bits set to `1`: `1111 1111 1111 1111 1111 1111 1111 1111 (0xFFFFFFFF)`.

Key point: Any negative number starts with MSB = 1, and the more negative it is, the more 1s it tends to have from the left.


# Multiplication
For a non-negative integer n:
```MLIR
n << k = n * (2^k)
```


# Get quotient
For two positive integers i and n, where $n = 2^k$:
```MLIR
i / n = i >> k
```

# Get remainder
For two positive integers i and n, where $n = 2^k$:
```mlir
i % n = i & (n - 1)
```
We can always factorize i into two parts:
- Higher bits: k, k + 1, k + 2, ...
- Lower bits: 0, ..., k - 1

The higher bits part can be represented as:

$`b_k2^k + b_{k+1}2^{k+1} + b_{k+2}2^{k+2} + ... = 2^k(b_k + b_{k+1}2^1 + b_{k+2}2^2 + ... )`$

So the higher bits part is divisible by $2^k$.

The lower bits part has a maximum value of $2^k - 1$, which is not divisble by $2^k$.

Therefore, the remainder is exactly the lower bits part.
