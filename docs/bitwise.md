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


# Division
For a non-negative integer n:
```MLIR
n >> k = floor(n / 2^k) = n / (2^k)
```

# Multiplication
For a non-negative integer n:
```MLIR
n << k = n * (2^k)
```
