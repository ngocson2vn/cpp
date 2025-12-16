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

# XOR
### XOR is associative
The bitwise XOR operation (often denoted as ⊕) is associative.<br/>
The equation $(a \oplus b) \oplus c = a \oplus (b \oplus c)$ holds for all integers $a$, $b$, and $c$.

Bitwise XOR operates independently on each bit of the binary representations of the numbers. For each corresponding bit position, the operation is equivalent to addition modulo 2 (i.e., 0 ⊕ 0 = 0, 0 ⊕ 1 = 1, 1 ⊕ 0 = 1, 1 ⊕ 1 = 0).

Addition modulo 2 is associative because $(x + y) \mod 2 + z \mod 2 = x + (y + z) \mod 2$ for any bits $x$, $y$, and $z$ (this follows from the associativity of addition in general).

Since this holds for every bit independently, the overall bitwise XOR operation is associative.

### Recall Modular Arithmetic Basics:
Modular arithmetic deals with remainders after division.<br/>
The expression $a \equiv b \pmod{m}$ means $a$ and $b$ leave the same remainder when divided by $m$, or equivalently, $a - b$ is divisible by $m$.

For addition, $(x + y) \mod m$ computes the sum first, then finds the remainder when divided by $m$.

### Apply to the Specific Case:
Compute the sum: $1 + 1 = 2$.<br/>
Divide by the modulus (2) and find the remainder: $2 \div 2 = 1$ with remainder 0 (since 2 is exactly divisible by 2).<br/>
Thus, $1 + 1 \equiv 0 \pmod{2}$, often written shorthand as "1 + 1 = 0 (mod 2)".