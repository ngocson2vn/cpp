# Check whether a positive integer n is power of 2
```cpp
bool isPowerOfTwo(int n) {
  return (n > 0) && ((n & (n - 1)) == 0);
}
```
