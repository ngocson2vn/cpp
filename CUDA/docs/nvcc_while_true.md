# Pure Compiler Memory Barrier
Given the following CUDA kernel:
```c
__global__ void makeAllSMsActive(uint64_t N) {
  uint64_t prev = 0;
  uint64_t next = 1;
  uint64_t tmp = 0;
  while (true) {
    for (uint64_t i = 1; i < N; i++) {
      tmp = prev + next;
      prev = next;
      next = tmp;
    }
  }
}
```
nvcc will remove `while (true)` because nothing is ever written to global memory, shared memory, or any other location visible outside the thread.

## Fix 1
```c
__global__ void makeAllSMsActive(uint64_t N) {
  uint64_t tmp = 0;
  uint64_t prev = 0;
  volatile uint64_t next = 1;
  while (true) {
    for (uint64_t i = 1; i < N; i++) {
      tmp = prev + next;
      prev = next;
      next = tmp;
    }
  }
}
```
Making `next` volatile works because volatile accesses are defined as **observable side effects** under the C++ abstract machine.
`volatile` indicates that the value of the object may be changed at any time, outside the control of the current code (e.g., by hardware or another thread).
Every read or write of a volatile object must actually occur, Therefore, the compiler is not allowed to optimize them away just because it thinks "no one will look at this value".

## Fix 2
```c
__global__ void makeAllSMsActive(uint64_t N) {
  uint64_t prev = 0;
  uint64_t next = 1;
  uint64_t tmp = 0;
  while (true) {
    for (uint64_t i = 1; i < N; i++) {
      tmp = prev + next;
      prev = next;
      next = tmp;
    }

    // Force memory synchronization fence to stop compiler movement
    asm volatile("" : : : "memory");
  }
}
```
"asm volatile("""" ::: ""memory"")" is a compiler barrier only which creates a hard sequencing point for the compiler's view of memory. 
All preceding memory operations in program order must be "complete" (from the compiler's perspective) before any subsequent ones begin.
