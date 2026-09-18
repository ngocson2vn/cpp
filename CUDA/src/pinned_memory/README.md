# Pinned Memory
```cpp
// Allocate pinned host memory with cudaMallocHost()
void* host_ptr;
cudaMallocHost(&host_ptr, bytes);

// H2D using cudaMemcpy as usually
void* dev_ptr;
cudaMalloc(&dev_ptr, bytes);
cudaMemcpy(dev_ptr, host_ptr, bytes, cudaMemcpyHostToDevice);
```
