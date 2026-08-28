# Common Issues
## AttributeError: '_OpNamespace' 'fbgemm' object has no attribute 'asynchronous_complete_cumsum'
```python
AttributeError: '_OpNamespace' 'fbgemm' object has no attribute 'asynchronous_complete_cumsum'
```
**Solution**:
```bash
# Install fbgemm_gpu==1.2.0 
pip3 install fbgemm_gpu==1.2.0

# Load libgomp.so.1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
```


## torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 14.06 GiB. GPU 0 has a total capacity of 83.05 GiB of which 11.15 GiB is free. 
Process 1056 has 24.71 GiB memory in use. 
Process 1061 has 25.33 GiB memory in use. 

Including non-PyTorch memory, this process has 21.79 GiB memory in use. Of the allocated memory 21.17 GiB is allocated by PyTorch, and 20.22 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
