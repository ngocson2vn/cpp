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

