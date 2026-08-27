# FBGEMM_GPU
## Build fbgemm_gpu for sm_120
```bash
#========================================================
# git clone
#========================================================
mkdir FBGEMM
cd FBGEMM
git init
git remote add origin https://github.com/pytorch/FBGEMM.git

# Fetch https://github.com/pytorch/FBGEMM/commit/625f9ce745b26119975c18f4ff185615a3b98f67
# https://github.com/pytorch/FBGEMM/tree/v1.2.0
git fetch --depth 1 origin 625f9ce745b26119975c18f4ff185615a3b98f67
git checkout -b v1.2.0 FETCH_HEAD


#========================================================
# Build
#========================================================
cd FBGEMM/fbgemm_gpu
git submodule sync && git submodule update --init --recursive
pip install -r requirements.txt

export MAX_JOBS=`nproc`
export USE_NCCL=0
export USE_FLASH_ATTENTION=0
export TORCH_CUDA_ARCH_LIST="12.0" # or include 12.0/12.9 as appropriate for sm_120

# Fix a known GCC 12 + C++20 + -O3 false positive
export CFLAGS+=" -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"
export CXXFLAGS+=" -Wno-error=maybe-uninitialized -Wno-error=uninitialized -Wno-error=restrict"

python setup.py bdist_wheel -DFBGEMM_BUILD_TARGET=default
```

## Test fbgemm_gpu
```bash
# Load libgomp.so.1
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1
python
```

Call `fbgemm.asynchronous_complete_cumsum(x)`:
```python
import torch
import fbgemm_gpu

x = torch.randint(0, 100, (1024,), device="cuda", dtype=torch.int32)
res = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
torch.cuda.synchronize()
print(res)
```
