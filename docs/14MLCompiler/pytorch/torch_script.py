"""
Python 3.11.10
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH
"""

import os
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.4/extras/CUPTI/lib64:" + os.environ["LD_LIBRARY_PATH"]

import torch

@torch.jit.script
def my_function(x):
    if x > 0:
        return x * 2
    else:
        return x * 3

print(my_function.graph)
