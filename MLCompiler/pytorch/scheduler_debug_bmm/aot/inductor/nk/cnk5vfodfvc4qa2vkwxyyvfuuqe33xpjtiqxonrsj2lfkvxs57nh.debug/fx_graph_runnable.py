
import os
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS'] = 'XTRITON'
os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '1'
os.environ['TORCH_LOGS'] = '+dynamo'
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = './scheduler_debug_bmm'
os.environ['DYNAMO_CKPT_PATH'] = './scheduler_debug_bmm/aot'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = './scheduler_debug_bmm/aot/inductor'
os.environ['INDUCTOR_POST_FUSION_SVG'] = '1'
os.environ['INDUCTOR_ORIG_FX_SVG'] = '1'
os.environ['TORCHINDUCTOR_PROLOGUE_FUSION'] = '1'
os.environ['TORCHINDUCTOR_DEBUG_FUSION'] = '1'
os.environ['TORCHDYNAMO_COMPILE_DYNAMIC_SHAPE'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.7.0a0+gitc88b85b.aml
# torch cuda version: 12.4
# torch git version: c88b85be511d3f8bce1109faf3697bce7abc4ee9


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Mar_28_02:18:24_PDT_2024 
# Cuda compilation tools, release 12.4, V12.4.131 
# Build cuda_12.4.r12.4/compiler.34097967_0 

# GPU Hardware Info: 
# NVIDIA A10 : 4 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        bmm = torch.ops.aten.bmm.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        sigmoid = torch.ops.aten.sigmoid.default(bmm);  bmm = None
        return (sigmoid,)
        
def load_args(reader):
    buf0 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf0, (3, 64, 64), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 23808, device=device(type='cuda', index=0))
    reader.tensor(buf1, (3, 31, 64), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)