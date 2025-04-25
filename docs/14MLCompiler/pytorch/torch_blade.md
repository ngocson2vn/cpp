# Torch Blade Workflow
GitHub repo: https://github.com/alibaba/BladeDISC/tree/main
<br/>

Given a PyTorch model: [test_torch_blade.py](./test_torch_blade.py)

# Frontend
```Python
import torch_blade
```
<br/>

**torch_blade/__init__.py**
```Python
try:
    import torch_blade.mlir  # noqa
except ImportError:
    pass
```
<br/>

**torch_blade/mlir/__init__.py**
```Python
from torch_blade.config import OptPipelines
from torch_blade.mlir.disc_engine_conversion import (  # noqa
    _compile_torchscript,  # noqa
    _optimize_mlir,
)

from .._torch_blade._mlir import *  # noqa

_DISC_NAME = backend_name()  # noqa
_is_available = True
_DISC_GROUP_NAME = _DISC_NAME.lower() + "_grp"
OptPipelines.register_pipeline(_DISC_NAME, _optimize_mlir)
```
_DISC_NAME = 'DISC'
<br/>


Entering Dynamo realm:<br/>
When `torch._dynamo.optimize(backend=compile_backend, dynamic=False)` is executed, the `compile_backend` will be invoked.


After some computations, the control will be transferred to torch_blade/optimization.py
```Python
# Get the registered _optimize_mlir function
optimizaiton_func = OptPipelines.pipelines[cfg.optimization_pipeline]

# Call _optimize_mlir function
optimizaiton_func(model, model_inputs)
```
<br/>

Split the FX graph into subgraphs (or clusters) of operations
<br/>

Compile subgraphs in parallel. The number of parallelisms is controlled by the env var `COMPILE_PARALLELISM`

**torch_blade/mlir/disc_engine_conversion.py**<br/>
For each subgraph, call function `try_cvt_to_disc_engine_func`
```Python
_compile_torchscript(subgraph, attr_name)
```
<br/>

Convert subgraph in TorchScript to MHLO:
```Python
mlir.cvt_torchscript_to_mhlo(graph)
```
This is a C API.
https://github.com/alibaba/BladeDISC/blob/58efe1a138cf32b11380710b51593642a0db18a4/pytorch_blade/pytorch_blade/compiler/mlir/pybind_functions.cpp#L26
<br/>


Next, lower MHLO to binary
```Python
mhlo_compile_cmd = os.path.join(pkg_path, "disc_compiler_main")

with open(compile_log, "w") as devnull:
    cfg = Config.get_current_context_or_new()
    cfg.disc_compile_for_multi_cuda_targets = tools.read_bool_from_env(
        "DISC_COMPILE_FOR_MULTI_CUDA_TARGETS",
        cfg.disc_compile_for_multi_cuda_targets,
    )
    env["TAO_MLIR_ENABLE_AMP"] = str(cfg.enable_mlir_amp).lower()
    env["DISC_CPU_FAST_MATH_LEVEL"] = str(cfg.disc_cpu_fast_math_level)
    # RUN: disc_compiler_main input_mlir_file.mlir output_file.so
    # redirect stdout to devnull
    extra_flags = []
    if cfg.disc_compile_for_multi_cuda_targets:
        extra_flags += ["--multi-cc-support"]
    subprocess.check_call(
        [
            mhlo_compile_cmd,
            inp_mlir_file.name,
            out_file_name,
            "--mlir-elide-elementsattrs-if-larger=8",
        ]
        + extra_flags,
        stdout=devnull,
        stderr=devnull,
        env=env,
    )
```

disc_compiler_main: https://github.com/alibaba/BladeDISC/blob/58efe1a138cf32b11380710b51593642a0db18a4/tao_compiler/mlir/disc/disc_compiler_main.cc
