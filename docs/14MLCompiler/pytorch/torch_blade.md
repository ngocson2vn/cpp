# Torch Blade Workflow
GitHub repo: https://github.com/alibaba/BladeDISC/tree/main
<br/>

Given a PyTorch model: [test_torch_blade.py](./test_torch_blade.py)

## Frontend
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
```Python
self.forward = torch._dynamo.optimize(backend=compile_backend, dynamic=False)(self.forward)
```
When `self.forward()` is called, the control will be transferred to <br/>
**torch/_dynamo/eval_frame.py**
```Python
set_eval_frame = torch._C._dynamo.eval_frame.set_eval_frame  # noqa: F401
...
prior = set_eval_frame(callback)
...
try:
    return fn(*args, **kwargs)
finally:
    ...
```
where, `set_eval_frame` is a C function defined in `eval_frame.c` <br/>

**torch/csrc/dynamo/eval_frame.c**
```C++
static PyObject* set_eval_frame(PyObject* new_callback, PyThreadState* tstate) {
  // Change the eval frame callback and return the old one
  //  - None: disables TorchDynamo
  //  - False: run-only mode (reuse existing compiles)
  //  - Python callable(): enables TorchDynamo
  PyObject* old_callback = eval_frame_callback_get();

  // owned by caller
  Py_INCREF(old_callback);

  if (old_callback != Py_None && new_callback == Py_None) {
    decrement_working_threads(tstate);
  } else if (old_callback == Py_None && new_callback != Py_None) {
    increment_working_threads(tstate);
  }

  Py_INCREF(new_callback);
  Py_DECREF(old_callback);

  // Set thread local callback. This will drive behavior of our shim, if/when it
  // is installed.
  eval_frame_callback_set(new_callback);

  return old_callback;
}
```
where, `new_callback` is a wrapper of the backend compiler function.
<br/>

increment_working_threads() -> enable_eval_frame_shim() which changes the bahavior of Python interpreter:
```C++
inline static void enable_eval_frame_shim(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x03090000
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      &custom_eval_frame_shim) {
    DEBUG_CHECK(previous_eval_frame == NULL);
    previous_eval_frame = _PyInterpreterState_GetEvalFrameFunc(tstate->interp);
    _PyInterpreterState_SetEvalFrameFunc(tstate->interp,
                                         &custom_eval_frame_shim);
  }
#else
  if (tstate->interp->eval_frame != &custom_eval_frame_shim) {
    // First call
    tstate->interp->eval_frame = &custom_eval_frame_shim;
  }
#endif
}
```
<br/>

When `fn(*args, **kwargs)` is executed, <u>**Python interpreter** will make a call chain:</u><br/>
custom_eval_frame_shim() -> _custom_eval_frame_shim() -> _custom_eval_frame() which calls `new_callback` to evaluate the function's frame.
```C++
PyObject* result =
    call_callback(callback, frame, cache_entry, frame_state);
```
<br/>

The `new_callback` calls Dynamo APIs to trace the frame's bytecode using `InstructionTranslator`:
```Python
tracer = InstructionTranslator(
    instructions,
    code,
    locals,
    globals,
    builtins,
    code_options,
    compiler_fn,
    one_graph,
    export,
    export_constraints,
    mutated_closure_cell_contents,
    frame_state=frame_state,
    speculation_log=speculation_log,
...
tracer.run()
)
```

Internally, this trace process builds an OutputGraph:
```Python
output=OutputGraph(
    code_options,
    compiler_fn,
    self,
    export,
    export_constraints,
    frame_state,
    local_scope=f_locals,
    global_scope=f_globals,
    f_code=f_code,
),
```
<br/>

When it reaches RETURN_VALUE opcode, it calls `self.output.compile_subgraph() -> self.compile_and_call_fx_graph()` to compile OutputGraph's graph:
```Python
gm = _make_graph_module(root, self.graph)
...
with self.restore_global_state():
    compiled_fn = self.call_user_compiler(gm)
```
where `gm` is a GraphModule which has a `torch.fx.graph.Graph` graph object:
```Python
(Pdb) p gm
GraphModule()

(Pdb) p gm.graph
<torch.fx.graph.Graph object at 0x7efc339ee350>

(Pdb) print(gm.graph)
graph():
    %l_y_ : torch.Tensor [num_users=1] = placeholder[target=L_y_]
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    %res : [num_users=1] = call_function[target=torch.multiply](args = (%l_x_, %l_y_), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.nn.functional.relu](args = (%res,), kwargs = {inplace: False})
    %res_1 : [num_users=1] = call_function[target=operator.truediv](args = (%relu, 2), kwargs = {})
    %pred : [num_users=1] = call_function[target=torch.sum](args = (%res_1,), kwargs = {})
    return (pred,)
```
<br/>

Step into `self.call_user_compiler(gm)`:
```Python
compiled_fn = compiler_fn(gm, self.example_inputs())
```
`compiler_fn` is our backend compiler function `infer_compile()`:
```Python
(Pdb) p compiler_fn.__name__
'infer_compile'
```
The control is transferred to the registered backend compiler function `infer_compile()` in <br/>
torch_blade/dynamo/__init__.py
```Python
from torch._dynamo.backends.registry import register_backend
...
def infer_compile(fx_g: fx.GraphModule, inps, use_ts=False) -> Callable:
    return _disc_compile(fx_g, inps, use_ts=False, is_training=False)
...
register_backend(name="disc_infer", compiler_fn=infer_compile)
```
<br/>

**Convert GraphModule to TorchScript**:
```Python
f = torch.jit.script(fx_g)

(Pdb) type(f)
<class 'torch.jit._script.RecursiveScriptModule'>

(Pdb) type(f.graph)
<class 'torch.Graph'>

(Pdb) p f.graph
graph(%self : __torch__.torch.fx.graph_module._lambda,
      %arg0_1.1 : Tensor,
      %arg1_1.1 : Tensor):
  %17 : NoneType = prim::Constant()
  %13 : int = prim::Constant[value=2]()
  %mul.1 : Tensor = aten::mul(%arg1_1.1, %arg0_1.1)
  %relu.1 : Tensor = aten::relu(%mul.1)
  %div.1 : Tensor = aten::div(%relu.1, %13)
  %sum_1.1 : Tensor = aten::sum(%div.1, %17)
  %21 : (Tensor) = prim::TupleConstruct(%sum_1.1)
  return (%21)
```
<br/>

**Compile RecursiveScriptModule**:
```Python
f = torch_blade._static_optimize(f, True, tuple(inps))
```


torch_blade/optimization.py
```Python
# Get the registered _optimize_mlir function
optimizaiton_func = OptPipelines.pipelines[cfg.optimization_pipeline]

# Call _optimize_mlir function
optimizaiton_func(model, model_inputs)
```
<br/>

## Middle-end
Split the FX graph into subgraphs (or clusters) of operations
<br/>

**Compile subgraphs in parallel**<br/>
The number of parallelisms is controlled by the env var `COMPILE_PARALLELISM`
**torch_blade/mlir/disc_engine_conversion.py**<br/>
For each subgraph, call function `try_cvt_to_disc_engine_func`
```Python
_compile_torchscript(subgraph, attr_name)
```
<br/>

**Convert subgraph in TorchScript to MHLO**
```Python
mlir.cvt_torchscript_to_mhlo(graph)
```
This is a C API.
https://github.com/alibaba/BladeDISC/blob/58efe1a138cf32b11380710b51593642a0db18a4/pytorch_blade/pytorch_blade/compiler/mlir/pybind_functions.cpp#L26
<br/>


**Call backend to lower MHLO to binary**
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
<br/>


**Register codegened shared object as an Engine**<br/>
The `_backends` module:
```Python
from torch_blade._torch_blade import _backends
```
This is a C++ module:
https://github.com/alibaba/BladeDISC/blob/58efe1a138cf32b11380710b51593642a0db18a4/pytorch_blade/pytorch_blade/pybind.cpp#L100-L102
```C++
using namespace torch::blade::backends;
py::module backends =
    m.def_submodule("_backends", "torch_blade python bindings to backends");
```

```Python
state = _backends.EngineState()
state.inputs = [_backends.TensorInfo(inp) for inp in subgraph.inputs()]
state.outputs = [_backends.TensorInfo(out) for out in subgraph.outputs()]
state.engine_bytes = so_bytes
state.model_proto = pb_bytes
state.backend_name = mlir.backend_name()
debug_fallback = tools.read_bool_from_env(
    "PY_TORCH_BLADE_DEBUG_ENABLE_ERROR_FALLBACK", False
)
if debug_fallback:
    fallback_bytes = _subgraph_to_bytes(subgraph, attr_name)
else:
    fallback_bytes = ""

# register engine into module, something like:
# __torch__.torch.classes.torch_blade.Engine = prim::GetAttr[name="disc_grp0"](%self)
with c_module_lock:
    eng_type = _backends.register_engine(
        c_module,
        state,
        attr_name,
        fallback_bytes,
        str(subgraph),
    )
logger.info("{} COMPILE SUCCESS".format(attr_name))
```
<br/>

**Update graph: replace group (subgraph) with engine**<br/>
clustering/support_group_conversion.py:
```Python
def replace_group_with_engine(
    graph,
    module_holder,
    node,
    attr_name,
    eng_type,
    group_inputs=True,
    engine_method_name="execute",
):
    attr = graph.create("prim::GetAttr")
    attr.addInput(module_holder)
    attr.s_("name", attr_name)
    attr.output().setType(eng_type)
    graph.appendNode(attr)
    attr.moveBefore(node)

    if group_inputs:
        # create input_list of self.engine.execute, something like:
        # %12 : Tensor[] = prim::ListConstruct(%10, %11)
        list_constuct = graph.create("prim::ListConstruct")
        for inp in node.inputs():
            list_constuct.addInput(inp)
        list_constuct.output().setType(torch_blade.tools.get_list_tensor_type())
        graph.appendNode(list_constuct)
        list_constuct.moveBefore(node)

    # create prim::CallMethod, something like:
    call_method = graph.create("prim::CallMethod")
    call_method.s_("name", engine_method_name)
    call_method.addInput(attr.output())
    if group_inputs:
        # %5 : Tensor[] = prim::CallMethod[name="execute"](%3, %input_list)
        call_method.addInput(list_constuct.output())
    else:
        # %5 : Tensor[] = prim::CallMethod[name="execute"](%3, %input1, %input2, ...)
        for inp in node.input_list():
            call_method.addInput(inp)

    call_method.output().setType(torch_blade.tools.get_list_tensor_type())
    graph.appendNode(call_method)
    call_method.moveBefore(node)

    # create prim::ListUnpack, something like:
    # %17 : Tensor, %18 : Tensor, %19 : Tensor = prim::ListUnpack(%16)
    list_unpack = graph.create("prim::ListUnpack")
    list_unpack.addInput(call_method.output())
    list_unpack.eraseOutput(0)
    graph.appendNode(list_unpack)
    list_unpack.moveBefore(node)

    for out in node.outputs():
        lu_out = list_unpack.addOutput()
        lu_out.setType(out.type())
        out.replaceAllUsesWith(lu_out)

    # destory node
    node.destroy()
    return attr
```
Output graph:
```Python
graph(%self : __torch__.torch.fx.graph_module.___torch_mangle_0._lambda,
      %arg0_1.1 : Half(8, 10, strides=[10, 1], requires_grad=0, device=cuda:0),
      %arg1_1.1 : Half(8, 10, strides=[10, 1], requires_grad=0, device=cuda:0)):
  %14 : __torch__.torch.classes.torch_blade.Engine = prim::GetAttr[name="disc_grpcluster0_0_len8_0"](%self)
  %15 : Tensor[] = prim::ListConstruct(%arg0_1.1, %arg1_1.1)
  %16 : Tensor[] = prim::CallMethod[name="execute"](%14, %15)
  %18 : Half(requires_grad=0, device=cuda:0) = prim::ListUnpack(%16)
  %11 : (Half(requires_grad=0, device=cuda:0)) = prim::TupleConstruct(%18)
  return (%11)
```
<br/>

**Return compiled RecursiveScriptModule**
```Python
(Pdb) p f
RecursiveScriptModule(original_name=_lambda)

(Pdb) p f.graph
graph(%self : __torch__.torch.fx.graph_module.___torch_mangle_0._lambda,
      %arg0_1.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu),
      %arg1_1.1 : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
  %12 : __torch__.torch.classes.torch_blade.Engine = prim::GetAttr[name="disc_grpcluster0_0_len6_0"](%self)
  %13 : Tensor[] = prim::ListConstruct(%arg1_1.1, %arg0_1.1)
  %14 : Tensor[] = prim::CallMethod[name="execute"](%12, %13)
  %16 : Float(requires_grad=0, device=cpu) = prim::ListUnpack(%14)
  %9 : (Float(requires_grad=0, device=cpu)) = prim::TupleConstruct(%16)
  return (%9)
```
<br/>

Back to self.compile_and_call_fx_graph()<br/>
**Create a OptimizedModule**
```Python
compiled_fn = disable(compiled_fn)

(Pdb) p compiled_fn
OptimizedModule(
  (_orig_mod): RecursiveScriptModule(original_name=_lambda)
)
```

**??? Unknown ???**
```Python
self.install_global_unsafe(name, compiled_fn)
```
where name is `__compiled_fn_1`

**Make call generated code**</br>
torch/_dynamo/output_graph.py
```Python
cg = PyCodegen(tx)
cg.make_call_generated_code(name)
instructions = cg.get_instructions()
return instructions
```
This snippet generates instructions that call `__compiled_fn_1`.

**Save compiled code** <br/>
torch/_dynamo/convert_frame.py<br/>
We save compiled code into a AOT directory:
```Python
dynamo_ckpt = DynamoCheckpoint()
...
output = tracer.output
...
for name, compiled_fn_value in output.compiled_funcs.items():
    dynamo_ckpt.jit_save(name, compiled_fn_value)
dynamo_ckpt.dill_save(instructions, "__instructions")
dynamo_ckpt.dill_save(code_options, "__code_options")
```

**Return a GuardedCode object to the caller in `torch/csrc/dynamo/eval_frame.c`**
torch/_dynamo/convert_frame.py<br/>
```Python
guarded_code = GuardedCode(out_code, check_fn.check_fn)
```
The `guarded_code` will be evaluated and then return the result to the caller.<br/>
The normal Python execution path is resumed.


## Backend
disc_compiler_main: https://github.com/alibaba/BladeDISC/blob/58efe1a138cf32b11380710b51593642a0db18a4/tao_compiler/mlir/disc/disc_compiler_main.cc
<br/>


## Runtime (RAL)
RAL: Runtime Abstraction Layer
<br/>

**pytorch_blade/compiler/backends/engine_class.cpp**
register_engine -> create_engine -> EngineClass::EngineClass(SerialType serialized) -> EngineInterface::CreateEngine(engine_state)
<br/>

**Serving stage**</br>
We skip code tracing by customizing `torch/_dynamo/convert_frame.py`:
```Python
if dynamo_ckpt.keep_trace():
    out_code = transform_code_object(code, transform)
else:
    out_code = transform_code_object(code, custom_transform)
```

The `custom_transform` function will load compiled code from the AOT directory:
```Python
for co_name in output.code_options["co_names"]:
    if not dynamo_ckpt.exists(co_name):
        continue
    this_compiled_fn = dynamo_ckpt.load(co_name)
    output.install_global_unsafe(co_name, this_compiled_fn)
...
instructions[:] = dynamo_ckpt.load("__instructions")
code_options.update(dynamo_ckpt.load("__code_options"))
```
<br/>

Next, create a GuardedCode:
```Python
check_fn = CheckFunctionManager(
    output,
    hooks.guard_fail_fn if hooks else None,
)

guarded_code = GuardedCode(out_code, check_fn.check_fn)
```
Then, the `guarded_code` is returned to the caller in `torch/csrc/dynamo/eval_frame.c`, which will execute the `guarded_code` and return result to the caller of `self.forward()` function.
<br/>
