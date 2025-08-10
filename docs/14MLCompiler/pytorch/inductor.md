# Call Stack
```Python
Traceback (most recent call last):
  File "/data00/home/son.nguyen/workspace/cpp/docs/14MLCompiler/pytorch/test_inductor.py", line 20, in <module>
    res = toy(x, y)
          ^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 749, in compile_wrapper
    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/output_graph.py", line 1871, in _call_user_compiler
    raise BackendCompilerFailed(
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/output_graph.py", line 1846, in _call_user_compiler
    compiled_fn = compiler_fn(gm, example_inputs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/repro/after_dynamo.py", line 150, in __call__
    compiled_gm = compiler_fn(gm, example_inputs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/__init__.py", line 2380, in __call__
    return compile_fx(model_, inputs_, config_patches=self.config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 2418, in compile_fx
    return aot_autograd(
           ^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/backends/common.py", line 109, in __call__
    cg = aot_module_simplified(gm, example_inputs, **self.kwargs)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 1199, in aot_module_simplified
    compiled_fn = AOTAutogradCache.load(
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/autograd_cache.py", line 1140, in load
    compiled_fn = dispatch_and_compile()
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 1184, in dispatch_and_compile
    compiled_fn, _ = create_aot_dispatcher_function(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 576, in create_aot_dispatcher_function
    return _create_aot_dispatcher_function(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 836, in _create_aot_dispatcher_function
    compiled_fn, fw_metadata = compiler_fn(
                               ^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py", line 245, in aot_dispatch_base
    compiled_fw = compiler(fw_module, updated_flat_args)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 483, in __call__
    return self.compiler_fn(gm, example_inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 2250, in fw_compiler_base
    return inner_compile(
           ^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 745, in compile_fx_inner
    return wrap_compiler_debug(_compile_fx_inner, compiler_name="inductor")(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_dynamo/repro/after_aot.py", line 124, in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/compile_fx.py", line 860, in _compile_fx_inner
    (key_info, cache_info) = FxGraphCache.prepare_key(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codecache.py", line 1474, in prepare_key
    key, debug_lines = compiled_fx_graph_hash(
                       ^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codecache.py", line 960, in compiled_fx_graph_hash
    details = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codecache.py", line 896, in __init__
    self.system_info = CacheBase.get_system()
                       ^^^^^^^^^^^^^^^^^^^^^^
  File "/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/_inductor/codecache.py", line 205, in get_system
    from triton.compiler.compiler import triton_key
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler' (/data00/home/son.nguyen/workspace/triton_dev/triton/python/triton/compiler/compiler.py)

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```
