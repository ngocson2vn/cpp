# GPU arch
```Bash
export TRITON_OVERRIDE_ARCH=sm86
```

# PTX version
```Python
# /data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python/triton/backends/nvidia/compiler.py
class CUDABackend(BaseBackend):
    def parse_options(self, opts) -> Any:
        ...
        # PTX version
        try:
            ptx_version = int(os.getenv("TRITON_OVERRIDE_PTX_VERSION", "unset"))
            args["ptx_version"] = ptx_version
        except:
            pass

        return CUDAOptions(**args)

```

# Errors
```Python
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] Triton compilation failed: Placeholder.DESCRIPTIVE_NAME
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel_0, XBLOCK : tl.constexpr):
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     pid = tl.program_id(0)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_0 = xnumel_0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_1 = num_xblocks_0 + tl.cdiv(456704, XBLOCK)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_2 = num_xblocks_1 + tl.cdiv(4096, XBLOCK)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     if pid < num_xblocks_0:
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         XBLOCK_0: tl.constexpr = 1
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 285
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         RBLOCK_0: tl.constexpr = 512
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         R0_BLOCK_0: tl.constexpr = 512
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         rnumel = r0_numel
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         RBLOCK: tl.constexpr = R0_BLOCK_0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK_0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = tl.full([1], xoffset, tl.int32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_index = tl.arange(0, R0_BLOCK_0)[:]
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_offset = 0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_mask = r0_index < r0_numel
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         roffset = r0_offset
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         rindex = r0_index
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_1 = r0_index
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x0 = xindex
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp0 = 284 + ((-1)*r0_1)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp1 = tl.full([1], 0, tl.int64)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp2 = tmp0 >= tmp1
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp3 = tl.full([1], 31, tl.int64)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp4 = tmp0 < tmp3
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp5 = 1.0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp7 = tl.where(tmp4, tmp5, tmp6)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp8 = tmp0 >= tmp3
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp9 = tl.full([1], 285, tl.int64)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp10 = tmp0 < tmp9
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp11 = tl.load(in_ptr0 + (254*x0 + (253 + ((-1)*r0_1))), xmask & r0_mask & tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp12 = 1.0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp13 = tmp11 == tmp12
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp14 = tmp13.to(tl.float32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp16 = tl.where(tmp8, tmp14, tmp15)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp17 = tl.where(tmp4, tmp7, tmp16)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp18 = tmp17.to(tl.float32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp19 = tl.broadcast_to(tmp18, [R0_BLOCK_0])
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp20, = tl.associative_scan((tmp19,), 0, _triton_helper_fn_add0)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr0 + (r0_1 + 285*x0), tmp20, xmask & r0_mask)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_1:
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xnumel = 456704
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x3 = xindex // 256
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x4 = xindex
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp21 = x3
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp22 = tl.full([1], 1781, tl.int64)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp23 = tmp21 < tmp22
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp24 = tl.load(in_ptr1 + (x4), xmask & tmp23, other=0.0).to(tl.float32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr1 + (x4), tmp24, xmask)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_2:
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_1
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xnumel = 4096
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x6 = xindex // 256
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x7 = xindex
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp25 = x6
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp26 = tl.full([1], 13, tl.int64)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp27 = tmp25 < tmp26
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp28 = tl.load(in_ptr2 + (x7), xmask & tmp27, other=0.0).to(tl.float32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr2 + (x7), tmp28, xmask)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     else:
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pass
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] 
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] metadata: {'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel_0': 'i32'}, 'device': 0, 'constants': {'XBLOCK': 1024}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'device_type': 'cuda', 'num_warps': 4, 'num_stages': 1, 'debug': True, 'cc': 89}
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] Traceback (most recent call last):
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]   File "/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 551, in _precompile_config
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     binary = triton.compile(*compile_args, **compile_kwargs)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]   File "/usr/local/lib/python3.11/dist-packages/triton/compiler/compiler.py", line 273, in compile
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     module = src.make_ir(options, codegen_fns, module_map, context)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]   File "/usr/local/lib/python3.11/dist-packages/triton/compiler/compiler.py", line 100, in make_ir
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] triton.compiler.errors.CompilationError: at 44:50:
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp9 = tl.full([1], 285, tl.int64)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp10 = tmp0 < tmp9
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp11 = tl.load(in_ptr0 + (254*x0 + (253 + ((-1)*r0_1))), xmask & r0_mask & tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp12 = 1.0
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp13 = tmp11 == tmp12
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp14 = tmp13.to(tl.float32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp16 = tl.where(tmp8, tmp14, tmp15)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp17 = tl.where(tmp4, tmp7, tmp16)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp18 = tmp17.to(tl.float32)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp19 = tl.broadcast_to(tmp18, [R0_BLOCK_0])
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp20, = tl.associative_scan((tmp19,), 0, _triton_helper_fn_add0)
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]                                                   ^
E0829 03:48:34.301000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] NameError('_triton_helper_fn_add0 is not defined')
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] Triton compilation failed: Placeholder.DESCRIPTIVE_NAME
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel_0, xnumel_1, xnumel_2, xnumel_3, xnumel_4, xnumel_5, xnumel_6, xnumel_7, XBLOCK : tl.constexpr):
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     pid = tl.program_id(0)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_2 = num_xblocks_1 + tl.cdiv(xnumel_2, XBLOCK)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_3 = num_xblocks_2 + tl.cdiv(xnumel_3, XBLOCK)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_4 = num_xblocks_3 + xnumel_4
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_5 = num_xblocks_4 + tl.cdiv(xnumel_5, XBLOCK)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_6 = num_xblocks_5 + tl.cdiv(xnumel_6, XBLOCK)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     num_xblocks_7 = num_xblocks_6 + tl.cdiv(xnumel_7, XBLOCK)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     if pid < num_xblocks_0:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_0
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x0 = xindex
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp0 = tl.load(in_ptr0 + (x0), xmask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp1 = tmp0.to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr0 + (x0), tmp1, xmask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_1:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_0
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x3 = xindex
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x1 = (xindex % 12800)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x2 = xindex // 12800
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp2 = tl.load(in_ptr1 + (x3), xmask).to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr1 + (x1 + 65024*x2), tmp2, xmask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_2:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_2
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x6 = xindex
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x4 = (xindex % 12800)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x5 = xindex // 12800
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp3 = tl.load(in_ptr2 + (x6), xmask).to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr2 + (x4 + 65024*x5), tmp3, xmask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_3:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_2
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_3
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x7 = (xindex % 285)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x8 = xindex // 285
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x9 = xindex
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp4 = tl.load(in_ptr3 + (284 + ((-1)*x7) + 285*x8), xmask, eviction_policy='evict_last').to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp5 = 0.0
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp6 = tmp4 > tmp5
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp7 = 1.0
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp8 = tl.where(tmp6, tmp7, tmp5)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr3 + (x9), tmp8, xmask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_4:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_3
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         XBLOCK_4: tl.constexpr = 1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 346
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         RBLOCK_4: tl.constexpr = 512
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         R0_BLOCK_4: tl.constexpr = 512
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         rnumel = r0_numel
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         RBLOCK: tl.constexpr = R0_BLOCK_4
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK_4
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = tl.full([1], xoffset, tl.int32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_4
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_index = tl.arange(0, R0_BLOCK_4)[:]
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_offset = 0
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_mask = r0_index < r0_numel
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         roffset = r0_offset
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         rindex = r0_index
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_11 = r0_index
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x10 = xindex
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp9 = 345 + ((-1)*r0_11)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp10 = tl.full([1], 0, tl.int64)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp11 = tmp9 >= tmp10
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp12 = tl.full([1], 31, tl.int64)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp13 = tmp9 < tmp12
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp14 = 1.0
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp16 = tl.where(tmp13, tmp14, tmp15)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp17 = tmp9 >= tmp12
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp18 = tl.full([1], 346, tl.int64)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp19 = tmp9 < tmp18
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp20 = tl.load(in_ptr4 + (315*x10 + (314 + ((-1)*r0_11))), xmask & r0_mask & tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp21 = 1.0
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp22 = tmp20 == tmp21
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp23 = tmp22.to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp25 = tl.where(tmp17, tmp23, tmp24)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp26 = tl.where(tmp13, tmp16, tmp25)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp27 = tmp26.to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp28 = tl.broadcast_to(tmp27, [R0_BLOCK_4])
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp29, = tl.associative_scan((tmp28,), 0, _triton_helper_fn_add0)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr4 + (r0_11 + 346*x10), tmp29, xmask & r0_mask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_5:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_4
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_5
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x14 = xindex
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x12 = (xindex % 25600)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x13 = xindex // 25600
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp30 = tl.load(in_ptr5 + (x14), xmask).to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr5 + (x12 + 80640*x13), tmp30, xmask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_6:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_5
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_6
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x17 = xindex
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x15 = (xindex % 12800)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x16 = xindex // 12800
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp31 = tl.load(in_ptr6 + (x17), xmask).to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr6 + (x15 + 80640*x16), tmp31, xmask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     elif pid < num_xblocks_7:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pid_offset = pid - num_xblocks_6
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         r0_numel = 1
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xoffset = pid_offset * XBLOCK
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xindex = xoffset + tl.arange(0, XBLOCK)[:]
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         xmask = xindex < xnumel_7
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x20 = xindex
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x18 = (xindex % 25600)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         x19 = xindex // 25600
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp32 = tl.load(in_ptr7 + (x20), xmask).to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tl.store(out_ptr7 + (x18 + 80640*x19), tmp32, xmask)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     else:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         pass
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] 
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] metadata: {'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'out_ptr5': '*bf16', 'out_ptr6': '*bf16', 'out_ptr7': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'xnumel_2': 'i32', 'xnumel_3': 'i32', 'xnumel_4': 'i32', 'xnumel_5': 'i32', 'xnumel_6': 'i32', 'xnumel_7': 'i32'}, 'device': 0, 'constants': {'XBLOCK': 1024}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})], 'device_type': 'cuda', 'num_warps': 4, 'num_stages': 1, 'debug': True, 'cc': 89}
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] Traceback (most recent call last):
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]   File "/usr/local/lib/python3.11/dist-packages/torch/_inductor/runtime/triton_heuristics.py", line 551, in _precompile_config
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     binary = triton.compile(*compile_args, **compile_kwargs)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]   File "/usr/local/lib/python3.11/dist-packages/triton/compiler/compiler.py", line 273, in compile
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     module = src.make_ir(options, codegen_fns, module_map, context)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]   File "/usr/local/lib/python3.11/dist-packages/triton/compiler/compiler.py", line 100, in make_ir
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]     return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] triton.compiler.errors.CompilationError: at 96:50:
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp18 = tl.full([1], 346, tl.int64)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp19 = tmp9 < tmp18
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp20 = tl.load(in_ptr4 + (315*x10 + (314 + ((-1)*r0_11))), xmask & r0_mask & tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp21 = 1.0
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp22 = tmp20 == tmp21
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp23 = tmp22.to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp25 = tl.where(tmp17, tmp23, tmp24)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp26 = tl.where(tmp13, tmp16, tmp25)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp27 = tmp26.to(tl.float32)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp28 = tl.broadcast_to(tmp27, [R0_BLOCK_4])
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]         tmp29, = tl.associative_scan((tmp28,), 0, _triton_helper_fn_add0)
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1]                                                   ^
E0829 03:48:52.463000 1991 torch/_inductor/runtime/triton_heuristics.py:553] [0/1] NameError('_triton_helper_fn_add0 is not defined')
```