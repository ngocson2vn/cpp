# BACKLOG
```
# How to fuse more nodes?
fuse(producer, consumer)

# How does Scheduler codegen a FusedSchedulerNode to a Triton kernel?
What exactly is the programming model of a Triton kernel? <br/>
- Each Triton kernel has a constexpr blockSize parameter which must be specified at compile time
- Each Triton kernel instance operates on a block of data. All operations are vectorize-wise.
```

### 3. Apply the gained knowledge to a custom fusion and lowering process


# Changes
```Python
# MLCompiler/pytorch/torch/_inductor/utils.py
def expr_fits_within_32bit(e: sympy.Expr) -> bool:
    if len(e.free_symbols) > 0 and config.relax_numel_check:
        return True


```
