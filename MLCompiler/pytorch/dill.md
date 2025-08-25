# dill dump and load
```Python
class SerializableCallable(torch.nn.Module):
    def __init__(self, compiled_fn):
        super().__init__()
        self.compiled_fn = compiled_fn

    def forward(self, *runtime_args):
        full_args = []
        full_args.extend(params_flat)
        full_args.extend(runtime_args)
        return self.compiled_fn(full_args)

    @staticmethod
    def serialize(object):
        mod = copy.copy(object)
        mod.current_callable = None
        return (mod,)

    @staticmethod
    def deserialize(compiled_fn):
        # breakpoint()
        path = get_path(compiled_fn.cache_key, "py")[2]
        compiled_fn.current_callable = PyCodeCache.load_by_key_path(
            compiled_fn.cache_key,
            path,
            compiled_fn.cache_linemap,
            compiled_fn.constants,
        ).call
        return SerializableCallable(compiled_fn)

    def __reduce__(self):
        return (SerializableCallable.deserialize, SerializableCallable.serialize(self.compiled_fn))
```

#### Typical sequence during pickling
1. You call `pickle.dumps(obj)` (or `dill.dumps(obj)`).
2. The pickler determines the protocol and invokes:
  - `obj.__reduce_ex__(protocol)` if present, else
  - `obj.__reduce__()`.
3. The returned tuple (e.g., `(callable, args, state, list_iter, dict_iter, buffer_iter)`) is serialized.

#### Typical sequence during unpickling (for context)
- The unpickler reads the tuple produced by `__reduce__` and executes the “recipe”:
  - Calls `callable(*args)` to create the object.
  - Applies `state` (via `__setstate__` if present, otherwise by default rules).
  - Iterates `list_iter`/`dict_iter` to populate containers, if provided.
- Note: No `__reduce__` is called here; only the reconstruction callable/state are used.
