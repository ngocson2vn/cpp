# Pdb and Autocomplete
Create ~/.pdbrc
```Python
# this adds tab completion
import rlcompleter
__import__('pdb').Pdb.complete = rlcompleter.Completer(locals()).complete
```

# pip install without cache
```Bash
pip3 install --no-cache-dir -I bytedlogid==0.2.1
```

# Update requirements of a wheel file
```Bash
wheel unpack ./file_name.whl

# Update file_name/file_name.dist-info/METADATA

wheel pack ./file_name
```

# Debugging
Using debugpy: [Example Debug Inductor](../.vscode/launch.json)
```json
        {
            "name": "Debug Inductor",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceRoot}/MLCompiler/pytorch/test_inductor.py",
            "cwd": "${workspaceRoot}/MLCompiler/pytorch",
            "env": {
                "PYTHONPATH": "/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python",
                "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64"
            }
        }
```

# Python 3.11
```
export PIP_BREAK_SYSTEM_PACKAGES=1
pip3 install/uninstall PACKAGE_NAME
```

# Function representation
<function make_pointwise.<locals>.inner at 0x7fb9f4eb6d40>
This tells you: “This is a function named `inner`, defined locally inside `make_pointwise`, and this specific function object currently lives at memory address 0x7fb9f4eb6d40.”
```Python
def make_pointwise(f):
    def inner(x):
        return f(x)
    return inner

g = make_pointwise(lambda x: x + 1)

print(g)                  # -> <function make_pointwise.<locals>.inner at 0x...>
print(g.__name__)         # -> inner
print(g.__qualname__)     # -> make_pointwise.<locals>.inner
print(g.__module__)       # -> __main__ (or the module name)
print(id(g))              # -> matches the hex address (in decimal form)
```

# Special separators in a function parameter list
In Python, the `/` and `*` separators in a function parameter list define how arguments can be passed to a function, specifically distinguishing between positional and keyword arguments. Here's a concise explanation:

- **`/`: Positional-only separator**  
  - Parameters before `/` must be passed as positional arguments (i.e., without specifying parameter names).
  - Example:
    ```python
    def func(a, b, /):
        return a + b
    func(1, 2)  # Valid
    func(a=1, b=2)  # Error: parameters before / cannot be keyword arguments
    ```
  - Purpose: Enforces positional argument passing, useful for clarity or when parameter names are irrelevant (e.g., in C-like APIs).

- **`*`: Keyword-only separator**  
  - Parameters after `*` must be passed as keyword arguments (i.e., using the parameter names).
  - Example:
    ```python
    def func(a, *, b, c):
        return a + b + c
    func(1, b=2, c=3)  # Valid
    func(1, 2, 3)  # Error: parameters after * must be keyword arguments
    ```
  - Purpose: Enforces keyword argument passing, improving readability and avoiding ambiguity when positional arguments could be confusing.

- **Combined usage**:  
  You can use both `/` and `*` in the same function to define positional-only, positional-or-keyword, and keyword-only parameters:
  ```python
  def func(a, b, /, c, *, d, e):
      return a + b + c + d + e
  func(1, 2, 3, d=4, e=5)  # Valid: a, b positional-only; c positional or keyword; d, e keyword-only
  ```

- **Key points**:
  - `/` was introduced in Python 3.8 (PEP 570).
  - `*` is used to mark the end of positional parameters and the start of keyword-only parameters.
  - Parameters between `/` and `*` (if both are used) can be passed either positionally or by keyword.

This syntax provides flexibility and clarity in function definitions, ensuring arguments are passed as intended.

# Decorator
```python
@functools.wraps(decomp_fn)
def wrapped(*args, **kwargs):
    # omit for brevity
```
The provided Python code snippet shows a **decorator** pattern using the `@functools.wraps` function from the `functools` module. Let’s break it down step by step:

### Code Explanation
1. **Decorator with `@functools.wraps`**:
   - The `@functools.wraps(decomp_fn)` is a decorator applied to the `wrapped` function. 
   - `functools.wraps` is a utility from the `functools` module that helps create well-behaved decorators by preserving the metadata (e.g., name, docstring, and other attributes) of the original function (`decomp_fn`) being wrapped.
   - Without `functools.wraps`, the `wrapped` function would lose the metadata of `decomp_fn`, and tools like debuggers or documentation generators might show the metadata of `wrapped` instead, which can be confusing.

2. **`decomp_fn`**:
   - `decomp_fn` is the original function that the decorator is wrapping. This is the function being modified or extended by the decorator.
   - The decorator (`wrapped`) will typically call `decomp_fn` somewhere in its body (though the body is omitted here for brevity).

3. **`wrapped` Function**:
   - The `wrapped` function is the wrapper function that replaces `decomp_fn` when the decorator is applied.
   - It accepts `*args` and `**kwargs`, which are Python’s way of handling variable positional and keyword arguments, respectively. This makes the decorator flexible, as it can wrap functions with any number of arguments.

4. **Purpose of the Decorator**:
   - Decorators are used to modify or extend the behavior of a function without changing its source code. For example, the `wrapped` function might:
     - Add pre- or post-processing logic.
     - Perform checks or logging.
     - Modify the input or output of `decomp_fn`.
   - Since the body is omitted (`# omit for brevity`), we don’t know the specific behavior, but it typically involves calling `decomp_fn(*args, **kwargs)` at some point to execute the original function.

5. **Why Use `functools.wraps`?**:
   - When you write a decorator, the wrapper function (`wrapped`) replaces the original function (`decomp_fn`). This can cause the original function’s metadata (like its `__name__`, `__doc__`, or `__module__`) to be replaced by the wrapper’s metadata.
   - `functools.wraps(decomp_fn)` ensures that the `wrapped` function copies the metadata of `decomp_fn`, making it appear as if the original function is still being used. For example:
     ```python
     def decomp_fn(x):
         """Original docstring."""
         return x

     @functools.wraps(decomp_fn)
     def wrapped(*args, **kwargs):
         return decomp_fn(*args, **kwargs)

     print(wrapped.__name__)  # Outputs: 'decomp_fn' (not 'wrapped')
     print(wrapped.__doc__)   # Outputs: 'Original docstring.'
     ```

### Example of a Complete Decorator

To illustrate how this might work in practice, here’s an example of what the code might look like with a complete implementation:

```python
import functools

def my_decorator(decomp_fn):
    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        print(f"Calling {decomp_fn.__name__} with args={args}, kwargs={kwargs}")
        result = decomp_fn(*args, **kwargs)
        print(f"Finished calling {decomp_fn.__name__}")
        return result
    return wrapped

@my_decorator
def example_function(x, y=10):
    """This is an example function."""
    return x + y

# Usage
result = example_function(5, y=20)
print(result)
```

**Output**:
```
Calling example_function with args=(5,), kwargs={'y': 20}
Finished calling example_function
25
```

### Key Points
- The `@functools.wraps(decomp_fn)` ensures the `wrapped` function retains the metadata of `decomp_fn`.
- The `wrapped` function is a generic wrapper that can handle any arguments passed to `decomp_fn` via `*args` and `**kwargs`.
- The actual behavior of the decorator depends on the code inside `wrapped`, which is omitted in your snippet.
- Decorators are commonly used for logging, timing, access control, or modifying function behavior.

If you have the omitted code or more context about what the decorator does, I can provide a more specific explanation!