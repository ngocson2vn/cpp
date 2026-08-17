# del object
In Python, given `buf16` holding a reference to a Python object. Next, perform the following assignment operation:
```python
buf18 = buf16
```
Now both `buf16` and `buf18` hold references to the same object. Will `del buf16` destroy the underlying object?
```python
del buf16
```

**No.** `del buf16` will **not** destroy the underlying object.

### Why

In Python, names (variables) are just references/bindings to objects. The object itself lives as long as its reference count is greater than zero.

1. `buf18 = buf16`  
   Creates a *second* reference to the exact same object.  
   Reference count of the object: +1

2. `del buf16`  
   Only removes the name `buf16` from the current namespace.  
   Reference count of the object: −1

Because `buf18` still points to the object, the reference count remains ≥ 1, so the object is **not** deallocated / garbage-collected.

### Minimal demonstration

```python
class Track:
    def __del__(self):
        print("object destroyed")

buf16 = Track()
buf18 = buf16
del buf16          # nothing is printed
print("buf18 still alive:", buf18 is not None)

del buf18          # now the object is destroyed
```

Output:

```
buf18 still alive: True
object destroyed
```

Only when the *last* remaining reference disappears does the object get destroyed.