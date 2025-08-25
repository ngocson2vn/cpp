import importlib.util, importlib.machinery as m

spec = importlib.util.find_spec("torch._C")
print(spec.origin)       # path to torch/_C.cpython-311-x86_64-linux-gnu.so
print(spec.loader)       # _frozen_importlib_external.ExtensionFileLoader

print()
print(f"EXTENSION_SUFFIXES: {m.EXTENSION_SUFFIXES}")
