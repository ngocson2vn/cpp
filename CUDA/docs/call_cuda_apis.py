import ctypes
try:
    libcudart = ctypes.CDLL('libcudart.so')
    device_id = 0
    result = libcudart.cudaSetDevice(0)
    if result == 0:
        print(f"Successfully called cudaSetDevice({device_id}) via ctypes.")
    else:
        print(f"cudaSetDevice failed with CUDA error code: {result}")
except OSError as e:
    print(f"Could not find or load libcudart.so: {e}")
