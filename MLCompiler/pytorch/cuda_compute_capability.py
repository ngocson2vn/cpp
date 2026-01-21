import pycuda.driver as drv
import pycuda.autoinit # This imports autoinit to initialize CUDA context automatically

print('Detected {} CUDA Capable device(s)\n'.format(drv.Device.count()))

i = 0
gpu_device = drv.Device(i)
print('Device {}: {}'.format(i, gpu_device.name()))

# Get the major and minor compute capability versions
major, minor = gpu_device.compute_capability()

# Format as a float (e.g., 8.6)
compute_capability = float('%d.%d' % (major, minor))

print('\t Compute Capability: {}'.format(compute_capability))
print('\t Total Memory: {} megabytes'.format(gpu_device.total_memory() // (1024**2)))
